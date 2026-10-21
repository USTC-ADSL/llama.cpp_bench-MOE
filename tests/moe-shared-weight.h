#pragma once
#include "ggml-shared-weight.h"

// MoE is a consumer of the model-independent 2D weight middleware. A bank
// contains all experts once; retained row slices are metadata, not copies.
struct moe_shared_weights {
    using handle = std::unique_ptr<ggml_shared_weight, decltype(&ggml_shared_weight_free)>;
    std::vector<handle> owners, slices;
    std::array<std::array<std::array<ggml_tensor *,3>,61>,3> weights{};
    ggml_context_ptr scalar_ctx = context(3);
    std::unique_ptr<shared_activation_buffer> scalar_storage;
    std::array<ggml_backend_buffer_ptr,3> scalar_imports;
    std::array<ggml_tensor *,3> scalar{};
    size_t payload_bytes=0, physical_bytes=0;
    int hidden=0, shared_hidden=0;

    moe_shared_weights(device * devs[3],source & src,int layer,
                       const resident_placement_model::availability & resident) {
        const char * parts[]={"gate","up","down"};
        for(int shared=0;shared<2;++shared) { for(int part=0;part<3;++part) {
            auto name="blk."+std::to_string(layer)+".ffn_"+parts[part]+(shared?"_shexp.weight":"_exps.weight");
            auto * t=src.tensor(name);
            require(t->type==GGML_TYPE_Q4_0 && t->ne[3]==1 && t->ne[2]==(shared?1:60),"shared MoE weight shape/type");
            if(part==0) { (shared?shared_hidden:hidden)=int(t->ne[1]); }
            auto raw=src.read(name);
            handle owner(ggml_shared_weight_create_with_layout(devs[1]->backend.get(),t->ne[0],t->ne[1]*t->ne[2],
                         raw.data(),raw.size(),GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2),ggml_shared_weight_free);
            require(bool(owner),"shared MoE bank creation failed");
            ggml_shared_weight_desc desc{}; require(ggml_shared_weight_describe(owner.get(),&desc),"bank descriptor");
            payload_bytes+=raw.size(); physical_bytes+=desc.storage.size;
            for(int e=0;e<t->ne[2];++e) {
                handle slice(ggml_shared_weight_slice_rows(owner.get(),e*t->ne[1],t->ne[1]),ggml_shared_weight_free);
                require(bool(slice),"shared MoE expert slice failed");
                ggml_shared_weight_desc sd{}; require(ggml_shared_weight_describe(slice.get(),&sd),"slice descriptor");
                require(sd.storage.host_ptr==desc.storage.host_ptr && sd.storage.dma_buf_fd==desc.storage.dma_buf_fd &&
                        sd.offset==size_t(e)*t->nb[2],"expert slice does not alias owner");
                for(int b=0;b<3;++b) { if(shared?resident.shared[b]:resident.experts[e][b]) {
                    auto * bound=ggml_shared_weight_bind(slice.get(),devs[b]->backend.get());
                    require(bound && bound->data==static_cast<uint8_t *>(desc.storage.host_ptr)+sd.offset,
                            "backend weight does not alias shared storage");
                    weights[b][shared?60:e][part]=bound;
                } }
                slices.push_back(std::move(slice));
            }
            owners.push_back(std::move(owner));
        } }
        // The 2048x1 F16 shared gate is already a common physical format. Its
        // one small DMA allocation is aliased by eligible backends unchanged.
        auto name="blk."+std::to_string(layer)+".ffn_gate_inp_shexp.weight";
        auto * t=src.tensor(name); auto raw=src.read(name);
        require(t->type==GGML_TYPE_F16 && t->ne[0]==2048 && t->ne[1]==1,"shared gate shape/type");
        scalar_storage=std::make_unique<shared_activation_buffer>(*devs[1],raw.size());
        std::memcpy(scalar_storage->desc.host_ptr,raw.data(),raw.size());
        payload_bytes+=raw.size(); physical_bytes+=scalar_storage->desc.size;
        for(int b=0;b<3;++b) { if(resident.shared[b]) {
            auto buffer=scalar_storage->get();
            if(b==0) { scalar_imports[b].reset(ggml_backend_cpu_buffer_from_ptr(scalar_storage->desc.host_ptr,scalar_storage->desc.size)); }
            if(b==2) { auto import=shared_activation_buffer::gpu_import(*devs[b]); require(import,"GPU scalar import missing");
                       scalar_imports[b].reset(import(devs[b]->backend.get(),&scalar_storage->desc)); }
            if(b!=1) { require(bool(scalar_imports[b]),"scalar alias import failed"); buffer=scalar_imports[b].get(); }
            scalar[b]=ggml_dup_tensor(scalar_ctx.get(),t);
            require(ggml_backend_tensor_alloc(buffer,scalar[b],scalar_storage->desc.host_ptr)==GGML_STATUS_SUCCESS,"scalar alias bind failed");
        } }
    }
};

struct moe_shared_job {
    device & dev;
    ggml_context_ptr ctx=context(600,512);
    ggml_cgraph * graph=nullptr;
    std::array<ggml_tensor *,61> inputs{}, outputs{};
    std::shared_ptr<shared_activation_buffer> arena;
    moe_shared_job(device & d,moe_shared_weights & weights,int backend,const std::array<int,61> & counts,
                   std::shared_ptr<shared_activation_buffer> storage) : dev(d),arena(std::move(storage)) {
        graph=ggml_new_graph_custom(ctx.get(),512,false);
        for(int e=0;e<61;++e) { if(counts[e]) {
            const auto & w=weights.weights[backend][e];
            require(w[0] && w[1] && w[2],"shared expert not bound on selected backend");
            auto * x=inputs[e]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,w[0]->ne[0],counts[e]);
            auto * u=ggml_mul_mat(ctx.get(),w[1],x);
            auto * g=ggml_mul_mat(ctx.get(),w[0],x);
            auto * out=ggml_mul_mat(ctx.get(),w[2],ggml_swiglu_split(ctx.get(),g,u));
            if(e==60) {
                auto * r=ggml_mul_mat(ctx.get(),weights.scalar[backend],x);
                out=ggml_mul(ctx.get(),out,ggml_div(ctx.get(),ggml_silu(ctx.get(),r),r));
            }
            outputs[e]=out; ggml_set_output(out);
            ggml_build_forward_expand(graph,out);
        } }
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * node=ggml_graph_node(graph,i);
            if(!ggml_backend_supports_op(dev.backend.get(),node)) {
                fprintf(stderr,"shared MoE unsupported %s %s\n",dev.name.c_str(),ggml_op_name(node->op));
                throw std::runtime_error("unsupported shared MoE graph");
            }
        }
        require(bool(arena),"shared MoE arena missing");
        // All inputs survive to their last consumers, all outputs to host
        // reduction. Intermediate projections reuse closed-lifetime regions.
        arena->bind(dev,ctx.get(),graph);
    }
};

struct moe_shared_execution {
    std::array<device *,3> devs;
    moe_shared_weights weights;
    std::array<std::shared_ptr<shared_activation_buffer>,3> arenas;
    std::array<std::map<std::array<int,61>,std::unique_ptr<moe_shared_job>>,3> jobs;
    moe_shared_execution(device * d[3],source & src,int layer,const resident_placement_model::availability & resident)
        : devs{d[0],d[1],d[2]},weights(d,src,layer,resident) {}

    std::vector<float> run(const float * x,const float * probs,const std::vector<int32_t> & ids,
                          const placement & plan,float * dst,size_t & allocations,json & last,int il,
                          const std::vector<int32_t> & expected) {
        const int nt=int(ids.size()/4); auto start=timer::now();
        std::array<std::vector<int>,60> pairs;
        std::array<std::array<int,61>,3> counts{};
        std::array<size_t,3> branch_pairs{};
        for(size_t p=0;p<ids.size();++p) {
            int e=ids[p]; require(e>=0 && e<60,"shared route ID out of range");
            pairs[e].push_back(int(p)); ++counts[plan.owner[e]][e]; ++branch_pairs[plan.owner[e]];
        }
        counts[plan.shared][60]=nt;
        moe_shared_job * work[3]{};
        for(int b=0;b<3;++b) { if(branch_pairs[b] || b==plan.shared) {
            auto & job=jobs[b][counts[b]];
            if(!job) { job=std::make_unique<moe_shared_job>(*devs[b],weights,b,counts[b],arenas[b]); ++allocations; }
            work[b]=job.get();
        } }
        const double gather=elapsed(start); start=timer::now();
        const double origin=std::chrono::duration<double,std::micro>(start.time_since_epoch()).count();
        std::array<double,3> branch_gather{}; std::array<size_t,3> copies{};
        std::array<std::array<double,3>,3> phases{}; std::array<std::array<double,2>,3> intervals{};
        auto execute=[&](int b) {
            intervals[b][0]=elapsed(start); auto begin=timer::now();
            auto & j=*work[b];
            for(int e=0;e<60;++e) { if(j.inputs[e]) {
                auto * input=static_cast<float *>(j.inputs[e]->data);
                for(size_t i=0;i<pairs[e].size();++i) { std::copy_n(x+(pairs[e][i]/4)*2048,2048,input+i*2048); }
                copies[b]+=pairs[e].size()*2048*sizeof(float);
            } }
            if(j.inputs[60]) { std::copy_n(x,nt*2048,static_cast<float *>(j.inputs[60]->data)); copies[b]+=nt*2048*sizeof(float); }
            branch_gather[b]=elapsed(begin); begin=timer::now();
            devs[b]->compute(j.graph); phases[b][1]=elapsed(begin); intervals[b][1]=elapsed(start);
        };
        std::future<void> futures[3]; int inline_b=-1;
        for(int b=0;b<3;++b) { if(work[b]) { inline_b=b; } }
        for(int b=0;b<3;++b) { if(work[b] && b!=inline_b) { futures[b]=std::async(std::launch::async,execute,b); } }
        execute(inline_b); for(auto & f:futures) { if(f.valid()) { f.get(); } }
        const double wall=elapsed(start); start=timer::now();
        std::vector<const float *> outputs(ids.size());
        for(int e=0;e<60;++e) { if(!pairs[e].empty()) {
            auto * out=static_cast<float *>(work[plan.owner[e]]->outputs[e]->data);
            for(size_t i=0;i<pairs[e].size();++i) { outputs[pairs[e][i]]=out+i*2048; }
        } }
        std::vector<float> result; if(!dst) { result.resize(nt*2048); dst=result.data(); }
        auto * shared=static_cast<float *>(work[plan.shared]->outputs[60]->data);
        for(int t=0;t<nt;++t) { for(int c=0;c<2048;++c) {
            float sum=0;
            for(int k=0;k<4;++k) { const int p=t*4+k; sum=std::fma(probs[t*60+ids[p]],outputs[p][c],sum); }
            dst[t*2048+c]=sum+shared[t*2048+c];
        } }
        std::array<int,60> totals{}; for(int e:ids) { ++totals[e]; }
        int changed=0; if(expected.size()==ids.size()) { for(size_t i=0;i<ids.size();++i) { changed+=ids[i]!=expected[i]; } }
        int active=0; for(auto n:totals) { active+=n>0; }
        last={{"layer",il},{"tokens",nt},{"gather_us",gather},{"expert_wall_us",wall},{"combine_us",elapsed(start)},
              {"backend_phases_us",phases},{"intervals_us",intervals},{"branch_origin_monotonic_us",origin},
              {"pairs",branch_pairs},{"changed_route_slots",changed},{"route_reference_available",expected.size()==ids.size()},
              {"reduce_threads",1},{"overlap_gather",true},{"backend_gather_us",branch_gather},{"expert_wall_includes_gather",true},
              {"npu_device_gather",false},{"backend_input_host_copy_bytes",copies},{"expert_token_counts",totals},
              {"allocations",allocations},{"shared_weight_matmuls",3*(active+1)},
              {"execution","grouped 2D expert graphs; one CPU4x8 bank per projection; host gather"}};
        return result;
    }
};
