#pragma once
#include "dynamic-moe-runtime.h"

// One immutable canonical routed bank per layer, shared by CPU and HTP. Only
// actual current-layer route counts enter the policy. No route capture input.
struct online_moe_runtime {
    struct layer_weights {
        std::unique_ptr<bank> source_htp;
        std::unique_ptr<bank> cpu_view;
    };
    source & src;
    device * devs[3];
    int max_tokens;
    fullmodel_execution_model model;
    json profiles=json::array();
    json online_calibration;
    std::array<layer_weights,24> layers;
    std::shared_ptr<shared_activation_buffer> arenas[3];
    std::unique_ptr<shared_bank> gpu_shared;
    json records=json::array();
    std::exception_ptr failure;
    size_t canonical_source_bytes=0, activation_bytes=0;
    size_t peak_shared_weights=0, peak_host_staging=0;
    double npu_conversion_us_per_expert;
    int forced_tail;
    bool reference_packed=std::getenv("MOE_ONLINE_REFERENCE_PACKED")!=nullptr;
    struct binding { online_moe_runtime * self; int layer; };
    std::array<binding,24> bindings;

    online_moe_runtime(source & s,device *devices[3],int nt,const json & calibration,
                       double conversion_us=300,int tail=-1)
        :src(s),max_tokens(nt),model(calibration,nt),npu_conversion_us_per_expert(conversion_us),forced_tail(tail) {
        online_calibration=calibration.value("online_weight_calibration",json());
        if(!online_calibration.is_null()) {
            require(online_calibration.at("tokens")==nt,"online weight calibration length not validated");
            for(const char *key:{"cpu_scale","cpu_overhead_us","npu_conversion_us_per_expert","npu_overhead_us","gpu_preparation_us"}) {
                double value=online_calibration.at(key); require(std::isfinite(value) && value>=0,"invalid online cost correction");
            }
        }
        require(nt>=64 && nt<=768,"online prototype supports prefill lengths 64..768");
        for(int b=0;b<3;++b) { devs[b]=devices[b]; }
        for(const auto &p:model.guard.at("profiles")) {
            if(p.at("tokens")==nt && p.at("family")=="cpu" && p.at("policy")=="overlap") { profiles.push_back(p); }
        }
        if(profiles.empty()) {
            auto interval=model.resolve_interval(nt);
            if(!interval.is_null()) { for(const auto &p:interval.at("profiles")) {
                if(p.at("family")=="cpu") { profiles.push_back(p); }
            } }
        }
        require(!online_calibration.is_null() || !profiles.empty() || forced_tail>=0,"no eligible online CPU/NPU/shared-GPU profile");
        for(int il=0;il<24;++il) {
            auto &l=layers[il]; l.source_htp=std::make_unique<bank>(*devs[1],src,il,nullptr,true);
            l.cpu_view=std::make_unique<bank>(*devs[0],*l.source_htp);
            canonical_source_bytes+=ggml_backend_buffer_get_size(l.source_htp->buf.get()); bindings[il]={this,il};
            // Source pages are no longer needed for routed weights. This releases
            // this mapping's resident references, not OS page cache or other maps.
            for(auto *t:l.source_htp->w) {
                size_t off=gguf_get_data_offset(src.gguf)+gguf_get_tensor_offset(src.gguf,gguf_find_tensor(src.gguf,t->name));
                size_t page=size_t(sysconf(_SC_PAGESIZE)),begin=(off+page-1)/page*page,end=(off+ggml_nbytes(t))/page*page;
                if(end>begin) { madvise(const_cast<uint8_t *>(src.mapped)+begin,end-begin,MADV_DONTNEED); }
            }
        }
        // Empty activation capacity is independent of Expert routes and weights.
        for(int b=0;b<2;++b) {
            auto &w=b==0?*layers[0].cpu_view:*layers[0].source_htp;
            oracle_job shape(w,nullptr,nt*4,1,nt,false,false,true,{},true);
            size_t bytes=shared_activation_buffer::required_size(*devs[b],shape.ctx.get(),shape.graph);
            arenas[b]=std::make_shared<shared_activation_buffer>(*devs[b],bytes); activation_bytes+=arenas[b]->desc.size;
        }
        // Initialize backend kernels and EMPTY target capacity as part of model
        // setup, just as native context allocation reserves its graph buffers.
        // No shared weight bytes, routes or prompt activations are prepared here.
        gpu_shared=std::make_unique<shared_bank>(*devs[2],src,0,true);
        std::vector<int> empty; bank dummy(*devs[2],src,0,&empty);
        oracle_job shared_shape(dummy,gpu_shared.get(),0,1,nt,false,false,true,{},true);
        size_t shared_bytes=shared_activation_buffer::required_size(*devs[2],shared_shape.ctx.get(),shared_shape.graph);
        arenas[2]=std::make_shared<shared_activation_buffer>(*devs[2],shared_bytes);
    }

    placement choose(int il,int nt,const std::vector<int> &counts,json &decision) {
        auto start=timer::now();
        placement best{"online",std::vector<int>(60,1),2};
        if(nt==1) {
            best.owner.assign(60,0); decision={{"reason","terminal one-token FFN on canonical CPU"},{"planning_us",elapsed(start)}}; return best;
        }
        std::vector<int> active; for(int e=0;e<60;++e) { if(counts[e]) { active.push_back(e); } }
        std::stable_sort(active.begin(),active.end(),[&](int a,int b) { return counts[a]!=counts[b]?counts[a]<counts[b]:a<b; });
        std::vector<int> tails;
        if(forced_tail>=0) { tails.push_back(std::min(forced_tail,int(active.size()))); }
        else if(!online_calibration.is_null()) {
            for(int k:online_calibration.at("candidate_tails")) {
                require(k>=0 && k<=60,"invalid online tail candidate");
                if(k<=int(active.size())) { tails.push_back(k); }
            }
        }
        else { for(const auto &p:profiles) { int k=p.at("tails").at(il); if(k<=int(active.size())) { tails.push_back(k); } } }
        if(tails.empty()) { tails.push_back(0); }
        double best_cost=std::numeric_limits<double>::infinity(); json candidates=json::array();
        for(int k:tails) {
            placement p{"online_cpu_tail"+std::to_string(k),std::vector<int>(60,1),2};
            for(int i=0;i<k;++i) { p.owner[active[i]]=0; }
            auto prediction=model.predict(nt,counts,p,k?"cpu":"branch",true);
            auto costs=prediction.at("backend_us").get<std::array<double,3>>();
            if(online_calibration.is_null()) { costs[1]+=npu_conversion_us_per_expert*(active.size()-k); }
            else {
                costs[0]=k?costs[0]*online_calibration.at("cpu_scale").get<double>()+online_calibration.at("cpu_overhead_us").get<double>():0;
                costs[1]=active.size()>size_t(k)?costs[1]+online_calibration.at("npu_conversion_us_per_expert").get<double>()*(active.size()-k)+online_calibration.at("npu_overhead_us").get<double>():0;
                costs[2]+=online_calibration.at("gpu_preparation_us").get<double>();
            }
            double score=*std::max_element(costs.begin(),costs.end())+prediction.at("host_us").get<double>();
            candidates.push_back({{"tail",k},{"cost_us",score},{"backend_us",costs}});
            if(score<best_cost) { best_cost=score; best=p; }
        }
        decision={{"candidates",candidates},{"planning_us",elapsed(start)},
            {"scope","current actual layer routes; layout/preparation-corrected branch cost; no future route or whole-request offline guard"}};
        return best;
    }

    void run(int il,int nt,const float *x,const float *probs,const std::vector<int32_t> &ids,float *result) {
        require(nt>0 && nt<=max_tokens && ids.size()==size_t(nt*4),"online FFN shape mismatch");
        const auto begin=timer::now(); std::vector<int> counts(60),pairs[3];
        for(int e:ids) { require(e>=0 && e<60,"invalid online route"); ++counts[e]; }
        json decision; auto plan=choose(il,nt,counts,decision);
        for(int p=0;p<nt*4;++p) { pairs[plan.owner[ids[p]]].push_back(p); }
        std::vector<float> weighted(size_t(nt)*4*2048),shared_result(size_t(nt)*2048);
        std::array<json,3> timings;
        auto branch=[&](int b) {
            const auto start=timer::now();
            if(b<2) {
                if(pairs[b].empty()) { timings[b]={{"total_us",0}}; return; }
                bank *selected=b==0?layers[il].cpu_view.get():layers[il].source_htp.get();
                std::unique_ptr<bank> reference;
                std::array<int,60> local; std::iota(local.begin(),local.end(),0);
                if(b==1 && reference_packed) {
                    // Validation oracle only: prepare exactly this layer's actual
                    // NPU experts using the unchanged CPU-to-tiled setter.
                    std::vector<int> subset; local.fill(-1);
                    for(int e=0;e<60;++e) { if(counts[e] && plan.owner[e]==1) {
                        local[e]=int(subset.size()); subset.push_back(e);
                    } }
                    reference=std::make_unique<bank>(*devs[1],src,il,&subset);
                    selected=reference.get();
                }
                auto &w=*selected;
                oracle_job job(w,nullptr,int(pairs[b].size()),1,nt,false,false,true,arenas[b],false,true);
                for(size_t row=0;row<pairs[b].size();++row) {
                    int p=pairs[b][row]; std::copy_n(x+(p/4)*2048,2048,job.input_data()+row*2048);
                    job.index_data()[row]=local[ids[p]];
                }
                double prepare=elapsed(start); auto phases=job.run(); auto reduce_start=timer::now();
                for(size_t row=0;row<pairs[b].size();++row) {
                    int p=pairs[b][row]; float weight=probs[(p/4)*60+ids[p]];
                    for(int c=0;c<2048;++c) { weighted[p*2048+c]=weight*job.output_data()[row*2048+c]; }
                }
                timings[b]={{"prepare_us",prepare},{"compute_us",phases[1]},{"scatter_us",elapsed(reduce_start)},{"total_us",elapsed(start)}};
            } else {
                // No retained converted shared layer: each first-pass conversion
                // and allocation is inside this timed branch, overlapping routed.
                require(std::getenv("GGML_OPENCL_FUSED_Q4_PREPARE"),"shared weight reuse requires fused GPU preparation");
                if(!gpu_shared) { gpu_shared=std::make_unique<shared_bank>(*devs[2],src,il,true); }
                auto &shared=*gpu_shared;
                const char *parts[]={"gate","up","down","gate_inp"};
                for(int i=0;i<4;++i) {
                    std::string name="blk."+std::to_string(il)+".ffn_"+parts[i]+"_shexp.weight";
                    auto *original=src.tensor(name); auto *target=shared.w[i];
                    require(ggml_are_same_shape(original,target) && original->type==target->type,"shared shape changed");
                    size_t offset=gguf_get_data_offset(src.gguf)+gguf_get_tensor_offset(src.gguf,gguf_find_tensor(src.gguf,name.c_str()));
                    require(offset<=src.mapped_size && ggml_nbytes(original)<=src.mapped_size-offset,"shared source bounds");
                    ggml_backend_tensor_set(target,src.mapped+offset,0,ggml_nbytes(original));
                }
                devs[2]->sync();
                const size_t weight_bytes=ggml_backend_buffer_get_size(shared.buf.get());
                peak_shared_weights=std::max(peak_shared_weights,weight_bytes);
                size_t staging=0; for(auto *w:shared.w) { staging=std::max(staging,2*ggml_nbytes(w)); }
                peak_host_staging=std::max(peak_host_staging,staging);
                double convert=elapsed(start);
                std::vector<int> empty; bank dummy(*devs[2],src,il,&empty);
                oracle_job shape(dummy,&shared,0,1,nt,false,false,true,{},true);
                size_t need=shared_activation_buffer::required_size(*devs[2],shape.ctx.get(),shape.graph);
                if(!arenas[2]) { arenas[2]=std::make_shared<shared_activation_buffer>(*devs[2],need); }
                require(need<=arenas[2]->desc.size,"shared arena grew past first prefill shape");
                oracle_job job(dummy,&shared,0,1,nt,false,false,true,arenas[2],false,true);
                job.set_shared_input(x,size_t(nt)*2048); double prepare=elapsed(start);
                auto phases=job.run(); std::copy_n(job.shared_output_data(),shared_result.size(),shared_result.data());
                timings[b]={{"conversion_us",convert},{"prepare_us",prepare},{"compute_us",phases[1]},{"total_us",elapsed(start)}};
            }
        };
        {
            std::future<void> cpu=std::async(std::launch::async,branch,0),gpu=std::async(std::launch::async,branch,2);
            // Futures join during normal and exception unwinding, before local
            // inputs/results or any graph/weight backing can be destroyed.
            branch(1); cpu.get(); gpu.get();
        }
        auto combine=timer::now();
        for(int t=0;t<nt;++t) { for(int c=0;c<2048;++c) {
            float sum=weighted[(t*4)*2048+c];
            for(int k=1;k<4;++k) { sum+=weighted[(t*4+k)*2048+c]; }
            result[t*2048+c]=sum+shared_result[t*2048+c];
        } }
        size_t additional=activation_bytes+(arenas[2]?arenas[2]->desc.size:0)+peak_shared_weights+peak_host_staging+
            (weighted.capacity()+shared_result.capacity())*sizeof(float);
        records.push_back({{"layer",il},{"tokens",nt},{"counts",counts},{"placement",plan_json(plan)},
            {"decision",decision},{"branches",timings},{"combine_us",elapsed(combine)},{"total_us",elapsed(begin)},
            {"accounted_additional_bytes",additional}});
    }
    static void compute(ggml_tensor *dst,const ggml_tensor *x,const ggml_tensor *p,const ggml_tensor *ids,int ith,int,void *opaque) {
        if(ith) { return; } auto &b=*static_cast<binding *>(opaque); auto &s=*b.self;
        try {
            if(s.failure) { std::rethrow_exception(s.failure); }
            require(x->type==GGML_TYPE_F32 && p->type==GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_is_contiguous(p),"online callback needs contiguous F32 inputs");
            s.run(b.layer,int(x->ne[1]),static_cast<const float *>(x->data),static_cast<const float *>(p->data),
                dense_read<int32_t>(const_cast<ggml_tensor *>(ids)),static_cast<float *>(dst->data));
        } catch(...) { s.failure=std::current_exception(); std::memset(dst->data,0,ggml_nbytes(dst)); }
    }
    static ggml_tensor *build(ggml_context *ctx,ggml_tensor *x,ggml_tensor *p,ggml_tensor *ids,int32_t layer,void *opaque) {
        auto &s=*static_cast<online_moe_runtime *>(opaque); require(layer>=0 && layer<24,"invalid layer");
        return ggml_map_custom3(ctx,x,p,ids,compute,1,&s.bindings[layer]);
    }
};
