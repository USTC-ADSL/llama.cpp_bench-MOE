#pragma once

#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"
#include <functional>

// Experimental runtime policy: invoked after actual router IDs are available.
// Backend IDs: CPU=0, HTP0=1, GPUOpenCL=2. It must not change model/context state.
struct moe_route_request {
    bool decode;
    int step, layer, tokens;
    std::array<int,60> counts{};
};
using moe_placement_policy = std::function<placement(const moe_route_request &)>;

struct dynamic_moe_runtime {
    struct slot {
        int layer = -1;
        std::vector<int> experts;
        bool has_shared = false;
        std::unique_ptr<bank> routed;
        std::unique_ptr<shared_bank> shared;
        std::unique_ptr<oracle_job> job;
        int rows = -1, tokens = -1;
        size_t bytes() const {
            size_t n = routed && routed->buf ? ggml_backend_buffer_get_size(routed->buf.get()) : 0;
            if (shared) {
                n += ggml_backend_buffer_get_size(shared->buf.get());
                if (shared->scalar_buf) { n += ggml_backend_buffer_get_size(shared->scalar_buf.get()); }
            }
            return n;
        }
        void clear() {
            job.reset(); shared.reset(); routed.reset(); experts.clear();
            layer = rows = tokens = -1; has_shared = false;
        }
    };
    source & src;
    device * devs[3];
    slot slots[3];
    moe_placement_policy policy;
    int group_size;
    bool decode = false;
    int step = 0;
    int verify_layer = -1;
    size_t peak_weight_bytes = 0;
    json records = json::array();
    std::exception_ptr failure;
    struct binding { dynamic_moe_runtime * runtime; int layer; };
    std::array<binding,24> bindings;

    dynamic_moe_runtime(source & s, device * devices[3], int group, moe_placement_policy p)
        : src(s), policy(std::move(p)), group_size(group) {
        require(group>0 && group<=60, "group size must be 1..60");
        require(src.mapped != nullptr, "dynamic runtime requires immutable mapped weights");
        for (int b=0;b<3;++b) { devs[b]=devices[b]; }
        for (int l=0;l<24;++l) { bindings[l]={this,l}; }
    }
    void clear() { for (auto & s:slots) { s.clear(); } }
    void account_weights() {
        size_t bytes=0;
        for (auto & s:slots) { bytes+=s.bytes(); }
        peak_weight_bytes=std::max(peak_weight_bytes,bytes);
    }
    void check() { if (failure) { std::rethrow_exception(failure); } }

    std::vector<float> run(int layer, int nt, const std::vector<float> & x,
                           const std::vector<float> & probs, const std::vector<int32_t> & ids) {
        auto start=timer::now();
        require(x.size()==size_t(nt)*2048 && probs.size()==size_t(nt)*60 && ids.size()==size_t(nt)*4,
                "dynamic FFN shape mismatch");
        moe_route_request request{decode,step,layer,nt,{}};
        for (int id:ids) { require(id>=0 && id<60,"invalid Expert ID"); ++request.counts[id]; }
        auto plan=policy(request);
        require(plan.owner.size()==60 && plan.shared>=0 && plan.shared<3,"invalid dynamic placement");
        std::vector<int> assigned[3];
        for (int e=0;e<60;++e) {
            require(plan.owner[e]>=0 && plan.owner[e]<3,"invalid dynamic backend");
            if (request.counts[e]) { assigned[plan.owner[e]].push_back(e); }
        }
        size_t waves=1;
        for (auto & a:assigned) { waves=std::max(waves,(a.size()+group_size-1)/group_size); }
        std::vector<float> result(x.size(),0);
        std::vector<float> weighted(ids.size()*2048);
        std::vector<float> shared_output(x.size());
        json wave_records=json::array();
        for (size_t wave=0;wave<waves;++wave) {
            auto prepare=timer::now();
            std::vector<int> pairs[3];
            bool active[3]{};
            int misses=0;
            for (int b=0;b<3;++b) {
                const size_t first=std::min(wave*group_size,assigned[b].size());
                const size_t end=std::min(first+group_size,assigned[b].size());
                std::vector<int> subset(assigned[b].begin()+first,assigned[b].begin()+end);
                const bool shared=wave==0 && b==plan.shared;
                auto & s=slots[b];
                if (subset.empty() && !shared) { s.clear(); continue; }
                active[b]=true;
                if (s.layer!=layer || s.experts!=subset || s.has_shared!=shared) {
                    // Release graph references before evicting converted weights.
                    s.clear();
                    s.routed=std::make_unique<bank>(*devs[b],src,layer,&subset);
                    account_weights();
                    if (shared) { s.shared=std::make_unique<shared_bank>(*devs[b],src,layer); }
                    account_weights();
                    s.layer=layer; s.experts=subset; s.has_shared=shared; ++misses;
                }
                std::array<int,60> local; local.fill(-1);
                for (size_t e=0;e<subset.size();++e) { local[subset[e]]=int(e); }
                for (size_t p=0;p<ids.size();++p) { if (local[ids[p]]>=0) { pairs[b].push_back(int(p)); } }
                int rows=int(pairs[b].size());
                if (!s.job || s.rows!=rows || s.tokens!=nt) {
                    s.job.reset();
                    s.job=std::make_unique<oracle_job>(*s.routed,s.shared.get(),rows,1,nt,false,b==0);
                    s.rows=rows; s.tokens=nt;
                }
                auto & j=*s.job;
                if (shared) { j.shared_input=x; }
                for (int r=0;r<rows;++r) {
                    int p=pairs[b][r];
                    std::copy_n(x.data()+(p/4)*2048,2048,j.input.data()+r*2048);
                    j.indices[r*j.stride]=local[ids[p]];
                }
            }
            size_t weights=0, activations=0;
            for (auto & s:slots) {
                weights+=s.bytes();
                if (s.job) { activations+=ggml_backend_buffer_get_size(s.job->buf.get()); }
            }
            peak_weight_bytes=std::max(peak_weight_bytes,weights);
            double prepare_us=elapsed(prepare);
            auto compute=timer::now();
            std::future<std::array<double,3>> tasks[3];
            std::array<double,3> phases[3]{};
            for (int b=0;b<3;++b) {
                if (active[b]) { tasks[b]=std::async(std::launch::async,[&,b] { return slots[b].job->run(); }); }
            }
            // All futures are joined before buffers can be evicted, including on exceptions.
            for (int b=0;b<3;++b) { if (tasks[b].valid()) { phases[b]=tasks[b].get(); } }
            double wall=elapsed(compute);
            auto combine=timer::now();
            for (int b=0;b<3;++b) {
                if (!active[b]) { continue; }
                auto & j=*slots[b].job;
                for (size_t r=0;r<pairs[b].size();++r) {
                    int p=pairs[b][r]; float weight=probs[(p/4)*60+ids[p]];
                    for (int c=0;c<2048;++c) { weighted[p*2048+c]=weight*j.output[r*2048+c]; }
                }
                if (wave==0 && b==plan.shared) {
                    shared_output=j.shared_output;
                }
            }
            wave_records.push_back({{"wave",wave},{"prepare_us",prepare_us},{"cache_misses",misses},
                {"weight_buffer_bytes",weights},{"activation_buffer_bytes",activations},
                {"backend_phases_us",phases},{"parallel_wall_us",wall},{"combine_us",elapsed(combine)}});
        }
        // Match native top-k addition order, with a separately rounded multiply.
        // Summing in backend/Expert-ID order perturbs subsequent quantization/routes.
        auto reduce=timer::now();
        for (int t=0;t<nt;++t) {
            for (int c=0;c<2048;++c) {
                float sum=weighted[(t*4)*2048+c];
                for (int k=1;k<4;++k) { sum+=weighted[(t*4+k)*2048+c]; }
                result[t*2048+c]=sum+shared_output[t*2048+c];
            }
        }
        double reduce_us=elapsed(reduce);
        json check=nullptr;
        if (layer==verify_layer || verify_layer==-2) {
            // Validation only: independent unsplit CPU FFN with identical inputs/routes.
            bank reference_bank(*devs[0],src,layer);
            shared_bank reference_shared(*devs[0],src,layer);
            oracle_job reference(reference_bank,&reference_shared,nt,4,nt,true);
            reference.input=x; reference.indices=ids; reference.run();
            std::vector<float> expected(x.size());
            for (size_t p=0;p<ids.size();++p) {
                for (int c=0;c<2048;++c) {
                    reference.output[p*2048+c]*=probs[(p/4)*60+ids[p]];
                }
            }
            for (int t=0;t<nt;++t) { for (int c=0;c<2048;++c) {
                float sum=reference.output[(t*4)*2048+c];
                for (int k=1;k<4;++k) { sum+=reference.output[(t*4+k)*2048+c]; }
                expected[t*2048+c]=sum+reference.shared_output[t*2048+c];
            } }
            check=validate_values(result,expected,nt,("dynamic_vs_unsplit_cpu_ffn_layer"+std::to_string(layer)).c_str());
        }
        records.push_back({{"layer",layer},{"tokens",nt},{"decode",decode},{"step",step},{"ffn_check",check},
            {"placement",plan_json(plan)},{"counts",request.counts},{"waves",wave_records},
            {"final_reduce_us",reduce_us},{"total_us",elapsed(start)}});
        return result;
    }
    static void compute(ggml_tensor * dst,const ggml_tensor * x,const ggml_tensor * p,
                        const ggml_tensor * ids,int ith,int,void * opaque) {
        if (ith) { return; }
        auto & binding=*static_cast<struct binding *>(opaque);
        auto & self=*binding.runtime;
        try {
            self.check();
            auto result=self.run(binding.layer,int(x->ne[1]),dense_read<float>(const_cast<ggml_tensor *>(x)),
                dense_read<float>(const_cast<ggml_tensor *>(p)),dense_read<int32_t>(const_cast<ggml_tensor *>(ids)));
            std::memcpy(dst->data,result.data(),result.size()*sizeof(float));
        } catch (...) {
            self.failure=std::current_exception();
            std::memset(dst->data,0,ggml_nbytes(dst));
        }
    }
    static ggml_tensor * build(ggml_context * ctx,ggml_tensor * x,ggml_tensor * p,
                               ggml_tensor * ids,int32_t layer,void * opaque) {
        auto & self=*static_cast<dynamic_moe_runtime *>(opaque);
        require(layer>=0 && layer<24,"invalid layer");
        return ggml_map_custom3(ctx,x,p,ids,compute,1,&self.bindings[layer]);
    }
};
