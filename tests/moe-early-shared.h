#pragma once

// N=1 experiment: identical CPU router dependency and resident CPU4x8 jobs.
// early submits NPU shared before router; late submits before CPU routed.
// The single shared worker is drained before engine/layer destruction.
struct moe_early_shared {
    partitioned_ffn & layer;
    moe_branch_worker & worker;
    bool & decode_active;
    std::string policy;
    moe_shared_job * shared_job=nullptr;
    bool pending=false;
    timer::time_point begin;
    double input_done=0,shared_copy_us=0,shared_start=0,shared_end=0,shared_compute=0;

    moe_early_shared(partitioned_ffn & l,moe_branch_worker & w,bool & active,const std::string & p)
        : layer(l),worker(w),decode_active(active),policy(p) {
        require(layer.single_layout && layer.resident.shared[0],"early shared requires CPU residency");
        for(const auto & e:layer.resident.experts) { require(e[0],"decode routed expert not CPU resident"); }
        auto & exec=*layer.single_layout;
        require(exec.weights.weights[1][60][0] && exec.weights.scalar[1],"NPU shared view not prepared");
        std::array<int,61> counts{}; counts[60]=1;
        auto & job=exec.jobs[1][{0,counts}];
        if(!job) {
            job=std::make_unique<moe_shared_job>(*exec.devs[1],exec.weights,1,counts,exec.arenas[1],0,exec.shared_scratch);
            ++layer.allocations;
        }
        shared_job=job.get();
    }

    void launch(const float * input) {
        require(!pending,"early shared scratch still in use");
        auto t=timer::now();
        std::copy_n(input,2048,static_cast<float *>(shared_job->inputs[60]->data));
        shared_copy_us=elapsed(t);
        if(std::getenv("MOE_EARLY_SHARED_TRACE")) { fprintf(stderr,"EARLY_SUBMIT layer=%d policy=%s\n",layer.il,policy.c_str()); }
        worker.submit([this] {
            if(std::getenv("MOE_EARLY_SHARED_TRACE")) { fprintf(stderr,"EARLY_COMPUTE_BEGIN layer=%d\n",layer.il); }
            shared_start=elapsed(begin); auto t=timer::now();
            shared_job->dev.compute(shared_job->graph);
            shared_compute=elapsed(t); shared_end=elapsed(begin);
            if(std::getenv("MOE_EARLY_SHARED_TRACE")) { fprintf(stderr,"EARLY_COMPUTE_END layer=%d\n",layer.il); }
        });
        pending=true;
    }

    static void input(ggml_tensor * dst,const ggml_tensor * src,int ith,int,void * opaque) {
        if(ith) { return; }
        auto & s=*static_cast<moe_early_shared *>(opaque);
        require(ggml_is_contiguous(src) && ggml_nelements(src)==2048 && src->type==GGML_TYPE_F32,"early shared input shape");
        require(!s.pending,"previous shared work not joined");
        s.begin=timer::now();
        s.shared_copy_us=s.shared_start=s.shared_end=s.shared_compute=0;
        std::memcpy(dst->data,src->data,2048*sizeof(float));
        if(s.decode_active && s.policy=="early") { s.launch(static_cast<const float *>(src->data)); }
        s.input_done=elapsed(s.begin);
    }

    static void finish(ggml_tensor * dst,const ggml_tensor * x,const ggml_tensor * probabilities,
                       const ggml_tensor * ids,int ith,int,void * opaque) {
        if(ith) { return; }
        auto & s=*static_cast<moe_early_shared *>(opaque); auto & l=s.layer;
        const double ready=elapsed(s.begin);
        require(ggml_is_contiguous(x) && ggml_is_contiguous(probabilities) && ggml_is_contiguous(ids) &&
                ggml_nelements(x)==2048 && ggml_nelements(ids)==4,"early shared FFN shapes");
        const float * input=static_cast<const float *>(x->data);
        const float * probs=static_cast<const float *>(probabilities->data);
        std::vector<int32_t> route(static_cast<const int32_t *>(ids->data),static_cast<const int32_t *>(ids->data)+4);
        auto * destination=static_cast<float *>(dst->data);
        if(!s.decode_active) {
            l.run_into(input,2048,probs,60,route,true,destination);
            return;
        }
        auto & exec=*l.single_layout;
        if(s.policy=="cpu" || s.policy=="regular") {
            placement p{"fixed_decode",std::vector<int>(60,0),s.policy=="cpu"?0:1};
            exec.run(input,probs,route,p,destination,l.allocations,l.last,l.il,{});
            l.last["decode_shared_launch"]=s.policy;
            l.last["router_window_us"]=ready-s.input_done;
            l.last["input_hook_us"]=s.input_done;
            return;
        }
        if(s.policy=="late") { s.launch(input); }
        require(s.pending,"shared branch was not submitted");
        std::array<int,61> counts{};
        for(int id:route) { require(id>=0 && id<60,"decode expert ID"); ++counts[id]; }
        require(std::set<int32_t>(route.begin(),route.end()).size()==4,"decode expects four distinct experts");
        auto prepare=timer::now();
        auto & cached=exec.jobs[0][{0,counts}];
        if(!cached) {
            cached=std::make_unique<moe_shared_job>(*exec.devs[0],exec.weights,0,counts,exec.arenas[0],0,exec.shared_scratch);
            ++l.allocations;
        }
        auto & job=*cached;
        const double routed_prepare=elapsed(prepare),cpu_start=elapsed(s.begin);
        auto t=timer::now();
        for(int id:route) { std::copy_n(input,2048,static_cast<float *>(job.inputs[id]->data)); }
        double cpu_gather=elapsed(t); t=timer::now();
        job.dev.compute(job.graph);
        const double cpu_compute=elapsed(t),cpu_end=elapsed(s.begin);
        s.worker.join(); s.pending=false;
        const double joined=elapsed(s.begin); t=timer::now();
        const float * shared=static_cast<const float *>(s.shared_job->outputs[60]->data);
        const float * out[4]; float weights[4];
        for(int k=0;k<4;++k) { out[k]=static_cast<const float *>(job.outputs[route[k]]->data); weights[k]=probs[route[k]]; }
        for(int c=0;c<2048;c+=4) {
#if defined(__aarch64__)
            float32x4_t sum=vdupq_n_f32(0);
            for(int k=0;k<4;++k) { sum=vfmaq_n_f32(sum,vld1q_f32(out[k]+c),weights[k]); }
            vst1q_f32(destination+c,vaddq_f32(sum,vld1q_f32(shared+c)));
#else
            for(int j=0;j<4;++j) {
                float sum=0; for(int k=0;k<4;++k) { sum=std::fma(weights[k],out[k][c+j],sum); }
                destination[c+j]=sum+shared[c+j];
            }
#endif
        }
        const double combine=elapsed(t);
        std::array<int,60> totals{}; for(int id:route) { ++totals[id]; }
        std::array<std::array<double,3>,3> phases{}; phases[0][1]=cpu_compute; phases[1][1]=s.shared_compute;
        std::array<std::array<double,2>,3> intervals{};
        intervals[0]={cpu_start,cpu_end}; intervals[1]={s.shared_start,s.shared_end};
        l.last={{"layer",l.il},{"tokens",1},{"pairs",{4,0,0}},{"expert_token_counts",totals},
            {"allocations",l.allocations},{"gather_us",routed_prepare},{"expert_wall_us",joined-ready},
            {"combine_us",combine},{"backend_phases_us",phases},{"intervals_us",intervals},
            {"backend_gather_us",{cpu_gather,s.shared_copy_us,0.0}},
            {"branch_origin_monotonic_us",std::chrono::duration<double,std::micro>(s.begin.time_since_epoch()).count()},
            {"decode_shared_launch",s.policy},{"router_window_us",ready-s.input_done},
            {"router_ready_us",ready},{"input_hook_us",s.input_done},
            {"shared_join_wait_us",joined-cpu_end},{"pre_router_to_join_us",joined},
            {"shared_started_before_router_ready",s.shared_start<ready},
            {"shared_scratch_owner",1},{"shared_weight_matmuls",15},{"reduce_threads",1},
            {"overlap_gather",true},{"expert_wall_includes_gather",true},{"branch_dispatch","shared_worker"},
            {"scope","intervals start at input hook before router; expert_wall starts at FFN callback; host graph intervals, not kernel timing"}};
    }
};
