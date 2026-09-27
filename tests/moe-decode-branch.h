#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

// One in-flight shared branch. Consecutive layers join before reusing arenas.
class moe_branch_worker {
    std::mutex mutex;
    std::condition_variable ready, completed;
    std::function<void()> task;
    std::exception_ptr error;
    bool busy=false, stop=false;
    std::thread worker;
public:
    moe_branch_worker() : worker([this] {
        std::unique_lock<std::mutex> lock(mutex);
        for(;;) {
            ready.wait(lock,[&] { return stop || bool(task); });
            if(stop) { return; }
            auto run=std::move(task); task={}; lock.unlock();
            std::exception_ptr failure;
            try { run(); } catch(...) { failure=std::current_exception(); }
            lock.lock(); error=failure; busy=false; completed.notify_one();
        }
    }) {}
    ~moe_branch_worker() {
        { std::unique_lock<std::mutex> lock(mutex); completed.wait(lock,[&] { return !busy; }); stop=true; }
        ready.notify_one(); worker.join();
    }
    void submit(std::function<void()> run) {
        { std::lock_guard<std::mutex> lock(mutex); require(!busy,"shared branch still in flight"); error=nullptr; task=std::move(run); busy=true; }
        ready.notify_one();
    }
    void join() {
        std::unique_lock<std::mutex> lock(mutex); completed.wait(lock,[&] { return !busy; });
        if(error) { std::rethrow_exception(error); }
    }
};

// Only nt=1 and identity CPU routed banks. CPU matmuls remain in llama's
// graph/threadpool. Shared accelerator work reads a separate activation arena.
struct moe_decode_branch {
    partitioned_ffn & layer;
    moe_branch_worker & worker;
    oracle_job * shared_job=nullptr;
    timer::time_point begin;
    double prepare_us=0,branch_begin_us=0,branch_end_us=0;
    std::array<double,3> shared_phases{};
    explicit moe_decode_branch(partitioned_ffn & l,moe_branch_worker & w) : layer(l),worker(w) {}

    static void launch(ggml_tensor * dst,const ggml_tensor * input,int ith,int,void * opaque) {
        if(ith) { return; }
        auto & s=*static_cast<moe_decode_branch *>(opaque); auto & l=s.layer;
        s.begin=timer::now(); s.shared_phases={}; s.branch_begin_us=s.branch_end_us=0;
        require(ggml_is_contiguous(input) && ggml_nelements(input)==2048,"decode branch shape");
        std::memcpy(dst->data,input->data,2048*sizeof(float));
        if(l.plan.shared!=0) {
            int b=l.plan.shared; auto & job=l.jobs[b][{0,1,1}];
            if(!job) {
                job=std::make_unique<oracle_job>(*l.banks[b],l.shared.get(),0,1,1,false,false,l.shared_activations,l.arenas[b],false,l.reuse_lifetimes,l.reservations[b][2].get());
                ++l.allocations;
            }
            s.shared_job=job.get(); job->set_shared_input(static_cast<const float *>(input->data),2048);
            s.worker.submit([&s] {
                s.branch_begin_us=elapsed(s.begin);
                s.shared_phases=s.shared_job->run();
                s.branch_end_us=elapsed(s.begin);
            });
        }
        s.prepare_us=elapsed(s.begin);
    }

    static void finish(ggml_tensor * dst,const ggml_tensor * routed,const ggml_tensor *,const ggml_tensor * ids,int ith,int,void * opaque) {
        if(ith) { return; }
        auto & s=*static_cast<moe_decode_branch *>(opaque); auto & l=s.layer;
        double cpu_end=elapsed(s.begin);
        if(l.plan.shared!=0) { s.worker.join(); }
        double joined=elapsed(s.begin); auto combine=timer::now();
        const float * x=static_cast<const float *>(routed->data); float * out=static_cast<float *>(dst->data);
        if(l.plan.shared==0) { std::memcpy(out,x,2048*sizeof(float)); }
        else { auto * shared=s.shared_job->shared_output_data(); for(int c=0;c<2048;++c) { out[c]=x[c]+shared[c]; } }
        std::array<int,60> counts{}; require(ggml_nelements(ids)==4,"decode route shape");
        for(int k=0;k<4;++k) { int id=static_cast<const int32_t *>(ids->data)[k]; require(id>=0 && id<60,"decode expert ID"); ++counts[id]; }
        std::array<std::array<double,3>,3> phases{}; phases[0][1]=cpu_end-s.prepare_us;
        std::array<std::array<double,2>,3> intervals{}; intervals[0]={0,cpu_end-s.prepare_us};
        if(l.plan.shared!=0) { phases[l.plan.shared]=s.shared_phases; intervals[l.plan.shared]={s.branch_begin_us-s.prepare_us,s.branch_end_us-s.prepare_us}; }
        l.last={{"layer",l.il},{"tokens",1},{"gather_us",s.prepare_us},{"expert_wall_us",joined-s.prepare_us},{"combine_us",elapsed(combine)},
                {"backend_phases_us",phases},{"intervals_us",intervals},{"pairs",{4,0,0}},
                {"expert_token_counts",counts},{"allocations",l.allocations},{"native_cpu_graph",true},
                {"shared_join_wait_us",joined-cpu_end},{"scope","CPU graph envelope includes router and aggregation; shared begins before routed; not hardware kernel timing"}};
    }

    ggml_tensor * build(ggml_context * ctx,ggml_tensor * input,ggml_tensor * probs,ggml_tensor * ids) {
        require(input->ne[1]==1 && layer.banks[0]->mapped_weights,"native decode requires mapped CPU routed bank");
        auto * start=ggml_map_custom1(ctx,input,launch,1,this);
        auto * x=ggml_reshape_3d(ctx,start,2048,1,1);
        auto & w=layer.banks[0]->w;
        auto * up=ggml_mul_mat_id(ctx,w[1],x,ids);
        auto * gate=ggml_mul_mat_id(ctx,w[0],x,ids);
        auto * down=ggml_mul_mat_id(ctx,w[2],ggml_swiglu_split(ctx,gate,up),ids);
        auto * weights=ggml_get_rows(ctx,ggml_reshape_3d(ctx,probs,1,60,1),ids);
        auto * weighted=ggml_mul(ctx,down,weights);
        auto * out=ggml_view_2d(ctx,weighted,2048,1,weighted->nb[2],0);
        for(int k=1;k<4;++k) { out=ggml_add(ctx,out,ggml_view_2d(ctx,weighted,2048,1,weighted->nb[2],k*weighted->nb[1])); }
        if(layer.plan.shared==0) {
            auto & sw=layer.shared->w;
            auto * su=ggml_mul_mat(ctx,sw[1],start); auto * sg=ggml_mul_mat(ctx,sw[0],start);
            auto * sd=ggml_mul_mat(ctx,sw[2],ggml_swiglu_split(ctx,sg,su));
            auto * sr=ggml_mul_mat(ctx,sw[3],start);
            auto * shared=ggml_mul(ctx,sd,ggml_div(ctx,ggml_silu(ctx,sr),sr));
            out=ggml_add(ctx,out,shared);
        }
        return ggml_map_custom3(ctx,out,start,ids,finish,1,this);
    }
};
