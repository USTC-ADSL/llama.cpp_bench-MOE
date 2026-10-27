#define EXPERT_ORACLE_NO_MAIN
#include "expert-oracle-probe.cpp"
#include "llama.h"
#include "moe-placement-model.h"
#include "moe-route-scheduler.h"
#include "moe-context-model.h"
#include "moe-execution-model.h"
#include "moe-token-workers.h"
#include "moe-runtime-state.h"
#include "moe-state-correction.h"
#include "moe-device-timing.h"
#include "moe-shared-weight.h"

#include <iomanip>
#include <sys/resource.h>
#include <set>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

template<class T> static std::vector<T> dense_read(ggml_tensor * t) {
    require(t->nb[0] == sizeof(T) && t->ne[2] == 1 && t->ne[3] == 1,"unexpected capture layout");
    std::vector<uint8_t> raw(ggml_nbytes(t));
    ggml_backend_tensor_get(t,raw.data(),0,raw.size());
    std::vector<T> result(t->ne[0]*t->ne[1]);
    for(int64_t row=0;row<t->ne[1];++row) { std::memcpy(result.data()+row*t->ne[0],raw.data()+row*t->nb[1],t->ne[0]*sizeof(T)); }
    return result;
}

template<class T> static void binary_write(std::ofstream & out,const std::vector<T> & values) {
    out.write(reinterpret_cast<const char *>(values.data()),values.size()*sizeof(T));
    require(out.good(),"binary write failed");
}
template<class T> static std::vector<T> binary_read(std::ifstream & in,size_t count) {
    std::vector<T> values(count);
    in.read(reinterpret_cast<char *>(values.data()),count*sizeof(T));
    require(in.good(),"binary read failed"); return values;
}

struct captured_layer {
    int nt = 0;
    std::vector<float> input,probs,output;
    std::vector<int32_t> ids;
};

struct router_placement_trace {
    json nodes=json::object();
    static bool callback(ggml_tensor * t,bool ask,void * user) {
        auto & self=*static_cast<router_placement_trace *>(user);
        const std::string name=t->name;
        const std::string key=name+"/n"+std::to_string(t->ne[1]);
        if(ask && name.find("ffn_moe_")!=std::string::npos && self.nodes.count(key)==0) {
            auto buffer_name=[](ggml_tensor * v) { return v && v->buffer?std::string(ggml_backend_buffer_name(v->buffer)):std::string("unbound/view"); };
            self.nodes[key]={{"op",ggml_op_name(t->op)},{"buffer",buffer_name(t)},{"src0_buffer",buffer_name(t->src[0])},{"rows",t->ne[1]}};
        }
        return false;
    }
};

struct capture_all {
    std::array<captured_layer,24> layers;
    std::set<int> selected;
    bool includes(int il) const { return selected.empty() || selected.count(il); }
    static bool callback(ggml_tensor * t,bool ask,void * opaque) {
        std::string name=t->name;
        auto dash=name.rfind('-');
        if(dash==std::string::npos) { return ask ? false : true; }
        if(name.find_first_not_of("0123456789",dash+1)!=std::string::npos) { return ask ? false : true; }
        std::string prefix=name.substr(0,dash);
        bool wanted=prefix=="ffn_norm" || prefix=="ffn_moe_probs" || prefix=="ffn_moe_topk" || prefix=="ffn_out";
        if(!wanted) { return !ask; }
        int il=std::stoi(name.substr(dash+1)); require(il>=0 && il<24,"invalid capture layer");
        auto & capture=*static_cast<capture_all *>(opaque);
        auto & l=capture.layers[il];
        l.nt=t->ne[1];
        if(ask) { return capture.includes(il); }
        if(!capture.includes(il)) { return true; }
        if(prefix=="ffn_norm") { l.input=dense_read<float>(t); }
        if(prefix=="ffn_moe_probs") { l.probs=dense_read<float>(t); }
        if(prefix=="ffn_moe_topk") { l.ids=dense_read<int32_t>(t); }
        if(prefix=="ffn_out") { l.output=dense_read<float>(t); }
        return true;
    }
};

struct partitioned_ffn {
    int il,nt;
    placement plan;
    std::unique_ptr<bank> banks[3];
    std::unique_ptr<shared_bank> shared;
    int primary_shared;
    std::unique_ptr<shared_bank> shared_variants[3];
    std::unique_ptr<moe_shared_execution> single_layout;
    const moe_route_scheduler * scheduler=nullptr;
    resident_placement_model::availability resident;
    std::array<int,3> resident_width{};
    std::map<std::tuple<int,int,int,bool>,std::unique_ptr<oracle_job>> jobs[3];
    std::array<std::array<int,60>,3> local{};
    std::vector<int32_t> expected_ids;
    json last;
    size_t allocations=0;
    bool shared_activations=false;
    bool direct_callback_io=false;
    bool fused_host_reduce=false;
    bool reuse_lifetimes=false;
    moe_token_workers * reduce_workers=nullptr;
    bool parallel_reduce=false;
    bool overlap_gather=false;
    // Fixed before arena preparation; never toggled within a warmed cohort.
    bool npu_device_gather=false;
    std::shared_ptr<shared_activation_buffer> arenas[3];
    // Separate contracts for split routed, native top-k, and shared-only graphs.
    std::unique_ptr<shared_activation_layout::reservation> reservations[3][2][3];
    shared_bank * shared_for(int b) { return b==primary_shared?shared.get():shared_variants[b].get(); }
    partitioned_ffn(device * devs[3],source & src,int layer,int tokens,const placement & p,bool compact,bool shared_buffers=false,
                    const resident_placement_model::availability * prepared=nullptr) : il(layer),nt(tokens),plan(p),primary_shared(p.shared),shared_activations(shared_buffers) {
        require(plan.owner.size()==60 && plan.shared>=0 && plan.shared<3,"invalid placement");
        const bool single=std::getenv("MOE_SHARED_WEIGHTS") && std::string(std::getenv("MOE_SHARED_WEIGHTS"))=="cpu4x8";
        for(int b=0;b<3;++b) {
            local[b].fill(-1); std::vector<int> subset;
            for(int e=0;e<60;++e) {
                require(plan.owner[e]>=0 && plan.owner[e]<3,"invalid expert owner");
                if(prepared?prepared->experts[e][b]:!compact || plan.owner[e]==b) {
                    local[b][e]=int(subset.size()); subset.push_back(e); resident.experts[e][b]=true;
                }
            }
            resident_width[b]=int(subset.size());
            if(!single) { banks[b]=std::make_unique<bank>(*devs[b],src,il,(compact || prepared) ? &subset : nullptr); }
        }
        if(!single) { shared=std::make_unique<shared_bank>(*devs[plan.shared],src,il); }
        resident.shared[plan.shared]=true;
        if(prepared) {
            require(moe_route_scheduler::feasible(plan,*prepared),"initial plan not resident");
            for(int b=0;b<3;++b) { if(b!=plan.shared && prepared->shared[b]) {
                if(!single) { shared_variants[b]=std::make_unique<shared_bank>(*devs[b],src,il); } resident.shared[b]=true;
            } }
        }
        if(single) { single_layout=std::make_unique<moe_shared_execution>(devs,src,il,resident); }
    }
    void set_shared(device * devs[3],source & src,int backend) {
        if(backend!=plan.shared) { shared.reset(); shared=std::make_unique<shared_bank>(*devs[backend],src,il); }
        plan.shared=backend;
        primary_shared=backend;
        resident.shared.fill(false); resident.shared[backend]=true;
        for(auto & cache:jobs) { cache.clear(); }
    }
    std::vector<float> run(const std::vector<float> & x,const std::vector<float> & probs,const std::vector<int32_t> & ids,bool record) {
        return run_into(x.data(),x.size(),probs.data(),probs.size(),ids,record,nullptr);
    }
    std::vector<float> run_into(const float * x,size_t elements,const float * probs,size_t probability_count,
                              const std::vector<int32_t> & ids,bool record,float * destination) {
        int nt=int(elements/2048);
        require((nt==this->nt || nt==1) && elements==size_t(2048*nt) && probability_count==size_t(60*nt) && ids.size()==size_t(4*nt),"FFN input shape mismatch");
        auto start=timer::now();
        double decision_us=0; moe_route_scheduler::result decision;
        if(scheduler) {
            std::vector<int> counts(60);
            for(int id:ids) { require(id>=0 && id<60,"invalid current route ID"); ++counts[id]; }
            decision=scheduler->choose(nt,counts,resident,&resident_width);
            plan=decision.selected;
            decision_us=elapsed(start);
        }
        if(single_layout) {
            auto result=single_layout->run(x,probs,ids,plan,destination,allocations,last,il,expected_ids);
            last["gather_us"]=last["gather_us"].get<double>()+decision_us;
            if(scheduler) {
                last["scheduling"]={{"decision_us",decision_us},{"included_in_gather",true},
                    {"evaluations",decision.evaluations},{"seed_us",decision.seed_us},
                    {"single_token_model",nt==1 && scheduler->single_token_model!=nullptr},
                    {"predicted_backend_us",decision.predicted.backend},{"predicted_host_us",decision.predicted.host},
                    {"predicted_total_us",decision.predicted.total},{"owner",plan.owner},{"shared",plan.shared},
                    {"resident_width",resident_width}};
            }
            return result;
        }
        std::vector<int> pairs[3];
        for(int p=0;p<4*nt;++p) { require(ids[p]>=0 && ids[p]<60,"invalid route ID"); pairs[plan.owner[ids[p]]].push_back(p); }
        bool native=false; for(auto & ps:pairs) { native|=ps.size()==ids.size(); }
        oracle_job * work[3]{};
        for(int b=0;b<3;++b) {
            if(pairs[b].empty() && b!=plan.shared) { continue; }
            int rows=pairs[b].empty()?0:native?nt:int(pairs[b].size()); int k=native?4:1;
            bool has_shared=b==plan.shared;
            auto & ptr=jobs[b][{rows,k,nt,has_shared}];
            // A compact one-Expert CPU repack tensor is 2D: its backend accepts
            // MUL_MAT, not MUL_MAT_ID. oracle_job uses the equivalent single-bank
            // path only when ne[2]==1 (and asserts k==1).
            if(!ptr) {
                auto * reserved=reservations[b][has_shared][!rows?2:native?1:0].get();
                require(!reuse_lifetimes || reserved,"missing activation layout reservation");
                ptr=std::make_unique<oracle_job>(*banks[b],has_shared?shared_for(b):nullptr,rows,k,nt,native,b==0,shared_activations,arenas[b],false,reuse_lifetimes,reserved,
                                               nullptr,nullptr,npu_device_gather && b==1 && rows>0 && !native);
                ++allocations;
            }
            work[b]=ptr.get();
        }
        std::array<double,3> branch_gather{};
        std::array<size_t,3> input_copy_bytes{};
        auto prepare=[&](int b) {
            auto begin=timer::now(); auto & j=*work[b];
            j.reset_index_padding(); j.set_shared_input(x,elements);
            if(j.sx && !j.sx->view_src) { input_copy_bytes[b]+=elements*sizeof(float); }
            if(j.input_rows) {
                std::copy_n(x,elements,j.input_data()); input_copy_bytes[b]+=elements*sizeof(float);
                auto * row_ids=static_cast<int32_t *>(j.input_rows->data);
                for(size_t i=0;i<pairs[b].size();++i) {
                    int p=pairs[b][i]; require(local[b][ids[p]]>=0,"expert not resident");
                    row_ids[i]=p/4; j.index_data()[i*j.stride]=local[b][ids[p]];
                }
            } else if(native && j.x) {
                std::copy_n(x,elements,j.input_data());
                input_copy_bytes[b]+=elements*sizeof(float);
                for(int t=0;t<nt;++t) { for(int kk=0;kk<4;++kk) {
                    require(local[b][ids[t*4+kk]]>=0,"native expert not resident");
                    j.index_data()[t*j.stride+kk]=local[b][ids[t*4+kk]];
                } }
            } else {
                input_copy_bytes[b]+=pairs[b].size()*2048*sizeof(float);
                for(size_t i=0;i<pairs[b].size();++i) {
                    int p=pairs[b][i]; require(local[b][ids[p]]>=0,"expert not resident");
                    std::copy_n(x+(p/4)*2048,2048,j.input_data()+i*2048); j.index_data()[i*j.stride]=local[b][ids[p]];
                }
            }
            branch_gather[b]=elapsed(begin);
        };
        // Banks and graph metadata are prepared serially. Each branch then owns
        // a disjoint backend arena and may gather while another backend computes.
        if(!overlap_gather) { for(int b=0;b<3;++b) { if(work[b]) { prepare(b); } } }
        double gather=elapsed(start); start=timer::now();
        const double branch_origin_us=std::chrono::duration<double,std::micro>(start.time_since_epoch()).count();
        std::array<double,3> phases[3]{}; std::array<double,2> intervals[3]{};
        std::future<void> futures[3]; int inline_b=-1;
        for(int b=0;b<3;++b) { if(work[b]) { inline_b=b; } }
        auto execute=[&](int b) { intervals[b][0]=elapsed(start); if(overlap_gather) { prepare(b); } phases[b]=work[b]->run(); intervals[b][1]=elapsed(start); };
        for(int b=0;b<3;++b) { if(work[b] && b!=inline_b) { futures[b]=std::async(std::launch::async,execute,b); } }
        execute(inline_b); for(auto & f:futures) { if(f.valid()) { f.get(); } }
        double wall=elapsed(start); start=timer::now();
        std::vector<const float *> outputs(ids.size());
        for(int b=0;b<3;++b) { for(size_t i=0;i<pairs[b].size();++i) { outputs[pairs[b][i]]=work[b]->output_data()+i*2048; } }
        std::vector<float> owned_result;
        if(!destination) { owned_result.resize(elements,0.0f); }
        else if(!fused_host_reduce) { std::fill_n(destination,elements,0.0f); }
        float * result=destination?destination:owned_result.data();
        if(fused_host_reduce) {
            auto reduce=[&](int begin,int end) { for(int t=begin;t<end;++t) {
                float w[4]; const float * o[4];
                for(int k=0;k<4;++k) { w[k]=probs[t*60+ids[t*4+k]]; o[k]=outputs[t*4+k]; }
                auto * shared_out=work[plan.shared]->shared_output_data()+t*2048;
                for(int c=0;c<2048;c+=4) {
#if defined(__aarch64__)
                    // Retain the existing AArch64 FMA/top-k order; write each output once.
                    auto sum=vdupq_n_f32(0.0f);
                    for(int k=0;k<4;++k) { sum=vfmaq_n_f32(sum,vld1q_f32(o[k]+c),w[k]); }
                    vst1q_f32(result+t*2048+c,vaddq_f32(sum,vld1q_f32(shared_out+c)));
#else
                    for(int lane=0;lane<4;++lane) {
                        float sum=0;
                        for(int k=0;k<4;++k) { sum=std::fma(w[k],o[k][c+lane],sum); }
                        result[t*2048+c+lane]=sum+shared_out[c+lane];
                    }
#endif
                }
            } };
            if(parallel_reduce && reduce_workers) { reduce_workers->run(nt,reduce); }
            else { reduce(0,nt); }
        } else {
            for(size_t p=0;p<ids.size();++p) {
                float weight=probs[(p/4)*60+ids[p]];
                for(int c=0;c<2048;++c) { result[(p/4)*2048+c]+=weight*outputs[p][c]; }
            }
            for(size_t i=0;i<elements;++i) { result[i]+=work[plan.shared]->shared_output_data()[i]; }
        }
        if(record) {
            std::array<int,60> counts{}; for(auto id:ids) { ++counts[id]; }
            int changed=0; if(expected_ids.size()==ids.size()) { for(size_t i=0;i<ids.size();++i) { changed+=ids[i]!=expected_ids[i]; } }
            last={{"layer",il},{"tokens",nt},{"gather_us",gather},{"expert_wall_us",wall},{"combine_us",elapsed(start)},
                  {"backend_phases_us",phases},{"intervals_us",intervals},{"branch_origin_monotonic_us",branch_origin_us},{"pairs",{pairs[0].size(),pairs[1].size(),pairs[2].size()}},
                  {"changed_route_slots",changed},{"route_reference_available",expected_ids.size()==ids.size()},
                  {"reduce_threads",parallel_reduce && reduce_workers && nt>=reduce_workers->count()?reduce_workers->count():1},
                  {"overlap_gather",overlap_gather},{"backend_gather_us",branch_gather},{"expert_wall_includes_gather",overlap_gather},
                  {"npu_device_gather",work[1] && work[1]->input_rows},{"backend_input_host_copy_bytes",input_copy_bytes},
                  {"expert_token_counts",counts},{"allocations",allocations}};
            if(scheduler) {
                last["scheduling"]={{"decision_us",decision_us},{"included_in_gather",true},
                    {"evaluations",decision.evaluations},{"seed_us",decision.seed_us},
                    {"single_token_model",nt==1 && scheduler->single_token_model!=nullptr},
                    {"predicted_backend_us",decision.predicted.backend},{"predicted_host_us",decision.predicted.host},
                    {"predicted_total_us",decision.predicted.total},{"owner",plan.owner},{"shared",plan.shared},
                    {"resident_width",resident_width}};
            }
        }
        return owned_result;
    }
};

#include "moe-decode-branch.h"
#include "moe-early-shared.h"

struct full_engine {
    std::unique_ptr<moe_token_workers> reduce_workers;
    std::unique_ptr<moe_token_workers> expert_workers;
    std::shared_ptr<shared_activation_buffer> arenas[3];
    std::vector<std::unique_ptr<partitioned_ffn>> layers;
    std::shared_ptr<shared_activation_buffer> shared_scratch;
    std::unique_ptr<moe_branch_worker> shared_worker;
    std::vector<std::unique_ptr<moe_decode_branch>> decode_branches;
    std::vector<std::unique_ptr<moe_early_shared>> early_shared_branches;
    bool decode_active=false;
    bool record=false;
    json physical_arena_sizes() const {
        json sizes=json::array(); std::set<const shared_activation_buffer *> seen;
        for(auto & arena:arenas) { if(arena && seen.insert(arena.get()).second) { sizes.push_back(arena->desc.size); } }
        return sizes;
    }
    ~full_engine() {
        // Also drain on exceptions between launch and finish: worker closures
        // reference decode_branches, whose state must remain alive until join.
        shared_worker.reset();
        decode_branches.clear();
        early_shared_branches.clear();
    }
    void prepare_arenas(device * devs[3]) {
        std::array<size_t,3> capacities{};
        if(!layers.empty() && layers[0]->single_layout) {
            const std::string scratch=std::getenv("MOE_SHARED_SCRATCH")?std::getenv("MOE_SHARED_SCRATCH"):"private";
            require(scratch=="private" || scratch=="common","unknown shared scratch mode");
            const std::string policy=std::getenv("MOE_SHARED_ARENA")?std::getenv("MOE_SHARED_ARENA"):"private";
            require(policy=="private" || policy=="pooled","unknown shared arena mode");
            const bool pooled=policy=="pooled";
            require(!pooled || !std::getenv("MOE_DECODE_SHARED_LAUNCH"),"pooled arena requires the synchronous layer dispatcher");
            size_t pool_capacity=0,alignment=128;
            for(auto * dev:std::array<device *,3>{devs[0],devs[1],devs[2]}) {
                alignment=std::max(alignment,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev->dev)));
            }
            size_t shared_capacity=0;
            const std::string dispatch=std::getenv("MOE_SHARED_DISPATCH")?std::getenv("MOE_SHARED_DISPATCH"):"async";
            require(dispatch=="async" || dispatch=="pool","unknown shared branch dispatch");
            if(dispatch=="pool") { expert_workers=std::make_unique<moe_token_workers>(3); }
            for(auto & l:layers) { l->single_layout->branch_workers=expert_workers.get(); }
            for(auto & l:layers) { for(int b=0;b<3;++b) {
                auto & w=l->single_layout->weights;
                // A sum of all non-view tensor sizes bounds any lifetime layout,
                // including dynamic expert counts. Padding covers every tensor.
                size_t bytes=1024*1024;
                if(l->resident_width[b]) { bytes+=size_t(4*l->nt)*4*(4096+3*w.hidden); }
                if(l->resident.shared[b]) {
                    size_t shared_bytes=size_t(l->nt)*(4*(6144+3*w.shared_hidden)+32);
                    if(scratch=="common") { shared_capacity=std::max(shared_capacity,size_t(1024*1024)+shared_bytes); }
                    else { bytes+=shared_bytes; }
                }
                if(b==1 && l->single_layout->npu_gather && l->resident_width[b] && l->nt>1) {
                    bytes+=size_t(4*l->nt)*sizeof(int32_t); // expert token-row maps
                    // When shared is eligible, its budget already contains the
                    // complete input, now aliased by all routed GET_ROWS nodes.
                    if(scratch=="common" || !l->resident.shared[b]) { bytes+=size_t(l->nt)*2048*sizeof(float); }
                }
                capacities[b]=std::max(capacities[b],bytes);
            } }
            if(pooled) {
                for(auto & l:layers) {
                    auto & exec=*l->single_layout;
                    // Across all three concurrent branches there are only 4N
                    // routed pairs and one shared expert, independent of owner.
                    size_t bytes=3*(1024*1024+alignment)+exec.routed_budget(size_t(4*l->nt));
                    if(scratch=="private") { bytes+=exec.shared_budget(l->nt); }
                    if(exec.npu_gather && l->resident_width[1] && l->nt>1) {
                        bytes+=size_t(4*l->nt)*sizeof(int32_t);
                        if(scratch=="common" || l->resident.shared[0] || l->resident.shared[2]) {
                            bytes+=size_t(l->nt)*2048*sizeof(float);
                        }
                    }
                    pool_capacity=std::max(pool_capacity,shared_activation_buffer::align(bytes,alignment));
                    exec.pooled_arena=true; exec.arena_alignment=alignment;
                }
                arenas[0]=std::make_shared<shared_activation_buffer>(*devs[1],pool_capacity);
            }
            for(int b=0;b<3;++b) {
                arenas[b]=pooled?arenas[0]:std::make_shared<shared_activation_buffer>(*devs[b],capacities[b]);
                if(pooled) { arenas[b]->view(*devs[b]); }
                for(auto & l:layers) { l->single_layout->arenas[b]=arenas[b]; }
            }
            if(scratch=="common") {
                require(shared_capacity>0,"common scratch has no eligible shared expert");
                shared_scratch=std::make_shared<shared_activation_buffer>(*devs[1],shared_capacity);
                for(int b=0;b<3;++b) { shared_scratch->view(*devs[b]); }
                for(auto & l:layers) { l->single_layout->shared_scratch=shared_scratch; }
            }
            return;
        }
        for(auto & l:layers) { for(int b=0;b<3;++b) {
            require(l->shared_activations,"arena requires shared layer");
            bool has_routed=bool(l->banks[b]->buf) || l->banks[b]->mapped_weights;
            for(int has_shared=0;has_shared<2;++has_shared) {
            if(has_shared && !l->resident.shared[b]) { continue; }
            if(!has_shared && (!has_routed || (!l->scheduler && l->plan.shared==b))) { continue; }
            if(!l->scheduler && has_shared && l->plan.shared!=b) { continue; }
            // k=1 with all 4*nt routed pairs bounds both split and native k=4 graphs.
            oracle_job layout(*l->banks[b],has_shared?l->shared_for(b):nullptr,
                              has_routed?4*l->nt:0,1,l->nt,false,b==0,true,{},true,false,nullptr,
                              nullptr,nullptr,l->npu_device_gather && b==1 && has_routed);
            capacities[b]=std::max(capacities[b],shared_activation_buffer::required_size(*devs[b],layout.ctx.get(),l->reuse_lifetimes?layout.graph:nullptr));
            if(l->reuse_lifetimes) {
                // Reuse the maximum-shape offsets, not a separately packed small
                // graph: best-fit capacity need not be monotonic in routed rows.
                l->reservations[b][has_shared][has_routed?0:2]=std::make_unique<shared_activation_layout::reservation>(
                    shared_activation_layout(*devs[b],layout.ctx.get(),layout.graph).freeze());
            }
            if(l->reuse_lifetimes && has_routed && l->banks[b]->w[0]->ne[2]>=4) {
                // Native k=4 has a different dependency graph (shared aliases x).
                // Reserve it separately rather than assume split liveness bounds it.
                oracle_job native(*l->banks[b],has_shared?l->shared_for(b):nullptr,
                                  l->nt,4,l->nt,true,false,true,{},true);
                capacities[b]=std::max(capacities[b],shared_activation_buffer::required_size(*devs[b],native.ctx.get(),native.graph));
                l->reservations[b][has_shared][1]=std::make_unique<shared_activation_layout::reservation>(
                    shared_activation_layout(*devs[b],native.ctx.get(),native.graph).freeze());
            }
            if(l->reuse_lifetimes && has_routed && has_shared) {
                oracle_job only_shared(*l->banks[b],l->shared_for(b),0,1,l->nt,false,b==0,true,{},true);
                auto reserved=shared_activation_layout(*devs[b],only_shared.ctx.get(),only_shared.graph).freeze();
                capacities[b]=std::max(capacities[b],reserved.bytes);
                l->reservations[b][has_shared][2]=std::make_unique<shared_activation_layout::reservation>(std::move(reserved));
            }
            }
        } }
        for(int b=0;b<3;++b) {
            if(capacities[b]) { arenas[b]=std::make_shared<shared_activation_buffer>(*devs[b],capacities[b]); }
            for(auto & l:layers) { l->arenas[b]=arenas[b]; }
        }
    }
    static void compute(ggml_tensor * dst,const ggml_tensor * x,const ggml_tensor * probabilities,const ggml_tensor * ids,int ith,int,void * opaque) {
        if(ith) { return; }
        auto & layer=*static_cast<partitioned_ffn *>(opaque);
        require(ggml_is_contiguous(x) && ggml_is_contiguous(probabilities),"noncontiguous FFN inputs");
        if(layer.direct_callback_io) {
            require(ggml_is_contiguous(dst) && dst->type==GGML_TYPE_F32 && x->type==GGML_TYPE_F32 && probabilities->type==GGML_TYPE_F32,
                    "direct callback requires contiguous F32 activations");
            std::vector<int32_t> route(ggml_nelements(ids));
            for(int64_t t=0;t<ids->ne[1];++t) { std::memcpy(route.data()+t*ids->ne[0],static_cast<char *>(ids->data)+t*ids->nb[1],ids->ne[0]*sizeof(int32_t)); }
            // The scheduler owns these host tensors until this synchronous callback returns.
            layer.run_into(static_cast<const float *>(x->data),ggml_nelements(x),static_cast<const float *>(probabilities->data),
                           ggml_nelements(probabilities),route,true,static_cast<float *>(dst->data));
            return;
        }
        std::vector<float> input(static_cast<float *>(x->data),static_cast<float *>(x->data)+ggml_nelements(x));
        std::vector<float> probs(static_cast<float *>(probabilities->data),static_cast<float *>(probabilities->data)+ggml_nelements(probabilities));
        std::vector<int32_t> route(ggml_nelements(ids));
        for(int64_t t=0;t<ids->ne[1];++t) { std::memcpy(route.data()+t*ids->ne[0],static_cast<char *>(ids->data)+t*ids->nb[1],ids->ne[0]*sizeof(int32_t)); }
        auto result=layer.run(input,probs,route,true);
        std::memcpy(dst->data,result.data(),result.size()*sizeof(float));
    }
    static ggml_tensor * build(ggml_context * ctx,ggml_tensor * input,ggml_tensor * probs,ggml_tensor * ids,int32_t il,void * opaque) {
        auto & engine=*static_cast<full_engine *>(opaque);
        require(il>=0 && size_t(il)<engine.layers.size(),"invalid custom layer");
        if(!engine.early_shared_branches.empty() && input->ne[1]==1) {
            return ggml_map_custom3(ctx,input,probs,ids,moe_early_shared::finish,1,engine.early_shared_branches[il].get());
        }
        if(!engine.decode_branches.empty() && input->ne[1]==1) { return engine.decode_branches[il]->build(ctx,input,probs,ids); }
        return ggml_map_custom3(ctx,input,probs,ids,compute,1,engine.layers[il].get());
    }
    static ggml_tensor * build_input(ggml_context * ctx,ggml_tensor * input,int32_t il,void * opaque) {
        auto & engine=*static_cast<full_engine *>(opaque);
        if(input->ne[1]!=1) { return input; }
        require(il>=0 && size_t(il)<engine.early_shared_branches.size(),"invalid early shared layer");
        return ggml_map_custom1(ctx,input,moe_early_shared::input,1,engine.early_shared_branches[il].get());
    }
};

static placement parse_plan(const json & p) { return {p.at("name"),p.at("owner").get<std::vector<int>>(),p.at("shared")}; }
static json plan_json(const placement & p) { return {{"name",p.name},{"owner",p.owner},{"shared",p.shared}}; }
static resident_placement_model::availability parse_resident(const json & j) {
    resident_placement_model::availability a;
    a.experts=j.at("experts").get<std::array<std::array<bool,3>,60>>();
    a.shared=j.at("shared").get<std::array<bool,3>>();
    for(const auto & row:a.experts) { require(row[0] || row[1] || row[2],"resident manifest omits expert"); }
    require(a.shared[0] || a.shared[1] || a.shared[2],"resident manifest omits shared");
    return a;
}

static int search_all(const char * model_path,const std::string & capture_path,const char * output,bool profile=false,int repetitions=3) {
    std::ifstream in(capture_path); json capture; in>>capture;
    ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(model_path,{true,&raw})); ggml_context_ptr meta(raw);
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(model_path,"rb"),fclose); require(gguf && meta && f,"search model open failed");
    source src{gguf.get(),meta.get(),f.get()}; device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
    json report={{"tokens",capture["tokens"]},{"token_ids",capture["token_ids"]},{"layers",json::array()},{"scope","offline real captured routes; bounded joint routed/shared placement search"}};
    report["optimized_profile"]=profile;
    json supplied;
    if(std::getenv("MOE_PROFILE_CANDIDATES")) {
        require(profile,"external candidates require profile mode");
        std::ifstream file(std::getenv("MOE_PROFILE_CANDIDATES")); file>>supplied;
        require(supplied.at("tokens")==capture.at("tokens"),"profile candidate length mismatch");
    }
    if(profile) { report["profile_contract"]="resident banks; shared activation arenas with lifetime reuse; direct output; fused reduction; overlapping gather; two warmups; weight preparation excluded"; }
    std::set<int> profile_layers;
    if(profile && std::getenv("MOE_PROFILE_LAYERS")) {
        std::istringstream list(std::getenv("MOE_PROFILE_LAYERS")); std::string value;
        while(std::getline(list,value,',')) { int il=std::stoi(value); require(il>=0 && il<24,"invalid profile layer"); profile_layers.insert(il); }
        require(!profile_layers.empty(),"empty profile layer selection");
    }
    for(int il=0;il<24;++il) {
        if(profile && !profile_layers.empty() && !profile_layers.count(il)) { continue; }
        int nt=capture["layers"][il]["tokens"];
        std::ifstream bin(capture_path+".layer"+std::to_string(il)+".bin",std::ios::binary);
        auto x=binary_read<float>(bin,2048*nt); auto probs=binary_read<float>(bin,60*nt);
        auto ids=binary_read<int32_t>(bin,4*nt); auto reference=binary_read<float>(bin,2048*nt);
        std::vector<int> counts(60); for(auto id:ids) { ++counts[id]; }
        auto candidates=model_candidates(counts);
        json layer_config;
        if(!supplied.is_null()) {
            for(const auto & l:supplied.at("layers")) { if(l.at("layer")==il) { layer_config=l; break; } }
            require(!layer_config.is_null(),"missing profile candidates for layer");
            candidates.clear(); for(const auto & p:layer_config.at("candidates")) { candidates.push_back(parse_plan(p)); }
            require(!candidates.empty(),"empty profile candidate list");
        }
        if(profile) {
            json result={{"layer",il},{"tokens",nt},{"counts",counts},{"ids",ids},{"candidates",json::array()}};
            for(const auto & p:candidates) {
                // Use exactly the resident confirmation layout; never time conversion.
                full_engine engine;
                resident_placement_model::availability prepared;
                bool use_union=layer_config.contains("resident");
                if(use_union) { prepared=parse_resident(layer_config.at("resident")); }
                engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,nt,p,true,true,use_union?&prepared:nullptr));
                auto & ff=*engine.layers.back(); ff.fused_host_reduce=true; ff.reuse_lifetimes=true; ff.overlap_gather=true;
                engine.prepare_arenas(devs);
                std::vector<float> actual(x.size());
                auto run=[&](bool record) { ff.run_into(x.data(),x.size(),probs.data(),probs.size(),ids,record,actual.data()); };
                run(false); run(false);
                auto check=chain_difference(actual,reference,nt);
                auto warmed=ff.allocations;
                json item=plan_json(p); item["check"]=check; item["samples"]=json::array(); item["resident_width"]=ff.resident_width;
                std::vector<double> times;
                for(int r=0;r<repetitions;++r) {
                    rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
                    auto start=timer::now(); run(true); double us=elapsed(start);
                    getrusage(RUSAGE_SELF,&after);
                    require(ff.allocations==warmed,"profile allocated after warmup");
                    auto sample=ff.last; sample["total_us"]=us;
                    sample["major_faults"]=after.ru_majflt-before.ru_majflt;
                    sample["minor_faults"]=after.ru_minflt-before.ru_minflt;
                    item["samples"].push_back(sample); times.push_back(us);
                }
                item["timing"]=stats(times,nt); result["candidates"].push_back(item);
                fprintf(stderr,"PROFILE layer=%d nt=%d plan=%s %.3f ms\n",il,nt,p.name.c_str(),item["timing"]["median_us"].get<double>()/1000);
            }
            report["layers"].push_back(result);
            std::ofstream out(output); out<<report.dump(2)<<'\n'; require(out.good(),"profile report write failed");
            continue;
        }
        partitioned_ffn ff(devs,src,il,nt,candidates[0],false);
        json result={{"layer",il},{"tokens",nt},{"counts",counts},{"ids",ids},{"candidates",json::array()}};
        double best=INFINITY,branch_best=INFINITY; placement winner,branch;
        for(const auto & p:candidates) {
            ff.set_shared(devs,src,p.shared); ff.plan=p;
            auto actual=ff.run(x,probs,ids,false);
            auto check=chain_difference(actual,reference,nt);
            std::vector<double> samples;
            for(int r=0;r<3;++r) { auto start=timer::now(); ff.run(x,probs,ids,false); samples.push_back(elapsed(start)); }
            double median=stats(samples,nt)["median_us"];
            json item=plan_json(p); item["timing"]=stats(samples,nt); item["check"]=check;
            result["candidates"].push_back(item);
            bool valid=check["passes_4percent_per_token"];
            if(valid && median<best) { best=median; winner=p; }
            if(valid && p.name.find("tail")==std::string::npos && median<branch_best) { branch_best=median; branch=p; }
        }
        require(std::isfinite(best) && std::isfinite(branch_best),"no valid FFN placement");
        result["selected"]=plan_json(winner); result["branch"]=plan_json(branch);
        // Validate compact resident banks against the same real FFN reference.
        for(auto & cache:ff.jobs) { cache.clear(); }
        for(auto & b:ff.banks) { b.reset(); } ff.shared.reset();
        partitioned_ffn compact(devs,src,il,nt,winner,true);
        result["compact_check"]=validate_values(compact.run(x,probs,ids,false),reference,nt,"compact_full_ffn");
        report["layers"].push_back(result);
        std::ofstream out(output); out<<report.dump(2)<<'\n'; require(out.good(),"search report write failed");
        fprintf(stderr,"FULL_SEARCH layer=%d nt=%d winner=%s %.3f ms\n",il,nt,winner.name.c_str(),best/1000);
    }
    return 0;
}

static int predict_all(const std::string & capture_path,const char * output,const json & preload_state=json(),double acquisition_us=0) {
    require(std::getenv("MOE_PLACEMENT_MODEL"),"prediction requires MOE_PLACEMENT_MODEL");
    std::ifstream model_file(std::getenv("MOE_PLACEMENT_MODEL")); json coefficients; model_file>>coefficients;
    bool contextual=coefficients.contains("base");
    resident_placement_model model(contextual?coefficients.at("base"):coefficients);
    std::ifstream capture_file(capture_path); json capture; capture_file>>capture;
    if(coefficients.value("model_type",std::string())=="branch_critical_path") {
        json layers=json::array();
        require(capture.at("layers").size()==24,"incomplete route capture");
        for(int il=0;il<24;++il) {
            int nt=capture.at("layers").at(il).at("tokens"); std::vector<int32_t> ids;
            if(capture.at("layers").at(il).contains("ids")) { ids=capture["layers"][il]["ids"].get<std::vector<int32_t>>(); }
            else {
                std::ifstream bin(capture_path+".layer"+std::to_string(il)+".bin",std::ios::binary);
                bin.seekg((2048+60)*size_t(nt)*sizeof(float)); ids=binary_read<int32_t>(bin,4*nt);
            }
            require(nt>0 && ids.size()==size_t(4*nt),"invalid captured shape");
            std::vector<int> counts(60); for(int id:ids) { require(id>=0 && id<60,"invalid captured ID"); ++counts[id]; }
            layers.push_back({{"layer",il},{"tokens",nt},{"counts",counts},{"ids",ids}});
        }
        auto start=timer::now(); fullmodel_execution_model execution(coefficients,capture.at("tokens"));
        auto report=execution.plan(capture,layers);
        if(coefficients.contains("runtime_state_correction")) {
            apply_moe_state_correction(report,coefficients.at("runtime_state_correction"),preload_state,moe_runtime_state::now_us());
            report["preload_state"]=preload_state; report["state_acquisition_us"]=acquisition_us;
        }
        double observed=elapsed(start)+acquisition_us;
        execution.finalize(report,observed);
        report["decision_device_total_us"]=elapsed(start)+acquisition_us;
        report["decision_timing_scope"]="state setup/acquisition when enabled, validation, candidate evaluation and plan construction; excludes route/model file IO and output serialization";
        std::ofstream out(output); out<<report.dump(2)<<'\n'; require(out.good(),"execution plan write failed"); return 0;
    }
    std::unique_ptr<fullmodel_context_model> context;
    if(contextual) { context=std::make_unique<fullmodel_context_model>(coefficients,capture.at("tokens")); }
    resident_placement_model::availability preparable;
    // Offline planner contract: weights are subsequently prepared on the selected
    // backend before full-model timing. This is NOT an assertion of online residency.
    for(auto & row:preparable.experts) { row.fill(true); } preparable.shared.fill(true);
    json report={{"tokens",capture["tokens"]},{"token_ids",capture["token_ids"]},{"layers",json::array()},
                 {"scope","offline route-conditioned model selection; weights prepared later; decision only timed"}};
    double custom_total=0,native_compute=0,uncertainty=0,decision_total=0;
    for(int il=0;il<24;++il) {
        int nt=capture["layers"][il]["tokens"];
        std::ifstream bin(capture_path+".layer"+std::to_string(il)+".bin",std::ios::binary);
        bin.seekg((2048+60)*size_t(nt)*sizeof(float)); auto ids=binary_read<int32_t>(bin,4*nt);
        std::vector<int> counts(60); for(int id:ids) { require(id>=0 && id<60,"invalid captured ID"); ++counts[id]; }
        resident_placement_model::prediction estimate; double prepare=0;
        auto start=timer::now(); placement selected;
        fullmodel_context_model::prediction corrected;
        if(context) {
            selected=context->choose(nt,counts,corrected); estimate=corrected.ffn;
            custom_total+=corrected.score; uncertainty+=corrected.uncertainty;
            native_compute+=model.predict(nt,counts,{"npu_shared_npu",std::vector<int>(60,1),1}).backend[1];
        } else { selected=model.choose(nt,counts,preparable,estimate,prepare); }
        double us=elapsed(start); decision_total+=us;
        report["layers"].push_back({{"layer",il},{"tokens",nt},{"counts",counts},{"ids",ids},{"selected",plan_json(selected)},
            {"decision_device_us",us},{"prediction",{{"backend_us",estimate.backend},{"host_us",estimate.host},{"total_us",estimate.total}}}});
        if(context) {
            auto & pred=report["layers"].back()["prediction"];
            pred["outside_share_us"]=corrected.outside; pred["uncertainty_share_us"]=corrected.uncertainty; pred["score_us"]=corrected.score;
        }
    }
    if(context) {
        auto start=timer::now(); report["path_decision"]=context->decide(custom_total,native_compute,uncertainty,decision_total);
        report["execution_mode"]=report["path_decision"]["execution_mode"]; decision_total+=elapsed(start);
        report["decision_device_total_us"]=decision_total;
    }
    std::ofstream out(output); out<<report.dump(2)<<'\n'; require(out.good(),"model plan write failed");
    return 0;
}

#ifndef FULLMODEL_ORACLE_NO_MAIN
int main(int argc,char ** argv) {
    try {
        require(argc==8,"usage: fullmodel-oracle MODEL PROMPT TOKENS OUTPUT MODE REPS PLANS_OR_DASH; modes capture,search,profile,predict,auto,cpu,gpu,npu,custom,branch,custom-npu");
        ggml_backend_load_all(); llama_backend_init();
        moe_device_timing device_timing(std::getenv("GGML_HEXAGON_TIMING") && std::string(std::getenv("GGML_HEXAGON_TIMING"))=="1");
        std::string mode=argv[5]; int nt=std::stoi(argv[3]),reps=std::stoi(argv[6]);
        require(nt>0 && reps>0,"tokens and repetitions must be positive");
        bool state_auto=std::getenv("MOE_RUNTIME_STATE_MODEL")!=nullptr;
        require(!state_auto || mode=="auto","runtime state model requires auto");
        moe_runtime_state runtime_state(state_auto || (std::getenv("MOE_RUNTIME_STATE") && std::string(std::getenv("MOE_RUNTIME_STATE"))=="1"));
        auto preload_state=runtime_state.sample();
        double acquisition_us=runtime_state.enabled?runtime_state.setup_us+preload_state.at("observation_us").get<double>():0;
        if(mode=="search" || mode=="profile") { return search_all(argv[1],argv[7],argv[4],mode=="profile",reps); }
        if(mode=="predict") { return predict_all(argv[7],argv[4],preload_state,acquisition_us); }
        bool automatic=mode=="auto"; json plans;
        if(automatic) {
            std::string plan_path=argv[7];
            if(state_auto) {
                setenv("MOE_PLACEMENT_MODEL",std::getenv("MOE_RUNTIME_STATE_MODEL"),1);
                plan_path=std::string(argv[4])+".plan.json";
                predict_all(argv[7],plan_path.c_str(),preload_state,acquisition_us);
            }
            std::ifstream in(plan_path); in>>plans;
            mode=plans.at("execution_mode").get<std::string>();
            require(mode=="npu" || mode=="custom","invalid pre-load execution mode");
            require(plans.at("tokens").get<int>()==nt && plans.at("layers").size()==24,"automatic plan shape mismatch");
        }
        bool custom=mode=="custom" || mode=="branch" || mode=="custom-npu";
        const bool single_layout=std::getenv("MOE_SHARED_WEIGHTS") && std::string(std::getenv("MOE_SHARED_WEIGHTS"))=="cpu4x8";
        require(!std::getenv("MOE_SHARED_WEIGHTS") || single_layout,"unknown shared weight layout");
        require(!single_layout || mode=="custom","shared weights require explicit custom mode");
        std::unique_ptr<resident_placement_model> route_model,decode_route_model;
        std::unique_ptr<moe_route_scheduler> route_scheduler;
        require(!std::getenv("MOE_DECODE_ROUTE_MODEL") || std::getenv("MOE_ROUTE_MODEL"),"decode route model requires prefill route model");
        if(std::getenv("MOE_ROUTE_MODEL")) {
            require(mode=="custom" && !automatic && !std::getenv("MOE_FEEDBACK_MODEL"),"current-route model requires explicit custom mode");
            std::ifstream file(std::getenv("MOE_ROUTE_MODEL")); json parameters; file>>parameters;
            route_model=std::make_unique<resident_placement_model>(parameters);
            require(!route_model->has_single_token_costs,"N=1 table must use the separate decode model option");
            const auto algorithm=parameters.value("algorithm",std::string("generic"));
            require(algorithm=="generic" || (algorithm=="shared_additive" && single_layout),"unsupported route search algorithm");
            if(std::getenv("MOE_DECODE_ROUTE_MODEL")) {
                std::ifstream decode_file(std::getenv("MOE_DECODE_ROUTE_MODEL")); json decode_parameters; decode_file>>decode_parameters;
                require(decode_parameters.at("calibration_tokens")==1,"decode route model must be calibrated for N=1");
                decode_route_model=std::make_unique<resident_placement_model>(decode_parameters);
                auto decode_algorithm=decode_parameters.value("algorithm",std::string("generic"));
                require(decode_route_model->has_single_token_costs?
                        single_layout && decode_algorithm=="single_token_exhaustive":decode_algorithm==algorithm,
                        "unsupported decode search algorithm or layout");
            }
            route_scheduler=std::make_unique<moe_route_scheduler>(*route_model,parameters.value("search_budget",2048),algorithm=="shared_additive",decode_route_model.get());
        }
        if(automatic && plans.value("model_type",std::string())=="branch_critical_path") {
            for(const char * name:{"MOE_ALTERNATE_CALLBACK_IO","MOE_ALTERNATE_HOST_REDUCE","MOE_ALTERNATE_REDUCE_THREADS","MOE_ALTERNATE_OVERLAP_GATHER"}) {
                require(!std::getenv(name) || std::string(std::getenv(name))!="1","auto execution policy cannot alternate");
            }
            const auto & policy=plans.at("execution_policy");
            require(policy.at("activation_lifetimes")==true && policy.at("overlap_gather").is_boolean(),"invalid automatic execution policy");
            for(const char * name:{"MOE_SHARED_ACTIVATIONS","MOE_DIRECT_CALLBACK_IO","MOE_ACTIVATION_ARENA","MOE_ACTIVATION_LIFETIMES","MOE_FUSED_HOST_REDUCE"}) { setenv(name,custom?"1":"0",1); }
            setenv("MOE_OVERLAP_GATHER",custom && policy.at("overlap_gather").get<bool>()?"1":"0",1);
            setenv("MOE_REDUCE_THREADS","1",1);
        }
        int quality_steps=std::getenv("MOE_QUALITY_STEPS")?std::stoi(std::getenv("MOE_QUALITY_STEPS")):0;
        int decode_reps=std::getenv("MOE_DECODE_REPS")?std::stoi(std::getenv("MOE_DECODE_REPS")):0;
        require(decode_reps>=0 && decode_reps<=32 && (!decode_reps || quality_steps>0),"invalid decode repetitions");
        require(quality_steps>=0 && quality_steps<=64 && (!quality_steps || mode!="capture"),"invalid quality steps");
        std::vector<llama_token> forced_tokens;
        if(std::getenv("MOE_QUALITY_TOKENS")) {
            std::ifstream in(std::getenv("MOE_QUALITY_TOKENS")); json forced; in>>forced;
            forced_tokens=forced.get<std::vector<llama_token>>();
            require(quality_steps>0 && int(forced_tokens.size())==quality_steps,"teacher forcing length mismatch");
        }
        bool transfer_placement=std::getenv("MOE_TRANSFER_PLACEMENT") && std::string(std::getenv("MOE_TRANSFER_PLACEMENT"))=="1";
        require(!transfer_placement || custom,"placement transfer requires custom mode");
        bool shared_activations=std::getenv("MOE_SHARED_ACTIVATIONS") && std::string(std::getenv("MOE_SHARED_ACTIVATIONS"))=="1";
        bool direct_callback_io=std::getenv("MOE_DIRECT_CALLBACK_IO") && std::string(std::getenv("MOE_DIRECT_CALLBACK_IO"))=="1";
        bool alternate_callback_io=std::getenv("MOE_ALTERNATE_CALLBACK_IO") && std::string(std::getenv("MOE_ALTERNATE_CALLBACK_IO"))=="1";
        bool activation_arena=std::getenv("MOE_ACTIVATION_ARENA") && std::string(std::getenv("MOE_ACTIVATION_ARENA"))=="1";
        bool activation_lifetimes=std::getenv("MOE_ACTIVATION_LIFETIMES") && std::string(std::getenv("MOE_ACTIVATION_LIFETIMES"))=="1";
        require(!activation_lifetimes || activation_arena,"activation lifetime reuse requires arena");
        bool fused_host_reduce=std::getenv("MOE_FUSED_HOST_REDUCE") && std::string(std::getenv("MOE_FUSED_HOST_REDUCE"))=="1";
        bool alternate_host_reduce=std::getenv("MOE_ALTERNATE_HOST_REDUCE") && std::string(std::getenv("MOE_ALTERNATE_HOST_REDUCE"))=="1";
        int reduce_threads=std::getenv("MOE_REDUCE_THREADS")?std::stoi(std::getenv("MOE_REDUCE_THREADS")):1;
        bool alternate_reduce_threads=std::getenv("MOE_ALTERNATE_REDUCE_THREADS") && std::string(std::getenv("MOE_ALTERNATE_REDUCE_THREADS"))=="1";
        require(reduce_threads>=1 && reduce_threads<=8 && (reduce_threads==1 || (custom && fused_host_reduce)),"parallel reduce requires fused custom path and 1..8 threads");
        require(!alternate_reduce_threads || (reduce_threads>1 && reps%4==0 && !alternate_callback_io && !alternate_host_reduce && !quality_steps),"invalid parallel reduce alternation");
        bool overlap_gather=std::getenv("MOE_OVERLAP_GATHER") && std::string(std::getenv("MOE_OVERLAP_GATHER"))=="1";
        bool npu_device_gather=std::getenv("MOE_NPU_DEVICE_GATHER") && std::string(std::getenv("MOE_NPU_DEVICE_GATHER"))=="1";
        require(!npu_device_gather || (custom && shared_activations && !quality_steps),"NPU device gather requires shared custom prefill");
        bool alternate_overlap_gather=std::getenv("MOE_ALTERNATE_OVERLAP_GATHER") && std::string(std::getenv("MOE_ALTERNATE_OVERLAP_GATHER"))=="1";
        require(!(overlap_gather || alternate_overlap_gather) || (custom && shared_activations),"gather overlap requires shared custom mode");
        require(!alternate_overlap_gather || (!overlap_gather && reps%4==0 && !alternate_reduce_threads && !alternate_callback_io && !alternate_host_reduce && !quality_steps),"invalid gather overlap alternation");
        require(!fused_host_reduce || custom,"fused reduction requires custom mode");
        require(!alternate_host_reduce || (custom && !fused_host_reduce && !alternate_callback_io && reps%4==0),
                "alternating reduction requires custom mode, no other alternation, and repetitions divisible by four");
        require(!activation_arena || (custom && shared_activations),"activation arena requires shared custom mode");
        require(!shared_activations || custom,"shared activations require custom mode");
        require(!direct_callback_io || custom,"direct callback requires custom mode");
        require(!alternate_callback_io || (custom && shared_activations && !direct_callback_io && reps%4==0),
                "alternating callback requires shared custom mode, no direct override, and repetitions divisible by four");
        bool accelerator=mode=="npu" || mode=="gpu" || custom;
        std::string dense_backend=std::getenv("MOE_DENSE_BACKEND")?std::getenv("MOE_DENSE_BACKEND"):(mode=="gpu"?"gpu":accelerator?"npu":"cpu");
        require(!std::getenv("MOE_DENSE_BACKEND") || (custom && !automatic && dense_backend=="cpu"),"dense override only supports explicit custom CPU");
        accelerator=dense_backend!="cpu";
        bool mapped_experts=std::getenv("MOE_CPU_MAPPED_EXPERTS") && std::string(std::getenv("MOE_CPU_MAPPED_EXPERTS"))=="1";
        require(!mapped_experts || (custom && !automatic && dense_backend=="cpu"),"mapped experts require custom CPU dense");
        bool native_decode_graph=std::getenv("MOE_NATIVE_DECODE_GRAPH") && std::string(std::getenv("MOE_NATIVE_DECODE_GRAPH"))=="1";
        const std::string decode_shared_launch=std::getenv("MOE_DECODE_SHARED_LAUNCH")?std::getenv("MOE_DECODE_SHARED_LAUNCH"):"off";
        require(decode_shared_launch=="off" || decode_shared_launch=="early" || decode_shared_launch=="late" ||
                decode_shared_launch=="cpu" || decode_shared_launch=="regular","unknown decode shared launch policy");
        require(decode_shared_launch=="off" || (single_layout && quality_steps && route_scheduler && !native_decode_graph),
                "decode shared launch requires shared layout, resident route model and decode steps");
        require(!single_layout || (shared_activations && activation_arena && activation_lifetimes && direct_callback_io &&
                fused_host_reduce && overlap_gather && reduce_threads==1 && !native_decode_graph && !automatic),
                "shared weights require resident shared-arena direct/fused/overlap execution without alternation or native decode graph");
        require(!route_scheduler || !native_decode_graph,"current-route search is not supported by fixed native decode branch");
        require(!native_decode_graph || (mapped_experts && shared_activations && activation_lifetimes),"native decode graph requires mapped CPU/shared lifetime buffers");
        require(custom || mode=="capture" || mode=="cpu" || mode=="npu" || mode=="gpu","invalid mode");
        if(custom && !automatic) { std::ifstream in(argv[7]); in>>plans; require((transfer_placement || plans["tokens"].get<int>()==nt) && plans["layers"].size()==24,"plan shape mismatch"); }
        int prefill_chunk=std::getenv("MOE_PREFILL_CHUNK")?std::stoi(std::getenv("MOE_PREFILL_CHUNK")):nt;
        require(prefill_chunk>0 && prefill_chunk<=nt && nt%prefill_chunk==0,"prefill chunk must divide input length");
        require(prefill_chunk==nt || (custom && transfer_placement && !automatic && !quality_steps && !route_scheduler && !std::getenv("MOE_FEEDBACK_MODEL")),
                "chunked prefill requires fixed transferred custom placement without decode or feedback");
        std::unique_ptr<fullmodel_context_model> feedback;
        if(std::getenv("MOE_FEEDBACK_MODEL")) {
            require(custom && !transfer_placement,"shadow feedback requires custom mode with matching route reference");
            std::ifstream in(std::getenv("MOE_FEEDBACK_MODEL")); json parameters; in>>parameters;
            feedback=std::make_unique<fullmodel_context_model>(parameters,nt);
        }
        const std::string router_backend=std::getenv("MOE_ROUTER_BACKEND")?std::getenv("MOE_ROUTER_BACKEND"):"default";
        require(router_backend=="default" || router_backend=="cpu" || router_backend=="gpu","unknown router backend");
        bool gpu_router=router_backend=="gpu",cpu_router=router_backend=="cpu";
        require(!cpu_router || (custom && !automatic),"CPU router requires explicit custom mode");
        require(decode_shared_launch=="off" || cpu_router,"early shared experiment requires explicit CPU router");
        require(!gpu_router || (custom && !automatic && dense_backend=="npu"),"GPU router requires explicit custom HTP dense mode");
        ggml_backend_dev_t target[]={ggml_backend_dev_by_name(mode=="gpu"?"GPUOpenCL":"HTP0"),gpu_router?ggml_backend_dev_by_name("GPUOpenCL"):nullptr,nullptr};
        require(!gpu_router || target[1],"router GPU unavailable");
        ggml_backend_ptr router_loader;
        if(gpu_router) {
            // Explicit tensor splits bypass device-memory discovery. Initialize
            // OpenCL before the loader queries its buffer alignment/size.
            router_loader.reset(ggml_backend_dev_init(target[1],nullptr));
            require(bool(router_loader),"router OpenCL initialization failed");
        }
        require(!accelerator || target[0],"requested accelerator unavailable");
        auto mp=llama_model_default_params(); mp.n_gpu_layers=accelerator?99:0;
        mp.devices=accelerator?target:nullptr;
        // Register both execution backends, while keeping all ordinary layers on HTP.
        const float router_split[]={1.0f,0.0f};
        if(gpu_router) {
            mp.tensor_split=router_split;
            // CPU overrides must remain canonical mmap sources, rather than
            // selecting OpenCL's host buffer and preparing unused expert copies.
            mp.no_host=true;
        }
        llama_model_tensor_buft_override overrides[]={
            {"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"))},
            {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_inp)_shexp\\.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"))},
            {nullptr,nullptr},
            {nullptr,nullptr},
            {nullptr,nullptr}};
        if(custom && dense_backend=="cpu") {
            // A plain CPU override otherwise re-selects CPU_REPACK globally,
            // duplicating all unused native Expert tensors. Optimize dense only.
            auto * cpu=ggml_backend_dev_by_name("CPU");
            auto extra=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(cpu),"ggml_backend_dev_get_extra_bufts"));
            require(extra!=nullptr,"CPU extra buffers unavailable"); auto types=extra(cpu);
            for(size_t i=0;types && types[i];++i) { if(std::string(ggml_backend_buft_name(types[i]))=="CPU_REPACK") {
                overrides[2]={"(blk\\.[0-9]+\\.attn_[qkv]\\.weight|blk\\.[0-9]+\\.attn_output\\.weight|^output\\.weight$)",types[i]}; break;
            } }
            require(overrides[2].pattern!=nullptr,"CPU_REPACK unavailable");
        }
        if(gpu_router) { overrides[2]={"blk\\.[0-9]+\\.ffn_gate_inp\\.weight",ggml_backend_dev_buffer_type(target[1])}; }
        if(cpu_router) {
            const int slot=overrides[2].pattern?3:2;
            overrides[slot]={"blk\\.[0-9]+\\.ffn_gate_inp\\.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"))};
        }
        if(custom) { mp.tensor_buft_overrides=overrides; mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP; }
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(model!=nullptr,"native model load failed");
        if(cpu_router) { require(llama_model_set_moe_router_cpu(model.get(),true),"model rejected CPU router policy"); }
        std::ifstream prompt_file(argv[2]); std::string prompt((std::istreambuf_iterator<char>(prompt_file)),{});
        auto * vocab=llama_model_get_vocab(model.get()); std::vector<llama_token> tokens(prompt.size()+32);
        int count=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
        require(count>=nt,"prompt too short"); tokens.resize(nt);
        if(automatic || (custom && !transfer_placement)) { require(tokens==plans["token_ids"].get<std::vector<int32_t>>(),"fixed prompt token mismatch"); }
        std::unique_ptr<device> devices[3];
        struct mapping {
            void * data=MAP_FAILED; size_t size=0;
            ~mapping() { if(data!=MAP_FAILED) { munmap(data,size); } }
        } weight_mapping;
        full_engine engine;
        double resident_preparation_us=0;
        if(reduce_threads>1) { engine.reduce_workers=std::make_unique<moe_token_workers>(reduce_threads); }
        if(custom) {
            auto preparation_start=timer::now();
            ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
            std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose); require(gguf && meta && f,"compact model open failed");
            source src{gguf.get(),meta.get(),f.get()};
            if(mapped_experts) {
                struct stat st{}; require(fstat(fileno(f.get()),&st)==0 && st.st_size>0,"mapped model stat failed");
                weight_mapping.size=size_t(st.st_size);
                weight_mapping.data=mmap(nullptr,weight_mapping.size,PROT_READ,MAP_SHARED,fileno(f.get()),0);
                require(weight_mapping.data!=MAP_FAILED,"model mmap failed");
                src.mapped=static_cast<const uint8_t *>(weight_mapping.data); src.mapped_size=weight_mapping.size; src.borrow_cpu_experts=true;
            }
            const char * names[]={"CPU","HTP0","GPUOpenCL"}; device * devs[3];
            for(int b=0;b<3;++b) { devices[b]=std::make_unique<device>(names[b]); devs[b]=devices[b].get(); }
            for(int il=0;il<24;++il) {
                placement p=mode=="custom-npu"?placement{"all_npu",std::vector<int>(60,1),1}:parse_plan(plans["layers"][il][mode=="branch"?"branch":"selected"]);
                if(native_decode_graph) { require(std::all_of(p.owner.begin(),p.owner.end(),[](int owner) { return owner==0; }),"native decode graph requires all routed experts on CPU"); }
                int layer_tokens=transfer_placement?(il==23?1:prefill_chunk):plans["layers"][il]["tokens"].get<int>();
                resident_placement_model::availability prepared;
                if(route_scheduler) { prepared=parse_resident(plans.at("layers").at(il).at("resident")); }
                const auto prefill_resident=prepared;
                // Decoder may use NPU shared in the final layer, while its
                // single-row prefill still obeys the original CPU-only policy.
                if(decode_shared_launch!="off") { prepared.shared[1]=true; }
                engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,layer_tokens,p,true,shared_activations,route_scheduler?&prepared:nullptr));
                if(decode_shared_launch!="off") { engine.layers.back()->resident=prefill_resident; }
                engine.layers.back()->scheduler=route_scheduler.get();
                engine.layers.back()->direct_callback_io=direct_callback_io;
                engine.layers.back()->fused_host_reduce=fused_host_reduce;
                engine.layers.back()->reuse_lifetimes=activation_lifetimes;
                engine.layers.back()->reduce_workers=engine.reduce_workers.get();
                engine.layers.back()->parallel_reduce=reduce_threads>1;
                engine.layers.back()->overlap_gather=overlap_gather;
                engine.layers.back()->npu_device_gather=npu_device_gather;
                if(!transfer_placement) { engine.layers.back()->expected_ids=plans["layers"][il]["ids"].get<std::vector<int32_t>>(); }
                fprintf(stderr,"RESIDENT layer=%d plan=%s\n",il,p.name.c_str());
            }
            if(activation_arena) { engine.prepare_arenas(devs); }
            if(native_decode_graph) {
                engine.shared_worker=std::make_unique<moe_branch_worker>();
                for(auto & l:engine.layers) { engine.decode_branches.push_back(std::make_unique<moe_decode_branch>(*l,*engine.shared_worker)); }
            }
            if(decode_shared_launch!="off") {
                engine.shared_worker=std::make_unique<moe_branch_worker>();
                for(auto & l:engine.layers) {
                    engine.early_shared_branches.push_back(std::make_unique<moe_early_shared>(*l,*engine.shared_worker,engine.decode_active,decode_shared_launch));
                }
                require(llama_model_set_moe_input_callback(model.get(),full_engine::build_input,&engine),"model rejected pre-router callback");
            }
            require(llama_model_set_moe_graph_callback(model.get(),full_engine::build,&engine),"model rejected FFN callback");
            resident_preparation_us=elapsed(preparation_start);
        }
        capture_all capture;
        if(std::getenv("MOE_CAPTURE_LAYERS")) {
            require(mode=="capture","capture layer selection requires capture mode");
            std::istringstream list(std::getenv("MOE_CAPTURE_LAYERS")); std::string value;
            while(std::getline(list,value,',')) {
                int il=std::stoi(value); require(il>=0 && il<24,"invalid capture layer selection");
                capture.selected.insert(il);
            }
            require(!capture.selected.empty(),"empty capture layer selection");
        }
        router_placement_trace router_trace;
        auto cp=llama_context_default_params(); cp.n_ctx=quality_steps?nt+quality_steps+32:nt; cp.n_batch=prefill_chunk; cp.n_ubatch=prefill_chunk;
        int host_threads=std::getenv("MOE_HOST_THREADS")?std::stoi(std::getenv("MOE_HOST_THREADS")):4;
        require(host_threads>=1 && host_threads<=8,"host threads must be in 1..8");
        cp.n_threads=host_threads; cp.n_threads_batch=host_threads; cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_kqv=accelerator; cp.op_offload=accelerator;
        if(mode=="capture") { cp.cb_eval=capture_all::callback; cp.cb_eval_user_data=&capture; }
        if(std::getenv("MOE_ROUTER_TRACE")) { require(custom,"router trace requires custom mode"); cp.cb_eval=router_placement_trace::callback; cp.cb_eval_user_data=&router_trace; }
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        require(ctx!=nullptr,"native context init failed");
        int nv=llama_vocab_n_tokens(vocab);
        for(auto t:forced_tokens) { require(t>=0 && t<nv,"teacher forcing token out of range"); }
        json report={{"tokens",nt},{"token_ids",tokens},{"mode",mode},{"samples",json::array()},
            {"scope","native complete model prefill including last-token logits and retained KV; dynamic real routing; weights resident; context reset outside timing"}};
        report["shared_activations"]=shared_activations;
        report["host_threads"]=host_threads;
        report["expert_cpu_threads"]=custom?4:0;
        report["prefill_chunk"]=prefill_chunk;
        report["prefill_chunks"]=nt/prefill_chunk;
        report["expert_weight_layout"]=single_layout?"single CPU4x8 V2 + canonical F16 scalar gate":"backend native banks";
        const bool shared_htp_gather=single_layout && !engine.layers.empty() && engine.layers[0]->single_layout->npu_gather;
        report["expert_execution"]=single_layout?"grouped 2D MUL_MAT graphs; host or HTP GET_ROWS gather":"native MUL_MAT_ID graphs";
        report["shared_input_gather"]=shared_htp_gather?"htp":"host";
        report["npu_device_gather_effective"]=false;
        report["cpu4x8_gather_fusion_requested"]=shared_htp_gather && std::getenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER") &&
            std::string(std::getenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER"))=="1";
        report["direct_callback_io"]=direct_callback_io;
        report["alternate_callback_io"]=alternate_callback_io;
        report["activation_arena"]=activation_arena;
        report["activation_lifetimes"]=activation_lifetimes;
        report["fused_host_reduce"]=fused_host_reduce;
        report["alternate_host_reduce"]=alternate_host_reduce;
        report["host_reduce_threads"]=reduce_threads;
        report["alternate_reduce_threads"]=alternate_reduce_threads;
        report["overlap_gather"]=overlap_gather;
        report["alternate_overlap_gather"]=alternate_overlap_gather;
        report["transfer_placement"]=transfer_placement;
        report["requested_mode"]=argv[5];
        report["dense_backend"]=dense_backend;
        report["router_backend_requested"]=router_backend;
        report["decode_shared_launch"]=decode_shared_launch;
        report["cpu_mapped_experts"]=mapped_experts;
        report["model_extra_bufts"]=mp.use_extra_bufts;
        report["native_decode_graph"]=native_decode_graph;
        report["shadow_feedback"]=bool(feedback);
        report["current_route_scheduler"]=bool(route_scheduler);
        report["single_token_route_model"]=bool(decode_route_model);
        if(decode_route_model) { report["single_token_route_model_scope"]="actual FFN N=1, including a single-row prefill layer; N>1 uses the original model; no weight or KV migration"; }
        report["resident_preparation_us"]=resident_preparation_us;
        report["resident_preparation_scope"]="custom backend initialization, weight bank import/layout conversion and activation arena preparation; outside llama_decode timing";
        if(route_scheduler) { report["scheduler_contract"]="current actual IDs; immutable explicitly prepared union banks; whole-expert move/swap search; planning included in llama_decode and layer gather; missing layouts ineligible; resident preparation excluded"; }
        report["process_identity"]={{"pid",getpid()},{"uid",getuid()},{"euid",geteuid()}};
        if(device_timing.read) { report["device_timing_scope"]="process-wide synchronized deltas; DSP batch includes preparation/cache flush; host queue read includes DSP wait; intervals overlap and are not additive; cycles/us is a DSP counter rate, not HMX clock readback"; }
        if(automatic) { report["path_decision"]=plans.at("path_decision"); }
        if(state_auto) {
            report["runtime_decision"]={{"state_correction",plans.at("state_correction")},{"decision_device_total_us",plans.at("decision_device_total_us")},
                {"selected_profile",plans.value("selected_profile",std::string())},{"custom_candidates",plans.at("custom_candidates")}};
        }
        if(automatic && plans.contains("execution_policy")) { report["planned_execution_policy"]=plans.at("execution_policy"); }
        if(feedback) { report["feedback_history"]=json::array(); }
        if(custom) { report["placement_source_tokens"]=plans["tokens"]; }
        if(custom) {
            report["placements"]=json::array();
            for(auto & l:engine.layers) {
                auto p=plan_json(l->plan); p["layer"]=l->il;
                if(l->single_layout) {
                    auto & w=l->single_layout->weights;
                    p["unique_weight_payload_bytes"]=w.payload_bytes;
                    p["unique_weight_allocation_bytes"]=w.physical_bytes;
                    p["weight_owner_allocations"]=w.owners.size()+1;
                    p["weight_slice_handles"]=w.slices.size();
                    p["alias_checks_passed"]=true;
                    p["resident_width"]=l->resident_width;
                    p["resident"]={{"experts",l->resident.experts},{"shared",l->resident.shared}};
                    report["placements"].push_back(p); continue;
                }
                std::array<size_t,3> bytes{},allocated_bytes{};
                for(int b=0;b<3;++b) { if(l->banks[b]->buf || l->banks[b]->mapped_weights) { for(auto * w:l->banks[b]->w) { bytes[b]+=ggml_nbytes(w); } } }
                for(int b=0;b<3;++b) {
                    if(l->banks[b]->buf) { allocated_bytes[b]+=ggml_backend_buffer_get_size(l->banks[b]->buf.get()); }
                    if(l->resident.shared[b]) {
                        auto * s=l->shared_for(b);
                        for(auto * w:s->w) { bytes[b]+=ggml_nbytes(w); }
                        allocated_bytes[b]+=ggml_backend_buffer_get_size(s->buf.get());
                        if(s->scalar_buf) { allocated_bytes[b]+=ggml_backend_buffer_get_size(s->scalar_buf.get()); }
                    }
                }
                p["resident_width"]=l->resident_width;
                p["resident"]={{"experts",l->resident.experts},{"shared",l->resident.shared}};
                p["resident_weight_bytes"]=bytes; p["resident_buffer_bytes"]=allocated_bytes; report["placements"].push_back(p);
            }
        }
        if(custom) {
            report["activation_arena_bytes"]=engine.physical_arena_sizes();
            report["activation_arena_accounting"]="unique physical backings; backend aliases counted once";
            report["activation_arena_policy"]=engine.layers[0]->single_layout && engine.layers[0]->single_layout->pooled_arena?"pooled":"private";
        }
        std::vector<double> times; std::vector<float> logits,paired_reference;
        if(runtime_state.enabled) {
            report["runtime_state"]={{"schema_version",1},{"setup_us",runtime_state.setup_us},{"preload",preload_state},
                {"scope","preload before weight preparation; per-invocation before/after decode; post-state and CPU deltas are diagnostics, not current-decision inputs"},
                {"warmups",json::array()}};
        }
        int runs=mode=="capture"?1:reps+2;
        size_t warmed_allocations=0;
        std::array<std::array<double,4>,24> residual;
        for(auto & row:residual) { row.fill(1.0); }
        size_t feedback_allocations=0;
        int feedback_observations=0;
        for(int r=0;r<runs;++r) {
            bool active_parallel=reduce_threads>1;
            if(alternate_reduce_threads) { int position=(r-2)%4; active_parallel=r<2?r==1:position==1 || position==2; }
            for(auto & l:engine.layers) { l->parallel_reduce=active_parallel; }
            bool active_overlap=overlap_gather;
            if(alternate_overlap_gather) { int position=(r-2)%4; active_overlap=r<2?r==1:position==1 || position==2; }
            for(auto & l:engine.layers) { l->overlap_gather=active_overlap; }
            bool active_direct=direct_callback_io;
            bool active_reduce=fused_host_reduce;
            if(alternate_host_reduce) {
                int position=(r-2)%4;
                active_reduce=r<2?r==1:position==1 || position==2;
                for(auto & l:engine.layers) { l->fused_host_reduce=active_reduce; }
            }
            if(alternate_callback_io) {
                int position=(r-2)%4;
                active_direct=r<2?r==1:position==1 || position==2;
                for(auto & l:engine.layers) { l->direct_callback_io=active_direct; }
            }
            std::array<resident_placement_model::prediction,24> predicted;
            std::array<double,24> adapted{};
            auto residual_before=residual;
            double feedback_cost=0;
            if(feedback) {
                auto start=timer::now();
                for(int il=0;il<24;++il) {
                    auto & l=*engine.layers[il]; std::vector<int> counts(60);
                    if(l.last.contains("expert_token_counts")) { counts=l.last["expert_token_counts"].get<std::vector<int>>(); }
                    else { for(int id:l.expected_ids) { ++counts[id]; } }
                    predicted[il]=feedback->predict(l.nt,counts,l.plan).ffn;
                    double critical=0;
                    for(int b=0;b<3;++b) { critical=std::max(critical,predicted[il].backend[b]*residual[il][b]); }
                    adapted[il]=critical+predicted[il].host*residual[il][3];
                }
                feedback_cost+=elapsed(start);
            }
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            json chunk_layers=json::array();
            auto state_before=runtime_state.sample();
            auto htp_before=device_timing.snapshot();
            rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
            const double request_begin_us=runtime_state.enabled?moe_runtime_state::now_us():0;
            auto start=timer::now(); int code=0;
            for(int offset=0;offset<nt;offset+=prefill_chunk) {
                auto batch=llama_batch_get_one(tokens.data()+offset,prefill_chunk);
                code=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
                require(code==0,"prefill chunk failed");
                if(prefill_chunk<nt) {
                    json layers=json::array(); for(auto & l:engine.layers) { layers.push_back(l->last); }
                    chunk_layers.push_back({{"offset",offset},{"layers",std::move(layers)}});
                }
            }
            auto * output=llama_get_logits_ith(ctx.get(),-1); require(code==0 && output,"native decode failed");
            logits.assign(output,output+nv); double us=elapsed(start);
            const double request_end_us=runtime_state.enabled?moe_runtime_state::now_us():0;
            getrusage(RUSAGE_SELF,&after);
            auto htp_after=device_timing.snapshot();
            auto state_after=runtime_state.sample();
            json state_row;
            if(runtime_state.enabled) {
                auto cpu_us=[](const timeval & v) { return double(v.tv_sec)*1000000+v.tv_usec; };
                double overhead=state_before.at("observation_us").get<double>()+state_after.at("observation_us").get<double>();
                state_row={{"before",state_before},{"after",state_after},{"observation_us",overhead},
                    {"request_begin_monotonic_us",request_begin_us},{"request_end_monotonic_us",request_end_us},
                    {"cpu_user_us",cpu_us(after.ru_utime)-cpu_us(before.ru_utime)},
                    {"cpu_system_us",cpu_us(after.ru_stime)-cpu_us(before.ru_stime)},
                    {"voluntary_switches",after.ru_nvcsw-before.ru_nvcsw},{"involuntary_switches",after.ru_nivcsw-before.ru_nivcsw},
                    {"total_with_observation_us",us+overhead}};
                if(device_timing.read) { state_row["htp_timing"]=device_timing.delta(htp_before,htp_after); }
                if(r<2) { report["runtime_state"]["warmups"].push_back({{"invocation",r},{"total_us",us},{"state",state_row}}); }
            }
            if(feedback) {
                // Predict BEFORE this invocation; update only AFTER observing it.
                // Ownership/banks never change. This is a causal shadow estimator.
                auto start=timer::now(); double fixed_sum=0,adapted_sum=0,actual_sum=0;
                std::array<std::array<double,4>,24> frozen_components{},measured_components{};
                size_t current_allocations=0;
                for(auto & l:engine.layers) { current_allocations+=l->allocations; }
                bool accept=r>0 && current_allocations==feedback_allocations;
                for(int il=0;il<24;++il) {
                    const auto & l=engine.layers[il]->last;
                    double actual=l["gather_us"].get<double>()+l["expert_wall_us"].get<double>()+l["combine_us"].get<double>();
                    std::array<double,4> observation{},prior{};
                    for(int b=0;b<3;++b) { observation[b]=l["backend_phases_us"][b][1]; prior[b]=predicted[il].backend[b]; }
                    observation[3]=actual-*std::max_element(observation.begin(),observation.begin()+3); prior[3]=predicted[il].host;
                    frozen_components[il]=prior; measured_components[il]=observation;
                    for(int b=0;b<4;++b) { if(accept && prior[b]>1) {
                        double ratio=std::max(.25,std::min(4.,observation[b]/prior[b]));
                        residual[il][b]=.5*residual[il][b]+.5*ratio;
                    } }
                    fixed_sum+=predicted[il].total; adapted_sum+=adapted[il]; actual_sum+=actual;
                }
                feedback_cost+=elapsed(start);
                report["feedback_history"].push_back({{"invocation",r},{"observations_before",feedback_observations},{"warmup",r<2},
                    {"update_accepted",accept},{"expert_graph_allocations",current_allocations},
                    {"frozen_ffn_predicted_us",fixed_sum},{"adaptive_ffn_predicted_us",adapted_sum},
                    {"actual_ffn_us",actual_sum},{"feedback_compute_us",feedback_cost},
                    {"state_before",residual_before},{"state_after",residual},
                    {"frozen_components_us",frozen_components},{"measured_components_us",measured_components}});
                feedback_allocations=current_allocations;
                if(accept) { ++feedback_observations; }
            }
            if(alternate_callback_io || alternate_host_reduce || alternate_reduce_threads || alternate_overlap_gather || activation_arena) {
                if(r==0) { paired_reference=logits; }
                else { require(std::memcmp(logits.data(),paired_reference.data(),logits.size()*sizeof(float))==0,
                               "alternating callback logits differ"); }
            }
            size_t allocations=0; for(auto & l:engine.layers) { allocations+=l->allocations; }
            if(r==1) { warmed_allocations=allocations; }
            if(r>=2) {
                require(allocations==warmed_allocations,"allocation occurred after warmup"); times.push_back(us);
                json row={{"round",r-2},{"total_us",us},{"minor_faults",after.ru_minflt-before.ru_minflt},{"major_faults",after.ru_majflt-before.ru_majflt}};
                row["direct_callback_io"]=active_direct;
                row["fused_host_reduce"]=active_reduce;
                row["host_reduce_threads"]=active_parallel?reduce_threads:1;
                row["overlap_gather"]=active_overlap;
                if(runtime_state.enabled) { row["runtime_state"]=state_row; }
                if(device_timing.read) { row["htp_timing"]=device_timing.delta(htp_before,htp_after); }
                if(feedback) { row["feedback_compute_us"]=feedback_cost; row["total_with_feedback_us"]=us+feedback_cost; }
                if(custom) { row["layers"]=json::array(); for(auto & l:engine.layers) { row["layers"].push_back(l->last); } }
                if(prefill_chunk<nt) { row["chunks"]=std::move(chunk_layers); row["layers_scope"]="last chunk only; use chunks for full-request phase totals"; }
                report["samples"].push_back(row);
            }
        }
        for(const auto & sample:report["samples"]) {
            auto observe=[&](const json & layers) {
                for(const auto & layer:layers) {
                    if(layer.value("npu_device_gather",false)) { report["npu_device_gather_effective"]=true; }
                }
            };
            if(sample.contains("chunks")) { for(const auto & chunk:sample["chunks"]) { observe(chunk.at("layers")); } }
            else if(sample.contains("layers")) { observe(sample["layers"]); }
        }
        if(mode=="capture") {
            report["layers"]=json::array();
            for(int il=0;il<24;++il) {
                auto & l=capture.layers[il];
                require(l.nt>0,"missing capture layer shape");
                if(!capture.includes(il)) {
                    report["layers"].push_back({{"layer",il},{"tokens",l.nt},{"captured",false}});
                    continue;
                }
                require(l.input.size()==size_t(2048*l.nt) && l.probs.size()==size_t(60*l.nt) && l.ids.size()==size_t(4*l.nt) && l.output.size()==l.input.size(),"incomplete layer capture");
                std::ofstream out(std::string(argv[4])+".layer"+std::to_string(il)+".bin",std::ios::binary);
                binary_write(out,l.input); binary_write(out,l.probs); binary_write(out,l.ids); binary_write(out,l.output);
                report["layers"].push_back({{"layer",il},{"tokens",l.nt}});
            }
        } else { report["timing"]=stats(times,nt); }
        if(std::getenv("MOE_ROUTER_TRACE")) { report["router_placement_trace"]=router_trace.nodes; }
        for(float value:logits) { require(std::isfinite(value),"nonfinite prefill logits"); }
        { std::ifstream status("/proc/self/status"); std::string line;
          while(std::getline(status,line)) { if(line.rfind("VmRSS:",0)==0 || line.rfind("VmHWM:",0)==0 || line.rfind("VmSize:",0)==0) {
              report["process_memory"].push_back(line);
          } } }
        std::ofstream logits_file(std::string(argv[4])+".logits.bin",std::ios::binary); binary_write(logits_file,logits);
        report["top1"]=std::max_element(logits.begin(),logits.end())-logits.begin();
        if(quality_steps) {
            fprintf(stderr,"QUALITY_BEGIN steps=%d shared_launch=%s\n",quality_steps,decode_shared_launch.c_str());
            // Retain the final prefill KV and resident banks. Only activation
            // graphs change shape; these diagnostic calls are outside timing.
            std::ofstream quality_file(std::string(argv[4])+".quality.bin",std::ios::binary);
            json quality={{"teacher_forced",!forced_tokens.empty()},{"steps",json::array()},
                          {"scope","same resident weights; fixed-length diagnostic continuation, including EOG tokens"}};
            std::vector<float> qlogits=logits;
            for(auto & l:engine.layers) { l->expected_ids.clear(); }
            engine.decode_active=true;
            for(int step=0;step<=quality_steps;++step) {
                for(float v:qlogits) { require(std::isfinite(v),"nonfinite quality logits"); }
                binary_write(quality_file,qlogits);
                llama_token predicted=std::max_element(qlogits.begin(),qlogits.end())-qlogits.begin();
                std::vector<char> piece(256); int length=llama_token_to_piece(vocab,predicted,piece.data(),piece.size(),0,true);
                if(length<0) { piece.resize(-length); length=llama_token_to_piece(vocab,predicted,piece.data(),piece.size(),0,true); }
                require(length>=0,"quality token piece conversion failed");
                json row={{"step",step},{"top1",predicted},{"piece_bytes",std::vector<uint8_t>(piece.begin(),piece.begin()+length)},
                          {"eog",llama_vocab_is_eog(vocab,predicted)}};
                if(step<quality_steps) {
                    llama_token input=forced_tokens.empty()?predicted:forced_tokens[step]; row["next_input_token"]=input;
                    auto batch=llama_batch_get_one(&input,1);
                    auto begin=timer::now(); int code=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
                    auto * output=llama_get_logits_ith(ctx.get(),-1); require(code==0 && output,"quality decode failed");
                    qlogits.assign(output,output+nv); row["next_decode_us"]=elapsed(begin);
                    if(custom) { row["next_layers"]=json::array(); for(auto & l:engine.layers) { row["next_layers"].push_back(l->last); } }
                }
                quality["steps"].push_back(row);
            }
            report["quality"]=quality;
        }
        if(decode_reps) {
            fprintf(stderr,"DECODE_BEGIN reps=%d shared_launch=%s\n",decode_reps,decode_shared_launch.c_str());
            // Freeze the initial diagnostic sequence, including for the CPU
            // teacher. Rebuild identical KV prefixes outside each timed sequence.
            std::vector<llama_token> teacher;
            for(int step=0;step<quality_steps;++step) { teacher.push_back(report["quality"]["steps"][step]["next_input_token"]); }
            json decode={{"teacher_tokens",teacher},{"warmup_sequences",2},{"samples",json::array()},
                {"scope","complete single-token llama_decode + synchronize; per-step logits copy/check, prefix rebuild and report serialization excluded; resident weights; real routing; no phase migration"}};
            std::vector<std::vector<float>> reference(quality_steps+1);
            std::vector<double> sequence_times,token_times;
            std::ofstream decode_file(std::string(argv[4])+".decode.bin",std::ios::binary);
            size_t warm_allocations=0;
            for(int repetition=-2;repetition<decode_reps;++repetition) {
                engine.decode_active=false;
                llama_memory_clear(llama_get_memory(ctx.get()),true);
                auto prefix_batch=llama_batch_get_one(tokens.data(),nt);
                auto prefix_start=timer::now();
                require(llama_decode(ctx.get(),prefix_batch)==0,"decode benchmark prefix failed"); llama_synchronize(ctx.get());
                double prefix_us=elapsed(prefix_start);
                engine.decode_active=true;
                json sequence={{"round",repetition},{"prefix_rebuild_us",prefix_us},{"steps",json::array()}};
                double total=0;
                for(int position=0;position<=quality_steps;++position) {
                    if(position>0) {
                        llama_token input=teacher[position-1]; auto batch=llama_batch_get_one(&input,1);
                        rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
                        auto begin=timer::now(); int code=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
                        double us=elapsed(begin); require(code==0,"decode benchmark failed");
                        getrusage(RUSAGE_SELF,&after); total+=us;
                        if(repetition>=0) {
                            token_times.push_back(us);
                            json row={{"step",position-1},{"total_us",us},{"minor_faults",after.ru_minflt-before.ru_minflt},{"major_faults",after.ru_majflt-before.ru_majflt}};
                            if(custom) { row["layers"]=json::array(); for(auto & l:engine.layers) { row["layers"].push_back(l->last); } }
                            sequence["steps"].push_back(row);
                        }
                    }
                    auto * output=llama_get_logits_ith(ctx.get(),-1); require(output!=nullptr,"missing decode benchmark logits");
                    for(int i=0;i<nv;++i) { require(std::isfinite(output[i]),"nonfinite decode benchmark logits"); }
                    if(repetition==0) { reference[position].assign(output,output+nv); binary_write(decode_file,reference[position]); }
                    else if(repetition>0) { require(std::memcmp(reference[position].data(),output,nv*sizeof(float))==0,"repeated decode logits differ"); }
                }
                size_t allocated=0; for(auto & l:engine.layers) { allocated+=l->allocations; }
                if(repetition==-1) { warm_allocations=allocated; }
                if(repetition>=0) {
                    require(allocated==warm_allocations,"decode graph allocated after warmup");
                    sequence["total_us"]=total; sequence["expert_graph_allocations"]=allocated;
                    sequence_times.push_back(total); decode["samples"].push_back(sequence);
                }
            }
            decode["sequence_timing"]=stats(sequence_times,quality_steps);
            decode["token_timing"]=stats(token_times,1);
            decode["all_repeat_logits_bitwise_equal"]=true;
            report["decode_benchmark"]=decode;
            fprintf(stderr,"DECODE_RESULT %s median_sequence_ms=%.3f tokens_s=%.3f\n",mode.c_str(),decode["sequence_timing"]["median_us"].get<double>()/1000,decode["sequence_timing"]["expert_tokens_per_second"].get<double>());
        }
        if(alternate_callback_io || alternate_host_reduce || alternate_reduce_threads || alternate_overlap_gather || activation_arena) { report["all_round_logits_bitwise_equal"]=true; }
        if(std::getenv("MOE_ROUTER_TRACE")) { report["router_placement_trace"]=router_trace.nodes; }
        size_t activation_bytes=0;
        size_t logical_activation_bytes=0;
        std::set<const shared_activation_buffer *> seen;
        for(auto & l:engine.layers) { for(auto & cache:l->jobs) { for(auto & entry:cache) {
            auto & j=*entry.second;
            logical_activation_bytes+=shared_activation_buffer::required_size(j.dev,j.ctx.get());
            if(j.shared_buffer) {
                if(seen.insert(j.shared_buffer.get()).second) { activation_bytes+=j.shared_buffer->desc.size; }
            } else { activation_bytes+=ggml_backend_buffer_get_size(j.buf.get()); }
        } } }
        report["expert_activation_buffer_bytes"]=activation_bytes;
        report["expert_activation_graph_bytes"]=logical_activation_bytes;
        report["arena_capacity_bytes"]=engine.physical_arena_sizes();
        report["shared_scratch_capacity_bytes"]=engine.shared_scratch?engine.shared_scratch->desc.size:0;
        report["shared_scratch_physical_backings"]=engine.shared_scratch?1:0;
        std::ifstream status("/proc/self/status"); report["process_status"]=std::string((std::istreambuf_iterator<char>(status)),{});
        std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"full-model report write failed");
        if(mode!="capture") { fprintf(stderr,"FULL_RESULT %s nt=%d median_ms=%.3f tokens_s=%.3f\n",mode.c_str(),nt,report["timing"]["median_us"].get<double>()/1000,report["timing"]["expert_tokens_per_second"].get<double>()); }
        // Backend-owned banks must be released before the devices themselves.
        ctx.reset(); engine.shared_worker.reset(); engine.decode_branches.clear(); engine.early_shared_branches.clear(); engine.layers.clear();
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
#endif
