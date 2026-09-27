#define EXPERT_ORACLE_NO_MAIN
#include "expert-oracle-probe.cpp"
#include "llama.h"
#include "moe-placement-model.h"
#include "moe-context-model.h"
#include "moe-execution-model.h"
#include "moe-token-workers.h"
#include "moe-runtime-state.h"
#include "moe-state-correction.h"

#include <iomanip>
#include <sys/resource.h>
#include <set>
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

struct capture_all {
    std::array<captured_layer,24> layers;
    static bool callback(ggml_tensor * t,bool ask,void * opaque) {
        std::string name=t->name;
        auto dash=name.rfind('-');
        if(dash==std::string::npos) { return ask ? false : true; }
        if(name.find_first_not_of("0123456789",dash+1)!=std::string::npos) { return ask ? false : true; }
        std::string prefix=name.substr(0,dash);
        bool wanted=prefix=="ffn_norm" || prefix=="ffn_moe_probs" || prefix=="ffn_moe_topk" || prefix=="ffn_out";
        if(ask) { return wanted; }
        if(!wanted) { return true; }
        int il=std::stoi(name.substr(dash+1)); require(il>=0 && il<24,"invalid capture layer");
        auto & l=static_cast<capture_all *>(opaque)->layers[il];
        l.nt=t->ne[1];
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
    std::map<std::tuple<int,int,int>,std::unique_ptr<oracle_job>> jobs[3];
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
    std::shared_ptr<shared_activation_buffer> arenas[3];
    // Separate contracts for split routed, native top-k, and shared-only graphs.
    std::unique_ptr<shared_activation_layout::reservation> reservations[3][3];
    partitioned_ffn(device * devs[3],source & src,int layer,int tokens,const placement & p,bool compact,bool shared_buffers=false) : il(layer),nt(tokens),plan(p),shared_activations(shared_buffers) {
        require(plan.owner.size()==60 && plan.shared>=0 && plan.shared<3,"invalid placement");
        for(int b=0;b<3;++b) {
            local[b].fill(-1); std::vector<int> subset;
            for(int e=0;e<60;++e) {
                require(plan.owner[e]>=0 && plan.owner[e]<3,"invalid expert owner");
                if(!compact || plan.owner[e]==b) { local[b][e]=int(subset.size()); subset.push_back(e); }
            }
            banks[b]=std::make_unique<bank>(*devs[b],src,il,compact ? &subset : nullptr);
        }
        shared=std::make_unique<shared_bank>(*devs[plan.shared],src,il);
    }
    void set_shared(device * devs[3],source & src,int backend) {
        if(backend!=plan.shared) { shared.reset(); shared=std::make_unique<shared_bank>(*devs[backend],src,il); }
        plan.shared=backend;
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
        std::vector<int> pairs[3];
        for(int p=0;p<4*nt;++p) { require(ids[p]>=0 && ids[p]<60,"invalid route ID"); pairs[plan.owner[ids[p]]].push_back(p); }
        bool native=false; for(auto & ps:pairs) { native|=ps.size()==ids.size(); }
        oracle_job * work[3]{};
        for(int b=0;b<3;++b) {
            if(pairs[b].empty() && b!=plan.shared) { continue; }
            int rows=pairs[b].empty()?0:native?nt:int(pairs[b].size()); int k=native?4:1;
            auto & ptr=jobs[b][{rows,k,nt}];
            // A compact one-Expert CPU repack tensor is 2D: its backend accepts
            // MUL_MAT, not MUL_MAT_ID. oracle_job uses the equivalent single-bank
            // path only when ne[2]==1 (and asserts k==1).
            if(!ptr) {
                auto * reserved=reservations[b][!rows?2:native?1:0].get();
                require(!reuse_lifetimes || reserved,"missing activation layout reservation");
                ptr=std::make_unique<oracle_job>(*banks[b],b==plan.shared?shared.get():nullptr,rows,k,nt,native,b==0,shared_activations,arenas[b],false,reuse_lifetimes,reserved);
                ++allocations;
            }
            work[b]=ptr.get();
        }
        std::array<double,3> branch_gather{};
        auto prepare=[&](int b) {
            auto begin=timer::now(); auto & j=*work[b];
            j.reset_index_padding(); j.set_shared_input(x,elements);
            if(native && j.x) {
                std::copy_n(x,elements,j.input_data());
                for(int t=0;t<nt;++t) { for(int kk=0;kk<4;++kk) { j.index_data()[t*j.stride+kk]=local[b][ids[t*4+kk]]; } }
            } else {
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
                  {"backend_phases_us",phases},{"intervals_us",intervals},{"pairs",{pairs[0].size(),pairs[1].size(),pairs[2].size()}},
                  {"changed_route_slots",changed},{"route_reference_available",expected_ids.size()==ids.size()},
                  {"reduce_threads",parallel_reduce && reduce_workers && nt>=reduce_workers->count()?reduce_workers->count():1},
                  {"overlap_gather",overlap_gather},{"backend_gather_us",branch_gather},{"expert_wall_includes_gather",overlap_gather},
                  {"expert_token_counts",counts},{"allocations",allocations}};
        }
        return owned_result;
    }
};

struct full_engine {
    std::unique_ptr<moe_token_workers> reduce_workers;
    std::shared_ptr<shared_activation_buffer> arenas[3];
    std::vector<std::unique_ptr<partitioned_ffn>> layers;
    bool record=false;
    void prepare_arenas(device * devs[3]) {
        std::array<size_t,3> capacities{};
        for(auto & l:layers) { for(int b=0;b<3;++b) {
            require(l->shared_activations,"arena requires shared layer");
            if(!l->banks[b]->buf && l->plan.shared!=b) { continue; }
            // k=1 with all 4*nt routed pairs bounds both split and native k=4 graphs.
            oracle_job layout(*l->banks[b],l->plan.shared==b?l->shared.get():nullptr,
                              l->banks[b]->buf?4*l->nt:0,1,l->nt,false,b==0,true,{},true);
            capacities[b]=std::max(capacities[b],shared_activation_buffer::required_size(*devs[b],layout.ctx.get(),l->reuse_lifetimes?layout.graph:nullptr));
            if(l->reuse_lifetimes) {
                // Reuse the maximum-shape offsets, not a separately packed small
                // graph: best-fit capacity need not be monotonic in routed rows.
                l->reservations[b][l->banks[b]->buf?0:2]=std::make_unique<shared_activation_layout::reservation>(
                    shared_activation_layout(*devs[b],layout.ctx.get(),layout.graph).freeze());
            }
            if(l->reuse_lifetimes && l->banks[b]->buf && l->banks[b]->w[0]->ne[2]>=4) {
                // Native k=4 has a different dependency graph (shared aliases x).
                // Reserve it separately rather than assume split liveness bounds it.
                oracle_job native(*l->banks[b],l->plan.shared==b?l->shared.get():nullptr,
                                  l->nt,4,l->nt,true,false,true,{},true);
                capacities[b]=std::max(capacities[b],shared_activation_buffer::required_size(*devs[b],native.ctx.get(),native.graph));
                l->reservations[b][1]=std::make_unique<shared_activation_layout::reservation>(
                    shared_activation_layout(*devs[b],native.ctx.get(),native.graph).freeze());
            }
            if(l->reuse_lifetimes && l->banks[b]->buf && l->plan.shared==b) {
                oracle_job only_shared(*l->banks[b],l->shared.get(),0,1,l->nt,false,b==0,true,{},true);
                auto reserved=shared_activation_layout(*devs[b],only_shared.ctx.get(),only_shared.graph).freeze();
                capacities[b]=std::max(capacities[b],reserved.bytes);
                l->reservations[b][2]=std::make_unique<shared_activation_layout::reservation>(std::move(reserved));
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
        return ggml_map_custom3(ctx,input,probs,ids,compute,1,engine.layers[il].get());
    }
};

static placement parse_plan(const json & p) { return {p.at("name"),p.at("owner").get<std::vector<int>>(),p.at("shared")}; }
static json plan_json(const placement & p) { return {{"name",p.name},{"owner",p.owner},{"shared",p.shared}}; }

static int search_all(const char * model_path,const std::string & capture_path,const char * output,bool profile=false,int repetitions=3) {
    std::ifstream in(capture_path); json capture; in>>capture;
    ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(model_path,{true,&raw})); ggml_context_ptr meta(raw);
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(model_path,"rb"),fclose); require(gguf && meta && f,"search model open failed");
    source src{gguf.get(),meta.get(),f.get()}; device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
    json report={{"tokens",capture["tokens"]},{"token_ids",capture["token_ids"]},{"layers",json::array()},{"scope","offline real captured routes; bounded joint routed/shared placement search"}};
    report["optimized_profile"]=profile;
    if(profile) { report["profile_contract"]="compact banks; shared activation arenas; direct output; fused reduction; two warmups; weight preparation excluded"; }
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
        if(profile) {
            json result={{"layer",il},{"tokens",nt},{"counts",counts},{"ids",ids},{"candidates",json::array()}};
            for(const auto & p:candidates) {
                // Use exactly the resident confirmation layout; never time conversion.
                full_engine engine;
                engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,nt,p,true,true));
                auto & ff=*engine.layers.back(); ff.fused_host_reduce=true;
                engine.prepare_arenas(devs);
                std::vector<float> actual(x.size());
                auto run=[&](bool record) { ff.run_into(x.data(),x.size(),probs.data(),probs.size(),ids,record,actual.data()); };
                run(false); run(false);
                auto check=chain_difference(actual,reference,nt);
                auto warmed=ff.allocations;
                json item=plan_json(p); item["check"]=check; item["samples"]=json::array();
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
        require(custom || mode=="capture" || mode=="cpu" || mode=="npu" || mode=="gpu","invalid mode");
        if(custom && !automatic) { std::ifstream in(argv[7]); in>>plans; require((transfer_placement || plans["tokens"].get<int>()==nt) && plans["layers"].size()==24,"plan shape mismatch"); }
        std::unique_ptr<fullmodel_context_model> feedback;
        if(std::getenv("MOE_FEEDBACK_MODEL")) {
            require(custom && !transfer_placement,"shadow feedback requires custom mode with matching route reference");
            std::ifstream in(std::getenv("MOE_FEEDBACK_MODEL")); json parameters; in>>parameters;
            feedback=std::make_unique<fullmodel_context_model>(parameters,nt);
        }
        ggml_backend_dev_t target[]={ggml_backend_dev_by_name(mode=="gpu"?"GPUOpenCL":"HTP0"),nullptr};
        require(!accelerator || target[0],"requested accelerator unavailable");
        auto mp=llama_model_default_params(); mp.n_gpu_layers=accelerator?99:0;
        mp.devices=accelerator?target:nullptr;
        llama_model_tensor_buft_override overrides[]={
            {"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"))},
            {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_inp)_shexp\\.weight",ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"))},
            {nullptr,nullptr}};
        if(custom) { mp.tensor_buft_overrides=overrides; mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP; }
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(model!=nullptr,"native model load failed");
        std::ifstream prompt_file(argv[2]); std::string prompt((std::istreambuf_iterator<char>(prompt_file)),{});
        auto * vocab=llama_model_get_vocab(model.get()); std::vector<llama_token> tokens(prompt.size()+32);
        int count=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
        require(count>=nt,"prompt too short"); tokens.resize(nt);
        if(automatic || (custom && !transfer_placement)) { require(tokens==plans["token_ids"].get<std::vector<int32_t>>(),"fixed prompt token mismatch"); }
        std::unique_ptr<device> devices[3];
        full_engine engine;
        if(reduce_threads>1) { engine.reduce_workers=std::make_unique<moe_token_workers>(reduce_threads); }
        if(custom) {
            ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
            std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose); require(gguf && meta && f,"compact model open failed");
            source src{gguf.get(),meta.get(),f.get()};
            const char * names[]={"CPU","HTP0","GPUOpenCL"}; device * devs[3];
            for(int b=0;b<3;++b) { devices[b]=std::make_unique<device>(names[b]); devs[b]=devices[b].get(); }
            for(int il=0;il<24;++il) {
                placement p=mode=="custom-npu"?placement{"all_npu",std::vector<int>(60,1),1}:parse_plan(plans["layers"][il][mode=="branch"?"branch":"selected"]);
                int layer_tokens=transfer_placement?(il==23?1:nt):plans["layers"][il]["tokens"].get<int>();
                engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,layer_tokens,p,true,shared_activations));
                engine.layers.back()->direct_callback_io=direct_callback_io;
                engine.layers.back()->fused_host_reduce=fused_host_reduce;
                engine.layers.back()->reuse_lifetimes=activation_lifetimes;
                engine.layers.back()->reduce_workers=engine.reduce_workers.get();
                engine.layers.back()->parallel_reduce=reduce_threads>1;
                engine.layers.back()->overlap_gather=overlap_gather;
                if(!transfer_placement) { engine.layers.back()->expected_ids=plans["layers"][il]["ids"].get<std::vector<int32_t>>(); }
                fprintf(stderr,"RESIDENT layer=%d plan=%s\n",il,p.name.c_str());
            }
            if(activation_arena) { engine.prepare_arenas(devs); }
            require(llama_model_set_moe_graph_callback(model.get(),full_engine::build,&engine),"model rejected FFN callback");
        }
        capture_all capture;
        auto cp=llama_context_default_params(); cp.n_ctx=quality_steps?nt+quality_steps+32:nt; cp.n_batch=nt; cp.n_ubatch=nt;
        cp.n_threads=4; cp.n_threads_batch=4; cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_kqv=accelerator; cp.op_offload=accelerator;
        if(mode=="capture") { cp.cb_eval=capture_all::callback; cp.cb_eval_user_data=&capture; }
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        require(ctx!=nullptr,"native context init failed");
        int nv=llama_vocab_n_tokens(vocab);
        for(auto t:forced_tokens) { require(t>=0 && t<nv,"teacher forcing token out of range"); }
        json report={{"tokens",nt},{"token_ids",tokens},{"mode",mode},{"samples",json::array()},
            {"scope","native complete model prefill including last-token logits and retained KV; dynamic real routing; weights resident; context reset outside timing"}};
        report["shared_activations"]=shared_activations;
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
        report["shadow_feedback"]=bool(feedback);
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
                std::array<size_t,3> bytes{};
                for(int b=0;b<3;++b) { if(l->banks[b]->buf) { for(auto * w:l->banks[b]->w) { bytes[b]+=ggml_nbytes(w); } } }
                for(auto * w:l->shared->w) { bytes[l->plan.shared]+=ggml_nbytes(w); }
                p["resident_weight_bytes"]=bytes; report["placements"].push_back(p);
            }
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
            auto batch=llama_batch_get_one(tokens.data(),nt);
            auto state_before=runtime_state.sample();
            rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
            auto start=timer::now(); int code=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
            auto * output=llama_get_logits_ith(ctx.get(),-1); require(code==0 && output,"native decode failed");
            logits.assign(output,output+nv); double us=elapsed(start);
            getrusage(RUSAGE_SELF,&after);
            auto state_after=runtime_state.sample();
            json state_row;
            if(runtime_state.enabled) {
                auto cpu_us=[](const timeval & v) { return double(v.tv_sec)*1000000+v.tv_usec; };
                double overhead=state_before.at("observation_us").get<double>()+state_after.at("observation_us").get<double>();
                state_row={{"before",state_before},{"after",state_after},{"observation_us",overhead},
                    {"cpu_user_us",cpu_us(after.ru_utime)-cpu_us(before.ru_utime)},
                    {"cpu_system_us",cpu_us(after.ru_stime)-cpu_us(before.ru_stime)},
                    {"voluntary_switches",after.ru_nvcsw-before.ru_nvcsw},{"involuntary_switches",after.ru_nivcsw-before.ru_nivcsw},
                    {"total_with_observation_us",us+overhead}};
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
                if(feedback) { row["feedback_compute_us"]=feedback_cost; row["total_with_feedback_us"]=us+feedback_cost; }
                if(custom) { row["layers"]=json::array(); for(auto & l:engine.layers) { row["layers"].push_back(l->last); } }
                report["samples"].push_back(row);
            }
        }
        if(mode=="capture") {
            report["layers"]=json::array();
            for(int il=0;il<24;++il) {
                auto & l=capture.layers[il];
                require(l.input.size()==size_t(2048*l.nt) && l.probs.size()==size_t(60*l.nt) && l.ids.size()==size_t(4*l.nt) && l.output.size()==l.input.size(),"incomplete layer capture");
                std::ofstream out(std::string(argv[4])+".layer"+std::to_string(il)+".bin",std::ios::binary);
                binary_write(out,l.input); binary_write(out,l.probs); binary_write(out,l.ids); binary_write(out,l.output);
                report["layers"].push_back({{"layer",il},{"tokens",l.nt}});
            }
        } else { report["timing"]=stats(times,nt); }
        std::ofstream logits_file(std::string(argv[4])+".logits.bin",std::ios::binary); binary_write(logits_file,logits);
        report["top1"]=std::max_element(logits.begin(),logits.end())-logits.begin();
        if(quality_steps) {
            // Retain the final prefill KV and resident banks. Only activation
            // graphs change shape; these diagnostic calls are outside timing.
            std::ofstream quality_file(std::string(argv[4])+".quality.bin",std::ios::binary);
            json quality={{"teacher_forced",!forced_tokens.empty()},{"steps",json::array()},
                          {"scope","same resident weights; fixed-length diagnostic continuation, including EOG tokens"}};
            std::vector<float> qlogits=logits;
            for(auto & l:engine.layers) { l->expected_ids.clear(); }
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
        if(alternate_callback_io || alternate_host_reduce || alternate_reduce_threads || alternate_overlap_gather || activation_arena) { report["all_round_logits_bitwise_equal"]=true; }
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
        report["arena_capacity_bytes"]=json::array();
        for(auto & arena:engine.arenas) { report["arena_capacity_bytes"].push_back(arena?arena->desc.size:0); }
        std::ifstream status("/proc/self/status"); report["process_status"]=std::string((std::istreambuf_iterator<char>(status)),{});
        std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"full-model report write failed");
        if(mode!="capture") { fprintf(stderr,"FULL_RESULT %s nt=%d median_ms=%.3f tokens_s=%.3f\n",mode.c_str(),nt,report["timing"]["median_us"].get<double>()/1000,report["timing"]["expert_tokens_per_second"].get<double>()); }
        // Backend-owned banks must be released before the devices themselves.
        ctx.reset(); engine.layers.clear();
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
#endif
