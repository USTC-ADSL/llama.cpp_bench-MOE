#define EXPERT_ORACLE_NO_MAIN
#include "expert-oracle-probe.cpp"
#include "llama.h"

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
    std::map<std::pair<int,int>,std::unique_ptr<oracle_job>> jobs[3];
    std::array<std::array<int,60>,3> local{};
    std::vector<int32_t> expected_ids;
    json last;
    size_t allocations=0;
    bool shared_activations=false;
    bool direct_callback_io=false;
    bool fused_host_reduce=false;
    std::shared_ptr<shared_activation_buffer> arenas[3];
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
        require(elements==size_t(2048*nt) && probability_count==size_t(60*nt) && ids.size()==size_t(4*nt),"FFN input shape mismatch");
        auto start=timer::now();
        std::vector<int> pairs[3];
        for(int p=0;p<4*nt;++p) { require(ids[p]>=0 && ids[p]<60,"invalid route ID"); pairs[plan.owner[ids[p]]].push_back(p); }
        bool native=false; for(auto & ps:pairs) { native|=ps.size()==ids.size(); }
        oracle_job * work[3]{};
        for(int b=0;b<3;++b) {
            if(pairs[b].empty() && b!=plan.shared) { continue; }
            int rows=pairs[b].empty()?0:native?nt:int(pairs[b].size()); int k=native?4:1;
            auto & ptr=jobs[b][{rows,k}];
            if(!ptr) { ptr=std::make_unique<oracle_job>(*banks[b],b==plan.shared?shared.get():nullptr,rows,k,nt,native,false,shared_activations,arenas[b]); ++allocations; }
            work[b]=ptr.get(); auto & j=*ptr; j.reset_index_padding(); j.set_shared_input(x,elements);
            if(native && j.x) {
                std::copy_n(x,elements,j.input_data());
                for(int t=0;t<nt;++t) { for(int kk=0;kk<4;++kk) { j.index_data()[t*j.stride+kk]=local[b][ids[t*4+kk]]; } }
            } else {
                for(size_t i=0;i<pairs[b].size();++i) {
                    int p=pairs[b][i]; require(local[b][ids[p]]>=0,"expert not resident");
                    std::copy_n(x+(p/4)*2048,2048,j.input_data()+i*2048); j.index_data()[i*j.stride]=local[b][ids[p]];
                }
            }
        }
        double gather=elapsed(start); start=timer::now();
        std::array<double,3> phases[3]{}; std::array<double,2> intervals[3]{};
        std::future<void> futures[3]; int inline_b=-1;
        for(int b=0;b<3;++b) { if(work[b]) { inline_b=b; } }
        auto execute=[&](int b) { intervals[b][0]=elapsed(start); phases[b]=work[b]->run(); intervals[b][1]=elapsed(start); };
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
            for(int t=0;t<nt;++t) {
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
            }
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
                  {"expert_token_counts",counts},{"allocations",allocations}};
        }
        return owned_result;
    }
};

struct full_engine {
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
                              l->banks[b]->buf?4*l->nt:0,1,l->nt,false,false,true,{},true);
            capacities[b]=std::max(capacities[b],shared_activation_buffer::required_size(*devs[b],layout.ctx.get()));
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

static int search_all(const char * model_path,const std::string & capture_path,const char * output) {
    std::ifstream in(capture_path); json capture; in>>capture;
    ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(model_path,{true,&raw})); ggml_context_ptr meta(raw);
    std::unique_ptr<FILE,decltype(&fclose)> f(fopen(model_path,"rb"),fclose); require(gguf && meta && f,"search model open failed");
    source src{gguf.get(),meta.get(),f.get()}; device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
    json report={{"tokens",capture["tokens"]},{"token_ids",capture["token_ids"]},{"layers",json::array()},{"scope","offline real captured routes; bounded joint routed/shared placement search"}};
    const char * names[]={"cpu","npu","gpu"};
    for(int il=0;il<24;++il) {
        int nt=capture["layers"][il]["tokens"];
        std::ifstream bin(capture_path+".layer"+std::to_string(il)+".bin",std::ios::binary);
        auto x=binary_read<float>(bin,2048*nt); auto probs=binary_read<float>(bin,60*nt);
        auto ids=binary_read<int32_t>(bin,4*nt); auto reference=binary_read<float>(bin,2048*nt);
        std::vector<int> counts(60),order; for(auto id:ids) { ++counts[id]; }
        for(int e=0;e<60;++e) { if(counts[e]) { order.push_back(e); } }
        std::sort(order.begin(),order.end(),[&](int a,int b) { return counts[a]==counts[b]?a<b:counts[a]<counts[b]; });
        std::vector<placement> candidates;
        for(int main=0;main<3;++main) { for(int shared=0;shared<3;++shared) {
            candidates.push_back({std::string(names[main])+"_shared_"+names[shared],std::vector<int>(60,main),shared});
        } }
        for(int other : {0,2}) { for(int count : {1,4,8,16,24,32}) {
            if(count>=int(order.size())) { continue; }
            for(int shared : {1,2}) {
                placement p{"npu_tail"+std::to_string(count)+"_"+names[other]+"_shared_"+names[shared],std::vector<int>(60,1),shared};
                for(int i=0;i<count;++i) { p.owner[order[i]]=other; } candidates.push_back(p);
            }
        } }
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

#ifndef FULLMODEL_ORACLE_NO_MAIN
int main(int argc,char ** argv) {
    try {
        require(argc==8,"usage: fullmodel-oracle MODEL PROMPT TOKENS OUTPUT MODE REPS PLANS_OR_DASH; modes capture,search,cpu,gpu,npu,custom,branch,custom-npu");
        ggml_backend_load_all(); llama_backend_init();
        std::string mode=argv[5]; int nt=std::stoi(argv[3]),reps=std::stoi(argv[6]);
        require(nt>0 && reps>0,"tokens and repetitions must be positive");
        if(mode=="search") { return search_all(argv[1],argv[7],argv[4]); }
        bool custom=mode=="custom" || mode=="branch" || mode=="custom-npu";
        bool transfer_placement=std::getenv("MOE_TRANSFER_PLACEMENT") && std::string(std::getenv("MOE_TRANSFER_PLACEMENT"))=="1";
        require(!transfer_placement || custom,"placement transfer requires custom mode");
        bool shared_activations=std::getenv("MOE_SHARED_ACTIVATIONS") && std::string(std::getenv("MOE_SHARED_ACTIVATIONS"))=="1";
        bool direct_callback_io=std::getenv("MOE_DIRECT_CALLBACK_IO") && std::string(std::getenv("MOE_DIRECT_CALLBACK_IO"))=="1";
        bool alternate_callback_io=std::getenv("MOE_ALTERNATE_CALLBACK_IO") && std::string(std::getenv("MOE_ALTERNATE_CALLBACK_IO"))=="1";
        bool activation_arena=std::getenv("MOE_ACTIVATION_ARENA") && std::string(std::getenv("MOE_ACTIVATION_ARENA"))=="1";
        bool fused_host_reduce=std::getenv("MOE_FUSED_HOST_REDUCE") && std::string(std::getenv("MOE_FUSED_HOST_REDUCE"))=="1";
        bool alternate_host_reduce=std::getenv("MOE_ALTERNATE_HOST_REDUCE") && std::string(std::getenv("MOE_ALTERNATE_HOST_REDUCE"))=="1";
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
        json plans; if(custom) { std::ifstream in(argv[7]); in>>plans; require((transfer_placement || plans["tokens"].get<int>()==nt) && plans["layers"].size()==24,"plan shape mismatch"); }
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
        if(custom && !transfer_placement) { require(tokens==plans["token_ids"].get<std::vector<int32_t>>(),"fixed prompt token mismatch"); }
        std::unique_ptr<device> devices[3];
        full_engine engine;
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
                if(!transfer_placement) { engine.layers.back()->expected_ids=plans["layers"][il]["ids"].get<std::vector<int32_t>>(); }
                fprintf(stderr,"RESIDENT layer=%d plan=%s\n",il,p.name.c_str());
            }
            if(activation_arena) { engine.prepare_arenas(devs); }
            require(llama_model_set_moe_graph_callback(model.get(),full_engine::build,&engine),"model rejected FFN callback");
        }
        capture_all capture;
        auto cp=llama_context_default_params(); cp.n_ctx=nt; cp.n_batch=nt; cp.n_ubatch=nt;
        cp.n_threads=4; cp.n_threads_batch=4; cp.type_k=GGML_TYPE_F16; cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_kqv=accelerator; cp.op_offload=accelerator;
        if(mode=="capture") { cp.cb_eval=capture_all::callback; cp.cb_eval_user_data=&capture; }
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        require(ctx!=nullptr,"native context init failed");
        int nv=llama_vocab_n_tokens(vocab);
        json report={{"tokens",nt},{"token_ids",tokens},{"mode",mode},{"samples",json::array()},
            {"scope","native complete model prefill including last-token logits and retained KV; dynamic real routing; weights resident; context reset outside timing"}};
        report["shared_activations"]=shared_activations;
        report["direct_callback_io"]=direct_callback_io;
        report["alternate_callback_io"]=alternate_callback_io;
        report["activation_arena"]=activation_arena;
        report["fused_host_reduce"]=fused_host_reduce;
        report["alternate_host_reduce"]=alternate_host_reduce;
        report["transfer_placement"]=transfer_placement;
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
        int runs=mode=="capture"?1:reps+2;
        size_t warmed_allocations=0;
        for(int r=0;r<runs;++r) {
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
            llama_memory_clear(llama_get_memory(ctx.get()),true);
            auto batch=llama_batch_get_one(tokens.data(),nt);
            rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
            auto start=timer::now(); int code=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
            auto * output=llama_get_logits_ith(ctx.get(),-1); require(code==0 && output,"native decode failed");
            logits.assign(output,output+nv); double us=elapsed(start);
            getrusage(RUSAGE_SELF,&after);
            if(alternate_callback_io || alternate_host_reduce || activation_arena) {
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
        if(alternate_callback_io || alternate_host_reduce || activation_arena) { report["all_round_logits_bitwise_equal"]=true; }
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
