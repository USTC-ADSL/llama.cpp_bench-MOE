#include "online-moe-runtime.h"
#include "ggml-accel-budget.h"
#define DYNAMIC_MOE_NO_MAIN
#include "dynamic-moe.cpp"

int main(int argc,char **argv) {
    try {
        require(argc==8,"usage: online-moe MODEL PROMPT TOKENS COST_MODEL OUTPUT MODE(native|online) TAIL(-1=model)");
        int nt=std::stoi(argv[3]),tail=std::stoi(argv[7]); bool online=std::string(argv[6])=="online";
        require(online || std::string(argv[6])=="native","invalid mode");
        require(nt>=64 && nt<=768 && tail>=-1 && tail<=60,"invalid length/tail");
        ggml_log_set([](ggml_log_level l,const char *s,void*) { if(l!=GGML_LOG_LEVEL_DEBUG) { fputs(s,stderr); } },nullptr);
        const auto init_start=timer::now(); ggml_backend_load_all(); llama_backend_init();
        mapped_model mapping(argv[1]); ggml_context *raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gf && meta,"GGUF metadata failed");
        source src{gf.get(),meta.get(),nullptr,static_cast<const uint8_t *>(mapping.data),mapping.size};
        std::unique_ptr<device> devices[3]; device *devs[3]{};
        std::unique_ptr<online_moe_runtime> runtime;
        if(online) {
            std::ifstream input(argv[4]); json calibration; input>>calibration;
            const char *names[]={"CPU","HTP0","GPUOpenCL"};
            for(int b=0;b<3;++b) { devices[b]=std::make_unique<device>(names[b]); devs[b]=devices[b].get(); }
            runtime=std::make_unique<online_moe_runtime>(src,devs,nt,calibration,300,tail);
        }
        double source_init_us=elapsed(init_start);
        ggml_backend_dev_t targets[]={ggml_backend_dev_by_name("HTP0"),nullptr}; require(targets[0],"NPU unavailable");
        auto mp=llama_model_default_params(); mp.n_gpu_layers=99; mp.devices=targets;
        auto cpu=ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU"));
        llama_model_tensor_buft_override overrides[]={
            {"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",cpu},
            {"blk\\.[0-9]+\\.ffn_(gate|up|down|gate_inp)_shexp\\.weight",cpu}, {nullptr,nullptr}, {nullptr,nullptr}};
        if(online && std::getenv("MOE_ONLINE_PACKED_HEAD")) { overrides[2]={"^output\\.weight$",devs[0]->buft}; }
        if(online) { mp.tensor_buft_overrides=overrides; mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP; }
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        require(bool(model),"model load failed");
        if(online) { require(llama_model_set_moe_graph_callback(model.get(),online_moe_runtime::build,runtime.get()),"callback rejected"); }
        auto cp=llama_context_default_params(); cp.n_ctx=nt+32; cp.n_batch=cp.n_ubatch=nt;
        cp.n_threads=cp.n_threads_batch=4; cp.type_k=cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_kqv=cp.op_offload=true;
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(llama_init_from_model(model.get(),cp),llama_free);
        require(bool(ctx),"context creation failed");
        double load_us=elapsed(init_start);
        std::ifstream prompt_file(argv[2]); require(bool(prompt_file),"prompt open failed");
        std::string prompt((std::istreambuf_iterator<char>(prompt_file)),{});
        std::vector<llama_token> tokens(prompt.size()+32);
        auto *vocab=llama_model_get_vocab(model.get());
        int n=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
        require(n>=nt,"prompt too short"); tokens.resize(nt);
        auto status=[] { std::ifstream f("/proc/self/status"); return std::string((std::istreambuf_iterator<char>(f)),{}); };
        json report={{"mode",argv[6]},{"tokens",tokens},{"load_us",load_us},{"source_and_backend_init_us",source_init_us},
            {"protocol","first prompt execution only; no warmup, capture, route replay or prepared target expert cache; complete llama_decode+synchronize"},
            {"before_status",status()},{"forced_tail",tail}};
        rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
        auto batch=llama_batch_get_one(tokens.data(),nt); auto start=timer::now();
        int rc=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get()); double total_us=elapsed(start);
        getrusage(RUSAGE_SELF,&after);
        if(runtime && runtime->failure) { std::rethrow_exception(runtime->failure); }
        require(rc==0,"prefill failed");
        auto *logits=llama_get_logits_ith(ctx.get(),-1); require(logits,"missing logits");
        int nv=llama_vocab_n_tokens(vocab); std::vector<float> values(logits,logits+nv);
        for(float v:values) { require(std::isfinite(v),"nonfinite logits"); }
        report["total_us"]=total_us; report["tokens_per_second"]=nt*1e6/total_us;
        report["top1"]=std::max_element(values.begin(),values.end())-values.begin(); report["vocab_size"]=nv;
        report["major_faults"]=after.ru_majflt-before.ru_majflt; report["minor_faults"]=after.ru_minflt-before.ru_minflt;
        report["maxrss_kib"]=after.ru_maxrss; report["after_status"]=status();
        auto budget=ggml_budget_snapshot();
        report["accelerator_accounting"]={{"mode",budget.mode},{"used",budget.used},{"peak",budget.peak},
            {"unknown",budget.unknown},{"live",budget.live},{"aliases",budget.aliases},{"coverage_gaps",budget.coverage_gaps}};
        if(runtime) {
            report["layers"]=runtime->records; report["canonical_routed_source_bytes"]=runtime->canonical_source_bytes;
            report["cpu_npu_activation_pool_bytes"]=runtime->activation_bytes;
            report["accounted_extra_allowance_bytes"]=budget.mode==1?json(budget.peak+runtime->activation_bytes+runtime->peak_host_staging+size_t(nt)*5*2048*sizeof(float)):json(nullptr);
            report["empty_shared_weight_capacity_bytes"]=ggml_backend_buffer_get_size(runtime->gpu_shared->buf.get());
            report["initial_converted_shared_layers"]=0;
            report["reference_packed_validation_only"]=runtime->reference_packed;
            report["memory_note"]="canonical DMA routed weights replace the complete expert representation and are aliased by CPU; bounded GPU shared conversion, activation pools and host staging reported separately; dense/KV, mapped-file residency and uninstrumented driver allocation are not an extra-memory guarantee";
        }
        std::ofstream logits_file(std::string(argv[5])+".logits.bin",std::ios::binary); binary_write(logits_file,values);
        std::ofstream out(argv[5]); out<<report.dump(2)<<'\n'; require(bool(out),"result write failed");
        fprintf(stderr,"ONLINE_PREFILL mode=%s nt=%d first_ms=%.3f top1=%d faults=%ld\n",argv[6],nt,total_us/1000,
            int(report["top1"]),after.ru_majflt-before.ru_majflt);
        ctx.reset(); model.reset(); runtime.reset(); return 0;
    } catch(const std::exception &e) { fprintf(stderr,"online-moe: %s\n",e.what()); return 1; }
}
