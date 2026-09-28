#include "online-moe-runtime.h"
#include "ggml-accel-budget.h"
#define DYNAMIC_MOE_NO_MAIN
#include "dynamic-moe.cpp"
#include "../src/llama-model.h"

// mincore observes file-cache residency without faulting source pages in. Count
// whole pages intersecting imported Expert matrices; shared boundary pages can
// be counted twice, so the result is a conservative source-duplication sample.
static json source_residency(source &src, bool include_shared) {
    size_t page=size_t(sysconf(_SC_PAGESIZE)),resident=0,queried=0;
    int calls=0,errors=0;
    for(int il=0;il<24;++il) { for(const char *part:{"gate","up","down","gate_inp"}) {
        for(bool shared:{false,true}) {
            if((shared && !include_shared) || (!shared && std::string(part)=="gate_inp")) { continue; }
            std::string name="blk."+std::to_string(il)+".ffn_"+part+(shared?"_shexp.weight":"_exps.weight");
            auto *t=src.tensor(name);
            size_t off=gguf_get_data_offset(src.gguf)+gguf_get_tensor_offset(src.gguf,gguf_find_tensor(src.gguf,name.c_str()));
            size_t first=off/page*page,last=(off+ggml_nbytes(t)+page-1)/page*page;
            std::vector<unsigned char> vec((last-first)/page);
            ++calls;
            if(mincore(const_cast<uint8_t *>(src.mapped)+first,last-first,vec.data())) { ++errors; continue; }
            queried+=last-first;
            for(auto v:vec) { resident+=(v&1)?page:0; }
        }
    } }
    return {{"resident_file_bytes_upper_sample",resident},{"queried_bytes",queried},{"calls",calls},{"errors",errors},
        {"scope","file-cache pages overlapping canonical imported experts; boundary pages may be double counted; sample is not a peak"}};
}

int main(int argc,char **argv) {
    try {
        require(argc==8,"usage: online-moe MODEL PROMPT TOKENS COST_MODEL OUTPUT MODE(native|online) TAIL(-1=model)");
        int nt=std::stoi(argv[3]),tail=std::stoi(argv[7]); bool online=std::string(argv[6])=="online";
        require(online || std::string(argv[6])=="native","invalid mode");
        require(nt>=64 && nt<=768 && tail>=-1 && tail<=60,"invalid length/tail");
        ggml_log_set([](ggml_log_level l,const char *s,void*) {
            if(l!=GGML_LOG_LEVEL_DEBUG || (std::getenv("MOE_DSP_TRACE") && (strstr(s,"profile-op") || strstr(s,"trace-evt")))) { fputs(s,stderr); }
        },nullptr);
        const auto init_start=timer::now(); ggml_backend_load_all(); llama_backend_init();
        mapped_model mapping(argv[1]); ggml_context *raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gf && meta,"GGUF metadata failed");
        source src{gf.get(),meta.get(),nullptr,static_cast<const uint8_t *>(mapping.data),mapping.size};
        const bool bounded_source=std::getenv("MOE_BOUNDED_SOURCE")!=nullptr;
        size_t source_release_bytes=0; int source_advice_calls=0,source_advice_failures=0;
        if(online && bounded_source) {
            src.release_after_import=[&](size_t off,size_t size) {
                size_t page=size_t(sysconf(_SC_PAGESIZE)),begin=(off+page-1)/page*page,end=(off+size)/page*page;
                if(end<=begin) { return; }
                ++source_advice_calls;
                int unmap_rc=madvise(static_cast<uint8_t *>(mapping.data)+begin,end-begin,MADV_DONTNEED);
                int cache_rc=posix_fadvise(mapping.fd,off_t(begin),off_t(end-begin),POSIX_FADV_DONTNEED);
                if(unmap_rc || cache_rc) { ++source_advice_failures; }
                else { source_release_bytes+=end-begin; }
            };
        }
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
        mp.no_mmap_prefetch=bounded_source;
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
        // Symmetric loaded-model control: restore only the host output weights,
        // without evaluating the prompt or preparing any Expert target weights.
        json output_residency={{"enabled",false},{"us",0.0},{"bytes",0}};
        if(std::getenv("MOE_RESIDENT_OUTPUT")) {
            auto *head=model->output;
            require(head && head->buffer && head->data &&
                    (ggml_backend_buffer_is_host(head->buffer) ||
                     (ggml_backend_buft_get_device(ggml_backend_buffer_get_type(head->buffer))==ggml_backend_dev_by_name("CPU") &&
                      std::string(ggml_backend_buffer_name(head->buffer))=="CPU_REPACK")),
                    "output residency control requires a host output tensor");
            size_t bytes=ggml_nbytes(head),page=size_t(sysconf(_SC_PAGESIZE));
            auto *data=static_cast<const volatile uint8_t *>(head->data);
            rusage rb{},ra{}; getrusage(RUSAGE_SELF,&rb); auto restore_start=timer::now();
            uint64_t checksum=0;
            for(size_t off=0;off<bytes;off+=page) { checksum+=data[off]; }
            if(bytes) { checksum+=data[bytes-1]; }
            double restore_us=elapsed(restore_start); getrusage(RUSAGE_SELF,&ra);
            output_residency={{"enabled",true},{"us",restore_us},{"bytes",bytes},{"checksum",checksum},
                {"buffer",ggml_backend_buffer_name(head->buffer)},
                {"major_faults",ra.ru_majflt-rb.ru_majflt},{"minor_faults",ra.ru_minflt-rb.ru_minflt}};
        }
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
        report["source_storage"]={{"no_mmap_prefetch",bounded_source},{"release_advice_calls",source_advice_calls},
            {"release_advice_failures",source_advice_failures},{"advised_bytes",source_release_bytes}};
        report["output_residency"]=output_residency;
        if(bounded_source) { report["source_cache_before"]=source_residency(src,bool(runtime && runtime->canonical_shared_source_bytes)); }
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
        report["restore_plus_prefill_us"]=total_us+output_residency.at("us").get<double>();
        report["top1"]=std::max_element(values.begin(),values.end())-values.begin(); report["vocab_size"]=nv;
        report["major_faults"]=after.ru_majflt-before.ru_majflt; report["minor_faults"]=after.ru_minflt-before.ru_minflt;
        report["maxrss_kib"]=after.ru_maxrss; report["after_status"]=status();
        if(bounded_source) { report["source_cache_after"]=source_residency(src,bool(runtime && runtime->canonical_shared_source_bytes)); }
        std::ifstream smaps("/proc/self/smaps_rollup");
        report["after_smaps_rollup"]=std::string((std::istreambuf_iterator<char>(smaps)),{});
        auto budget=ggml_budget_snapshot();
        report["accelerator_accounting"]={{"mode",budget.mode},{"used",budget.used},{"peak",budget.peak},
            {"unknown",budget.unknown},{"live",budget.live},{"aliases",budget.aliases},{"coverage_gaps",budget.coverage_gaps}};
        if(runtime) {
            report["layers"]=runtime->records; report["canonical_routed_source_bytes"]=runtime->canonical_source_bytes;
            report["canonical_shared_source_bytes"]=runtime->canonical_shared_source_bytes;
            report["cpu_npu_activation_pool_bytes"]=runtime->activation_bytes;
            report["accounted_extra_allowance_bytes"]=budget.mode==1?json(budget.peak+runtime->activation_bytes+runtime->peak_host_staging+size_t(nt)*5*2048*sizeof(float)):json(nullptr);
            report["empty_shared_weight_capacity_bytes"]=ggml_backend_buffer_get_size(runtime->gpu_shared->buf.get());
            report["initial_converted_shared_layers"]=0;
            report["reference_packed_validation_only"]=runtime->reference_packed;
            if(runtime->prefetch) {
                const auto &p=*runtime->prefetch;
                require(!p.pending.valid(),"prefetch work left after terminal layer");
                report["prefetch"]={{"capacity",p.capacity},{"method",p.predictor.at("method")},
                    {"target_bytes",p.target_bytes},{"source_import_alias_peak_bytes",p.source_import_peak},
                    {"preparations",p.records},{"initial_prepared_experts",0}};
                if(budget.mode==1) {
                    size_t raw=report.at("accounted_extra_allowance_bytes");
                    require(raw>=p.source_import_peak,"import alias budget attribution failed");
                    report["prefetch"]["allowance_minus_known_source_alias_bytes"]=raw-p.source_import_peak;
                    report["prefetch"]["accounting_note"]="raw observer counts DMA source import again; subtracted only one overlapping full-layer source import, whose backing is the existing canonical source; target imports remain charged; not a driver/physical-memory guarantee";
                }
            }
            report["memory_note"]="canonical DMA routed weights replace the complete expert representation and are aliased by CPU; bounded GPU shared conversion, activation pools and host staging reported separately; dense/KV, mapped-file residency and uninstrumented driver allocation are not an extra-memory guarantee";
        }
        std::ofstream logits_file(std::string(argv[5])+".logits.bin",std::ios::binary); binary_write(logits_file,values);
        std::ofstream out(argv[5]); out<<report.dump(2)<<'\n'; require(bool(out),"result write failed");
        fprintf(stderr,"ONLINE_PREFILL mode=%s nt=%d first_ms=%.3f top1=%d faults=%ld\n",argv[6],nt,total_us/1000,
            int(report["top1"]),after.ru_majflt-before.ru_majflt);
        ctx.reset(); model.reset(); runtime.reset(); return 0;
    } catch(const std::exception &e) { fprintf(stderr,"online-moe: %s\n",e.what()); return 1; }
}
