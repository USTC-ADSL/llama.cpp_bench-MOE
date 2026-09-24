#include "dynamic-moe-runtime.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct mapped_model {
    int fd = -1;
    void * data = MAP_FAILED;
    size_t size = 0;
    explicit mapped_model(const char * path) {
        fd=open(path,O_RDONLY); require(fd>=0,"open model mapping failed");
        struct stat st{};
        if (fstat(fd,&st)!=0 || st.st_size<=0) { close(fd); throw std::runtime_error("model stat failed"); }
        size=size_t(st.st_size); data=mmap(nullptr,size,PROT_READ,MAP_SHARED,fd,0);
        if (data==MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
    }
    ~mapped_model() { if (data!=MAP_FAILED) { munmap(data,size); } if (fd>=0) { close(fd); } }
};

static int backend_id(const std::string & name) {
    if (name=="cpu") { return 0; }
    if (name=="npu") { return 1; }
    if (name=="gpu") { return 2; }
    throw std::runtime_error("unknown backend: "+name);
}

static placement configured_plan(const json & config,const moe_route_request & request) {
    const auto & phase=config.at(request.decode?"decode":"prefill");
    auto selected=phase;
    // Layer and step overrides let offline plans exercise the same runtime interface.
    if (phase.contains("layers")) {
        auto key=std::to_string(request.layer);
        if (phase["layers"].contains(key)) { selected.merge_patch(phase["layers"][key]); }
    }
    if (phase.contains("steps")) {
        auto key=std::to_string(request.step);
        if (phase["steps"].contains(key)) { selected.merge_patch(phase["steps"][key]); }
    }
    placement p{"configured",std::vector<int>(60,backend_id(selected.at("routed"))),backend_id(selected.at("shared"))};
    if (selected.contains("groups")) {
        std::array<bool,60> seen{};
        for (auto & group:selected["groups"]) {
            int b=backend_id(group.at("backend"));
            for (int e:group.at("experts").get<std::vector<int>>()) {
                require(e>=0 && e<60 && !seen[e],"invalid or overlapping Expert group");
                seen[e]=true; p.owner[e]=b;
            }
        }
    }
    return p;
}

#ifndef DYNAMIC_MOE_NO_MAIN
int main(int argc,char ** argv) {
    try {
        require(argc==8,"usage: dynamic-moe MODEL PROMPT TOKENS DECODE_STEPS POLICY_JSON OUTPUT MODE(dynamic|native)");
        int nt=std::stoi(argv[3]),steps=std::stoi(argv[4]);
        require(nt>0 && steps>=0,"invalid sequence lengths");
        std::ifstream config_file(argv[5]); json config; config_file>>config;
        bool dynamic=std::string(argv[7])=="dynamic";
        require(dynamic || std::string(argv[7])=="native","invalid mode");
        ggml_backend_load_all(); llama_backend_init();
        mapped_model mapping(argv[1]);
        ggml_context * raw=nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gguf && meta,"model metadata failed");
        source src{gguf.get(),meta.get(),nullptr,static_cast<const uint8_t *>(mapping.data),mapping.size};
        std::unique_ptr<device> devices[3]; device * devs[3]{};
        std::unique_ptr<dynamic_moe_runtime> runtime;
        if (dynamic) {
            const char * names[]={"CPU","HTP0","GPUOpenCL"};
            for (int b=0;b<3;++b) { devices[b]=std::make_unique<device>(names[b]); devs[b]=devices[b].get(); }
            runtime=std::make_unique<dynamic_moe_runtime>(src,devs,config.value("group_size",8),
                [&](const moe_route_request & request) { return configured_plan(config,request); });
            runtime->verify_layer=config.value("verify_layer",-1);
        }
        ggml_backend_dev_t target[2]{};
        std::vector<std::string> patterns;
        std::vector<llama_model_tensor_buft_override> overrides;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(nullptr,llama_model_free);
        std::unique_ptr<llama_context,decltype(&llama_free)> ctx(nullptr,llama_free);
        capture_all capture;
        const char * names[]={"CPU","HTP0","GPUOpenCL"};
        auto load=[&](int backend) {
            auto mp=llama_model_default_params(); mp.n_gpu_layers=backend?99:0;
            target[0]=ggml_backend_dev_by_name(names[backend]);
            require(target[0]!=nullptr,"dense backend unavailable");
            mp.devices=backend?target:nullptr;
            patterns={"blk\\.[0-9]+\\.ffn_(gate|up|down)_exps\\.weight",
                "blk\\.[0-9]+\\.ffn_(gate|up|down|gate_inp)_shexp\\.weight"};
            // Keep dense CPU matmuls on the native packed path while unused Expert
            // tensors remain immutable mmap views instead of full packed replicas.
            if (dynamic && backend==0) {
                for (int64_t i=0;i<gguf_get_n_tensors(gguf.get());++i) {
                    std::string name=gguf_get_tensor_name(gguf.get(),i);
                    auto * t=src.tensor(name);
                    if (t->type!=GGML_TYPE_Q4_0 || ggml_n_dims(t)!=2 || name=="token_embd.weight" ||
                        name.find("_shexp.")!=std::string::npos || name.find("_exps.")!=std::string::npos) { continue; }
                    std::string pattern="^";
                    for (char ch:name) { if (ch=='.') { pattern+='\\'; } pattern+=ch; }
                    patterns.push_back(pattern+"$");
                }
            }
            overrides.clear();
            for (size_t i=0;i<patterns.size();++i) {
                overrides.push_back({patterns[i].c_str(),i<2?ggml_backend_dev_buffer_type(ggml_backend_dev_by_name("CPU")):devs[0]->buft});
            }
            overrides.push_back({nullptr,nullptr});
            if (dynamic) { mp.tensor_buft_overrides=overrides.data(); mp.use_extra_bufts=false; mp.load_mode=LLAMA_LOAD_MODE_MMAP; }
            model.reset(llama_model_load_from_file(argv[1],mp)); require(model!=nullptr,"model load failed");
            if (dynamic) { require(llama_model_set_moe_graph_callback(model.get(),dynamic_moe_runtime::build,runtime.get()),"callback rejected"); }
            auto cp=llama_context_default_params(); cp.n_ctx=nt+steps+32; cp.n_batch=nt; cp.n_ubatch=nt;
            cp.n_threads=cp.n_threads_batch=4; cp.type_k=cp.type_v=GGML_TYPE_F16;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.offload_kqv=backend!=0; cp.op_offload=backend!=0;
            if (config.value("capture_prefill",false)) { cp.cb_eval=capture_all::callback; cp.cb_eval_user_data=&capture; }
            ctx.reset(llama_init_from_model(model.get(),cp)); require(ctx!=nullptr,"context creation failed");
        };
        int prefill_backend=backend_id(config.at("prefill").at("dense"));
        int decode_backend=backend_id(config.at("decode").at("dense"));
        auto loading=timer::now(); load(prefill_backend); double load_us=elapsed(loading);
        std::ifstream prompt_file(argv[2]); require(prompt_file.good(),"prompt open failed");
        std::string prompt((std::istreambuf_iterator<char>(prompt_file)),{});
        std::vector<llama_token> tokens(prompt.size()+32);
        auto * vocab=llama_model_get_vocab(model.get());
        int n=llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),tokens.size(),true,true);
        require(n>=nt,"prompt too short"); tokens.resize(nt);
        int nv=llama_vocab_n_tokens(vocab);
        json report={{"mode",argv[7]},{"config",config},{"tokens",tokens},{"load_us",load_us},
            {"scope","complete Transformer prefill and decode; conversion included; explicit activation copies"},
            {"samples",json::array()}};
        std::ofstream logits_file(std::string(argv[6])+".logits.bin",std::ios::binary);
        llama_token next=0;
        for (int step=0;step<=steps;++step) {
            if (step==1 && prefill_backend!=decode_backend) {
                auto switching=timer::now();
                std::vector<uint8_t> state(llama_state_get_size(ctx.get()));
                require(llama_state_get_data(ctx.get(),state.data(),state.size())==state.size(),"state export failed");
                ctx.reset(); model.reset(); if (runtime) { runtime->clear(); }
                load(decode_backend);
                require(llama_state_set_data(ctx.get(),state.data(),state.size())==state.size(),"state import failed");
                std::vector<uint8_t> restored(llama_state_get_size(ctx.get()));
                require(llama_state_get_data(ctx.get(),restored.data(),restored.size())==restored.size(),"restored state export failed");
                require(restored==state,"phase switch changed serialized KV/state");
                report["phase_switch"]={{"us",elapsed(switching)},{"state_bytes",state.size()},
                    {"state_roundtrip_equal",true},
                    {"method","serialize KV and state; release old model/context; load phase backend; restore"}};
            }
            if (runtime) { runtime->decode=step>0; runtime->step=step; runtime->records=json::array(); }
            // Optional teacher forcing keeps all backend comparisons on identical tokens.
            if (step>0 && config.contains("decode_tokens")) { next=config.at("decode_tokens").at(step-1); }
            require(step==0 || (next>=0 && next<nv),"invalid decode token");
            auto batch=step==0?llama_batch_get_one(tokens.data(),nt):llama_batch_get_one(&next,1);
            rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
            auto start=timer::now(); int rc=llama_decode(ctx.get(),batch); llama_synchronize(ctx.get());
            double us=elapsed(start); if (runtime) { runtime->check(); }
            require(rc==0,"inference failed");
            auto * logits=llama_get_logits_ith(ctx.get(),-1); require(logits!=nullptr,"missing logits");
            std::vector<float> values(logits,logits+nv);
            for (float v:values) { require(std::isfinite(v),"nonfinite logits"); }
            binary_write(logits_file,values);
            int top1=int(std::max_element(values.begin(),values.end())-values.begin());
            getrusage(RUSAGE_SELF,&after);
            json sample={{"step",step},{"input_token",step?json(next):json(nullptr)},
                {"tokens",step?1:nt},{"total_us",us},{"top1",top1},
                {"major_faults",after.ru_majflt-before.ru_majflt}};
            if (runtime) { sample["layers"]=runtime->records; }
            report["samples"].push_back(std::move(sample)); next=top1;
            if (step==0 && config.value("capture_prefill",false)) {
                std::ofstream captured(std::string(argv[6])+".capture.bin",std::ios::binary);
                for (auto & layer:capture.layers) {
                    binary_write(captured,layer.input); binary_write(captured,layer.probs);
                    binary_write(captured,layer.ids); binary_write(captured,layer.output);
                }
            }
            fprintf(stderr,"DYNAMIC step=%d tokens=%d elapsed_ms=%.3f top1=%d\n",step,step?1:nt,us/1000,top1);
        }
        if (runtime) { report["peak_converted_weight_buffer_bytes"]=runtime->peak_weight_bytes; }
        report["memory_note"]="buffer bytes exclude source mmap residency, driver scratch, conversion staging, host vectors, dense weights, KV and optional validation reference banks; not a physical RAM cap";
        std::ifstream status("/proc/self/status"); report["process_status"]=std::string((std::istreambuf_iterator<char>(status)),{});
        report["vocab_size"]=nv;
        std::ofstream out(argv[6]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
        ctx.reset(); model.reset(); runtime.reset();
        return 0;
    } catch (const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
#endif
