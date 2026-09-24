#define DYNAMIC_MOE_NO_MAIN
#include "dynamic-moe.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==6,"usage: dynamic-moe-check MODEL CAPTURE TOKENS LAYER OUTPUT");
        int nt=std::stoi(argv[3]),layer=std::stoi(argv[4]);
        require(nt>0 && layer>=0 && layer<24,"invalid capture selection");
        std::ifstream in(argv[2],std::ios::binary);
        in.seekg(size_t(layer)*nt*(2048+60+4+2048)*sizeof(float));
        if (layer==23) { nt=1; }
        auto x=binary_read<float>(in,2048*nt),probs=binary_read<float>(in,60*nt);
        auto ids=binary_read<int32_t>(in,4*nt); auto expected=binary_read<float>(in,2048*nt);
        ggml_backend_load_all(); llama_backend_init(); mapped_model mapping(argv[1]);
        ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gguf && meta,"metadata load failed");
        source src{gguf.get(),meta.get(),nullptr,static_cast<const uint8_t *>(mapping.data),mapping.size};
        device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
        json report=json::array();
        for (int group : {1,8,60}) { for (int b=0;b<3;++b) {
            dynamic_moe_runtime runtime(src,devs,group,[&](const moe_route_request &) {
                return placement{"check",std::vector<int>(60,b),0};
            });
            auto actual=runtime.run(layer,nt,x,probs,ids);
            auto check=chain_difference(actual,expected,nt);
            report.push_back({{"layer",layer},{"group",group},{"routed_backend",b},{"shared_backend",0},{"check",check}});
            fprintf(stderr,"CHECK group=%d backend=%d %s\n",group,b,check.dump().c_str());
        } }
        std::ofstream out(argv[5]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
        return 0;
    } catch (const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
