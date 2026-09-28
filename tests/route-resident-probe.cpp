// Device contract: immutable union banks, shared/no-shared graph cache entries,
// and repeated whole-expert migrations must match fresh compact-bank execution.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==4,"usage: route-resident-probe MODEL CAPTURE OUTPUT");
        ggml_backend_load_all();
        json capture; std::ifstream(argv[2])>>capture;
        int nt=capture.at("layers").at(0).at("tokens");
        std::ifstream input(std::string(argv[2])+".layer0.bin",std::ios::binary);
        auto x=binary_read<float>(input,2048*nt),probs=binary_read<float>(input,60*nt);
        auto ids=binary_read<int32_t>(input,4*nt);
        ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose); require(gguf && meta && file,"source open failed");
        source src{gguf.get(),meta.get(),file.get()}; device cpu("CPU"),npu("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&npu,&gpu};
        resident_placement_model::availability all; for(auto & row:all.experts) { row.fill(true); } all.shared.fill(true);
        json coefficients={{"version",1},{"backends",std::vector<std::vector<double>>(3,std::vector<double>(26))},{"host",std::vector<double>(8)}};
        resident_placement_model model(coefficients); moe_route_scheduler search(model);
        placement initial{"native",std::vector<int>(60,1),1};
        full_engine engine; engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,0,nt,initial,true,true,&all));
        auto & union_ff=*engine.layers[0]; union_ff.reuse_lifetimes=true; union_ff.fused_host_reduce=true; union_ff.overlap_gather=true;
        union_ff.npu_device_gather=std::getenv("MOE_NPU_DEVICE_GATHER") && std::string(std::getenv("MOE_NPU_DEVICE_GATHER"))=="1";
        union_ff.scheduler=&search; engine.prepare_arenas(devs); union_ff.scheduler=nullptr;
        std::vector<placement> cases;
        for(int b=0;b<3;++b) { for(int s=0;s<3;++s) { cases.push_back({"native",std::vector<int>(60,b),s}); } }
        for(int s=0;s<3;++s) {
            placement p{"threeway",std::vector<int>(60),s}; for(int e=0;e<60;++e) { p.owner[e]=e%3; } cases.push_back(p);
        }
        std::vector<std::vector<float>> references;
        json report={{"tokens",nt},{"npu_device_gather",union_ff.npu_device_gather},{"cases",json::array()}};
        for(const auto & p:cases) {
            full_engine compact; compact.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,0,nt,p,true,true));
            auto & ff=*compact.layers[0]; ff.reuse_lifetimes=true; ff.fused_host_reduce=true; ff.overlap_gather=true; compact.prepare_arenas(devs);
            auto expected=ff.run(x,probs,ids,false);
            union_ff.plan=p; auto actual=union_ff.run(x,probs,ids,true);
            auto check=chain_difference(actual,expected,nt);
            require(check.at("worst_token_nrmse").get<double>()<1e-4,"union/compact numerical mismatch");
            references.push_back(actual); report["cases"].push_back({{"plan",plan_json(p)},{"check",check}});
        }
        size_t warmed=union_ff.allocations;
        for(int i=int(cases.size())-1;i>=0;--i) {
            union_ff.plan=cases[i]; auto actual=union_ff.run(x,probs,ids,true);
            require(std::memcmp(actual.data(),references[i].data(),actual.size()*sizeof(float))==0,"cache revisit changed output");
        }
        require(union_ff.allocations==warmed,"cache revisit allocated graph");
        report["cached_graphs"]=warmed; report["reverse_revisit_bitwise_equal"]=true;
        std::ofstream output(argv[3]); output<<report.dump(2)<<'\n'; require(output.good(),"contract report failed");
        engine.layers.clear();
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
