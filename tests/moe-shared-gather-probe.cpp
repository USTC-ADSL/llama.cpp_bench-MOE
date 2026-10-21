// Compare host materialization with HTP GET_ROWS using identical weights/routes.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

static json compare_gather(moe_shared_execution & shared,const placement & plan,
                           const std::vector<float> & x,const std::vector<float> & probs,
                           const std::vector<int32_t> & ids,int layer,int repeats) {
    size_t allocations=0; json phases;
    std::vector<float> actual(x.size()),baseline;
    auto run=[&] { shared.run(x.data(),probs.data(),ids,plan,actual.data(),allocations,phases,layer,{}); };
    for(int mode=0;mode<2;++mode) {
        shared.npu_gather=mode; run(); run();
        const size_t pairs=phases["pairs"][1].get<size_t>();
        const size_t nt=ids.size()/4;
        require(phases["npu_device_gather"].get<bool>()==bool(mode && nt>1 && pairs && pairs+(plan.shared==1?nt:0)>nt),
                "device gather sparse fallback mismatch");
        if(mode==0) { baseline=actual; }
        require(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))==0,"device gather changed output bits");
    }
    const auto warmed=allocations;
    json result={{"tokens",ids.size()/4},{"placement",plan_json(plan)},{"bitwise_equal",true},{"samples",json::array()}};
    for(int round=0;round<repeats;++round) { for(int step=0;step<2;++step) {
        int mode=round%2?1-step:step; shared.npu_gather=mode;
        auto begin=timer::now(); run(); double us=elapsed(begin);
        require(allocations==warmed,"gather probe allocated after warmup");
        require(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))==0,"timed gather changed output bits");
        auto sample=phases; sample["total_us"]=us; sample["mode"]=mode; sample["round"]=round;
        result["samples"].push_back(sample);
    } }
    return result;
}

int main(int argc,char ** argv) {
    try {
        require(argc==5,"usage: moe-shared-gather-probe MODEL CAPTURE ROUTE_MODEL OUTPUT");
        setenv("MOE_SHARED_GATHER","htp",1); // Reserve both graph variants before timing.
        ggml_backend_load_all(); ggml_context * raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gf && meta && f,"gather probe model open failed"); source src{gf.get(),meta.get(),f.get()};
        json capture,parameters; std::ifstream(argv[2])>>capture; std::ifstream(argv[3])>>parameters;
        resident_placement_model costs(parameters);
        moe_route_scheduler scheduler(costs,parameters.value("search_budget",2048),true);
        device cpu("CPU"),htp("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&htp,&gpu};
        json report={{"scope","same CPU4x8 weights; host versus HTP gather; async dispatch; AB/BA"},
                     {"indexed_fusion_requested",std::getenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER") && std::string(std::getenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER"))=="1"},
                     {"cases",json::array()}};
        for(int il:{0,12,23}) {
            int nt=capture.at("layers").at(il).at("tokens");
            std::ifstream bin(std::string(argv[2])+".layer"+std::to_string(il)+".bin",std::ios::binary);
            auto x=binary_read<float>(bin,nt*2048),probs=binary_read<float>(bin,nt*60);
            auto ids=binary_read<int32_t>(bin,nt*4); auto reference=binary_read<float>(bin,nt*2048);
            resident_placement_model::availability resident;
            for(auto & row:resident.experts) { row.fill(true); } resident.shared.fill(true);
            std::vector<int> counts(60); for(int e:ids) { ++counts[e]; }
            auto plan=scheduler.choose(nt,counts,resident).selected;
            full_engine engine;
            engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,nt,plan,true,true,&resident));
            engine.prepare_arenas(devs); auto & shared=*engine.layers[0]->single_layout; shared.vector_reduce=true;
            if(il==0) {
                for(int n:{1,2,3,17,32,33}) { for(int shared_backend:{0,1}) {
                    std::vector<float> sx(n*2048),sp(n*60,0); std::vector<int32_t> si(n*4);
                    for(size_t i=0;i<sx.size();++i) { sx[i]=.1f*std::sin(float(i%1009)*.07f); }
                    for(int t=0;t<n;++t) { for(int k=0;k<4;++k) {
                        int e=(t*7+k*13)%60; si[t*4+k]=e; sp[t*60+e]=.25f;
                    } }
                    placement p{"boundary",std::vector<int>(60,1),shared_backend};
                    auto item=compare_gather(shared,p,sx,sp,si,il,2); item["case"]="boundary";
                    report["cases"].push_back(item);
                } }
                // Identical HTP expert counts but different complete input
                // lengths must not alias a cached full-input tensor.
                for(int n:{2,3,17,2}) {
                    std::vector<int32_t> si{0,1,2,40,0,1,41,42};
                    for(int t=2;t<n;++t) { si.insert(si.end(),{40,41,42,43}); }
                    std::vector<float> sx(n*2048),sp(n*60,0);
                    for(size_t i=0;i<sx.size();++i) { sx[i]=.1f*std::cos(float(i%997)*.09f); }
                    for(int t=0;t<n;++t) { for(int k=0;k<4;++k) { sp[t*60+si[t*4+k]]=.25f; } }
                    placement p{"cache-length",std::vector<int>(60,0),0}; p.owner[0]=p.owner[1]=p.owner[2]=1;
                    auto item=compare_gather(shared,p,sx,sp,si,il,2); item["case"]="cache-length";
                    report["cases"].push_back(item);
                }
                // Same shapes/counts, changed input and row maps, then reuse the
                // original map. Also cover the exact sparse-copy threshold.
                for(int extra:{0,1,1,0}) {
                    const int n=17;
                    std::vector<float> sx(n*2048),sp(n*60,0); std::vector<int32_t> si(n*4);
                    for(size_t i=0;i<sx.size();++i) { sx[i]=.1f*std::sin(float(i+extra*27)*.03f); }
                    for(int t=0;t<n;++t) {
                        si[t*4]=0; si[t*4+1]=extra && t==0?1:40; si[t*4+2]=41; si[t*4+3]=42;
                    }
                    static int permutation=0;
                    if(++permutation%2==0) { std::rotate(si.begin(),si.begin()+12,si.end()); }
                    for(int t=0;t<n;++t) { for(int k=0;k<4;++k) { sp[t*60+si[t*4+k]]=.25f; } }
                    placement p{"threshold-cache",std::vector<int>(60,0),0}; p.owner[0]=p.owner[1]=1;
                    auto item=compare_gather(shared,p,sx,sp,si,il,2); item["case"]="threshold-cache";
                    report["cases"].push_back(item);
                    p.owner.assign(60,0); p.shared=1;
                    item=compare_gather(shared,p,sx,sp,si,il,2); item["case"]="shared-only";
                    report["cases"].push_back(item);
                }
                if(std::getenv("MOE_GATHER_PROBE_SMALL")) {
                    std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"small report write failed"); return 0;
                }
                // The arena and graph metadata must fit all sixty active experts.
                std::vector<int32_t> si(1024*4); std::vector<float> sp(1024*60,0);
                for(size_t i=0;i<si.size();++i) { si[i]=i%60; sp[(i/4)*60+si[i]]=.25f; }
                placement p{"all-experts",std::vector<int>(60,1),1};
                auto all=compare_gather(shared,p,x,sp,si,il,2); all["case"]="all-experts";
                report["cases"].push_back(all);
            }
            auto item=compare_gather(shared,plan,x,probs,ids,il,12); item["case"]="captured"; item["layer"]=il;
            size_t check_allocations=0; json trace;
            auto actual=shared.run(x.data(),probs.data(),ids,plan,nullptr,check_allocations,trace,il,{});
            item["reference_check"]=chain_difference(actual,reference,nt);
            require(item["reference_check"]["passes_4percent_per_token"],"captured reference threshold exceeded");
            item["payload_bytes"]=shared.weights.payload_bytes;
            item["arena_capacity_bytes"]={engine.arenas[0]->desc.size,engine.arenas[1]->desc.size,engine.arenas[2]->desc.size};
            report["cases"].push_back(item);
            std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"gather report write failed");
            fprintf(stderr,"GATHER layer=%d tokens=%d bitwise_equal=yes\n",il,nt);
        }
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"gather probe: %s\n",e.what()); return 1; }
}
