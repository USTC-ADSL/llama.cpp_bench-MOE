// Compare orchestration changes with fixed real inputs and identical resident
// weights. One layer at a time avoids loading the complete model.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==5,"usage: moe-shared-overhead-probe MODEL CAPTURE ROUTE_MODEL OUTPUT");
        ggml_backend_load_all();
        ggml_context * raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gf && meta && f,"probe model open failed"); source src{gf.get(),meta.get(),f.get()};
        json capture,parameters;
        std::ifstream(argv[2])>>capture; std::ifstream(argv[3])>>parameters;
        resident_placement_model costs(parameters);
        moe_route_scheduler scheduler(costs,parameters.value("search_budget",2048),true);
        device cpu("CPU"),htp("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&htp,&gpu};
        moe_token_workers pool(3);
        json report={{"scope","fixed captured inputs, one resident layer; async versus persistent dispatch; alternating order"},
                     {"layers",json::array()}};
        for(int il:{0,12,23}) {
            int nt=capture.at("layers").at(il).at("tokens");
            std::ifstream bin(std::string(argv[2])+".layer"+std::to_string(il)+".bin",std::ios::binary);
            auto x=binary_read<float>(bin,nt*2048),probs=binary_read<float>(bin,nt*60);
            auto ids=binary_read<int32_t>(bin,nt*4);
            auto reference=binary_read<float>(bin,nt*2048);
            resident_placement_model::availability resident;
            for(auto & row:resident.experts) { row.fill(true); } resident.shared.fill(true);
            std::vector<int> counts(60); for(int e:ids) { ++counts[e]; }
            auto selected=scheduler.choose(nt,counts,resident).selected;
            full_engine engine;
            engine.layers.push_back(std::make_unique<partitioned_ffn>(devs,src,il,nt,selected,true,true,&resident));
            engine.prepare_arenas(devs);
            auto & ff=*engine.layers[0]; auto & shared=*ff.single_layout;
            shared.vector_reduce=true;
            std::vector<float> actual(x.size()),baseline;
            auto run=[&] { ff.run_into(x.data(),x.size(),probs.data(),probs.size(),ids,true,actual.data()); };
            // Warm all modes before measurement; caches are shared by variants.
            for(int mode=0;mode<2;++mode) {
                shared.branch_workers=mode?&pool:nullptr;
                run(); run();
                if(mode==0) { baseline=actual; }
                require(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))==0,"overhead variant changed output bits");
            }
            size_t warmed=ff.allocations;
            json layer={{"layer",il},{"tokens",nt},{"placement",plan_json(selected)},
                        {"reference_check",chain_difference(baseline,reference,nt)},
                        {"payload_bytes",shared.weights.payload_bytes},{"samples",json::array()}};
            size_t rows=0,runs=0;
            std::array<int,60> previous; previous.fill(-2);
            for(size_t p=0;p<ids.size();++p) { int e=ids[p],t=int(p/4); ++rows; runs+=t!=previous[e]+1; previous[e]=t; }
            layer["gather_rows"]=rows; layer["consecutive_runs"]=runs;
            for(int round=0;round<12;++round) { for(int step=0;step<2;++step) {
                int mode=round%2 ? 1-step : step;
                shared.branch_workers=mode?&pool:nullptr;
                auto begin=timer::now(); run(); double us=elapsed(begin);
                require(ff.allocations==warmed,"graph allocated after warmup");
                require(std::memcmp(actual.data(),baseline.data(),actual.size()*sizeof(float))==0,"measured variant changed output bits");
                auto sample=ff.last; sample["total_us"]=us; sample["mode"]=mode; sample["round"]=round;
                layer["samples"].push_back(sample);
            } }
            report["layers"].push_back(layer);
            std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"probe output failed");
            fprintf(stderr,"OVERHEAD layer=%d nt=%d variants=2 samples=24 bitwise_equal=yes\n",il,nt);
        }
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"overhead probe: %s\n",e.what()); return 1; }
}
