// Disjoint pooled regions must survive route/owner changes and graph-cache reuse.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: moe-pooled-activation-probe MODEL OUTPUT");
        setenv("MOE_SHARED_WEIGHTS","cpu4x8",1); setenv("MOE_SHARED_REDUCE","neon",1);
        setenv("MOE_SHARED_GATHER","htp",1); setenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER","1",1);
        setenv("MOE_SHARED_DISPATCH","async",1);
        ggml_backend_load_all(); ggml_context * raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose);
        require(gf && meta && file,"pooled probe model open failed"); source src{gf.get(),meta.get(),file.get()};
        device cpu("CPU"),htp("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&htp,&gpu};
        resident_placement_model::availability resident;
        for(auto & row:resident.experts) { row.fill(true); } resident.shared.fill(true);
        placement initial{"pooled",std::vector<int>(60,1),1};
        json report={{"cases",json::array()},{"capacities",json::array()}};
        moe_token_workers workers(3);
        bool saw_nonzero=false,saw_offset_change=false;
        for(bool common:{false,true}) {
            full_engine baseline,pooled;
            for(auto * engine:{&baseline,&pooled}) {
                setenv("MOE_SHARED_SCRATCH",common?"common":"private",1);
                setenv("MOE_SHARED_ARENA",engine==&pooled?"pooled":"private",1);
                engine->layers.push_back(std::make_unique<partitioned_ffn>(devs,src,0,2048,initial,true,true,&resident));
                engine->prepare_arenas(devs);
            }
            auto & a=*baseline.layers[0]->single_layout; auto & b=*pooled.layers[0]->single_layout;
            for(int backend=0;backend<3;++backend) {
                require(pooled.arenas[backend].get()==pooled.arenas[0].get(),"pool has multiple backings");
                require(ggml_backend_buffer_get_base(pooled.arenas[0]->view(*devs[backend]))==pooled.arenas[0]->desc.host_ptr,
                        "pooled backend import is not an alias");
            }
            auto total=[](const full_engine & engine) {
                size_t n=engine.shared_scratch?engine.shared_scratch->desc.size:0;
                for(const auto & bytes:engine.physical_arena_sizes()) { n+=bytes.get<size_t>(); }
                return n;
            };
            require(total(pooled)<total(baseline),"pool did not reduce physical capacity");
            report["capacities"].push_back({{"common_shared_scratch",common},{"private_bytes",total(baseline)},
                                            {"pooled_bytes",total(pooled)},{"routed_backings",1}});
            size_t alloc_a=0,alloc_b=0;
            std::map<std::array<int,61>,std::set<size_t>> htp_offsets;
            for(int nt:{1,17,128,2048}) {
                std::vector<float> x(nt*2048),probs(nt*60,0),expected(x.size()),actual(x.size());
                std::vector<int32_t> ids(nt*4);
                for(int t=0;t<nt;++t) { for(int k=0;k<4;++k) {
                    int e=(t*7+k*13)%60; ids[t*4+k]=e; probs[t*60+e]=.25f;
                } }
                size_t warmed_a=0,warmed_b=0;
                for(int round=0;round<3;++round) {
                    // Both dispatch modes must preserve disjoint partitions.
                    b.branch_workers=round==1?&workers:nullptr;
                    for(int topology=0;topology<5;++topology) { for(int owner:{0,2,1}) {
                        placement p{"pooled-test",std::vector<int>(60),owner};
                        for(int e=0;e<60;++e) {
                            p.owner[e]=topology<3?topology:(e<20?0:e<40?1:2);
                            if(topology==4 && e>=10 && e<20) { p.owner[e]=2; }
                        }
                        for(size_t i=0;i<x.size();++i) { x[i]=.1f*std::sin(float((i+round*29+owner*17)%1009)*.07f); }
                        json old_sample,new_sample;
                        a.run(x.data(),probs.data(),ids,p,expected.data(),alloc_a,old_sample,0,{});
                        b.run(x.data(),probs.data(),ids,p,actual.data(),alloc_b,new_sample,0,{});
                        require(std::memcmp(actual.data(),expected.data(),x.size()*sizeof(float))==0,"pooled activation changed output bits");
                        const auto & pool=new_sample.at("pooled_arena");
                        size_t end=0;
                        for(int backend=0;backend<3;++backend) {
                            size_t offset=pool["partition_offsets"][backend],bytes=pool["partition_bytes"][backend];
                            if(!bytes) { continue; }
                            require(offset>=end && pool["layout_bytes"][backend].get<size_t>()<=bytes,"pooled partitions overlap or overflow");
                            end=offset+bytes; saw_nonzero|=offset!=0;
                        }
                        require(end<=pool["capacity_bytes"].get<size_t>(),"pooled capacity exceeded");
                        std::array<int,61> counts{};
                        for(int e:ids) { if(p.owner[e]==1) { ++counts[e]; } }
                        if(owner==1) { counts[60]=nt; }
                        if(new_sample["pairs"][1].get<int>() || owner==1) {
                            htp_offsets[counts].insert(pool["partition_offsets"][1].get<size_t>());
                            saw_offset_change|=htp_offsets[counts].size()>1;
                        }
                        if(round) { require(alloc_a==warmed_a && alloc_b==warmed_b,"cached route allocated another graph after warmup"); }
                        report["cases"].push_back({{"tokens",nt},{"topology",topology},{"shared",owner},{"round",round},
                            {"common_shared_scratch",common},{"bitwise_equal",true},{"pooled_arena",pool}});
                    } }
                    warmed_a=alloc_a; warmed_b=alloc_b;
                }
            }
        }
        require(saw_nonzero && saw_offset_change,"probe did not exercise nonzero and changing offsets");
        report["same_shape_different_offsets"]=true;
        std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(out.good(),"pooled probe report write failed");
        fprintf(stderr,"POOLED cases=%zu all bitwise equal; offset changes and cached revisits passed\n",report["cases"].size());
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"pooled probe: %s\n",e.what()); return 1; }
}
