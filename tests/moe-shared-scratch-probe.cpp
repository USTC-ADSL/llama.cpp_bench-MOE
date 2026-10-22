// Exclusive shared scratch must survive owner switches and concurrent routed work.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: moe-shared-scratch-probe MODEL OUTPUT");
        setenv("MOE_SHARED_WEIGHTS","cpu4x8",1); setenv("MOE_SHARED_REDUCE","neon",1);
        setenv("MOE_SHARED_GATHER","htp",1); setenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER","1",1);
        ggml_backend_load_all(); ggml_context * raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose);
        require(gf && meta && file,"scratch probe model open failed"); source src{gf.get(),meta.get(),file.get()};
        device cpu("CPU"),htp("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&htp,&gpu};
        resident_placement_model::availability resident;
        for(auto & row:resident.experts) { row.fill(true); } resident.shared.fill(true);
        placement initial{"scratch",std::vector<int>(60,1),1};
        full_engine baseline,common;
        for(auto * engine:{&baseline,&common}) {
            setenv("MOE_SHARED_SCRATCH",engine==&common?"common":"private",1);
            engine->layers.push_back(std::make_unique<partitioned_ffn>(devs,src,0,1024,initial,true,true,&resident));
            engine->prepare_arenas(devs);
        }
        auto & a=*baseline.layers[0]->single_layout; auto & b=*common.layers[0]->single_layout;
        require(common.shared_scratch && !baseline.shared_scratch,"scratch policy ignored");
        for(int backend=0;backend<3;++backend) {
            require(ggml_backend_buffer_get_base(common.shared_scratch->view(*devs[backend]))==common.shared_scratch->desc.host_ptr,
                    "shared activation view is not a physical alias");
            require(common.arenas[backend]->desc.host_ptr!=common.shared_scratch->desc.host_ptr,"routed aliases shared scratch");
        }
        json report={{"cases",json::array()},{"physical_shared_backings",1},{"backend_aliases",3},
                     {"common_scratch_bytes",common.shared_scratch->desc.size}};
        size_t original_bytes=0,common_bytes=common.shared_scratch->desc.size;
        for(int backend=0;backend<3;++backend) { original_bytes+=baseline.arenas[backend]->desc.size;common_bytes+=common.arenas[backend]->desc.size; }
        report["private_capacity_bytes"]=original_bytes;report["common_capacity_bytes"]=common_bytes;
        require(common_bytes<original_bytes,"common scratch did not reduce capacity");
        size_t alloc_a=0,alloc_b=0;
        for(int nt:{1,2,17,32,33,128,256,512,1024}) {
            std::vector<float> x(nt*2048),probs(nt*60,0),expected(x.size()),actual(x.size());
            std::vector<int32_t> ids(nt*4);
            for(int t=0;t<nt;++t) { for(int k=0;k<4;++k) {
                int e=(t*7+k*13)%60;ids[t*4+k]=e;probs[t*60+e]=.25f;
            } }
            for(int topology=0;topology<4;++topology) {
                placement p{"scratch-test",std::vector<int>(60),0};
                for(int e=0;e<60;++e) { p.owner[e]=topology<3?topology:e%3; }
                size_t warmed_a=0,warmed_b=0;
                for(int round=0;round<3;++round) {
                    // Revisit all owners after other backends have overwritten
                    // common storage; change values so stale cached data fail.
                    for(int owner:{0,2,1}) {
                        p.shared=owner;
                        for(size_t i=0;i<x.size();++i) { x[i]=.1f*std::sin(float((i+round*29+owner*17)%1009)*.07f); }
                        json old_sample,new_sample;
                        a.run(x.data(),probs.data(),ids,p,expected.data(),alloc_a,old_sample,0,{});
                        b.run(x.data(),probs.data(),ids,p,actual.data(),alloc_b,new_sample,0,{});
                        require(std::memcmp(actual.data(),expected.data(),x.size()*sizeof(float))==0,"common scratch changed output bits");
                        require(new_sample.at("shared_scratch_owner")==owner,"wrong common scratch writer");
                        if(round) { require(alloc_a==warmed_a && alloc_b==warmed_b,"new job allocated after all owners warmed"); }
                        report["cases"].push_back({{"tokens",nt},{"topology",topology},{"shared",owner},{"round",round},
                            {"bitwise_equal",true},{"shared_layout_bytes",new_sample["shared_scratch_layout_bytes"]},
                            {"routed_layout_bytes",new_sample["routed_layout_bytes"]}});
                    }
                    warmed_a=alloc_a;warmed_b=alloc_b;
                }
            }
        }
        std::ofstream out(argv[2]);out<<report.dump(2)<<'\n';require(out.good(),"scratch probe report write failed");
        fprintf(stderr,"SCRATCH cases=%zu physical_backings=1 aliases=3 capacity=%zu->%zu all bitwise equal\n",
                report["cases"].size(),original_bytes,common_bytes);
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"scratch probe: %s\n",e.what());return 1; }
}
