#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: moe-shared-weight-probe MODEL OUTPUT");
        ggml_backend_load_all();
        ggml_context * raw=nullptr; gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gf && meta && f,"probe model open failed"); source src{gf.get(),meta.get(),f.get()};
        device cpu("CPU"),htp("HTP0"),gpu("GPUOpenCL"); device * devs[]={&cpu,&htp,&gpu};
        resident_placement_model::availability resident;
        for(auto & e:resident.experts) { e.fill(true); } resident.shared.fill(true);
        moe_shared_execution shared(devs,src,0,resident);
        for(int b=0;b<3;++b) { shared.arenas[b]=std::make_shared<shared_activation_buffer>(*devs[b],32*1024*1024); }
        json report={{"scope","real layer0 Q4_0 weights; synthetic F32 input; first/middle/last expert offsets; native FFN reference"},
                     {"unique_payload_bytes",shared.weights.payload_bytes},{"cases",json::array()}};
        for(int b=0;b<3;++b) {
            std::vector<int> subset{0,7,31,59}; bank native(*devs[b],src,0,&subset); shared_bank sw(*devs[b],src,0);
            for(int nt:{1,17,32}) {
                std::vector<float> x(2048*nt),probs(60*nt,0);
                std::vector<int32_t> ids(4*nt);
                for(size_t i=0;i<x.size();++i) { x[i]=.1f*std::sin(float(i%1009)*.07f); }
                for(int t=0;t<nt;++t) { for(int k=0;k<4;++k) { ids[t*4+k]=subset[k]; probs[t*60+subset[k]]=.25f; } }
                placement p{"probe",std::vector<int>(60,b),b}; size_t allocations=0; json phases;
                auto out=shared.run(x.data(),probs.data(),ids,p,nullptr,allocations,phases,0,{});
                oracle_job reference(native,&sw,nt,4,nt,true,false,true);
                std::copy(x.begin(),x.end(),reference.input_data()); reference.set_shared_input(x);
                for(int t=0;t<nt;++t) { for(int k=0;k<4;++k) { reference.index_data()[t*reference.stride+k]=k; } }
                reference.run(); std::vector<float> ref(x.size());
                for(int t=0;t<nt;++t) { for(int c=0;c<2048;++c) {
                    float sum=0; for(int k=0;k<4;++k) { sum=std::fma(.25f,reference.output_data()[(t*4+k)*2048+c],sum); }
                    ref[t*2048+c]=sum+reference.shared_output_data()[t*2048+c];
                } }
                auto check=validate_values(out,ref,nt,devs[b]->name.c_str());
                report["cases"].push_back({{"backend",devs[b]->name},{"tokens",nt},{"validation",check}});
            }
        }
        std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(out.good(),"probe output write failed");
        return 0;
    } catch(const std::exception & e) { fprintf(stderr,"shared MoE probe: %s\n",e.what()); return 1; }
}
