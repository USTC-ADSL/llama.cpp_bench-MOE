// One resident layer, fixed 128-token data tiled at larger lengths. Measures
// backend windows independently of routing-distribution and process changes.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc,char ** argv) {
    try {
        require(argc==4 || argc==5,"usage: moe-route-unit-probe MODEL CAPTURE128_LAYER0_BIN OUTPUT [CPU|HTP0|GPUOpenCL]");
        ggml_backend_load_all();
        std::ifstream input(argv[2],std::ios::binary);
        auto base_x=binary_read<float>(input,2048*128),base_probs=binary_read<float>(input,60*128);
        auto base_ids=binary_read<int32_t>(input,4*128);
        auto base_ref=binary_read<float>(input,2048*128);
        ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose); require(gguf && meta && file,"source open failed");
        source src{gguf.get(),meta.get(),file.get()};
        std::string backend=argc==5?argv[4]:"HTP0";
        require(backend=="CPU" || backend=="HTP0" || backend=="GPUOpenCL","unsupported probe backend");
        std::unique_ptr<device> backing_device;
        if(backend!="HTP0") { backing_device=std::make_unique<device>("HTP0"); }
        device dev(backend.c_str()); bank weights(dev,src,0); shared_bank shared(dev,src,0);
        const int modes=backend=="HTP0"?3:2;
        moe_device_timing timing(true);
        struct test_case {
            int nt,mode; std::unique_ptr<oracle_job> job;
            std::vector<float> x,probs,reference,result,first;
            std::vector<int32_t> ids;
            json samples=json::array(),check;
        };
        std::vector<test_case> cases;
        for(int nt:{128,768,1536}) { for(int mode=0;mode<modes;++mode) {
            bool native=mode==0,device_gather=mode==2;
            test_case c; c.nt=nt; c.mode=mode;
            for(int copy=0;copy<nt/128;++copy) {
                c.x.insert(c.x.end(),base_x.begin(),base_x.end());
                c.probs.insert(c.probs.end(),base_probs.begin(),base_probs.end());
                c.ids.insert(c.ids.end(),base_ids.begin(),base_ids.end());
                c.reference.insert(c.reference.end(),base_ref.begin(),base_ref.end());
            }
            c.result.resize(c.x.size());
            oracle_job layout(weights,&shared,native?nt:4*nt,native?4:1,nt,native,false,true,{},true,false,nullptr,nullptr,nullptr,device_gather);
            auto reserved=shared_activation_layout(dev,layout.ctx.get(),layout.graph).freeze();
            auto arena=std::make_shared<shared_activation_buffer>(dev,reserved.bytes);
            c.job=std::make_unique<oracle_job>(weights,&shared,native?nt:4*nt,native?4:1,nt,native,false,true,arena,false,true,&reserved,nullptr,nullptr,device_gather);
            cases.push_back(std::move(c));
        } }
        auto run=[&](test_case & c,bool record) {
            auto & j=*c.job; auto snapshot=timing.snapshot();
            rusage before{},after{}; getrusage(RUSAGE_SELF,&before);
            auto begin=timer::now(); j.reset_index_padding(); j.set_shared_input(c.x);
            size_t copied=0;
            if(j.sx && !j.sx->view_src) { copied+=c.x.size()*sizeof(float); }
            if(c.mode!=1) { std::copy(c.x.begin(),c.x.end(),j.input_data()); copied+=c.x.size()*sizeof(float); }
            for(int p=0;p<4*c.nt;++p) {
                if(c.mode==1) { std::copy_n(c.x.data()+(p/4)*2048,2048,j.input_data()+p*2048); copied+=2048*sizeof(float); }
                if(c.mode==2) { static_cast<int32_t *>(j.input_rows->data)[p]=p/4; }
                j.index_data()[c.mode==0?(p/4)*j.stride+p%4:p*j.stride]=c.ids[p];
            }
            double gather=elapsed(begin); begin=timer::now(); auto phases=j.run(); double compute=elapsed(begin);
            auto counters=timing.delta(snapshot,timing.snapshot()); begin=timer::now();
            for(int t=0;t<c.nt;++t) { for(int ch=0;ch<2048;++ch) {
                float sum=0; for(int k=0;k<4;++k) { int p=t*4+k; sum=std::fma(c.probs[t*60+c.ids[p]],j.output_data()[p*2048+ch],sum); }
                c.result[t*2048+ch]=sum+j.shared_output_data()[t*2048+ch];
            } }
            double reduce=elapsed(begin); getrusage(RUSAGE_SELF,&after);
            if(record) {
                require(c.first==c.result,"same-case repeated output changed");
                c.samples.push_back({{"gather_us",gather},{"compute_us",compute},{"reduce_us",reduce},{"htp_timing",counters},
                                     {"input_host_copy_bytes",copied},{"minor_faults",after.ru_minflt-before.ru_minflt},{"major_faults",after.ru_majflt-before.ru_majflt}});
            }
        };
        for(auto & c:cases) { run(c,false); run(c,false); c.first=c.result; c.check=chain_difference(c.result,c.reference,c.nt); }
        for(int sweep=0;sweep<6;++sweep) {
            for(size_t j=0;j<cases.size();++j) { size_t index=sweep%2?cases.size()-1-j:j; run(cases[index],true); }
        }
        json report={{"backend",backend},{"scope","single resident layer0; identical 128-token input/router pattern tiled; native/split/device-gather do equal top4 and shared work; alternating forward/reverse case order"},{"cases",json::array()}};
        for(size_t i=0;i<cases.size();++i) {
            auto & c=cases[i]; std::vector<int> counts(60); for(int id:c.ids) { ++counts[id]; }
            auto equivalence=chain_difference(c.first,cases[(i/modes)*modes].first,c.nt);
            require(equivalence["worst_token_nrmse"].get<double>()<1e-4,"native/split/device gather output mismatch");
            report["cases"].push_back({{"tokens",c.nt},{"mode",std::array<const char *,3>{"native","split","device_gather"}[c.mode]},
                {"counts",counts},{"pairs",c.ids.size()},{"reference_check",c.check},{"native_equivalence",equivalence},{"samples",c.samples}});
        }
        std::ofstream output(argv[3]); output<<report.dump(2)<<'\n'; require(output.good(),"probe report failed");
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
