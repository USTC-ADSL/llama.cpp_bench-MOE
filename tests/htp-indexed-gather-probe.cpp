// Device regression: nonmonotonic/duplicate token maps, graph-cache revisits,
// observable gather output, unretained inputs, and gate/up NX enabled/disabled.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

struct indexed_case {
    ggml_context_ptr ctx = context(64,64);
    ggml_tensor * x, * rows, * ids, * gather;
    std::array<ggml_tensor *,2> out;
    ggml_cgraph * graph;
    std::shared_ptr<shared_activation_buffer> arena;
    std::string name;
    bool fuse;
    indexed_case(device & dev, bank & w, std::string name, bool fuse, bool retained, bool visible, bool interposed=false)
        : name(std::move(name)), fuse(fuse) {
        constexpr int nt=16, pairs=71;
        x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,nt);
        rows=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,pairs);
        ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,1,pairs);
        if(retained) { ggml_set_output(x); ggml_set_output(rows); }
        gather=ggml_get_rows(ctx.get(),x,rows);
        if(visible) { ggml_set_output(gather); }
        ggml_set_name(gather,(this->name+"-gather").c_str());
        auto * act=ggml_reshape_3d(ctx.get(),gather,2048,1,pairs);
        graph=ggml_new_graph_custom(ctx.get(),32,false);
        if(interposed) {
            ggml_build_forward_expand(graph,gather);
            ggml_build_forward_expand(graph,ggml_sqr(ctx.get(),x));
        }
        for(int i=0;i<2;++i) {
            out[i]=ggml_mul_mat_id(ctx.get(),w.w[i],act,ids);
            ggml_set_output(out[i]);
            ggml_set_name(out[i],(this->name+"-projection"+std::to_string(i)).c_str());
            ggml_build_forward_expand(graph,out[i]);
        }
        arena=std::make_shared<shared_activation_buffer>(dev,shared_activation_buffer::required_size(dev,ctx.get(),graph));
        arena->bind(dev,ctx.get(),graph);
    }
    std::vector<float> run(device & dev,int pass) {
        setenv("GGML_HEXAGON_FUSE_MOE_GATHER",fuse?"1":"0",1);
        std::vector<float> data(2048*16);
        for(size_t i=0;i<data.size();++i) { data[i]=std::sin(float(i+pass*17)*0.03f); }
        std::copy(data.begin(),data.end(),static_cast<float *>(x->data));
        for(int p=0;p<71;++p) {
            static_cast<int32_t *>(rows->data)[p]=(p*7+3+pass*5)%16;
            static_cast<int32_t *>(ids->data)[p]=(p*13+pass*3)%60;
        }
        require(ggml_backend_graph_compute(dev.backend.get(),graph)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_synchronize(dev.backend.get());
        if(gather->flags & GGML_TENSOR_FLAG_OUTPUT) {
            for(int p=0;p<71;++p) {
                int t=(p*7+3+pass*5)%16;
                require(std::equal(data.data()+2048*t,data.data()+2048*(t+1),static_cast<float *>(gather->data)+2048*p),"observable GET_ROWS was skipped");
            }
        }
        std::vector<float> result;
        for(auto * t:out) { auto * d=static_cast<float *>(t->data); result.insert(result.end(),d,d+ggml_nelements(t)); }
        return result;
    }
};

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: htp-indexed-gather-probe MODEL REPORT");
        ggml_backend_load_all();
        ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose); require(gguf && meta && file,"source open failed");
        source src{gguf.get(),meta.get(),file.get()}; device dev("HTP0"); bank w(dev,src,0);
        indexed_case reference(dev,w,"reference",false,true,false);
        std::vector<std::unique_ptr<indexed_case>> cases;
        cases.push_back(std::make_unique<indexed_case>(dev,w,"fused",true,true,false));
        cases.push_back(std::make_unique<indexed_case>(dev,w,"unretained",true,false,false));
        cases.push_back(std::make_unique<indexed_case>(dev,w,"visible",true,true,true));
        cases.push_back(std::make_unique<indexed_case>(dev,w,"interposed",true,true,false,true));
        json report={{"cases",json::array()}};
        for(int pass:{0,1,2,1,0}) {
            auto expected=reference.run(dev,pass);
            auto next_expected=reference.run(dev,pass+1);
            for(auto & c:cases) {
                auto actual=c->run(dev,pass);
                require(actual==expected,"indexed/fallback projection changed output");
                require(c->run(dev,pass+1)==next_expected,"cached graph used stale row indices");
                report["cases"].push_back({{"name",c->name},{"pass",pass},{"bitwise_equal",true}});
            }
        }
        std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(out.good(),"report failed");
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
