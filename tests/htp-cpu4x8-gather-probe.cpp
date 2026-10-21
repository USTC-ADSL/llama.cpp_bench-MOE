// Indexed ordinary CPU4x8 matmul: exact reference, cache refresh and rewrite guards.
#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

struct cpu4x8_indexed_case {
    ggml_context_ptr ctx=context(64,64);
    ggml_tensor * x, * rows, * gather;
    std::array<ggml_tensor *,2> out;
    ggml_cgraph * graph;
    std::shared_ptr<shared_activation_buffer> arena;
    std::string name;
    bool fuse;
    cpu4x8_indexed_case(device & dev,const std::array<ggml_tensor *,2> & weights,
                        int count,std::string label,bool enabled,bool retained=true,bool visible=false,
                        bool interposed=false,bool extra_user=false) : name(std::move(label)),fuse(enabled) {
        x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,17);
        rows=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,count);
        if(retained) { ggml_set_output(x); ggml_set_output(rows); }
        gather=ggml_get_rows(ctx.get(),x,rows);
        if(visible) { ggml_set_output(gather); }
        ggml_set_name(gather,(name+"-gather").c_str());
        graph=ggml_new_graph_custom(ctx.get(),32,false);
        if(interposed) {
            ggml_build_forward_expand(graph,gather);
            ggml_build_forward_expand(graph,ggml_sqr(ctx.get(),x));
        }
        for(int i=0;i<2;++i) {
            out[i]=ggml_mul_mat(ctx.get(),weights[i],gather); ggml_set_output(out[i]);
            ggml_set_name(out[i],(name+"-projection"+std::to_string(i)).c_str());
            ggml_build_forward_expand(graph,out[i]);
        }
        if(extra_user) { ggml_build_forward_expand(graph,ggml_sqr(ctx.get(),gather)); }
        arena=std::make_shared<shared_activation_buffer>(dev,shared_activation_buffer::required_size(dev,ctx.get(),graph));
        arena->bind(dev,ctx.get(),graph);
    }
    std::vector<float> run(device & dev,int pass) {
        setenv("GGML_HEXAGON_FUSE_CPU4X8_GATHER",fuse?"1":"0",1);
        std::vector<float> data(2048*17);
        for(size_t i=0;i<data.size();++i) { data[i]=std::sin(float(i+pass*17)*.03f); }
        std::copy(data.begin(),data.end(),static_cast<float *>(x->data));
        for(int p=0;p<rows->ne[0];++p) { static_cast<int32_t *>(rows->data)[p]=(p*7+3+pass*5)%17; }
        require(ggml_backend_graph_compute(dev.backend.get(),graph)==GGML_STATUS_SUCCESS,"indexed compute failed");
        ggml_backend_synchronize(dev.backend.get());
        if(gather->flags & GGML_TENSOR_FLAG_OUTPUT) {
            for(int p=0;p<rows->ne[0];++p) {
                int t=(p*7+3+pass*5)%17;
                require(std::equal(data.data()+2048*t,data.data()+2048*(t+1),static_cast<float *>(gather->data)+2048*p),
                        "visible gather skipped");
            }
        }
        std::vector<float> result;
        for(auto * t:out) { auto * ptr=static_cast<float *>(t->data); result.insert(result.end(),ptr,ptr+ggml_nelements(t)); }
        return result;
    }
};

int main(int argc,char ** argv) {
    try {
        require(argc==3,"usage: htp-cpu4x8-gather-probe MODEL REPORT");
        ggml_backend_load_all(); ggml_context * raw=nullptr;
        gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE,decltype(&fclose)> f(fopen(argv[1],"rb"),fclose);
        require(gf && meta && f,"model open failed"); source src{gf.get(),meta.get(),f.get()}; device dev("HTP0");
        std::vector<moe_shared_weights::handle> owners; std::array<ggml_tensor *,2> weights;
        for(int i=0;i<2;++i) {
            const std::string name=i?"blk.0.ffn_up_exps.weight":"blk.0.ffn_gate_exps.weight";
            auto * t=src.tensor(name); auto bytes=src.read(name);
            moe_shared_weights::handle owner(ggml_shared_weight_create_with_layout(dev.backend.get(),t->ne[0],t->ne[1],
                bytes.data(),t->nb[2],GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2),ggml_shared_weight_free);
            require(bool(owner),"weight allocation failed"); weights[i]=ggml_shared_weight_bind(owner.get(),dev.backend.get());
            require(weights[i],"weight import failed"); owners.push_back(std::move(owner));
        }
        json report={{"cases",json::array()}};
        for(int count:{1,2,31,32,33,71,1025}) {
            cpu4x8_indexed_case reference(dev,weights,count,"reference",false);
            std::vector<std::unique_ptr<cpu4x8_indexed_case>> cases;
            cases.push_back(std::make_unique<cpu4x8_indexed_case>(dev,weights,count,"fused",true));
            cases.push_back(std::make_unique<cpu4x8_indexed_case>(dev,weights,count,"unretained",true,false));
            cases.push_back(std::make_unique<cpu4x8_indexed_case>(dev,weights,count,"visible",true,true,true));
            cases.push_back(std::make_unique<cpu4x8_indexed_case>(dev,weights,count,"interposed",true,true,false,true));
            cases.push_back(std::make_unique<cpu4x8_indexed_case>(dev,weights,count,"extra-user",true,true,false,false,true));
            for(int pass:{0,1,2,1,0}) {
                auto expected=reference.run(dev,pass),next=reference.run(dev,pass+1);
                for(auto & c:cases) {
                    require(c->run(dev,pass)==expected,"indexed result mismatch");
                    require(c->run(dev,pass+1)==next,"cached row map or input stale");
                    report["cases"].push_back({{"name",c->name},{"count",count},{"pass",pass},{"bitwise_equal",true}});
                }
            }
        }
        std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(out.good(),"report write failed");
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
