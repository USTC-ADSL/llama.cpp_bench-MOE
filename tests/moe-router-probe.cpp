// Real router weights and captured normalized inputs; no expert computation.
#include "expert-probe-common.h"
#include "ggml-opencl.h"
#include "ggml-hexagon.h"

struct router_job {
    device & d;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * w, * x, * projection, * probs, * ids;
    ggml_cgraph * graph;
    std::vector<double> samples;
    std::vector<double> device_samples;
    ggml_backend_opencl_get_compute_timing_fn cl_timing=nullptr;
    ggml_backend_hexagon_get_timing_fn htp_timing=nullptr;
    bool supported=true;
    json unsupported=json::array();
    router_job(device & dev, ggml_type type, int nt, bool full,
               const std::vector<uint8_t> & weights, const std::vector<float> & input):d(dev) {
        ctx.reset(ggml_init({32*ggml_tensor_overhead()+ggml_graph_overhead_custom(32,false),nullptr,true}));
        require(bool(ctx),"router metadata allocation failed");
        w=ggml_new_tensor_2d(ctx.get(),type,2048,60);
        x=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,2048,nt);
        projection=ggml_mul_mat(ctx.get(),w,x);
        probs=full?ggml_soft_max(ctx.get(),projection):nullptr;
        ids=full?ggml_argsort_top_k(ctx.get(),probs,4):nullptr;
        graph=ggml_new_graph_custom(ctx.get(),32,false);
        ggml_build_forward_expand(graph,full?ids:projection);
        for(int n=0;n<ggml_graph_n_nodes(graph);++n) {
            auto * node=ggml_graph_node(graph,n);
            if(!ggml_backend_supports_op(d.backend.get(),node)) {
                supported=false; unsupported.push_back(ggml_op_name(node->op));
            }
        }
        if(!supported) { return; }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(),d.backend.get()));
        require(bool(buffer),"router allocation failed");
        ggml_backend_tensor_set(w,weights.data(),0,weights.size());
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float)); d.sync();
        auto reg=ggml_backend_dev_backend_reg(d.dev);
        cl_timing=reinterpret_cast<ggml_backend_opencl_get_compute_timing_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_get_compute_timing"));
        htp_timing=reinterpret_cast<ggml_backend_hexagon_get_timing_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hexagon_get_timing"));
    }
    void run(bool record) {
        if(!supported) { return; }
        ggml_opencl_compute_timing cb{},ca{}; ggml_hexagon_timing hb{},ha{};
        bool cl_active=cl_timing && cl_timing(d.backend.get(),&cb);
        bool htp_active=htp_timing && htp_timing(&hb,sizeof(hb)) && hb.enabled && hb.profile_mode;
        auto begin=timer::now(); d.compute(graph); double us=elapsed(begin),device_us=-1;
        if(cl_active) { require(cl_timing(d.backend.get(),&ca) && ca.kernels>cb.kernels,"missing GPU events"); device_us=double(ca.kernel_ns-cb.kernel_ns)/1000; }
        if(htp_active) { require(htp_timing(&ha,sizeof(ha)) && ha.counters[GGML_HEXAGON_TIMING_OPS]>hb.counters[GGML_HEXAGON_TIMING_OPS],"missing HTP op timing"); device_us=double(ha.counters[GGML_HEXAGON_TIMING_OP_US]-hb.counters[GGML_HEXAGON_TIMING_OP_US]); }
        if(record) { samples.push_back(us); }
        if(record && device_us>=0) { device_samples.push_back(device_us); }
    }
    std::vector<float> values(ggml_tensor * t) {
        std::vector<float> v(ggml_nelements(t)); ggml_backend_tensor_get(t,v.data(),0,v.size()*sizeof(float)); return v;
    }
};

int main(int argc,char ** argv) {
    try {
        require(argc==6,"usage: moe-router-probe MODEL CAPTURE128_LAYER_BIN LAYER OUTPUT REPS");
        const int layer=std::stoi(argv[3]),reps=std::stoi(argv[5]);
        require(layer>=0 && layer<24 && reps>=8,"invalid router probe arguments");
        ggml_backend_load_all(); device htp("HTP0"),gpu("GPUOpenCL");
        ggml_context * raw=nullptr; gguf_context_ptr gguf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gguf && meta,"model metadata unavailable");
        const auto name="blk."+std::to_string(layer)+".ffn_gate_inp.weight";
        auto * tensor=ggml_get_tensor(meta.get(),name.c_str());
        require(tensor && tensor->ne[0]==2048 && tensor->ne[1]==60,"unexpected router weight shape");
        require(tensor->type==GGML_TYPE_F32 || tensor->type==GGML_TYPE_F16,"unexpected router weight type");
        std::vector<uint8_t> weights(ggml_nbytes(tensor));
        std::ifstream model(argv[1],std::ios::binary);
        model.seekg(gguf_get_data_offset(gguf.get())+gguf_get_tensor_offset(gguf.get(),gguf_find_tensor(gguf.get(),name.c_str())));
        model.read(reinterpret_cast<char *>(weights.data()),weights.size()); require(model.good(),"router weights read failed");
        std::vector<float> wf(2048*60),base(2048*128);
        if(tensor->type==GGML_TYPE_F32) { std::memcpy(wf.data(),weights.data(),weights.size()); }
        else { ggml_get_type_traits(tensor->type)->to_float(weights.data(),wf.data(),wf.size()); }
        std::ifstream capture(argv[2],std::ios::binary); capture.read(reinterpret_cast<char *>(base.data()),base.size()*sizeof(float));
        require(capture.good(),"captured normalized input read failed");
        std::vector<float> base_ref(60*128);
        for(int t=0;t<128;++t) { for(int e=0;e<60;++e) {
            double v=0; for(int k=0;k<2048;++k) { v+=double(wf[e*2048+k])*base[t*2048+k]; }
            base_ref[t*60+e]=float(v);
        } }
        json report={{"layer",layer},{"weight_type",ggml_type_name(tensor->type)},
            {"scope","resident router projection or projection+softmax+argsort_top_k; synchronized host wall; no transfer/readback in timing; real 128-token inputs tiled; unsupported ops reported without fallback"},
            {"warmups",8},{"reps",reps},{"cases",json::array()}};
        for(int nt:{1,128,768,1024,2048,4096}) {
            std::vector<float> input(2048*nt),reference(60*nt);
            for(int t=0;t<nt;++t) { std::copy_n(base.data()+(t%128)*2048,2048,input.data()+t*2048); std::copy_n(base_ref.data()+(t%128)*60,60,reference.data()+t*60); }
            for(bool full:{false,true}) {
                router_job a(htp,tensor->type,nt,full,weights,input),b(gpu,tensor->type,nt,full,weights,input);
                for(int r=-8;r<reps;++r) { if(r%2) { b.run(r>=0); a.run(r>=0); } else { a.run(r>=0); b.run(r>=0); } }
                json row={{"tokens",nt},{"mode",full?"full_router":"projection"},{"backends",json::object()}};
                for(auto * j:{&a,&b}) {
                    json result={{"supported",j->supported},{"unsupported_ops",j->unsupported}};
                    if(j->supported) {
                        result["timing"]=stats(j->samples,nt);
                        if(!j->device_samples.empty()) { result["device_timing"]=stats(j->device_samples,nt); }
                        result["projection_check"]=validate_values(j->values(j->projection),reference,nt,j->d.name.c_str());
                        if(full) {
                            auto probs=j->values(j->probs); std::vector<int32_t> ids(nt*4);
                            // argsort_top_k is a 4-column view of a 60-column
                            // sorted tensor, so the row stride is not 4*sizeof(int).
                            std::vector<uint8_t> raw_ids(ggml_nbytes(j->ids));
                            ggml_backend_tensor_get(j->ids,raw_ids.data(),0,raw_ids.size());
                            for(int t=0;t<nt;++t) { std::memcpy(ids.data()+t*4,raw_ids.data()+t*j->ids->nb[1],4*sizeof(int32_t)); }
                            size_t mismatch=0; double worst_sum=0;
                            for(int t=0;t<nt;++t) {
                                std::array<int,60> order; std::iota(order.begin(),order.end(),0);
                                std::partial_sort(order.begin(),order.begin()+4,order.end(),[&](int l,int r){return reference[t*60+l]>reference[t*60+r];});
                                double sum=0; for(int e=0;e<60;++e) { require(std::isfinite(probs[t*60+e]),"nonfinite router probability"); sum+=probs[t*60+e]; }
                                worst_sum=std::max(worst_sum,std::abs(sum-1));
                                for(int k=0;k<4;++k) { require(ids[t*4+k]>=0 && ids[t*4+k]<60,"invalid router ID"); mismatch+=ids[t*4+k]!=order[k]; }
                            }
                            require(worst_sum<1e-3,"router probabilities not normalized");
                            result["top4_position_mismatches"]=mismatch; result["top4_positions"]=nt*4; result["worst_probability_sum_error"]=worst_sum;
                        }
                    }
                    row["backends"][j->d.name]=result;
                }
                report["cases"].push_back(row);
            }
        }
        std::ofstream out(argv[4]); out<<report.dump(2)<<'\n'; require(out.good(),"router report write failed");
    } catch(const std::exception & e) { fprintf(stderr,"ERROR: %s\n",e.what()); return 1; }
}
