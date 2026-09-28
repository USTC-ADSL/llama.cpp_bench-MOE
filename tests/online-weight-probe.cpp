// Prototype gate for canonical-source HMX. No prompt routes or model warmup.
#define main previous_weight_prepare_main
#include "weight-prepare-probe.cpp"
#undef main
#include "ggml-shared-dma.h"

struct online_matrix {
    device & d;
    ggml_context_ptr weights, nodes;
    ggml_backend_buffer_ptr wb, ab;
    ggml_tensor * w, * x, * ids, * out;
    ggml_cgraph * graph;
    std::vector<float> input;
    std::vector<int32_t> routes;
    bool supported;
    online_matrix(device & dev, const source_matrix & m, int tokens, bool canonical) : d(dev) {
        weights.reset(ggml_init({ggml_tensor_overhead(),nullptr,true}));
        nodes.reset(ggml_init({8*ggml_tensor_overhead()+ggml_graph_overhead_custom(8,false),nullptr,true}));
        require(weights && nodes,"metadata allocation failed");
        w=ggml_new_tensor_3d(weights.get(),GGML_TYPE_Q4_0,m.width,m.height,2);
        ggml_set_name(w,m.name.c_str());
        wb.reset(ggml_backend_alloc_ctx_tensors_from_buft(weights.get(),d.buft));
        require(bool(wb),"weight allocation failed");
        ggml_backend_buffer_set_usage(wb.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(canonical) {
            using mark_fn=bool (*)(ggml_tensor *);
            auto mark=reinterpret_cast<mark_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(d.dev),"ggml_backend_hexagon_tensor_use_canonical_q4_0"));
            require(mark && mark(w),"canonical API rejected weight");
        }
        x=ggml_new_tensor_3d(nodes.get(),GGML_TYPE_F32,m.width,1,tokens);
        ids=ggml_new_tensor_2d(nodes.get(),GGML_TYPE_I32,1,tokens);
        out=ggml_mul_mat_id(nodes.get(),w,x,ids);
        supported=ggml_backend_supports_op(d.backend.get(),out);
        graph=ggml_new_graph_custom(nodes.get(),8,false); ggml_build_forward_expand(graph,out);
        ab.reset(ggml_backend_alloc_ctx_tensors_from_buft(nodes.get(),ggml_backend_dev_buffer_type(d.dev)));
        require(bool(ab),"activation allocation failed");
        input.resize(size_t(m.width)*tokens); routes.resize(tokens);
        for(int t=0;t<tokens;++t) {
            routes[t]=t%2;
            for(int k=0;k<m.width;++k) { input[t*m.width+k]=std::sin(float(k)*.13f+float(t)*.31f); }
        }
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(ids,routes.data(),0,routes.size()*sizeof(int32_t));
    }
    double prepare(const std::vector<uint8_t> & bytes) {
        auto start=clock_type::now(); ggml_backend_tensor_set(w,bytes.data(),0,bytes.size()); d.sync(); return elapsed(start);
    }
    double compute() {
        require(supported,"unsupported canonical shape"); auto start=clock_type::now();
        require(ggml_backend_graph_compute(d.backend.get(),graph)==GGML_STATUS_SUCCESS,"matrix compute failed");
        d.sync(); return elapsed(start);
    }
    std::vector<float> get() {
        std::vector<float> values(ggml_nelements(out));
        ggml_backend_tensor_get(out,values.data(),0,values.size()*sizeof(float)); return values;
    }
};

static json canonical_contract(device &d) {
    using mark_fn=bool (*)(ggml_tensor *);
    auto mark=reinterpret_cast<mark_fn>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(d.dev),"ggml_backend_hexagon_tensor_use_canonical_q4_0"));
    require(mark,"canonical API missing");
    ggml_context_ptr ctx(ggml_init({32*ggml_tensor_overhead(),nullptr,true}));
    auto *raw=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_Q4_0,128,32,2);
    auto *packed=ggml_dup_tensor(ctx.get(),raw);
    auto *bad=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_Q4_0,32,32,2);
    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),d.buft));
    require(bool(buf),"contract buffer failed");
    ggml_backend_buffer_set_usage(buf.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    require(mark(raw) && !mark(bad),"canonical shape guard failed");
    std::vector<uint8_t> bytes(ggml_nbytes(raw)),got(bytes.size());
    for(size_t i=0;i<bytes.size();++i) { bytes[i]=uint8_t(i*17+3); }
    ggml_backend_tensor_set(raw,bytes.data(),0,bytes.size());
    require(mark(raw),"canonical idempotence failed");
    ggml_backend_tensor_set_2d(raw,bytes.data()+raw->nb[1],raw->nb[1]*2,raw->nb[1],3,raw->nb[1]*2,raw->nb[1]);
    auto expected=bytes;
    for(int i=0;i<3;++i) { std::copy_n(bytes.data()+(i+1)*raw->nb[1],raw->nb[1],expected.data()+(2+2*i)*raw->nb[1]); }
    ggml_backend_tensor_get(raw,got.data(),0,got.size()); require(got==expected,"canonical 2D update repacked bytes");
    std::vector<uint8_t> rows(3*raw->nb[1]);
    ggml_backend_tensor_get_2d(raw,rows.data(),2*raw->nb[1],raw->nb[1],3,raw->nb[1]*2,raw->nb[1]);
    require(std::equal(rows.begin(),rows.end(),bytes.begin()+raw->nb[1]),"canonical 2D read mismatch");
    ggml_backend_tensor_copy_async(d.backend.get(),d.backend.get(),raw,packed); d.sync();
    require(!mark(packed),"already written packed weight was reinterpreted");
    ggml_backend_tensor_get(packed,got.data(),0,got.size()); require(got==expected,"canonical-to-packed copy mismatch");
    ggml_backend_tensor_copy_async(d.backend.get(),d.backend.get(),packed,raw); d.sync();
    ggml_backend_tensor_get(raw,got.data(),0,got.size()); require(got==expected,"packed-to-canonical copy mismatch");
    auto *view=ggml_view_3d(ctx.get(),raw,128,32,1,raw->nb[1],raw->nb[2],raw->nb[2]);
    require(!mark(view),"canonical view marking accepted");
    auto *x=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,128,1,8);
    auto *ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,1,8);
    require(!ggml_backend_supports_op(d.backend.get(),ggml_mul_mat_id(ctx.get(),view,x,ids)),"canonical view compute accepted");
    auto *single=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,128,1,1);
    auto *single_ids=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,1,1);
    require(!ggml_backend_supports_op(d.backend.get(),ggml_mul_mat_id(ctx.get(),raw,single,single_ids)),"canonical HVX fallback accepted");
    auto *dense=ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,128,8,2);
    require(!ggml_backend_supports_op(d.backend.get(),ggml_mul_mat(ctx.get(),raw,dense)),"canonical ordinary matmul accepted");
    return {{"strided_set_get_exact",true},{"cross_layout_async_copy_fallback_exact",true},
        {"written_packed_mark_rejected",true},{"unsupported_shape_view_matmul_hvx_rejected",true}};
}

static json gpu_htp_pack_probe(online_matrix &canonical,online_matrix &packed,
        const std::vector<uint8_t> &bytes,int reps) {
    device gpu("GPUOpenCL");
    using export_fn=bool (*)(ggml_backend_buffer_t,ggml_backend_shared_dma_buffer_desc *);
    using import_fn=ggml_backend_buffer_t (*)(ggml_backend_t,const ggml_backend_shared_dma_buffer_desc *);
    using pack_fn=bool (*)(ggml_backend_t,ggml_backend_buffer_t,size_t,ggml_backend_buffer_t,size_t,uint32_t,uint32_t,uint32_t,uint64_t);
    auto export_dma=reinterpret_cast<export_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(canonical.d.dev),"ggml_backend_hexagon_buffer_export_shared_dma"));
    auto reg=ggml_backend_dev_backend_reg(gpu.dev);
    auto import=reinterpret_cast<import_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_import_shared_dma_compute_buffer"));
    auto pack=reinterpret_cast<pack_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_pack_q4_0_htp"));
    require(export_dma && import && pack,"GPU HTP conversion APIs missing");
    ggml_backend_shared_dma_buffer_desc sd{},td{};
    require(export_dma(canonical.wb.get(),&sd) && export_dma(packed.wb.get(),&td),"conversion DMA export failed");
    ggml_backend_buffer_ptr source(import(gpu.backend.get(),&sd)),target(import(gpu.backend.get(),&td));
    require(source && target,"conversion DMA import failed");
    const size_t so=static_cast<uint8_t *>(canonical.w->data)-static_cast<uint8_t *>(sd.host_ptr);
    const size_t to=static_cast<uint8_t *>(packed.w->data)-static_cast<uint8_t *>(td.host_ptr);
    auto run=[&](uint64_t mask) { require(pack(gpu.backend.get(),source.get(),so,target.get(),to,
        canonical.w->ne[0],canonical.w->ne[1],2,mask),"GPU HTP conversion rejected"); };
    run(2); std::vector<uint8_t> restored(bytes.size());
    ggml_backend_tensor_get(packed.w,restored.data(),0,restored.size());
    require(std::equal(bytes.begin()+bytes.size()/2,bytes.end(),restored.begin()),"sparse GPU conversion expert mapping failed");
    run(3); ggml_backend_tensor_get(packed.w,restored.data(),0,restored.size());
    require(restored==bytes,"GPU HTP layout mismatch");
    canonical.compute(); packed.compute(); require(canonical.get()==packed.get(),"GPU HTP converted result differs");
    std::vector<double> convert,combined;
    for(int r=0;r<reps;++r) { auto begin=clock_type::now(); run(3); double us=elapsed(begin);
        packed.compute(); combined.push_back(elapsed(begin)); convert.push_back(us); }
    return {{"sparse_mapping_and_roundtrip_exact",true},{"compute_bitwise_equal",true},
        {"gpu_prepare",stats(convert,bytes.size())},{"gpu_prepare_npu_compute",stats(combined,bytes.size())}};
}

static json mixed_matrix_check(online_matrix &canonical, online_matrix &packed) {
    ggml_context_ptr ctx(ggml_init({2*ggml_tensor_overhead(),nullptr,true}));
    auto *map=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_I32,2);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(),canonical.d.buft));
    require(bool(buffer),"mixed map allocation failed");
    canonical.compute(); const auto reference=canonical.get();
    canonical.out->src[3]=packed.w; canonical.out->src[4]=map;
    require(ggml_backend_supports_op(canonical.d.backend.get(),canonical.out),"mixed matrix unsupported");
    ggml_graph_clear(canonical.graph); ggml_build_forward_expand(canonical.graph,canonical.out);
    for (const std::array<int32_t,2> values : {std::array<int32_t,2>{-1,-1}, {-1,1}, {0,-1}, {0,1}}) {
        ggml_backend_tensor_set(map,values.data(),0,sizeof(values));
        canonical.compute(); require(canonical.get()==reference,"mixed matrix output differs");
    }
    auto *bad=ggml_new_tensor_1d(ctx.get(),GGML_TYPE_F32,2);
    canonical.out->src[4]=bad;
    require(!ggml_backend_supports_op(canonical.d.backend.get(),canonical.out),"mixed map type guard failed");
    canonical.out->src[3]=canonical.w; canonical.out->src[4]=map;
    require(!ggml_backend_supports_op(canonical.d.backend.get(),canonical.out),"mixed packed layout guard failed");
    canonical.out->src[3]=canonical.out->src[4]=nullptr;
    ggml_graph_clear(canonical.graph); ggml_build_forward_expand(canonical.graph,canonical.out);
    return {{"all_miss_partial_all_hit_exact",true},{"type_and_layout_guards",true}};
}

int main(int argc,char ** argv) {
    try {
        require(argc==4,"usage: online-weight-probe MODEL OUTPUT REPS");
        int reps=std::stoi(argv[3]); require(reps>=4 && reps<=1000,"invalid reps");
        ggml_log_set([](ggml_log_level l,const char *s,void*) {
            if(l!=GGML_LOG_LEVEL_DEBUG || (std::getenv("MOE_DSP_TRACE") && (strstr(s,"profile-op") || strstr(s,"trace-evt")))) { fputs(s,stderr); }
        },nullptr);
        ggml_context *raw=nullptr; gguf_context_ptr gf(gguf_init_from_file(argv[1],{true,&raw})); ggml_context_ptr meta(raw);
        require(gf && meta,"GGUF metadata failed");
        std::unique_ptr<FILE,decltype(&fclose)> file(fopen(argv[1],"rb"),fclose); require(bool(file),"source open failed");
        ggml_backend_load_all(); device htp("HTP0");
        json report={{"scope","two actual experts; canonical GGUF in shared DMA versus CPU repack to tiled HTP; no prompt or whole-model claim"},
            {"reps",reps},{"warmups",2},{"cases",json::array()}};
        report["canonical_contract"]=canonical_contract(htp);
        for(const char *part:{"gate","up","down"}) {
            source_matrix m; m.name=std::string("blk.0.ffn_")+part+"_exps.weight";
            auto *t=ggml_get_tensor(meta.get(),m.name.c_str()); require(t && t->type==GGML_TYPE_Q4_0,"missing Q4_0 weight");
            m.width=t->ne[0]; m.height=t->ne[1]; m.bytes=t->nb[2]; m.experts=t->ne[2];
            m.offset=gguf_get_data_offset(gf.get())+gguf_get_tensor_offset(gf.get(),gguf_find_tensor(gf.get(),m.name.c_str()));
            std::vector<uint8_t> bytes(2*m.bytes),one(m.bytes);
            for(int e=0;e<2;++e) { read_matrix(file.get(),m,e+7,one); std::copy(one.begin(),one.end(),bytes.begin()+e*m.bytes); }
            for(int nt:{8,32,128,256}) {
                online_matrix packed(htp,m,nt,false),canonical(htp,m,nt,true);
                const double initialize_us=canonical.prepare(bytes); packed.prepare(bytes);
                std::vector<uint8_t> restored(bytes.size());
                ggml_backend_tensor_get(canonical.w,restored.data(),0,restored.size());
                require(restored==bytes,"canonical source bytes changed");
                json item={{"matrix",part},{"tokens",nt},{"experts",2},{"canonical_supported",canonical.supported},
                    {"canonical_initial_copy_us",initialize_us},{"source_bytes",bytes.size()},
                    {"weight_buffer_bytes",ggml_backend_buffer_get_size(canonical.wb.get())}};
                if(canonical.supported) {
                    canonical.compute(); packed.compute(); auto a=canonical.get(),b=packed.get();
                    double error=0,signal=0; for(size_t i=0;i<a.size();++i) {
                        require(std::isfinite(a[i]) && std::isfinite(b[i]),"nonfinite output");
                        error+=std::pow(double(a[i])-b[i],2); signal+=double(b[i])*b[i];
                    }
                    require(signal>1e-10,"vacuous output"); double nrmse=std::sqrt(error/signal);
                    item["versus_packed_nrmse"]=nrmse; item["bitwise_equal"]=std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0;
                    require(nrmse<1e-6,"canonical compute differs from unchanged tiled path");
                    std::vector<float> row(m.width); auto dequant=ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float;
                    double re=0,rs=0;
                    for(int p:{0,nt-1}) { for(int r=0;r<m.height;++r) {
                        dequant(bytes.data()+(p%2)*m.bytes+r*ggml_row_size(GGML_TYPE_Q4_0,m.width),row.data(),m.width);
                        double expected=0; for(int k=0;k<m.width;++k) { expected+=double(row[k])*canonical.input[p*m.width+k]; }
                        re+=std::pow(a[p*m.height+r]-expected,2); rs+=expected*expected;
                    } }
                    double rn=std::sqrt(re/rs); item["versus_fp64_nrmse"]=rn; require(rn<.02,"canonical reference check failed");
                    std::vector<double> pc,cc,pp,total;
                    for(int r=-2;r<reps;++r) {
                        double cu,pu,prep,combined;
                        auto packed_run=[&] { pu=packed.compute(); auto start=clock_type::now(); prep=packed.prepare(bytes); packed.compute(); combined=elapsed(start); };
                        if((r+2)%2) { packed_run(); cu=canonical.compute(); }
                        else { cu=canonical.compute(); packed_run(); }
                        if(r>=0) { cc.push_back(cu); pc.push_back(pu); pp.push_back(prep); total.push_back(combined); }
                        require(canonical.get()==a,"canonical repeated output changed");
                    }
                    item["canonical_compute"]=stats(cc,bytes.size()); item["packed_compute"]=stats(pc,bytes.size());
                    item["host_prepare"]=stats(pp,bytes.size()); item["host_prepare_compute"]=stats(total,bytes.size());
                    if(nt==128) { item["gpu_htp_pack"]=gpu_htp_pack_probe(canonical,packed,bytes,reps); }
                    item["mixed_layout"]=mixed_matrix_check(canonical,packed);
                    fprintf(stderr,"ONLINE_WEIGHT %s nt=%d canonical=%.3f tiled=%.3f prepare_compute=%.3f us nrmse=%.9g\n",part,nt,
                        double(item["canonical_compute"]["median_us"]),double(item["packed_compute"]["median_us"]),double(item["host_prepare_compute"]["median_us"]),nrmse);
                }
                report["cases"].push_back(item); std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(bool(out),"write failed");
            }
        }
        // Dense shared-Expert GPU preparation: independently compare the old
        // three-pass path, fused conversion, raw round trips and FP64 dots.
        device gpu("GPUOpenCL"); report["gpu_shared"]=json::array();
        for(const char *part:{"gate","up","down"}) {
            source_matrix m; m.name=std::string("blk.0.ffn_")+part+"_shexp.weight";
            auto *t=ggml_get_tensor(meta.get(),m.name.c_str()); require(t && t->type==GGML_TYPE_Q4_0,"missing shared weight");
            m.width=t->ne[0]; m.height=t->ne[1]; m.bytes=ggml_nbytes(t); m.experts=1;
            m.offset=gguf_get_data_offset(gf.get())+gguf_get_tensor_offset(gf.get(),gguf_find_tensor(gf.get(),m.name.c_str()));
            std::vector<uint8_t> bytes(m.bytes),back(m.bytes); read_matrix(file.get(),m,0,bytes);
            std::vector<double> old_us,new_us; json old_check,new_check;
            std::vector<float> previous;
            for(int r=0;r<reps;++r) {
                // Fresh target avoids reusing layout metadata as if it were an
                // ordinary tensor descriptor. Allocation is measured separately.
                for(int which=0;which<2;++which) {
                    int fused=(r%2)?1-which:which;
                    if(fused) { setenv("GGML_OPENCL_FUSED_Q4_PREPARE","1",1); } else { unsetenv("GGML_OPENCL_FUSED_Q4_PREPARE"); }
                    ready_matrix ready(gpu,m,1); auto start=clock_type::now(); ready.set(bytes); double us=elapsed(start);
                    (fused?new_us:old_us).push_back(us);
                    if(r==0) {
                        ready.get(back); require(back==bytes,"GPU shared canonical round trip mismatch");
                        auto check=ready.validate(bytes); if(fused) { new_check=check; } else { old_check=check; }
                        std::vector<float> output(ggml_nelements(ready.out)); ggml_backend_tensor_get(ready.out,output.data(),0,output.size()*sizeof(float));
                        if(previous.empty()) { previous=output; } else { require(output==previous,"GPU fused shared output differs"); }
                    }
                }
            }
            setenv("GGML_OPENCL_FUSED_Q4_PREPARE","1",1);
            ready_matrix reused(gpu,m,1); std::vector<double> reused_us;
            for(int r=0;r<reps;++r) {
                auto changing=bytes; changing[2+18*(r%32)]^=0x17;
                auto start=clock_type::now(); reused.set(changing); reused_us.push_back(elapsed(start));
                reused.get(back); require(back==changing,"reused GPU layout returned stale bytes"); reused.validate(changing);
            }
            report["gpu_shared"].push_back({{"matrix",part},{"old_prepare",stats(old_us,m.bytes)},
                {"fused_prepare",stats(new_us,m.bytes)},{"reused_fused_prepare",stats(reused_us,m.bytes)},
                {"old_validation",old_check},{"fused_validation",new_check}});
            fprintf(stderr,"GPU_PREPARE %s old=%.3f fused=%.3f us\n",part,double(report["gpu_shared"].back()["old_prepare"]["median_us"]),double(report["gpu_shared"].back()["fused_prepare"]["median_us"]));
            std::ofstream out(argv[2]); out<<report.dump(2)<<'\n'; require(bool(out),"write failed");
        }
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"online-weight-probe: %s\n",e.what()); return 1; }
}
