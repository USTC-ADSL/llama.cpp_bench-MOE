// Actual GGUF Q4_0 bytes, identical weights/activations/ids for CPU and target.
// Predeclared acceptance: finite outputs, relative RMSE <= .02 and
// max absolute error / RMS(reference) <= .10. Never adjust after observing results.
#include "ggml.h"
#include "gguf.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "json.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
using json = nlohmann::ordered_json;
struct weight { std::string name; int64_t k, m, experts; std::vector<char> bytes; };

json output_stats(const std::vector<float> & values) {
    double lo=INFINITY, hi=-INFINITY, sq=0; size_t nonzero=0, nonfinite=0;
    for (float v : values) {
        if (!std::isfinite(v)) { ++nonfinite; continue; }
        lo=std::min(lo,double(v)); hi=std::max(hi,double(v));
        sq+=double(v)*v; nonzero+=v!=0;
    }
    return {{"min",lo},{"max",hi},{"rms",std::sqrt(sq/values.size())},
            {"nonzero_fraction",double(nonzero)/values.size()},{"nonfinite_count",nonfinite}};
}

std::vector<float> compute(ggml_backend_t backend, const std::vector<weight> & ws,
                          int n, int used, bool broadcast, bool weight_context) {
    auto ctx = ggml_init({16*1024*1024, nullptr, true});
    auto wctx = weight_context ? ggml_init({16*1024*1024, nullptr, true}) : ctx;
    auto x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ws[0].k, broadcast ? 1 : used, n);
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, n);
    ggml_set_name(x, "E0_fixed_activation"); ggml_set_name(ids, "E0_fixed_ids");
    std::vector<ggml_tensor *> weights, outputs;
    auto graph = ggml_new_graph_custom(ctx, 128, false);
    for (const auto & w : ws) {
        auto a = ggml_new_tensor_3d(wctx, GGML_TYPE_Q4_0, w.k, w.m, w.experts);
        ggml_set_name(a, w.name.c_str());
        auto y = ggml_mul_mat_id(ctx, a, x, ids);
        ggml_set_name(y, ("E0_"+w.name).c_str());
        if (!ggml_backend_supports_op(backend, y)) {
            if (weight_context) ggml_free(wctx);
            ggml_free(ctx); throw std::runtime_error("unsupported MUL_MAT_ID");
        }
        weights.push_back(a); outputs.push_back(y);
        ggml_build_forward_expand(graph, y);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    auto wbuffer = weight_context ? ggml_backend_alloc_ctx_tensors(wctx, backend) : nullptr;
    if (!buffer || (weight_context && !wbuffer)) {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (wbuffer) ggml_backend_buffer_free(wbuffer);
        if (weight_context) ggml_free(wctx);
        ggml_free(ctx); throw std::runtime_error("allocation_failed");
    }
    if (wbuffer) ggml_backend_buffer_set_usage(wbuffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (size_t i=0; i<ws.size(); ++i)
        ggml_backend_tensor_set(weights[i], ws[i].bytes.data(), 0, ws[i].bytes.size());
    std::vector<float> activation(ggml_nelements(x));
    // Deterministic, bounded, nonconstant input; no RNG drift across backends.
    for (size_t i=0; i<activation.size(); ++i) activation[i] = std::sin(float(i%8191)*.031f)*.5f;
    std::vector<int32_t> routes(used*n);
    for (int t=0; t<n; ++t) for (int e=0; e<used; ++e)
        routes[t*used+e] = (t*7+e*13)%ws[0].experts;
    ggml_backend_tensor_set(x, activation.data(), 0, activation.size()*4);
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*4);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
        throw std::runtime_error("compute_failed");
    ggml_backend_synchronize(backend);
    std::vector<float> out;
    for (auto y : outputs) {
        size_t off=out.size(); out.resize(off+ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data()+off, 0, ggml_nbytes(y));
    }
    if (wbuffer) ggml_backend_buffer_free(wbuffer);
    if (weight_context) ggml_free(wctx);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) return 2;
    gguf_context * meta = gguf_init_from_file(argv[1], {true, nullptr});
    if (!meta) return 3;
    json info;
    info["model_path"] = argv[1];
    for (int64_t i=0; i<gguf_get_n_kv(meta); ++i) {
        std::string key=gguf_get_key(meta,i);
        if (key.find("qwen2moe.")==0 || key=="general.architecture" || key=="general.file_type") {
            auto type=gguf_get_kv_type(meta,i);
            if (type==GGUF_TYPE_STRING) info["metadata"][key]=gguf_get_val_str(meta,i);
            else if (type==GGUF_TYPE_UINT32) info["metadata"][key]=gguf_get_val_u32(meta,i);
            else if (type==GGUF_TYPE_FLOAT32) info["metadata"][key]=gguf_get_val_f32(meta,i);
        }
    }
    std::vector<weight> weights;
    std::ifstream file(argv[1], std::ios::binary);
    for (const char * suffix : {"gate", "up", "down"}) {
        std::string name=std::string("blk.0.ffn_")+suffix+"_exps.weight";
        auto id=gguf_find_tensor(meta, name.c_str());
        if (id<0) return 4;
        const auto * ne=gguf_get_tensor_ne(meta,id);
        auto type=gguf_get_tensor_type(meta,id);
        info["tensors"][name]={{"shape",{ne[0],ne[1],ne[2],ne[3]}},{"type",ggml_type_name(type)},
            {"offset",gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,id)},
            {"bytes",gguf_get_tensor_size(meta,id)}};
        if (type!=GGML_TYPE_Q4_0) return 5;
        weight w{name,ne[0],ne[1],ne[2],{}};
        if (argc>2) {
            w.bytes.resize(gguf_get_tensor_size(meta,id));
            file.seekg(gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,id));
            file.read(w.bytes.data(), w.bytes.size());
            if (!file) return 6;
        }
        weights.push_back(std::move(w));
    }
    std::cout << info.dump() << std::endl;
    if (argc==2) { gguf_free(meta); return 0; }
    int used=gguf_get_val_u32(meta,gguf_find_key(meta,"qwen2moe.expert_used_count"));
    ggml_backend_load_all();
    auto cpu=ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(cpu,4);
    ggml_backend_cpu_set_use_ref(cpu,true);
    auto target=ggml_backend_init_by_name(argv[2],nullptr);
    if (!target) { std::cout << "{\"status\":\"unsupported_backend\"}" << std::endl; return 7; }
    bool pass=true;
    const bool weight_context = std::getenv("E0_WEIGHT_CONTEXT") && std::string(std::getenv("E0_WEIGHT_CONTEXT")) == "1";
    for (int n : {1,128}) for (int kind=0; kind<2; ++kind) {
        std::vector<weight> ws = kind==0 ? std::vector<weight>{weights[0],weights[1]} : std::vector<weight>{weights[2]};
        json result={{"backend",argv[2]},{"n",n},{"used",used},{"operation",kind==0?"gate_up_NX_candidate":"down_MUL_MAT_ID"},
                     {"weight_context",weight_context},{"relative_rmse_limit",.02},{"max_error_over_ref_rms_limit",.10}};
        try {
            auto ref=compute(cpu,ws,n,used,kind==0,weight_context);
            auto got=compute(target,ws,n,used,kind==0,weight_context);
            result["reference_stats"]=output_stats(ref);
            result["output_stats"]=output_stats(got);
            double sq=0, refsq=0, maxerr=0; bool finite=true;
            for (size_t i=0;i<ref.size();++i) {
                finite &= std::isfinite(ref[i]) && std::isfinite(got[i]);
                double d=double(got[i])-ref[i]; sq+=d*d; refsq+=double(ref[i])*ref[i];
                maxerr=std::max(maxerr,std::abs(d));
            }
            double rrmse=std::sqrt(sq/std::max(refsq,1e-30));
            double mx=maxerr/std::max(std::sqrt(refsq/ref.size()),1e-15);
            bool ok=finite && rrmse<=.02 && mx<=.10;
            result["finite"]=finite; result["relative_rmse"]=rrmse;
            result["max_error_over_ref_rms"]=mx; result["elements"]=ref.size();
            result["status"]=ok?"PASS":"FAIL"; pass &= ok;
        } catch (const std::exception & e) { result["status"]="BLOCKED"; result["reason"]=e.what(); pass=false; }
        std::cout << result.dump() << std::endl;
    }
    ggml_backend_free(target); ggml_backend_free(cpu); gguf_free(meta);
    return pass?0:1;
}
