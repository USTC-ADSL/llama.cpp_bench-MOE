#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"
#include "ggml-shared-weight.h"
#include "ggml-opencl.h"
#include "ggml-hexagon.h"
#include "gguf.h"
#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
using timer = std::chrono::steady_clock;
static void check(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}
static void quiet(enum ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); }
}
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return (v[(v.size()-1)/2] + v[v.size()/2])/2;
}
struct job {
    ggml_backend_t backend;
    ggml_context_ptr wc, gc;
    ggml_backend_buffer_ptr wb, gb;
    ggml_tensor * w, * x, * y;
    ggml_cgraph * graph;
    ggml_backend_opencl_get_compute_timing_fn cl_timing=nullptr;
    ggml_backend_hexagon_get_timing_fn htp_timing=nullptr;
    double device_us=-1;
    job(ggml_backend_t b, ggml_backend_buffer_type_t wt, int k, int m, int n,
        const std::vector<uint8_t> & raw, const std::vector<float> & input, ggml_tensor * shared=nullptr) : backend(b) {
        wc.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        gc.reset(ggml_init({8*ggml_tensor_overhead()+ggml_graph_overhead_custom(8,false), nullptr, true}));
        check(wc && gc, "metadata allocation failed");
        w = shared?shared:ggml_new_tensor_2d(wc.get(), GGML_TYPE_Q4_0, k, m);
        x = ggml_new_tensor_2d(gc.get(), GGML_TYPE_F32, k, n);
        y = ggml_mul_mat(gc.get(), w, x);
        if (!shared) {
            wb.reset(ggml_backend_alloc_ctx_tensors_from_buft(wc.get(), wt));
            check(wb != nullptr, "weight allocation failed");
            ggml_backend_buffer_set_usage(wb.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        check(ggml_backend_supports_op(b,y), "MUL_MAT unsupported");
        gb.reset(ggml_backend_alloc_ctx_tensors(gc.get(), b));
        check(gb != nullptr, "compute allocation failed");
        if (!shared) { ggml_backend_tensor_set(w, raw.data(), 0, raw.size()); }
        ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
        ggml_backend_synchronize(b);
        graph = ggml_new_graph_custom(gc.get(),8,false);
        ggml_build_forward_expand(graph,y);
        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(b));
        cl_timing=reinterpret_cast<ggml_backend_opencl_get_compute_timing_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_get_compute_timing"));
        htp_timing=reinterpret_cast<ggml_backend_hexagon_get_timing_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_hexagon_get_timing"));
    }
    double run() {
        ggml_opencl_compute_timing cl_before{},cl_after{};
        ggml_hexagon_timing htp_before{},htp_after{};
        bool cl_active=cl_timing && cl_timing(backend,&cl_before);
        bool htp_active=htp_timing && htp_timing(&htp_before,sizeof(htp_before)) && htp_before.enabled && htp_before.profile_mode;
        auto t = timer::now();
        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute failed");
        ggml_backend_synchronize(backend);
        double wall_us=std::chrono::duration<double,std::micro>(timer::now()-t).count();
        device_us=-1;
        if(cl_active) {
            check(cl_timing(backend,&cl_after) && cl_after.kernels>cl_before.kernels,"missing GPU timing events");
            device_us=double(cl_after.kernel_ns-cl_before.kernel_ns)/1000;
        }
        if(htp_active) {
            check(htp_timing(&htp_after,sizeof(htp_after)) &&
                htp_after.counters[GGML_HEXAGON_TIMING_OPS]>htp_before.counters[GGML_HEXAGON_TIMING_OPS],"missing DSP timing events");
            device_us=double(htp_after.counters[GGML_HEXAGON_TIMING_OP_US]-htp_before.counters[GGML_HEXAGON_TIMING_OP_US]);
        }
        return wall_us;
    }
    double validate(const std::vector<float> & ref) {
        std::vector<float> out(ref.size());
        ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
        double e=0,s=0;
        for (size_t i=0;i<out.size();++i) {
            check(std::isfinite(out[i]),"nonfinite output");
            e += std::pow(double(out[i])-ref[i],2); s += double(ref[i])*ref[i];
        }
        check(s>0,"vacuous reference");
        double err=std::sqrt(e/s);
        check(err<0.02,"NRMSE exceeds 2% reference bound");
        return err;
    }
};
struct cpu_pool {
    ggml_backend_t backend=nullptr;
    ggml_threadpool_t pool=nullptr;
    void (*set)(ggml_backend_t,ggml_threadpool_t)=nullptr;
    void (*free)(ggml_threadpool_t)=nullptr;
    cpu_pool(ggml_backend_t b,ggml_backend_reg_t reg,int threads) {
        auto create=reinterpret_cast<ggml_threadpool_t (*)(ggml_threadpool_params *)>(ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_new"));
        set=reinterpret_cast<void (*)(ggml_backend_t,ggml_threadpool_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_cpu_set_threadpool"));
        free=reinterpret_cast<void (*)(ggml_threadpool_t)>(ggml_backend_reg_get_proc_address(reg,"ggml_threadpool_free"));
        if(!create || !set || !free) { return; }
        auto params=ggml_threadpool_params_default(threads);
        if(const char * mask=std::getenv("GGML_SHARED_TEST_CPUMASK")) {
            unsigned long long bits=std::stoull(mask,nullptr,16);
            check(bits && __builtin_popcountll(bits)>=threads,"affinity mask has too few CPUs");
            for(int i=0;i<64 && i<GGML_MAX_N_THREADS;++i) { params.cpumask[i]=(bits>>i)&1; }
            params.strict_cpu=true;
        }
        pool=create(&params); check(pool,"threadpool creation failed");
        backend=b; set(backend,pool);
    }
    ~cpu_pool() { if(pool) { set(backend,nullptr); free(pool); } }
};
int main(int argc,char **argv) {
    try {
        check(argc>=3,"usage: shared-weight-probe BACKEND OUTPUT [REPS=24] [THREADS=4]");
        int reps=argc>3?std::stoi(argv[3]):24, threads=argc>4?std::stoi(argv[4]):4;
        check(reps>=4 && threads>0,"invalid repetitions/threads");
        ggml_log_set(quiet,nullptr);
        ggml_backend_load_all();
        auto dev=ggml_backend_dev_by_name(argv[1]); check(dev,"backend unavailable");
        ggml_backend_ptr b(ggml_backend_dev_init(dev,nullptr)); check(b!=nullptr,"backend init failed");
        auto reg=ggml_backend_dev_backend_reg(dev);
        cpu_pool pool(b.get(),reg,threads);
        bool shared_mode=std::getenv("GGML_SHARED_WEIGHT_TEST")!=nullptr;
        const auto layout=std::getenv("GGML_SHARED_WEIGHT_CPU4X8")?
            GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2:GGML_SHARED_WEIGHT_Q4_0_32X32_V1;
        check(layout!=GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2 || (shared_mode && std::string(argv[1])!="HTP0"),
            "CPU4x8 candidate does not yet support HTP compute");
        bool edges=std::getenv("GGML_SHARED_WEIGHT_EDGES")!=nullptr;
        const char * model=std::getenv("GGML_SHARED_WEIGHT_MODEL");
        check(!model || !edges,"model and synthetic edge mode are separate suites");
        gguf_context_ptr model_gguf;
        ggml_context_ptr model_meta;
        std::ifstream model_file;
        if(model) {
            ggml_context * meta=nullptr;
            model_gguf.reset(gguf_init_from_file(model,{true,&meta}));
            model_meta.reset(meta);
            model_file.open(model,std::ios::binary);
            check(model_gguf && model_meta && bool(model_file),"cannot read GGUF metadata");
        }
        ggml_backend_ptr htp_owner, other_cpu, other_gpu;
        ggml_backend_t htp=nullptr,cpu=nullptr,gpu=nullptr;
        if(shared_mode) {
            auto init=[](const char * name) { auto d=ggml_backend_dev_by_name(name); check(d,"shared backend missing"); return ggml_backend_dev_init(d,nullptr); };
            if(std::string(argv[1])=="HTP0") { htp=b.get(); } else { htp_owner.reset(init("HTP0")); htp=htp_owner.get(); }
            if(std::string(argv[1])=="CPU") { cpu=b.get(); } else { other_cpu.reset(init("CPU")); cpu=other_cpu.get(); }
            if(std::string(argv[1])=="GPUOpenCL") { gpu=b.get(); } else { other_gpu.reset(init("GPUOpenCL")); gpu=other_gpu.get(); }
            check(htp && cpu && gpu,"shared backend init failed");
            ggml_shared_weight_free(nullptr);
            ggml_shared_weight_desc invalid{};
            uint8_t dummy=0;
            check(!ggml_shared_weight_describe(nullptr,&invalid) && !ggml_shared_weight_bind(nullptr,cpu),"null handle accepted");
            check(!ggml_shared_weight_create(nullptr,32,32,&dummy,576) &&
                !ggml_shared_weight_create(htp,0,32,&dummy,0) &&
                !ggml_shared_weight_create(htp,33,32,&dummy,576) &&
                !ggml_shared_weight_create(htp,32,32,&dummy,575),"invalid create accepted");
            for(auto backend:{cpu,gpu}) {
                auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                auto import=reinterpret_cast<ggml_backend_import_shared_weight_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_import_shared_weight"));
                check(import && !import(backend,nullptr) && !import(backend,&invalid),"invalid import accepted");
            }
        }
        auto set_threads=reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_set_n_threads"));
        if(set_threads) { set_threads(b.get(),threads); }
        std::vector<ggml_backend_buffer_type_t> types{ggml_backend_dev_buffer_type(dev)};
        if(std::string(argv[1])=="CPU") {
            auto extras=reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_dev_get_extra_bufts"));
            if(extras) { auto list=extras(dev); for(size_t i=0;list && list[i];++i) {
                if(std::string(ggml_backend_buft_name(list[i]))=="CPU_REPACK") { types.push_back(list[i]); }
            } }
        }
        json result={{"backend",argv[1]},{"threads",threads},{"reps",reps},
            {"persistent_cpu_pool",pool.pool!=nullptr},
            {"cpu_affinity",std::getenv("GGML_SHARED_TEST_CPUMASK")?std::getenv("GGML_SHARED_TEST_CPUMASK"):"default"},
            {"timing","single MUL_MAT graph + completion; weight loading/repack and activation upload excluded; invocation scratch included"},
            {"input","deterministic synthetic Q4_0; Qwen representative matrix shapes; no MoE routing"}, {"cases",json::array()}};
        if(model) { result["input"]={{"model",model},{"expert",7},{"activation","deterministic synthetic F32"},{"routing",false}}; }
        const std::vector<std::pair<int,int>> shapes=edges?std::vector<std::pair<int,int>>{{128,32},{256,96}}:
            std::vector<std::pair<int,int>>{{2048,1408},{1408,2048}};
        for(auto dims:shapes) {
            int k=dims.first,m=dims.second;
            std::vector<float> source(size_t(k)*m),dequant(source.size());
            for(size_t i=0;i<source.size();++i) { source[i]=0.1f*std::sin(float(i%10007)*0.071f)+0.03f*std::cos(float(i%997)*0.13f); }
            std::vector<uint8_t> raw(ggml_row_size(GGML_TYPE_Q4_0,k)*m);
            std::string source_name="synthetic";
            if(model) {
                source_name=k==2048?"blk.0.ffn_gate_exps.weight":"blk.0.ffn_down_exps.weight";
                auto * tensor=ggml_get_tensor(model_meta.get(),source_name.c_str());
                auto index=gguf_find_tensor(model_gguf.get(),source_name.c_str());
                check(tensor && index>=0 && tensor->type==GGML_TYPE_Q4_0 && tensor->ne[0]==k && tensor->ne[1]==m &&
                    tensor->ne[2]>7 && tensor->ne[3]==1 && tensor->nb[2]==raw.size(),"model tensor mismatch");
                auto offset=gguf_get_data_offset(model_gguf.get())+gguf_get_tensor_offset(model_gguf.get(),index)+7*raw.size();
                model_file.seekg(offset); model_file.read(reinterpret_cast<char *>(raw.data()),raw.size());
                check(bool(model_file),"short model tensor read");
            } else {
                ggml_quantize_chunk(GGML_TYPE_Q4_0,source.data(),raw.data(),0,m,k,nullptr);
            }
            std::unique_ptr<ggml_shared_weight,decltype(&ggml_shared_weight_free)> owner(nullptr,ggml_shared_weight_free);
            ggml_tensor * shared=nullptr;
            ggml_shared_weight_desc desc{};
            if(shared_mode) {
                owner.reset(ggml_shared_weight_create_with_layout(htp,k,m,raw.data(),raw.size(),layout)); check(owner!=nullptr,"shared create failed");
                check(ggml_shared_weight_describe(owner.get(),&desc),"describe failed");
                for(auto backend:{htp,cpu,gpu}) {
                    if(layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2 && backend==htp) {
                        check(!ggml_shared_weight_bind(owner.get(),backend),"unsupported CPU4x8 backend accepted");
                        continue;
                    }
                    auto * t=ggml_shared_weight_bind(owner.get(),backend); check(t,"shared bind failed");
                    check(ggml_shared_weight_bind(owner.get(),backend)==t,"repeated bind creates new alias");
                    check(t->data==static_cast<char *>(desc.storage.host_ptr)+desc.offset,"shared pointer differs");
                    std::vector<uint8_t> recovered(raw.size());
                    ggml_backend_tensor_get(t,recovered.data(),0,recovered.size());
                    check(recovered==raw,"shared layout byte roundtrip differs");
                    if(backend!=htp) {
                        uint8_t partial[37]; ggml_backend_tensor_get(t,partial,17,sizeof(partial));
                        check(std::memcmp(partial,raw.data()+17,sizeof(partial))==0,"partial byte roundtrip differs");
                        ggml_backend_tensor_get(t,partial,raw.size()-sizeof(partial),sizeof(partial));
                        check(std::memcmp(partial,raw.data()+raw.size()-sizeof(partial),sizeof(partial))==0,"tail byte roundtrip differs");
                        auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
                        auto import=reinterpret_cast<ggml_backend_import_shared_weight_t>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_import_shared_weight"));
                        auto bad=desc; bad.storage.size=raw.size()-1;
                        check(!import(backend,&bad),"undersized import accepted");
                        bad=desc; bad.layout=0;
                        check(!import(backend,&bad),"unknown layout accepted");
                        bad=desc; bad.offset=1;
                        check(!import(backend,&bad),"misaligned import accepted");
                        ggml_context_ptr negctx(ggml_init({12*ggml_tensor_overhead()+ggml_graph_overhead_custom(8,false),nullptr,true}));
                        auto * unsupported=ggml_dup(negctx.get(),t);
                        check(!ggml_backend_supports_op(backend,unsupported),"shared DUP advertised as supported");
                        auto * graph=ggml_new_graph_custom(negctx.get(),8,false);
                        ggml_build_forward_expand(graph,unsupported);
                        check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_FAILED,"shared DUP did not reject before execution");
                        auto * view=ggml_view_2d(negctx.get(),t,t->ne[0],t->ne[1],t->nb[1],0);
                        auto * x=ggml_new_tensor_2d(negctx.get(),GGML_TYPE_F32,k,1);
                        auto * mm=ggml_mul_mat(negctx.get(),view,x);
                        check(!ggml_backend_supports_op(backend,mm),"shared weight view accepted");
                    }
                }
                shared=ggml_shared_weight_bind(owner.get(),b.get()); check(shared,"target binding failed");
                result["shared_storage"]={{"layout",desc.layout},{"weight_bytes",raw.size()},
                    {"allocation_bytes",desc.storage.size},{"backend_bindings",layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2?2:3},
                    {"pointer_alias",true},{"roundtrip_exact",true}};
            }
            auto to_float=ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float;
            for(int r=0;r<m;++r) { to_float(raw.data()+ggml_row_size(GGML_TYPE_Q4_0,k)*r,dequant.data()+size_t(r)*k,k); }
            const std::vector<int> batches=edges?std::vector<int>{1,2,3,4,5,7,8,9,31,33}:std::vector<int>{1,32,128};
            for(int n:batches) {
                std::vector<float> input(size_t(k)*n),ref(size_t(m)*n);
                for(size_t i=0;i<input.size();++i) { input[i]=std::sin(float(i%10009)*0.13f)+0.1f*std::cos(float(i%997)*0.021f); }
                for(int t=0;t<n;++t) { for(int r=0;r<m;++r) {
                    double acc=0; for(int c=0;c<k;++c) { acc+=double(dequant[size_t(r)*k+c])*input[size_t(t)*k+c]; }
                    ref[size_t(t)*m+r]=float(acc);
                } }
                std::vector<std::unique_ptr<job>> jobs;
                std::vector<std::vector<double>> samples(types.size());
                std::vector<std::vector<double>> device_samples(types.size());
                json rows=json::array();
                for(auto type:types) {
                    jobs.emplace_back(new job(b.get(),type,k,m,n,raw,input));
                    for(int i=0;i<8;++i) { jobs.back()->run(); }
                    rows.push_back({{"buffer",ggml_backend_buft_name(type)}, {"nrmse",jobs.back()->validate(ref)},
                                    {"weight_buffer_bytes",ggml_backend_buffer_get_size(jobs.back()->wb.get())}});
                }
                if(shared) {
                    jobs.emplace_back(new job(b.get(),nullptr,k,m,n,raw,input,shared));
                    for(int i=0;i<8;++i) { jobs.back()->run(); }
                    if(n==1) {
                        // Deliberate test-only mutation of otherwise immutable storage,
                        // after all work completes. A hidden private copy would keep the
                        // old result. Restore before formal timing; never a public update API.
                        auto flip=[&] {
                            auto * packed=static_cast<uint8_t *>(desc.storage.host_ptr)+desc.offset;
                            if(layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2) {
                                for(size_t block=0;block<raw.size()/72;++block) {
                                    for(int row=0;row<4;++row) { packed[block*72+row*2+1]^=0x80; }
                                }
                            } else { for(size_t tile=0;tile<raw.size()/576;++tile) {
                                for(int row=0;row<32;++row) { packed[tile*576+513+row*2]^=0x80; }
                            } }
                        };
                        std::vector<float> neg=ref; for(auto &v:neg) { v=-v; }
                        flip(); jobs.back()->run(); jobs.back()->validate(neg);
                        flip(); jobs.back()->run(); jobs.back()->validate(ref);
                        result["shared_storage"]["post_bind_alias_compute_verified"]=true;
                    }
                    rows.push_back({{"buffer",layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2?"SHARED_Q4_0_CPU4X8_V2":"SHARED_Q4_0_32X32_V1"},
                                    {"nrmse",jobs.back()->validate(ref)},
                                    {"weight_buffer_bytes",0},{"shared_owner_bytes",desc.storage.size}});
                    samples.emplace_back();
                    device_samples.emplace_back();
                }
                // Alternate direction each round to reduce order/temperature bias.
                for(int r=0;r<reps;++r) { for(size_t j=0;j<jobs.size();++j) {
                    size_t i=r%2?jobs.size()-1-j:j; samples[i].push_back(jobs[i]->run());
                    if(jobs[i]->device_us>=0) { device_samples[i].push_back(jobs[i]->device_us); }
                } }
                for(size_t i=0;i<jobs.size();++i) {
                    rows[i]["samples_us"]=samples[i]; rows[i]["median_us"]=median(samples[i]);
                    if(!device_samples[i].empty()) {
                        rows[i]["device_samples_us"]=device_samples[i]; rows[i]["device_median_us"]=median(device_samples[i]);
                        rows[i]["device_timing"]=jobs[i]->cl_timing?"sum kernel event END-START; includes activation preparation":"DSP op usecs; includes DMA/decode";
                    }
                    fprintf(stderr,"%s %s K=%d M=%d N=%d median=%.3f us\n",argv[1],rows[i]["buffer"].get<std::string>().c_str(),k,m,n,median(samples[i]));
                }
                result["cases"].push_back({{"k",k},{"m",m},{"n",n},{"source_tensor",source_name},{"variants",rows}});
                std::ofstream file(argv[2]); check(bool(file),"cannot write result"); file<<result.dump(2)<<'\n';
            }
        }
        return 0;
    } catch(const std::exception &e) { fprintf(stderr,"shared-weight-probe: %s\n",e.what()); return 1; }
}
