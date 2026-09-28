#pragma once
#include "ggml-shared-dma.h"

// Experimental one-layer lookahead. All OpenCL calls are serialized: background
// packing starts after shared compute; the following GPU branch joins it first.
// NPU consumes a slot only if its future was already ready on layer entry.
struct online_expert_prefetch {
    using export_fn=bool (*)(ggml_backend_buffer_t,ggml_backend_shared_dma_buffer_desc *);
    using import_fn=ggml_backend_buffer_t (*)(ggml_backend_t,const ggml_backend_shared_dma_buffer_desc *);
    using pack_fn=bool (*)(ggml_backend_t,ggml_backend_buffer_t,size_t,ggml_backend_buffer_t,size_t,uint32_t,uint32_t,uint32_t,uint64_t);
    struct slot {
        std::unique_ptr<bank> weights;
        ggml_backend_buffer_ptr imported; // destroyed before backing weights
        ggml_backend_shared_dma_buffer_desc desc{};
        int layer=-1;
        std::vector<int> ids;
    };
    device &gpu;
    std::array<bank *,24> sources;
    std::array<slot,2> slots;
    export_fn export_dma;
    import_fn import_dma;
    pack_fn pack;
    json predictor, records=json::array();
    int capacity, pending_layer=-1;
    size_t target_bytes=0, source_import_peak=0;
    decltype(timer::now()) epoch;
    std::future<json> pending; // joins before any slot/source/backend can die

    online_expert_prefetch(device &npu,device &gpu_,source &src,const std::array<bank *,24> &banks,int nt)
        :gpu(gpu_),sources(banks) {
        std::ifstream f(std::getenv("MOE_PREFETCH_MODEL")); require(bool(f),"prefetch predictor open failed"); f>>predictor;
        require(predictor.at("version")==1 && predictor.at("tokens")==nt,"prefetch predictor length/version mismatch");
        require(predictor.at("method")=="prior","prototype supports the training-selected layer prior only");
        require(predictor.at("layers").size()==22,"prefetch layer coverage mismatch");
        capacity=std::getenv("MOE_PREFETCH_EXPERTS")?std::stoi(std::getenv("MOE_PREFETCH_EXPERTS")):4;
        require(capacity>0 && capacity<=12,"prefetch capacity exceeds bounded two-slot design");
        for(int il=0;il<22;++il) {
            const auto &p=predictor.at("layers").at(il);
            require(p.at("source_layer")==il && p.at("target_layer")==il+1 && p.at("bias").size()==60,
                    "prefetch predictor layer contract");
            for(double v:p.at("bias")) { require(std::isfinite(v),"nonfinite prefetch score"); }
        }
        export_dma=reinterpret_cast<export_fn>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(npu.dev),"ggml_backend_hexagon_buffer_export_shared_dma"));
        auto reg=ggml_backend_dev_backend_reg(gpu.dev);
        import_dma=reinterpret_cast<import_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_import_shared_dma_compute_buffer"));
        pack=reinterpret_cast<pack_fn>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_opencl_pack_q4_0_htp"));
        require(export_dma && import_dma && pack,"prefetch shared-DMA APIs unavailable");
        std::vector<int> ids(capacity); std::iota(ids.begin(),ids.end(),0);
        for(auto &s:slots) {
            s.weights=std::make_unique<bank>(npu,src,0,&ids,false,true);
            require(export_dma(s.weights->buf.get(),&s.desc),"prefetch target DMA export failed");
            s.imported.reset(import_dma(gpu.backend.get(),&s.desc)); require(bool(s.imported),"prefetch target DMA import failed");
            target_bytes+=ggml_backend_buffer_get_size(s.weights->buf.get());
        }
    }
    void begin_request(decltype(timer::now()) start) { epoch=start; }
    double time_us() const { return elapsed(epoch); }
    bool ready(int layer) {
        if(!pending.valid() || pending_layer!=layer || pending.wait_for(std::chrono::microseconds(0))!=std::future_status::ready) { return false; }
        records.push_back(pending.get()); pending_layer=-1;
        require(slots[layer%2].layer==layer,"prefetch slot layer mismatch");
        return true;
    }
    double join_gpu() {
        const auto start=timer::now();
        if(pending.valid()) { records.push_back(pending.get()); pending_layer=-1; }
        return elapsed(start);
    }
    void launch_next(int current) {
        if(current>=22) { return; } // terminal one-token FFN uses CPU
        require(!pending.valid(),"OpenCL preparation caller overlap");
        auto scores=predictor.at("layers").at(current).at("bias").get<std::vector<double>>();
        std::vector<int> ids(60); std::iota(ids.begin(),ids.end(),0);
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b) { return scores[a]>scores[b]; });
        ids.resize(capacity); std::sort(ids.begin(),ids.end());
        const int target=current+1; auto &s=slots[target%2]; s.ids=ids; s.layer=target;
        source_import_peak=std::max(source_import_peak,ggml_backend_buffer_get_size(sources[target]->buf.get()));
        pending_layer=target;
        pending=std::async(std::launch::async,[this,target,ids] {
            const double start=time_us(); auto &dst=slots[target%2]; auto &src=*sources[target];
            ggml_backend_shared_dma_buffer_desc sd{};
            require(export_dma(src.buf.get(),&sd),"prefetch source DMA export failed");
            ggml_backend_buffer_ptr imported(import_dma(gpu.backend.get(),&sd));
            require(bool(imported),"prefetch source DMA import failed");
            uint64_t mask=0; for(int e:ids) { mask|=uint64_t(1)<<e; }
            const double conversion_start=time_us();
            for(int i=0;i<3;++i) {
                auto *w=src.w[i]; auto *t=dst.weights->w[i];
                require(w->ne[0]==t->ne[0] && w->ne[1]==t->ne[1] && w->ne[2]==60,"prefetch tensor shape mismatch");
                size_t so=static_cast<uint8_t *>(w->data)-static_cast<uint8_t *>(sd.host_ptr);
                size_t to=static_cast<uint8_t *>(t->data)-static_cast<uint8_t *>(dst.desc.host_ptr);
                require(pack(gpu.backend.get(),imported.get(),so,dst.imported.get(),to,w->ne[0],w->ne[1],60,mask),"GPU prefetch pack rejected");
            }
            // pack waits for every GPU event; publish readiness only after the
            // imported source view is released and all target writes are visible.
            imported.reset(); const double end=time_us();
            return json{{"source_layer",target-1},{"target_layer",target},{"ids",ids},
                {"start_us",start},{"conversion_start_us",conversion_start},{"end_us",end},
                {"total_us",end-start},{"source_import_alias_bytes",sd.size},
                {"target_bytes",ggml_backend_buffer_get_size(dst.weights->buf.get())}};
        });
    }
};
