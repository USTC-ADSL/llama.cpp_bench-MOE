#pragma once

#include "expert-probe-common.h"
#include "ggml-shared-dma.h"

struct shared_activation_buffer {
    // Imported views are destroyed before the backing and its host mapping.
    ggml_backend_buffer_ptr backing;
    ggml_backend_buffer_ptr imported;
    ggml_backend_shared_dma_buffer_desc desc{};
    using export_fn = bool (*)(ggml_backend_buffer_t, ggml_backend_shared_dma_buffer_desc *);
    using import_fn = ggml_backend_buffer_t (*)(ggml_backend_t, const ggml_backend_shared_dma_buffer_desc *);

    static import_fn gpu_import(device & gpu) {
        return reinterpret_cast<import_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(gpu.dev), "ggml_backend_opencl_import_shared_dma_compute_buffer"));
    }
    shared_activation_buffer(device & dev, size_t bytes) {
        auto htp=ggml_backend_dev_by_name("HTP0"); require(htp!=nullptr,"shared activation requires HTP0");
        auto export_dma=reinterpret_cast<export_fn>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(htp),"ggml_backend_hexagon_buffer_export_shared_dma"));
        require(export_dma!=nullptr,"HTP shared export API missing");
        backing.reset(ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(htp),bytes));
        require(backing && export_dma(backing.get(),&desc),"shared backing allocation/export failed");
        ggml_backend_buffer_set_usage(backing.get(),GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        if (dev.name=="GPUOpenCL") {
            auto import=gpu_import(dev); require(import!=nullptr,"OpenCL DMA import API missing");
            imported.reset(import(dev.backend.get(),&desc));
            require(imported!=nullptr,"OpenCL IO-coherent DMA import unavailable");
        } else if (dev.name=="CPU") {
            imported.reset(ggml_backend_dev_buffer_from_host_ptr(dev.dev,desc.host_ptr,desc.size,desc.size));
            require(imported!=nullptr,"CPU shared activation view failed");
        } else { require(dev.name=="HTP0","unsupported shared activation backend"); }
        ggml_backend_buffer_set_usage(get(),GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    ggml_backend_buffer_t get() const { return imported?imported.get():backing.get(); }
    static size_t align(size_t bytes,size_t alignment) { return (bytes+alignment-1)/alignment*alignment; }
    static size_t required_size(device & dev,ggml_context * ctx) {
        size_t alignment=std::max<size_t>(128,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev.dev)));
        size_t bytes=0;
        for (auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            if (!t->view_src) { bytes=align(bytes,alignment)+ggml_nbytes(t); }
        }
        return align(bytes,alignment);
    }
    void bind(device & dev,ggml_context * ctx) {
        size_t alignment=std::max<size_t>(128,ggml_backend_buft_get_alignment(ggml_backend_dev_buffer_type(dev.dev)));
        require(required_size(dev,ctx)<=desc.size,"shared activation arena capacity exceeded");
        size_t offset=0;
        for (auto * t=ggml_get_first_tensor(ctx);t;t=ggml_get_next_tensor(ctx,t)) {
            if (t->view_src) { require(ggml_backend_view_init(t)==GGML_STATUS_SUCCESS,"shared view init failed"); }
            else {
                offset=align(offset,alignment);
                require(ggml_backend_tensor_alloc(get(),t,static_cast<uint8_t *>(desc.host_ptr)+offset)==GGML_STATUS_SUCCESS,
                        "shared tensor allocation failed");
                offset+=ggml_nbytes(t);
            }
        }
    }
    static std::unique_ptr<shared_activation_buffer> allocate(device & dev,ggml_context * ctx) {
        auto buffer=std::make_unique<shared_activation_buffer>(dev,required_size(dev,ctx));
        buffer->bind(dev,ctx);
        return buffer;
    }
};
