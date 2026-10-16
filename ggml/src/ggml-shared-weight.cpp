#include "ggml-shared-weight.h"
#include "ggml-shared-weight-impl.h"
#include "ggml-alloc.h"
#include "ggml-cpp.h"

#include <memory>
#include <vector>

struct shared_weight_binding {
    ggml_backend_t backend;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * tensor = nullptr;
};

struct ggml_shared_weight {
    ggml_shared_weight_desc desc{};
    ggml_backend_buffer_ptr raw_storage;
    ggml_backend_t allocator = nullptr;
    std::vector<std::unique_ptr<shared_weight_binding>> bindings;
};

static void * proc(ggml_backend_t backend, const char * name) {
    return ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), name);
}

static std::unique_ptr<shared_weight_binding> make_binding(ggml_backend_t backend, uint32_t k, uint32_t m) {
    auto b = std::unique_ptr<shared_weight_binding>(new shared_weight_binding);
    b->backend = backend;
    b->ctx.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!b->ctx) { return nullptr; }
    b->tensor = ggml_new_tensor_2d(b->ctx.get(), GGML_TYPE_Q4_0, k, m);
    if (!b->tensor) { return nullptr; }
    ggml_set_name(b->tensor, "shared_weight_q4_0_32x32_v1");
    return b;
}

ggml_shared_weight_t ggml_shared_weight_create(ggml_backend_t allocator, uint32_t k, uint32_t m,
                                             const void * canonical, size_t size) {
    return ggml_shared_weight_create_with_layout(allocator,k,m,canonical,size,GGML_SHARED_WEIGHT_Q4_0_32X32_V1);
}

ggml_shared_weight_t ggml_shared_weight_create_with_layout(ggml_backend_t allocator, uint32_t k, uint32_t m,
                                             const void * canonical, size_t size, ggml_shared_weight_layout layout) {
    if (!allocator || !canonical || !k || !m || k % 32 || m % 32 ||
        k > INT32_MAX || m > INT32_MAX || size_t(k/32) > SIZE_MAX / 18 / m || size != size_t(k/32)*m*18) {
        return nullptr;
    }
    using export_fn = bool (*)(ggml_backend_buffer_t, ggml_backend_shared_dma_buffer_desc *);
    auto export_dma = reinterpret_cast<export_fn>(proc(allocator,"ggml_backend_hexagon_buffer_export_shared_dma"));
    if (!export_dma) { return nullptr; }
    auto w = std::unique_ptr<ggml_shared_weight>(new ggml_shared_weight);
    if(layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2) {
        // Bind the allocator only through its explicit CPU4x8 marker.
        w->allocator=allocator;
        w->raw_storage.reset(ggml_backend_alloc_buffer(allocator,size));
        if(!w->raw_storage || !export_dma(w->raw_storage.get(),&w->desc.storage)) { return nullptr; }
        w->desc.layout=layout; w->desc.k=k; w->desc.m=m;
        if(!ggml_shared_weight_desc_valid(&w->desc)) { return nullptr; }
        auto * dst=static_cast<uint8_t *>(w->desc.storage.host_ptr);
        const auto * src=static_cast<const uint8_t *>(canonical);
        for(size_t group=0;group<m/4;++group) { for(size_t b=0;b<k/32;++b) {
            auto * out=dst+(group*(k/32)+b)*72;
            for(size_t r=0;r<4;++r) {
                const auto * in=src+((group*4+r)*(k/32)+b)*18;
                std::memcpy(out+r*2,in,2);
                for(size_t j=0;j<16;++j) { out[8+(j/8)*32+r*8+j%8]=in[2+j]^0x88; }
            }
        } }
        return w.release();
    }
    if(layout!=GGML_SHARED_WEIGHT_Q4_0_32X32_V1) { return nullptr; }
    auto b = make_binding(allocator,k,m);
    if (!b) { return nullptr; }
    b->buffer.reset(ggml_backend_alloc_ctx_tensors(b->ctx.get(),allocator));
    if (!b->buffer || !export_dma(b->buffer.get(),&w->desc.storage)) { return nullptr; }
    const uintptr_t base = reinterpret_cast<uintptr_t>(w->desc.storage.host_ptr);
    const uintptr_t data = reinterpret_cast<uintptr_t>(b->tensor->data);
    if (!base || data < base) { return nullptr; }
    w->desc.offset = data-base;
    w->desc.layout = GGML_SHARED_WEIGHT_Q4_0_32X32_V1;
    w->desc.k=k; w->desc.m=m;
    if (!ggml_shared_weight_desc_valid(&w->desc)) { return nullptr; }
    ggml_backend_buffer_set_usage(b->buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_set(b->tensor,canonical,0,size);
    ggml_backend_synchronize(allocator);
    w->bindings.push_back(std::move(b));
    return w.release();
}

ggml_tensor * ggml_shared_weight_bind(ggml_shared_weight_t w, ggml_backend_t backend) {
    if (!w || !backend) { return nullptr; }
    for (const auto & b : w->bindings) { if (b->backend==backend) { return b->tensor; } }
    if(w->raw_storage && backend==w->allocator) {
        using marker_fn=bool (*)(ggml_tensor *);
        auto mark=reinterpret_cast<marker_fn>(proc(backend,"ggml_backend_hexagon_tensor_use_cpu4x8_q4_0"));
        if(!mark || w->desc.k%128) { return nullptr; }
        auto b=make_binding(backend,w->desc.k,w->desc.m);
        if(!b || ggml_backend_tensor_alloc(w->raw_storage.get(),b->tensor,w->desc.storage.host_ptr)!=GGML_STATUS_SUCCESS ||
           !mark(b->tensor)) { return nullptr; }
        ggml_set_name(b->tensor,"shared_weight_q4_0_cpu4x8_v2");
        ggml_backend_buffer_set_usage(w->raw_storage.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        auto * result=b->tensor;
        w->bindings.push_back(std::move(b));
        return result;
    }
    auto import = reinterpret_cast<ggml_backend_import_shared_weight_t>(proc(backend,"ggml_backend_import_shared_weight"));
    if (!import) { return nullptr; }
    auto b=make_binding(backend,w->desc.k,w->desc.m);
    if (!b) { return nullptr; }
    if(w->desc.layout==GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2) { ggml_set_name(b->tensor,"shared_weight_q4_0_cpu4x8_v2"); }
    b->buffer.reset(import(backend,&w->desc));
    if (!b->buffer) { return nullptr; }
    if (ggml_backend_tensor_alloc(b->buffer.get(),b->tensor,
                static_cast<char *>(ggml_backend_buffer_get_base(b->buffer.get()))+w->desc.offset)!=GGML_STATUS_SUCCESS) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(b->buffer.get(),GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * result=b->tensor;
    w->bindings.push_back(std::move(b));
    return result;
}

bool ggml_shared_weight_describe(ggml_shared_weight_t w, ggml_shared_weight_desc * desc) {
    if (!w || !desc) { return false; }
    *desc=w->desc;
    return true;
}

void ggml_shared_weight_free(ggml_shared_weight_t w) {
    if (!w) { return; }
    for (auto & b : w->bindings) { ggml_backend_synchronize(b->backend); }
    while (!w->bindings.empty()) { w->bindings.pop_back(); }
    delete w;
}
