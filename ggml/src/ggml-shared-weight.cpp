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
    if (!allocator || !canonical || !k || !m || k % 32 || m % 32 ||
        k > INT32_MAX || m > INT32_MAX || size_t(k/32) > SIZE_MAX / 18 / m || size != size_t(k/32)*m*18) {
        return nullptr;
    }
    using export_fn = bool (*)(ggml_backend_buffer_t, ggml_backend_shared_dma_buffer_desc *);
    auto export_dma = reinterpret_cast<export_fn>(proc(allocator,"ggml_backend_hexagon_buffer_export_shared_dma"));
    if (!export_dma) { return nullptr; }
    auto w = std::unique_ptr<ggml_shared_weight>(new ggml_shared_weight);
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
    auto import = reinterpret_cast<ggml_backend_import_shared_weight_t>(proc(backend,"ggml_backend_import_shared_weight"));
    if (!import) { return nullptr; }
    auto b=make_binding(backend,w->desc.k,w->desc.m);
    if (!b) { return nullptr; }
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
