#pragma once

#include "ggml-backend.h"
#include "llama-mmap.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// Model-owned policy and storage. Backends retain ownership of layout conversion.
class llama_weight_manager {
public:
    explicit llama_weight_manager(const char * config_path);
    ~llama_weight_manager();
    void validate_devices(const std::vector<ggml_backend_dev_t> & devices, int n_layers) const;

    ggml_backend_dev_t layer_device(int layer, ggml_backend_dev_t fallback) const;
    ggml_backend_dev_t weight_device(const std::string & name, ggml_backend_dev_t fallback) const;
    bool is_swap(const std::string & name) const;
    void register_weight(ggml_tensor * tensor, uint32_t file, size_t offset, ggml_backend_dev_t device);
    void adopt_files(llama_files files);

    void check_fixed(ggml_backend_buffer_type_t buft, size_t bytes) const;
    void add_fixed(ggml_backend_buffer_t buffer);
    void set_runtime(const std::map<ggml_backend_buffer_type_t, size_t> & bytes);
    void attach_context(const void * context);
    void detach_context(const void * context);
    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    ggml_backend_sched_weight_provider provider();
    void write_report() const;

    // Explicit controls for future policies. Synchronous; evict refuses pinned data.
    void prefetch(ggml_backend_t backend, const ggml_tensor * weight, const std::vector<int32_t> & experts = {});
    void evict(const ggml_tensor * weight);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
