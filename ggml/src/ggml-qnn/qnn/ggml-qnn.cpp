#include "backend-ops.hpp"
#include "common.hpp"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include "logger.hpp"
#include "qnn-lib-path.hpp"
#include "tensor.hpp"
#include "utils.hpp"

#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

namespace {

const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev);

qnn::ggml_backend_qnn_device_context * get_device_context(ggml_backend_dev_t dev) {
    return reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(dev->context);
}

qnn::qnn_buffer_interface * get_buffer_context(ggml_backend_buffer_t buffer) {
    return reinterpret_cast<qnn::qnn_buffer_interface *>(buffer->context);
}

bool is_qnn_device(ggml_backend_dev_t dev) {
    return dev != nullptr && dev->iface.get_name == ggml_backend_qnn_device_get_name;
}

bool qnn_lib_dir_has_required_host_libs(const std::string & path, const char * backend_lib_name) {
    if (path.empty() || backend_lib_name == nullptr || backend_lib_name[0] == '\0') {
        return false;
    }

#ifdef _WIN32
    constexpr const char * system_lib_name = "QnnSystem.dll";
#else
    constexpr const char * system_lib_name = "libQnnSystem.so";
#endif

    const std::filesystem::path dir(path);
    std::error_code             system_ec;
    std::error_code             backend_ec;
    return std::filesystem::exists(dir / system_lib_name, system_ec) &&
           std::filesystem::exists(dir / backend_lib_name, backend_ec);
}

bool qnn_aot_env_enabled() {
    const char * value = std::getenv("GGML_QNN_AOT_CONFIG");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

bool ggml_backend_qnn_try_initialize_aot_runtime(qnn::ggml_backend_qnn_device_context * dev_ctx) {
    if (dev_ctx == nullptr || dev_ctx->device != QNN_BACKEND_NPU || !dev_ctx->instance) {
        return false;
    }

    if (!qnn_aot_env_enabled()) {
        dev_ctx->aot_mode = false;
        dev_ctx->aot_runtime.reset();
        return false;
    }

    const std::string aot_config_path = std::getenv("GGML_QNN_AOT_CONFIG");
    std::string       aot_model_dir;
    if (const char * env_aot_model_dir = std::getenv("GGML_QNN_AOT_MODEL_DIR");
        env_aot_model_dir && env_aot_model_dir[0] != '\0') {
        aot_model_dir = env_aot_model_dir;
    } else {
        aot_model_dir = std::filesystem::path(aot_config_path).parent_path().string();
    }

    const bool same_runtime =
        dev_ctx->aot_runtime != nullptr &&
        dev_ctx->aot_runtime->is_enabled() &&
        dev_ctx->aot_config_path == aot_config_path &&
        dev_ctx->aot_model_dir == aot_model_dir;
    if (same_runtime) {
        dev_ctx->aot_mode = true;
        return true;
    }

    const bool already_attempted_same_config =
        dev_ctx->aot_init_attempted &&
        dev_ctx->aot_runtime == nullptr &&
        dev_ctx->aot_attempted_config_path == aot_config_path &&
        dev_ctx->aot_attempted_model_dir == aot_model_dir;
    if (already_attempted_same_config) {
        return false;
    }

    dev_ctx->aot_init_attempted        = true;
    dev_ctx->aot_attempted_config_path = aot_config_path;
    dev_ctx->aot_attempted_model_dir   = aot_model_dir;
    dev_ctx->aot_mode                  = true;
    dev_ctx->aot_config_path           = aot_config_path;
    dev_ctx->aot_model_dir             = aot_model_dir;
    dev_ctx->aot_runtime.reset();

    const bool config_exists = std::filesystem::exists(dev_ctx->aot_config_path);
    const bool model_dir_exists = !dev_ctx->aot_model_dir.empty() && std::filesystem::exists(dev_ctx->aot_model_dir);
    if (!config_exists || !model_dir_exists) {
        QNN_LOG_WARN("[aot] config/model dir unavailable for %s: config=%s exists=%d model_dir=%s exists=%d\n",
                     qnn::get_backend_name(dev_ctx->device),
                     dev_ctx->aot_config_path.c_str(),
                     (int) config_exists,
                     dev_ctx->aot_model_dir.c_str(),
                     (int) model_dir_exists);
        dev_ctx->aot_mode = false;
        return false;
    }

    auto aot_runtime = std::make_unique<qnn::qnn_aot_runtime>(dev_ctx->instance, dev_ctx->device);
    if (!aot_runtime->initialize(dev_ctx->aot_config_path, dev_ctx->aot_model_dir)) {
        QNN_LOG_WARN("[aot] failed to initialize AoT runtime from %s (model_dir=%s)\n",
                     dev_ctx->aot_config_path.c_str(),
                     dev_ctx->aot_model_dir.c_str());
        dev_ctx->aot_mode = false;
        return false;
    }

    dev_ctx->aot_runtime = std::move(aot_runtime);
    QNN_LOG_INFO("[aot] enabled AoT runtime with config %s (model_dir=%s)\n",
                 dev_ctx->aot_config_path.c_str(),
                 dev_ctx->aot_model_dir.c_str());
    return true;
}

struct ggml_backend_qnn_buffer_type_context {
    std::string name;
    bool        is_host = false;
};

ggml_backend_qnn_buffer_type_context * get_buffer_type_context(ggml_backend_buffer_type_t buft) {
    return reinterpret_cast<ggml_backend_qnn_buffer_type_context *>(buft->context);
}

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend buffer object
 * -----------------------------------------------------------------------------------------------
 */
void ggml_backend_qnn_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = get_buffer_context(buffer);
    delete ctx;
}

void * ggml_backend_qnn_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = get_buffer_context(buffer);
    return ctx->get_buffer();
}

ggml_status ggml_backend_qnn_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(tensor);
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_qnn_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                        ggml_tensor *         tensor,
                                        const void *          data,
                                        size_t                offset,
                                        size_t                size) {
    GGML_UNUSED(buffer);
    memcpy((char *) tensor->data + offset, data, size);
}

void ggml_backend_qnn_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                        const ggml_tensor *   tensor,
                                        void *                data,
                                        size_t                offset,
                                        size_t                size) {
    GGML_UNUSED(buffer);
    memcpy(data, (const char *) tensor->data + offset, size);
}

bool ggml_backend_qnn_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_UNUSED(buffer);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }

    return false;
}

void ggml_backend_qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = get_buffer_context(buffer);
    memset(ctx->get_buffer(), value, ctx->get_size());
}

constexpr const ggml_backend_buffer_i ggml_backend_qnn_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_qnn_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_qnn_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_qnn_buffer_init_tensor,
    /* .memset_tensor   = */ nullptr,
    /* .set_tensor      = */ ggml_backend_qnn_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_qnn_buffer_get_tensor,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ ggml_backend_qnn_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_qnn_buffer_clear,
    /* .reset           = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend object
 * -----------------------------------------------------------------------------------------------
 */
const char * ggml_backend_qnn_buffer_type_name(ggml_backend_buffer_type_t buft) {
    auto * type_ctx = get_buffer_type_context(buft);
    if (type_ctx != nullptr && !type_ctx->name.empty()) {
        return type_ctx->name.c_str();
    }

    auto * dev_ctx = get_device_context(buft->device);
    return qnn::get_backend_name(dev_ctx->device);
}

ggml_backend_buffer_t ggml_backend_qnn_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto * dev_ctx  = get_device_context(buft->device);
    auto * type_ctx = get_buffer_type_context(buft);

    size = std::max<size_t>(size, 1);

    qnn::qnn_buffer_interface * ctx = nullptr;
    if (type_ctx != nullptr && type_ctx->is_host && dev_ctx->device == QNN_BACKEND_NPU) {
        auto host_pool = std::make_unique<qnn::qnn_htp_buffer_pool>(
            dev_ctx->instance,
            size);
        if (!host_pool->is_valid()) {
            return nullptr;
        }
        ctx = host_pool.release();
    } else {
        ctx = new qnn::qnn_mem_buffer(size);
    }

    if (!ctx->is_valid()) {
        delete ctx;
        return nullptr;
    }

    QNN_LOG_DEBUG("[%s]alloc buffer: %p, size: %ld\n", ggml_backend_qnn_buffer_type_name(buft),
                  (void *) ctx->get_buffer(), (long) size);
    return ggml_backend_buffer_init(buft, ggml_backend_qnn_buffer_interface, ctx, size);
}

size_t ggml_backend_qnn_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    auto * type_ctx = get_buffer_type_context(buft);
    return type_ctx != nullptr && type_ctx->is_host ? 64 : 32;
}

size_t ggml_backend_qnn_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    // TODO: get the max size from device
    return 1024L * 1024 * 1024;
}

bool ggml_backend_qnn_buffer_is_host(ggml_backend_buffer_type_t buft) {
    auto * type_ctx = get_buffer_type_context(buft);
    return type_ctx != nullptr && type_ctx->is_host;
}

const char * ggml_backend_qnn_name(ggml_backend_t backend) {
    auto * device_ctx = get_device_context(backend->device);
    return device_ctx->name.c_str();
}

void ggml_backend_qnn_free(ggml_backend_t backend) {
    auto * device_ctx = get_device_context(backend->device);
    QNN_LOG_INFO("idx %d, name:%s\n", device_ctx->device, device_ctx->name.c_str());

    if (device_ctx->aot_runtime != nullptr) {
        device_ctx->aot_runtime->reset_state();
    }
    device_ctx->aot_mode = false;
    device_ctx->qnn_graph_cache.clear();

    delete backend;
}

bool ggml_backend_qnn_aot_has_pending_generic_kv_writeback(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }

    auto * device_ctx = get_device_context(backend->device);
    return device_ctx != nullptr &&
           device_ctx->aot_runtime != nullptr &&
           device_ctx->aot_runtime->has_pending_generic_kv_writeback();
}

bool ggml_backend_qnn_aot_flush_pending_generic_kv_writeback(ggml_backend_t backend) {
    if (backend == nullptr) {
        return false;
    }

    auto * device_ctx = get_device_context(backend->device);
    if (device_ctx == nullptr || device_ctx->aot_runtime == nullptr) {
        return false;
    }

    return device_ctx->aot_runtime->flush_pending_generic_kv_writeback();
}

ggml_guid_t ggml_backend_qnn_guid() {
    static ggml_guid guid = { 0x1a, 0x2b, 0x3c, 0x4d, 0x5e, 0x6f, 0x70, 0x81,
                              0x92, 0xa3, 0xb4, 0xc5, 0xd6, 0xe7, 0xf8, 0x09 };
    return &guid;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return ggml_guid_matches(backend->guid, ggml_backend_qnn_guid());
}

bool ggml_backend_qnn_cpy_tensor_async(ggml_backend_t      backend_src,
                                       ggml_backend_t      backend_dst,
                                       const ggml_tensor * src,
                                       ggml_tensor *       dst) {
    GGML_UNUSED(backend_src);
    GGML_UNUSED(backend_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);

    QNN_LOG_DEBUG("opy form %s to %s, src_is_qnn: %d, dst_is_qnn: %d\n", ggml_get_name(src), ggml_get_name(dst),
                  (int) ggml_backend_is_qnn(backend_src), (int) ggml_backend_is_qnn(backend_dst));
    return false;
}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type ggml_backend_qnn_buffer_types[QNN_BACKEND_COUNT];
    static ggml_backend_qnn_buffer_type_context ggml_backend_qnn_buffer_type_ctx[QNN_BACKEND_COUNT];
    auto *                          dev_ctx = get_device_context(dev);
    if (!ggml_backend_qnn_buffer_types[dev_ctx->device].device) {
        ggml_backend_qnn_buffer_type_ctx[dev_ctx->device] = {
            /* .name    = */ qnn::get_backend_name(dev_ctx->device),
            /* .is_host = */ false,
        };
        ggml_backend_qnn_buffer_types[dev_ctx->device] = {
            /* .iface   = */ {
                              /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                              /* .alloc_buffer     = */
                ggml_backend_qnn_buffer_type_alloc_buffer,  /* .get_alignment    = */
                ggml_backend_qnn_buffer_type_get_alignment, /* .get_max_size     = */
                ggml_backend_qnn_buffer_type_get_max_size, /* .get_alloc_size   = */ nullptr,          // defaults to ggml_nbytes
                              /* .is_host          = */ ggml_backend_qnn_buffer_is_host,
                              },
            /* .device */
            dev,
            /* .context = */ &ggml_backend_qnn_buffer_type_ctx[dev_ctx->device],
        };
    } else {
        GGML_ASSERT(ggml_backend_qnn_buffer_types[dev_ctx->device].device == dev);
    }

    return &ggml_backend_qnn_buffer_types[dev_ctx->device];
}

ggml_backend_buffer_type_t ggml_backend_qnn_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    static ggml_backend_buffer_type ggml_backend_qnn_host_buffer_types[QNN_BACKEND_COUNT];
    static ggml_backend_qnn_buffer_type_context ggml_backend_qnn_host_buffer_type_ctx[QNN_BACKEND_COUNT];

    auto * dev_ctx = get_device_context(dev);
    if (dev_ctx->device != QNN_BACKEND_NPU) {
        return nullptr;
    }

    if (!ggml_backend_qnn_host_buffer_types[dev_ctx->device].device) {
        ggml_backend_qnn_host_buffer_type_ctx[dev_ctx->device] = {
            /* .name    = */ std::string(qnn::get_backend_name(dev_ctx->device)) + "-host",
            /* .is_host = */ true,
        };
        ggml_backend_qnn_host_buffer_types[dev_ctx->device] = {
            /* .iface   = */ {
                              /* .get_name         = */ ggml_backend_qnn_buffer_type_name,
                              /* .alloc_buffer     = */ ggml_backend_qnn_buffer_type_alloc_buffer,
                              /* .get_alignment    = */ ggml_backend_qnn_buffer_type_get_alignment,
                              /* .get_max_size     = */ ggml_backend_qnn_buffer_type_get_max_size,
                              /* .get_alloc_size   = */ nullptr,
                              /* .is_host          = */ ggml_backend_qnn_buffer_is_host,
                              },
            /* .device  = */ dev,
            /* .context = */ &ggml_backend_qnn_host_buffer_type_ctx[dev_ctx->device],
        };
    } else {
        GGML_ASSERT(ggml_backend_qnn_host_buffer_types[dev_ctx->device].device == dev);
    }

    return &ggml_backend_qnn_host_buffer_types[dev_ctx->device];
}

ggml_status ggml_backend_qnn_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    return qnn::device_compute_graph(get_device_context(backend->device), cgraph) ? GGML_STATUS_SUCCESS :
                                                                                    GGML_STATUS_FAILED;
}

constexpr const ggml_backend_i ggml_backend_qnn_interface = {
    /* .get_name                = */ ggml_backend_qnn_name,
    /* .free                    = */ ggml_backend_qnn_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ ggml_backend_qnn_cpy_tensor_async,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_qnn_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

/*
 * -----------------------------------------------------------------------------------------------
 * qnn backend device object
 * -----------------------------------------------------------------------------------------------
 */
const char * ggml_backend_qnn_device_get_name(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    return qnn::get_backend_name(dev_ctx->device);
}

const char * ggml_backend_qnn_device_get_description(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    return dev_ctx->description.empty() ? qnn::get_backend_desc(dev_ctx->device) : dev_ctx->description.c_str();
}

void ggml_backend_qnn_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    GGML_UNUSED(dev);
    *free  = common::get_system_free_memory_in_bytes();
    *total = common::get_system_total_memory_in_bytes();
    QNN_LOG_DEBUG("free memory: %ldMB, total memory: %ldMB\n", (*free / 1048576), (*total) / 1048576);
}

enum ggml_backend_dev_type ggml_backend_qnn_device_get_type(ggml_backend_dev_t dev) {
    return qnn::get_device_caps(get_device_context(dev)->device).type;
}

void ggml_backend_qnn_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    auto * dev_ctx = get_device_context(dev);
    props->name        = ggml_backend_qnn_device_get_name(dev);
    props->description = ggml_backend_qnn_device_get_description(dev);
    props->type        = ggml_backend_qnn_device_get_type(dev);
    ggml_backend_qnn_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* async                */ false,
        /* host_buffer          */ dev_ctx->device == QNN_BACKEND_NPU,
        /* buffer_from_host_ptr */ false,
        /* events               */ false,
    };
}

ggml_backend_t ggml_backend_qnn_init_with_device_context(ggml_backend_dev_t dev, const char * extend_lib_search_path) {
    auto *     dev_ctx = get_device_context(dev);
    const auto device  = dev_ctx->device;

#ifdef _WIN32
    constexpr char path_list_separator = ';';
    const char *   loader_path         = std::getenv("PATH");
#else
    constexpr char path_list_separator = ':';
    const char *   loader_path         = std::getenv("LD_LIBRARY_PATH");
#endif

    const bool  missing_search_path = extend_lib_search_path == nullptr || extend_lib_search_path[0] == '\0';
    const auto  cwd                 = qnn::qnn_current_working_directory();
    std::string resolved_lib_search_path = qnn::resolve_qnn_lib_search_path(
            extend_lib_search_path,
            std::getenv("GGML_QNN_LIB_SEARCH_PATH"),
            std::getenv("REMOTE_LIB_DIR"),
            std::getenv("REMOTE_BIN_DIR"),
            loader_path,
            cwd.c_str(),
            GGML_QNN_DEFAULT_LIB_SEARCH_PATH,
            path_list_separator,
            [device](const std::string & path) {
                return qnn_lib_dir_has_required_host_libs(path, qnn::get_device_caps(device).lib_name);
            });
    extend_lib_search_path = resolved_lib_search_path.c_str();

    if (missing_search_path) {
        QNN_LOG_WARN("extend_lib_search_path is nullptr, resolved qnn lib search path to %s\n",
                     extend_lib_search_path[0] != '\0' ? extend_lib_search_path : "<empty>");
    }

    QNN_LOG_DEBUG("device %s\n", qnn::get_backend_name(device));
    QNN_LOG_DEBUG("extend_lib_search_path %s\n", extend_lib_search_path);

    if (!dev_ctx->instance) {
        auto instance = std::make_shared<qnn::qnn_instance>(extend_lib_search_path, device);
        if (!instance->qnn_init(nullptr)) {
            QNN_LOG_WARN("failed to init qnn backend %s\n", qnn::get_backend_name(device));
            return nullptr;
        }

        auto qnn_interface = instance->get_qnn_interface();
        if (!qnn_interface) {
            QNN_LOG_WARN("qnn subsystem failure\n");
            return nullptr;
        }

        dev_ctx->instance      = std::move(instance);
        dev_ctx->qnn_interface = std::move(qnn_interface);
        dev_ctx->socinfo       = dev_ctx->instance->get_soc_info();

        if (dev_ctx->cpu_fallback_backend == nullptr) {
            dev_ctx->cpu_fallback_backend = ggml_backend_cpu_init();
        }
    }

    auto instance      = dev_ctx->instance;
    auto qnn_interface = dev_ctx->qnn_interface;
    if (!instance || !qnn_interface) {
        QNN_LOG_WARN("qnn subsystem failure\n");
        return nullptr;
    }

    std::string device_name = qnn::get_backend_name(device);
    QNN_LOG_INFO("qnn device name %s\n", device_name.c_str());
    const auto & device_caps          = qnn::get_device_caps(device);
    if (dev_ctx->cpu_fallback_backend != nullptr) {
        ggml_backend_cpu_set_n_threads(dev_ctx->cpu_fallback_backend, (int) dev_ctx->threads);
    } else {
        QNN_LOG_WARN("[aot] failed to initialize CPU fallback backend\n");
    }

    if (device == QNN_BACKEND_NPU) {
        if (ggml_backend_qnn_try_initialize_aot_runtime(dev_ctx) && dev_ctx->aot_runtime != nullptr) {
            dev_ctx->aot_runtime->reset_state();
        }
    }
    dev_ctx->supported_types          = device_caps.supported_types;
    dev_ctx->cpu_preprocess_types     = device_caps.cpu_preprocess_types;
    dev_ctx->max_tensor_size_in_bytes = device_caps.max_tensor_size_in_bytes;
    {
        char buffer[256];
        snprintf(buffer, sizeof(buffer), "%s(%s)", qnn::get_chipset_desc(dev_ctx->socinfo.soc_model),
                 qnn::get_backend_desc(dev_ctx->device));
        dev_ctx->description = buffer;
    }

#ifdef GGML_HEXAGON_ENABLE_QUANTIZED_TENSORS
    // TODO: remove npu from here if hardware quantization is supported
    dev_ctx->enable_cpu_dequantize = device == QNN_BACKEND_CPU;
#endif

    ggml_backend_t qnn_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_qnn_guid(),
        /* .iface     = */ ggml_backend_qnn_interface,
        /* .device    = */ dev,
        /* .context   = */ nullptr,
    };

    return qnn_backend;
}

ggml_backend_t ggml_backend_qnn_device_init(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_qnn_init_with_device_context(dev, params);
}

ggml_backend_buffer_type_t ggml_backend_qnn_device_get_buffer_type(ggml_backend_dev_t dev) {
    auto * dev_ctx = get_device_context(dev);
    (void) dev_ctx;
    return ggml_backend_qnn_buffer_type(dev);
}

ggml_backend_buffer_t ggml_backend_qnn_device_buffer_from_ptr(ggml_backend_dev_t dev,
                                                              void *             ptr,
                                                              size_t             size,
                                                              size_t             max_tensor_size) {
    // TODO
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

bool ggml_backend_qnn_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    // Note that this function could be called before the device context is initialized
    auto * device_ctx = get_device_context(dev);
    ggml_backend_qnn_try_initialize_aot_runtime(device_ctx);
    return qnn::device_supports_op(device_ctx, op);
}

bool ggml_backend_qnn_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    auto * device_ctx = get_device_context(dev);
    ggml_backend_qnn_try_initialize_aot_runtime(device_ctx);
    if (buft == nullptr) {
        return false;
    }

    ggml_backend_dev_t buft_dev = ggml_backend_buft_get_device(buft);
    if (is_qnn_device(buft_dev)) {
        auto * buft_dev_ctx = get_device_context(buft_dev);
        if (buft_dev_ctx != nullptr && buft_dev_ctx->device == device_ctx->device) {
            return true;
        }
    }

    if (device_ctx->device == QNN_BACKEND_NPU) {
        const char * aot_config = std::getenv("GGML_QNN_AOT_CONFIG");
        if (aot_config != nullptr && aot_config[0] != '\0' && std::strcmp(aot_config, "0") != 0) {
            // Reserve-time scheduler queries can happen before the AoT runtime is fully initialized.
            // In workflow1 the precompiled QNN context owns transformer weights, so the GGUF weight
            // buffer types should not fragment the transformer stage while we are still assigning splits.
            return true;
        }
    }

    if (device_ctx->aot_mode && device_ctx->aot_runtime) {
        // AoT transformer execution consumes the precompiled QNN context and host-visible stage inputs,
        // not the original ggml static weights referenced by the graph. Treat all buffer types as
        // compatible here so the scheduler does not fragment the transformer stage around unused weight buffers.
        return true;
    }

    return ggml_backend_buft_is_host(buft);
}

bool ggml_backend_qnn_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto * device_ctx = get_device_context(dev);
#ifndef NDEBUG
    QNN_LOG_DEBUG("[%s][%s]offload op\n", qnn::get_backend_name(device_ctx->device), ggml_op_name(op->op));
#endif
    if (device_ctx->aot_mode && device_ctx->aot_runtime) {
        return false;
    }
    GGML_UNUSED(op);
    return false;
}


constexpr const ggml_backend_device_i ggml_backend_qnn_device_interface = {
    /* .get_name             = */ ggml_backend_qnn_device_get_name,
    /* .get_description      = */ ggml_backend_qnn_device_get_description,
    /* .get_memory           = */ ggml_backend_qnn_device_get_memory,
    /* .get_type             = */ ggml_backend_qnn_device_get_type,
    /* .get_props            = */ ggml_backend_qnn_device_get_props,
    /* .init_backend         = */ ggml_backend_qnn_device_init,
    /* .get_buffer_type      = */ ggml_backend_qnn_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_qnn_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_qnn_device_supports_op,
    /* .supports_buft        = */ ggml_backend_qnn_device_supports_buft,
    /* .offload_op           = */ ggml_backend_qnn_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

class qnn_device_proxy : public backend_device_proxy {
  public:
    explicit qnn_device_proxy(backend_index_type device) {
        const auto & device_caps = qnn::get_device_caps(device);
        _device_context          = std::make_unique<qnn::ggml_backend_qnn_device_context>(
            /* .device   = */ device,  // init from the last device, i.e. NPU
            /* .threads  = */ 1,       // TODO: fix this
            /* .name     = */ qnn::get_backend_name(device),
            /* .supported_types = */ device_caps.supported_types);
    }

    ~qnn_device_proxy() override {
        // Workflow1 keeps a process-lifetime QNN backend/device/AoT runtime to avoid
        // repeated-init instability. On Android we also observed multiple exit-time
        // aborts inside libQnnSystem.so while destructing these objects during static
        // teardown. Releasing ownership here intentionally leaks the process-lifetime
        // device context so the OS can reclaim it on exit without running the fragile
        // QNN teardown path.
        (void) _device_context.release();
    }

    const ggml_backend_device_i & get_iface() const { return ggml_backend_qnn_device_interface; }

    void * get_context() { return _device_context.get(); }

  private:
    std::unique_ptr<qnn::ggml_backend_qnn_device_context> _device_context;
};

}  // namespace

bool ggml_backend_qnn_aot_has_pending_generic_kv_writeback(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr) {
        return false;
    }

    auto * device_ctx = reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(backend->device->context);
    return device_ctx != nullptr &&
           device_ctx->aot_runtime != nullptr &&
           device_ctx->aot_runtime->has_pending_generic_kv_writeback();
}

bool ggml_backend_qnn_aot_flush_pending_generic_kv_writeback(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr) {
        return false;
    }

    auto * device_ctx = reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(backend->device->context);
    if (device_ctx == nullptr || device_ctx->aot_runtime == nullptr) {
        return false;
    }

    return device_ctx->aot_runtime->flush_pending_generic_kv_writeback();
}

bool ggml_backend_qnn_aot_reset_state(ggml_backend_t backend) {
    if (backend == nullptr || backend->device == nullptr) {
        return false;
    }

    auto * device_ctx = reinterpret_cast<qnn::ggml_backend_qnn_device_context *>(backend->device->context);
    if (device_ctx == nullptr || device_ctx->aot_runtime == nullptr) {
        return false;
    }

    device_ctx->aot_runtime->reset_state();
    return true;
}

backend_device_proxy_ptr create_qnn_backend_context(backend_index_type device) {
    if (device >= QNN_BACKEND_COUNT) {
        QNN_LOG_ERROR("[qnn]invalid device %d\n", device);
        return backend_device_proxy_ptr();
    }

#ifndef GGML_QNN_ENABLE_CPU_BACKEND
    if (device == QNN_BACKEND_CPU) {
        /*
                     * here we skip the initialization of CPU device,
                     *   cause it'll block unsupported ops fallback to ggml cpu backend
                     */
        GGML_LOG_DEBUG("qnn backend registry skip CPU device\n");
        return backend_device_proxy_ptr();
    }
#endif

    return std::make_unique<qnn_device_proxy>(device);
}

// Force the QNN registry implementation from runtime-common into libggml-qnn.
ggml_backend_reg_t ggml_backend_qnn_link_anchor(void);

ggml_backend_reg_t ggml_backend_qnn_link_anchor(void) {
    return ggml_backend_qnn_reg();
}

GGML_BACKEND_DL_IMPL(ggml_backend_qnn_reg)
