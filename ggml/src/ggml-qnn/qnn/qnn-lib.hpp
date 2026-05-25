#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// header file of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
// https://qpm.qualcomm.com/#/main/tools/details/qualcomm_ai_engine_direct
#include <HTP/QnnHtpDevice.h>
#include <HTP/QnnHtpGraph.h>
#include <QnnBackend.h>
#include <QnnCommon.h>
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnInterface.h>
#include <QnnProperty.h>
#include <QnnTensor.h>
#include <QnnTypes.h>
#include <System/QnnSystemInterface.h>

#include "dyn-lib-loader.hpp"
#include "qnn-types.hpp"
#include "rpc-mem.hpp"
#include "utils.hpp"

namespace qnn {

// =================================================================================================
//
// wrapper class of Qualcomm QNN(Qualcomm Neural Network, aka Qualcomm AI Engine Direct) SDK
// ref:https://github.com/pytorch/executorch/tree/main/backends/qualcomm
// =================================================================================================

// TODO: fix this for other compilers
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wextra-semi"
#pragma GCC diagnostic ignored "-Wpedantic"

class qnn_system_interface {
#define DEFINE_SHIM_FUNCTION_SYS_INTERFACE(F, pointer_name)                                                  \
    template <typename... Args> inline auto qnn_##F(Args... args) const {                                    \
        return (_qnn_sys_interface.QNN_SYSTEM_INTERFACE_VER_NAME.pointer_name)(std::forward<Args>(args)...); \
    }

  public:
    qnn_system_interface(const QnnSystemInterface_t & qnn_sys_interface, common::dl_handler_t lib_handle);
    ~qnn_system_interface();

    bool is_valid() const { return _qnn_system_handle != nullptr; }

    // QnnSystem
    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_create, systemContextCreate);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_get_binary_info, systemContextGetBinaryInfo);

    DEFINE_SHIM_FUNCTION_SYS_INTERFACE(system_context_free, systemContextFree);

  private:
    qnn_system_interface(const qnn_system_interface &) = delete;
    void operator=(const qnn_system_interface &)       = delete;
    qnn_system_interface(qnn_system_interface &&)      = delete;
    void operator=(qnn_system_interface &&)            = delete;

    const QnnSystemInterface_t _qnn_sys_interface = {};
    common::dl_handler_t       _lib_handle        = nullptr;
    QnnSystemContext_Handle_t  _qnn_system_handle = nullptr;
};

class qnn_interface {
#define DEFINE_SHIM_FUNCTION_INTERFACE(F, pointer_name)                                           \
    template <typename... Args> inline auto qnn_##F(Args... args) const {                         \
        return (_qnn_interface.QNN_INTERFACE_VER_NAME.pointer_name)(std::forward<Args>(args)...); \
    }

  public:
    qnn_interface(const QnnInterface_t & qnn_interface) : _qnn_interface(qnn_interface) {}

    // QnnBackend
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_create, backendCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_free, backendFree);
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_register_op_package, backendRegisterOpPackage);
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_validate_op_config, backendValidateOpConfig);
    DEFINE_SHIM_FUNCTION_INTERFACE(backend_get_api_version, backendGetApiVersion);

    // QnnDevice
    DEFINE_SHIM_FUNCTION_INTERFACE(device_create, deviceCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(device_free, deviceFree);
    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_infrastructure, deviceGetInfrastructure);
    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_platform_info, deviceGetPlatformInfo);
    DEFINE_SHIM_FUNCTION_INTERFACE(device_free_platform_info, deviceFreePlatformInfo);
    DEFINE_SHIM_FUNCTION_INTERFACE(device_get_info, deviceGetInfo);

    // QnnContext
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create, contextCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary_size, contextGetBinarySize);
    DEFINE_SHIM_FUNCTION_INTERFACE(context_get_binary, contextGetBinary);
    DEFINE_SHIM_FUNCTION_INTERFACE(context_create_from_binary, contextCreateFromBinary);
    DEFINE_SHIM_FUNCTION_INTERFACE(context_free, contextFree);

    // QnnGraph
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_create, graphCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_add_node, graphAddNode);
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_finalize, graphFinalize);
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_execute, graphExecute);
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_retrieve, graphRetrieve);
    DEFINE_SHIM_FUNCTION_INTERFACE(graph_set_config, graphSetConfig);

    // QnnLog
    DEFINE_SHIM_FUNCTION_INTERFACE(log_create, logCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(log_free, logFree);
    DEFINE_SHIM_FUNCTION_INTERFACE(log_set_log_level, logSetLogLevel);

    // QnnProfile
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_create, profileCreate);
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_set_config, profileSetConfig);
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_events, profileGetEvents);
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_sub_events, profileGetSubEvents);
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_get_event_data, profileGetEventData);
    DEFINE_SHIM_FUNCTION_INTERFACE(profile_free, profileFree);

    // QnnMem
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_register, memRegister);
    DEFINE_SHIM_FUNCTION_INTERFACE(mem_de_register, memDeRegister);

    // QnnProperty
    DEFINE_SHIM_FUNCTION_INTERFACE(property_has_capability, propertyHasCapability);

    // QnnTensor
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_context_tensor, tensorCreateContextTensor);
    DEFINE_SHIM_FUNCTION_INTERFACE(tensor_create_graph_tensor, tensorCreateGraphTensor);

    uint32_t get_backend_id() const { return _qnn_interface.backendId; }

  private:
    qnn_interface(const qnn_interface &)  = delete;
    void operator=(const qnn_interface &) = delete;
    qnn_interface(qnn_interface &&)       = delete;
    void operator=(qnn_interface &&)      = delete;

    const QnnInterface_t _qnn_interface = {};
};

#pragma GCC diagnostic pop

using qnn_interface_ptr = std::shared_ptr<qnn_interface>;

class qnn_instance {
  public:
    using BackendIdType = decltype(QnnInterface_t{}.backendId);

    explicit qnn_instance(const std::string & lib_path, backend_index_type device);

    ~qnn_instance() {}

    bool qnn_init(const QnnSaver_Config_t ** saver_config);
    bool qnn_finalize();

    qnn_interface_ptr get_qnn_interface() {
        if (!_qnn_interface) {
            QNN_LOG_WARN("pls check why _qnn_interface is not loaded\n");
        }
        return _qnn_interface;
    }

    std::shared_ptr<qnn_system_interface> get_qnn_system_interface() {
        if (!_qnn_sys_interface) {
            QNN_LOG_WARN("pls check why _qnn_sys_interface is not loaded\n");
        }
        return _qnn_sys_interface;
    }

    Qnn_LogHandle_t get_qnn_log_handle() { return _qnn_log_handle; }

    Qnn_DeviceHandle_t get_qnn_device_handle() { return _qnn_device_handle; }

    Qnn_BackendHandle_t get_qnn_backend_handle() { return _qnn_backend_handle; }

    Qnn_ContextHandle_t get_qnn_context_handle() { return _qnn_context_handle; }

    Qnn_GraphHandle_t get_qnn_graph_handle() { return _qnn_graph_handle; }

    int init_htp_perfinfra();
    int set_htp_power_workpoint();

    std::string & get_qnn_graph_name() { return _graph_name; }

    void * alloc_rpcmem(size_t bytes, size_t alignment) {
        if (!_rpc_mem) {
            QNN_LOG_WARN("rpc memory not initialized\n");
            return nullptr;
        }

        auto   allocate_bytes = static_cast<int64_t>(bytes + alignment);
        void * buf            = _rpc_mem->alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int) allocate_bytes);
        if (!buf) {
            QNN_LOG_WARN("failed to allocate rpc memory, size: %d MB\n", (int) (allocate_bytes / (1 << 20)));
            return nullptr;
        }

        auto aligned_buf = reinterpret_cast<void *>(qnn::align_to(alignment, reinterpret_cast<intptr_t>(buf)));
        bool status      = _rpcmem_store_map.insert(std::pair<void *, void *>(aligned_buf, buf)).second;
        if (!status) {
            QNN_LOG_WARN("failed to allocate rpc memory\n");
            _rpc_mem->free(buf);
        }

        return aligned_buf;
    }

    void free_rpcmem(void * buf) {
        if (!_rpc_mem) {
            QNN_LOG_WARN("rpc memory not initialized\n");
        } else if (_rpcmem_store_map.count(buf) == 0) {
            QNN_LOG_WARN("no allocated tensor\n");
        } else {
            _rpc_mem->free(_rpcmem_store_map[buf]);
            _rpcmem_store_map.erase(buf);
        }
    }

    int rpcmem_to_fd(void * buf) {
        int fd = -1;
        if (!_rpc_mem) {
            QNN_LOG_WARN("rpc memory not initialized\n");
        } else if (_rpcmem_store_map.count(buf) == 0) {
            QNN_LOG_WARN("no allocated tensor\n");
        } else {
            buf = _rpcmem_store_map[buf];
            fd  = _rpc_mem->to_fd(buf);
        }
        return fd;
    }

    Qnn_MemHandle_t register_rpcmem(void * p_data, const uint32_t rank, uint32_t * dimensions,
                                    Qnn_DataType_t data_type, Qnn_ContextHandle_t context_handle = nullptr) {
        if (!p_data) {
            QNN_LOG_WARN("invalid param\n");
            return nullptr;
        }

        if (!_rpc_mem) {
            QNN_LOG_WARN("rpc memory not initialized\n");
            return nullptr;
        }

        if (is_rpcmem_registered(p_data)) {
            QNN_LOG_WARN("rpc memory already registered\n");
            return _qnn_rpc_buffer_to_handles[p_data];
        }

        auto mem_fd = rpcmem_to_fd(p_data);
        if (mem_fd == -1) {
            QNN_LOG_WARN("failed to get file descriptor\n");
            return nullptr;
        }

        QNN_LOG_DEBUG("mem_fd %d\n", mem_fd);
        Qnn_MemDescriptor_t descriptor = {
            { rank, dimensions, nullptr },
            data_type, QNN_MEM_TYPE_ION, { { mem_fd } }
        };
        auto effective_context = context_handle ? context_handle : _qnn_context_handle;
        if (!effective_context) {
            QNN_LOG_WARN("failed to register shared memory, context handle is null\n");
            return nullptr;
        }

        Qnn_MemHandle_t handle = nullptr;
        auto            error  = _qnn_interface->qnn_mem_register(effective_context, &descriptor,
                                                                  /*numDescriptors=*/1, &handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to register shared memory, error %d, %s\n", (int) QNN_GET_ERROR_CODE(error),
                         strerror(error));
            return nullptr;
        }

        _qnn_rpc_buffer_to_handles.insert({ p_data, handle });
        QNN_LOG_DEBUG("successfully register shared memory handler: %p\n", handle);
        return handle;
    }

    void unregister_rpcmem(Qnn_MemHandle_t mem_handle) {
        auto error = _qnn_interface->qnn_mem_de_register(&mem_handle, 1);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to unregister shared memory, error %d\n", (int) QNN_GET_ERROR_CODE(error));
        }

        auto it = std::find_if(_qnn_rpc_buffer_to_handles.begin(), _qnn_rpc_buffer_to_handles.end(),
                               [mem_handle](const auto & kv) { return kv.second == mem_handle; });
        if (it == _qnn_rpc_buffer_to_handles.end()) {
            QNN_LOG_WARN("failed to find shared memory handler: %p\n", mem_handle);
            return;
        }

        _qnn_rpc_buffer_to_handles.erase(it);
    }

    bool is_rpcmem_allocated(void * buf) { return _rpcmem_store_map.count(buf) != 0; }

    bool is_rpcmem_registered(void * buf) { return _qnn_rpc_buffer_to_handles.count(buf) != 0U; }

    const qnn::qcom_socinfo & get_soc_info() { return _soc_info; }

    bool has_custom_op_package() const { return _has_custom_op_package; }

  private:
    int  load_system();
    bool load_backend(std::string & lib_path, const QnnSaver_Config_t ** /*saver_config*/);
    void unload_backend();

  private:
    static constexpr const int _required_num_providers = 1;

    std::string   _additional_lib_load_path;
    std::string   _backend_lib_name;
    BackendIdType _backend_id;

#ifdef NDEBUG
    QnnLog_Level_t _qnn_log_level = QNN_LOG_LEVEL_INFO;  // TODO: should we consider changing this dynamically?
#else
    QnnLog_Level_t _qnn_log_level = QNN_LOG_LEVEL_DEBUG;
#endif

    std::shared_ptr<qnn::qnn_system_interface> _qnn_sys_interface;
    std::shared_ptr<qnn::qnn_interface>        _qnn_interface;

    Qnn_GraphHandle_t                   _qnn_graph_handle   = nullptr;
    Qnn_LogHandle_t                     _qnn_log_handle     = nullptr;
    Qnn_DeviceHandle_t                  _qnn_device_handle  = nullptr;
    Qnn_BackendHandle_t                 _qnn_backend_handle = nullptr;
    Qnn_ContextHandle_t                 _qnn_context_handle = nullptr;
    QnnHtpDevice_PerfInfrastructure_t * _qnn_htp_perfinfra  = nullptr;
    uint32_t                            _qnn_power_configid = 1;

    std::unordered_map<void *, Qnn_MemHandle_t> _qnn_rpc_buffer_to_handles;

    std::mutex                                                _init_mutex;
    std::unordered_map<BackendIdType, common::dl_handler_t>   _loaded_lib_handle;
    std::unordered_map<std::string, BackendIdType>            _lib_path_to_backend_id;
    std::unordered_map<BackendIdType, const QnnInterface_t *> _loaded_backend;

    std::unique_ptr<common::rpc_mem>   _rpc_mem;
    std::unordered_map<void *, void *> _rpcmem_store_map;

    std::string _graph_name;

    qnn::qcom_socinfo _soc_info = {};

    bool                 _has_custom_op_package      = false;
    common::dl_handler_t _custom_op_extra_lib_handle = nullptr;
};

using qnn_instance_ptr = std::shared_ptr<qnn_instance>;

struct device_caps {
    const char *               lib_name;
    enum ggml_backend_dev_type type;

    // TODO: should we get this from device?
    uint64_t supported_types;

    // TODO: should we merge this with supported_types?
    uint64_t cpu_preprocess_types;

    // TODO: should we get this from device?
    size_t max_tensor_size_in_bytes;
};

const device_caps & get_device_caps(backend_index_type device);

}  // namespace qnn
