#pragma once

#include <array>
#include <cstring>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <HTP/QnnHtpMem.h>

#include "ggml-backend-impl.h"
#include "ggml.h"
#include "logger.hpp"
#include "qnn-lib.hpp"

namespace qnn {

/**
 * @brief An interface for managing generic QNN buffers.
 *
 * This abstract class defines the interface for managing generic memory buffers in a QNN context.
 */
class qnn_buffer_interface {
  public:
    virtual ~qnn_buffer_interface() = default;

    /**
     * @brief Checks if the buffer is valid.
     *
     * This pure virtual function must be implemented by derived classes to check
     * the validity of the buffer.
     *
     * @return true if the buffer is valid, false otherwise.
     */
    virtual bool is_valid() const = 0;

    /**
     * @brief Gets the buffer pointer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * a pointer to the buffer.
     *
     * @return A pointer to the buffer.
     */
    virtual uint8_t * get_buffer() = 0;

    /**
     * @brief Gets the buffer pointer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * a pointer to the buffer.
     *
     * @return A pointer to the buffer.
     */
    virtual size_t get_size() const = 0;

    /**
     * @brief Gets the QNN memory handle associated with the buffer.
     *
     * This pure virtual function must be implemented by derived classes to return
     * the memory handle associated with the buffer.
     *
     * @return The memory handle, or null if no valid QNN memory handle is attached.
     */
    virtual Qnn_MemHandle_t get_mem_handle() const = 0;
};

using qnn_buffer_ptr = std::shared_ptr<qnn_buffer_interface>;


class qnn_shared_buffer_allocator {
  public:
    qnn_shared_buffer_allocator(qnn_instance_ptr qnn_instance, size_t size, size_t alignment = 64) :
        _size(size),
        _alignment(alignment),
        _qnn_instance(std::move(qnn_instance)) {
        _data = static_cast<uint8_t *>(_qnn_instance->alloc_rpcmem(size, alignment));
        if (!_data) {
            QNN_LOG_WARN("failed to allocate shared rpcmem, size=%zu\n", size);
            return;
        }

        _fd = _qnn_instance->rpcmem_to_fd(_data);
        if (_fd == -1) {
            QNN_LOG_WARN("failed to get shared rpcmem fd\n");
            _qnn_instance->free_rpcmem(_data);
            _data = nullptr;
            return;
        }
    }

    ~qnn_shared_buffer_allocator() {
        if (_qnn_instance && _data) {
            _qnn_instance->free_rpcmem(_data);
        }
    }

    bool is_valid() const { return _data != nullptr && _fd != -1; }
    uint8_t * base() const { return _data; }
    size_t size() const { return _size; }
    int fd() const { return _fd; }

    size_t allocate(size_t bytes) {
        const size_t aligned = static_cast<size_t>(qnn::align_to(_alignment, static_cast<intptr_t>(_offset)));
        if (aligned + bytes > _size) {
            return SIZE_MAX;
        }
        _offset = aligned + bytes;
        return aligned;
    }

  private:
    size_t           _size      = 0;
    size_t           _offset    = 0;
    size_t           _alignment = 64;
    int              _fd        = -1;
    uint8_t *        _data      = nullptr;
    qnn_instance_ptr _qnn_instance;

    DISABLE_COPY(qnn_shared_buffer_allocator);
    DISABLE_MOVE(qnn_shared_buffer_allocator);
};

class qnn_htp_shared_buffer : public qnn_buffer_interface {
  public:
    qnn_htp_shared_buffer(qnn_instance_ptr qnn_instance,
                          std::shared_ptr<qnn_shared_buffer_allocator> allocator,
                          size_t size,
                          Qnn_DataType_t data_type,
                          Qnn_ContextHandle_t context_handle) :
        _size(size),
        _allocator(std::move(allocator)),
        _qnn_instance(std::move(qnn_instance)) {
        if (!_allocator || !_allocator->is_valid() || !_qnn_instance) {
            QNN_LOG_WARN("invalid shared buffer allocator\n");
            return;
        }

        _offset = _allocator->allocate(size);
        if (_offset == SIZE_MAX) {
            QNN_LOG_WARN("no free memory in shared buffer allocator, size=%zu total=%zu\n", size, _allocator->size());
            return;
        }

        const size_t type_size = std::max<size_t>(1, qnn::qnn_datatype_size(data_type));
        uint32_t shape[1] = { static_cast<uint32_t>(std::max<size_t>(1, (size + type_size - 1) / type_size)) };
        QnnMemHtp_Descriptor_t htp_mem_desc = {
            .type = QNN_HTP_MEM_SHARED_BUFFER,
            .size = _allocator->size(),
            .sharedBufferConfig = {
                .fd = _allocator->fd(),
                .offset = _offset,
            },
        };
        Qnn_MemDescriptor_t mem_desc = {
            .memShape = {
                .numDim = 1,
                .dimSize = shape,
                .shapeConfig = nullptr,
            },
            .dataType = data_type,
            .memType = QNN_MEM_TYPE_CUSTOM,
            .customInfo = &htp_mem_desc,
        };

        auto qnn_interface = _qnn_instance->get_qnn_interface();
        if (!qnn_interface || !context_handle) {
            QNN_LOG_WARN("failed to register shared buffer, qnn interface/context invalid\n");
            return;
        }

        auto error = qnn_interface->qnn_mem_register(context_handle, &mem_desc, 1, &_mem_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to register shared buffer, err=%d (%s)\n", (int) error, get_qnn_error_string(error));
            _mem_handle = nullptr;
            return;
        }
    }

    ~qnn_htp_shared_buffer() {
        if (_qnn_instance && _mem_handle) {
            auto qnn_interface = _qnn_instance->get_qnn_interface();
            if (qnn_interface) {
                auto error = qnn_interface->qnn_mem_de_register(&_mem_handle, 1);
                if (error != QNN_SUCCESS) {
                    QNN_LOG_WARN("failed to unregister shared buffer, error=%d (%s)\n",
                                 (int) error, get_qnn_error_string(error));
                }
            }
            _mem_handle = nullptr;
        }
    }

    bool is_valid() const override { return _allocator && _allocator->is_valid() && _mem_handle != nullptr; }
    uint8_t * get_buffer() override { return _allocator ? _allocator->base() + _offset : nullptr; }
    size_t get_size() const override { return _size; }
    Qnn_MemHandle_t get_mem_handle() const override { return _mem_handle; }

  private:
    size_t                                      _size       = 0;
    uint64_t                                    _offset     = 0;
    Qnn_MemHandle_t                             _mem_handle = nullptr;
    std::shared_ptr<qnn_shared_buffer_allocator> _allocator;
    qnn_instance_ptr                            _qnn_instance;

    DISABLE_COPY(qnn_htp_shared_buffer);
    DISABLE_MOVE(qnn_htp_shared_buffer);
};

class qnn_htp_shared_buffer_view : public qnn_buffer_interface {
  public:
    qnn_htp_shared_buffer_view(qnn_instance_ptr qnn_instance,
                               std::shared_ptr<qnn_shared_buffer_allocator> allocator,
                               size_t offset,
                               size_t size,
                               uint32_t rank,
                               const uint32_t * dimensions,
                               Qnn_DataType_t data_type,
                               Qnn_ContextHandle_t context_handle) :
        _size(size),
        _offset(offset),
        _rank(std::max<uint32_t>(1, std::min<uint32_t>(rank, GGML_MAX_DIMS))),
        _allocator(std::move(allocator)),
        _qnn_instance(std::move(qnn_instance)) {
        if (!_allocator || !_allocator->is_valid() || !_qnn_instance) {
            QNN_LOG_WARN("invalid shared buffer view allocator\n");
            return;
        }

        _dimensions.fill(1);
        if (dimensions != nullptr) {
            for (uint32_t i = 0; i < _rank; ++i) {
                _dimensions[i] = std::max<uint32_t>(dimensions[i], 1);
            }
        } else {
            const size_t type_size = std::max<size_t>(1, qnn::qnn_datatype_size(data_type));
            _dimensions[0] = static_cast<uint32_t>(std::max<size_t>(1, (size + type_size - 1) / type_size));
        }

        QnnMemHtp_Descriptor_t htp_mem_desc = {
            .type = QNN_HTP_MEM_SHARED_BUFFER,
            .size = _allocator->size(),
            .sharedBufferConfig = {
                .fd = _allocator->fd(),
                .offset = static_cast<uint64_t>(_offset),
            },
        };
        Qnn_MemDescriptor_t mem_desc = {
            .memShape = {
                .numDim = _rank,
                .dimSize = _dimensions.data(),
                .shapeConfig = nullptr,
            },
            .dataType = data_type,
            .memType = QNN_MEM_TYPE_CUSTOM,
            .customInfo = &htp_mem_desc,
        };

        auto qnn_interface = _qnn_instance->get_qnn_interface();
        if (!qnn_interface || !context_handle) {
            QNN_LOG_WARN("failed to register shared buffer view, qnn interface/context invalid\n");
            return;
        }

        auto error = qnn_interface->qnn_mem_register(context_handle, &mem_desc, 1, &_mem_handle);
        if (error != QNN_SUCCESS) {
            QNN_LOG_WARN("failed to register shared buffer view, err=%d (%s)\n",
                         (int) error,
                         get_qnn_error_string(error));
            _mem_handle = nullptr;
            return;
        }
    }

    ~qnn_htp_shared_buffer_view() {
        if (_qnn_instance && _mem_handle) {
            auto qnn_interface = _qnn_instance->get_qnn_interface();
            if (qnn_interface) {
                auto error = qnn_interface->qnn_mem_de_register(&_mem_handle, 1);
                if (error != QNN_SUCCESS) {
                    QNN_LOG_WARN("failed to unregister shared buffer view, error=%d (%s)\n",
                                 (int) error,
                                 get_qnn_error_string(error));
                }
            }
            _mem_handle = nullptr;
        }
    }

    bool is_valid() const override { return _allocator && _allocator->is_valid() && _mem_handle != nullptr; }
    uint8_t * get_buffer() override { return _allocator ? _allocator->base() + _offset : nullptr; }
    size_t get_size() const override { return _size; }
    Qnn_MemHandle_t get_mem_handle() const override { return _mem_handle; }

  private:
    size_t                                      _size       = 0;
    size_t                                      _offset     = 0;
    uint32_t                                    _rank       = 1;
    std::array<uint32_t, GGML_MAX_DIMS>         _dimensions = {};
    Qnn_MemHandle_t                             _mem_handle = nullptr;
    std::shared_ptr<qnn_shared_buffer_allocator> _allocator;
    qnn_instance_ptr                            _qnn_instance;

    DISABLE_COPY(qnn_htp_shared_buffer_view);
    DISABLE_MOVE(qnn_htp_shared_buffer_view);
};

class qnn_htp_buffer_pool : public qnn_buffer_interface {
  public:
    qnn_htp_buffer_pool(qnn_instance_ptr qnn_instance,
                        size_t size,
                        size_t alignment = 64) :
        _size(size),
        _allocator(std::make_shared<qnn_shared_buffer_allocator>(qnn_instance, size, alignment)),
        _qnn_instance(std::move(qnn_instance)) {}

    bool is_valid() const override {
        return _allocator && _allocator->is_valid();
    }

    uint8_t * get_buffer() override {
        return _allocator ? _allocator->base() : nullptr;
    }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

    qnn_buffer_ptr get_tensor_view(
            const ggml_tensor * tensor,
            uint32_t rank,
            const uint32_t * dimensions,
            Qnn_DataType_t data_type,
            Qnn_ContextHandle_t context_handle) {
        if (!tensor || !tensor->data || !is_valid() || context_handle == nullptr) {
            return nullptr;
        }

        std::array<uint32_t, GGML_MAX_DIMS> normalized_dimensions = {};
        normalized_dimensions.fill(1);
        const uint32_t normalized_rank = std::max<uint32_t>(1, std::min<uint32_t>(rank, GGML_MAX_DIMS));
        if (dimensions != nullptr) {
            for (uint32_t i = 0; i < normalized_rank; ++i) {
                normalized_dimensions[i] = std::max<uint32_t>(dimensions[i], 1);
            }
        }

        const size_t nbytes = std::max<size_t>(1, ggml_nbytes(tensor));
        const qnn_htp_tensor_view_key key = {
            tensor->data,
            nbytes,
            context_handle,
            normalized_rank,
            normalized_dimensions,
            data_type,
        };
        auto it = _tensor_views.find(key);
        if (it != _tensor_views.end()) {
            return it->second;
        }

        auto * base = _allocator->base();
        auto * ptr  = static_cast<uint8_t *>(tensor->data);

        if (ptr < base || ptr + nbytes > base + _size) {
            QNN_LOG_WARN("tensor buffer view out of range: tensor=%s ptr=%p base=%p bytes=%zu pool=%zu\n",
                         ggml_get_name(tensor),
                         (void *) ptr,
                         (void *) base,
                         nbytes,
                         _size);
            return nullptr;
        }

        const size_t offset = static_cast<size_t>(ptr - base);
        auto view = std::make_shared<qnn_htp_shared_buffer_view>(
            _qnn_instance,
            _allocator,
            offset,
            nbytes,
            rank,
            dimensions,
            data_type,
            context_handle);
        if (!view->is_valid()) {
            return nullptr;
        }

        _tensor_views.emplace(key, view);
        return view;
    }

  private:
    struct qnn_htp_tensor_view_key {
        const void *                     data            = nullptr;
        size_t                           nbytes          = 0;
        Qnn_ContextHandle_t              context_handle  = nullptr;
        uint32_t                         rank            = 1;
        std::array<uint32_t, GGML_MAX_DIMS> dimensions   = {};
        Qnn_DataType_t                   data_type       = QNN_DATATYPE_UNDEFINED;

        bool operator==(const qnn_htp_tensor_view_key & other) const {
            return data == other.data &&
                   nbytes == other.nbytes &&
                   context_handle == other.context_handle &&
                   rank == other.rank &&
                   dimensions == other.dimensions &&
                   data_type == other.data_type;
        }
    };

    struct qnn_htp_tensor_view_key_hash {
        size_t operator()(const qnn_htp_tensor_view_key & key) const {
            size_t hash = std::hash<const void *>()(key.data);
            hash ^= std::hash<size_t>()(key.nbytes) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<Qnn_ContextHandle_t>()(key.context_handle) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<uint32_t>()(key.rank) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= std::hash<int>()(static_cast<int>(key.data_type)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            for (uint32_t dim : key.dimensions) {
                hash ^= std::hash<uint32_t>()(dim) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };

    size_t                                       _size           = 0;
    std::shared_ptr<qnn_shared_buffer_allocator> _allocator;
    qnn_instance_ptr                             _qnn_instance;
    std::unordered_map<qnn_htp_tensor_view_key, qnn_buffer_ptr, qnn_htp_tensor_view_key_hash> _tensor_views;

    DISABLE_COPY(qnn_htp_buffer_pool);
    DISABLE_MOVE(qnn_htp_buffer_pool);
};

inline qnn_htp_buffer_pool * get_qnn_htp_buffer_pool(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return nullptr;
    }

    const char * buffer_name = ggml_backend_buffer_name(tensor->buffer);
    if (buffer_name == nullptr || std::strcmp(buffer_name, "qnn-npu-host") != 0) {
        return nullptr;
    }

    return reinterpret_cast<qnn_htp_buffer_pool *>(tensor->buffer->context);
}

inline qnn_buffer_ptr try_get_qnn_host_buffer_view(const ggml_tensor * tensor,
                                                   Qnn_ContextHandle_t context_handle,
                                                   uint32_t            rank,
                                                   const uint32_t *    dimensions,
                                                   Qnn_DataType_t      data_type) {
    auto * pool = get_qnn_htp_buffer_pool(tensor);
    if (pool == nullptr || !pool->is_valid() || context_handle == nullptr) {
        return nullptr;
    }

    auto buffer = pool->get_tensor_view(
        tensor,
        rank,
        dimensions,
        data_type,
        context_handle);
    if (buffer && buffer->get_mem_handle() != nullptr) {
        QNN_LOG_DEBUG("reuse qnn shared-host buffer for tensor(%s)\n", ggml_get_name(tensor));
    }

    return buffer;
}

/**
 * @brief A class for managing QNN RPC memory buffers.
 *
 * This class is responsible for allocating, registering, and managing a buffer in RPC memory.
 * It ensures that the buffer is properly allocated and registered with the QNN instance, and
 * handles cleanup of the buffer and its associated memory handle upon destruction.
 */
class qnn_rpc_buffer : public qnn_buffer_interface {
  public:
    qnn_rpc_buffer(qnn_instance_ptr qnn_instance, const size_t size, const uint32_t rank, uint32_t * dimensions,
                   Qnn_DataType_t data_type, Qnn_ContextHandle_t context_handle = nullptr) :
        _size(size),
        _qnn_instance(qnn_instance) {
        _qnn_rpc_buffer     = static_cast<uint8_t *>(qnn_instance->alloc_rpcmem(size, alignof(uint8_t *)));
        _qnn_rpc_mem_handle = qnn_instance->register_rpcmem(_qnn_rpc_buffer, rank, dimensions, data_type, context_handle);
        if (!_qnn_rpc_buffer || !_qnn_rpc_mem_handle) {
            QNN_LOG_WARN("Failed to register RPC memory: buffer or memory handle is null\n");
            // let the destructor free the buffer
            return;
        }

        QNN_LOG_DEBUG("alloc rpcmem(%p) successfully, size %d\n", (void *) _qnn_rpc_buffer, (int) size);
    }

    ~qnn_rpc_buffer() {
        if (_qnn_instance) {
            if (_qnn_rpc_mem_handle) {
                _qnn_instance->unregister_rpcmem(_qnn_rpc_mem_handle);
            }

            if (_qnn_rpc_buffer) {
                _qnn_instance->free_rpcmem(_qnn_rpc_buffer);
            }
        }
    }

    bool is_valid() const override { return _qnn_rpc_buffer && _qnn_rpc_mem_handle; }

    uint8_t * get_buffer() override { return _qnn_rpc_buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return _qnn_rpc_mem_handle; }

  private:
    size_t           _size               = 0;
    uint8_t *        _qnn_rpc_buffer     = nullptr;
    Qnn_MemHandle_t  _qnn_rpc_mem_handle = nullptr;
    qnn_instance_ptr _qnn_instance;

    DISABLE_COPY(qnn_rpc_buffer);
    DISABLE_MOVE(qnn_rpc_buffer);
};

/**
 * @brief A class for managing QNN memory buffers allocated in regular memory.
 *
 * This class is responsible for allocating, managing, and freeing memory buffers
 * in regular (non-RPC) memory. It implements the qnn_buffer_interface to provide
 * a consistent interface for buffer management.
 */
class qnn_mem_buffer : public qnn_buffer_interface {
  public:
    explicit qnn_mem_buffer(const uint8_t * data, size_t size) {
        _buffer = reinterpret_cast<uint8_t *>(qnn::page_align_alloc(size));

        if (!_buffer) {
            QNN_LOG_WARN("failed to allocate %.2f MiB\n", float(size / (1 << 20)));
            return;
        }

        _size = size;

        if (data) {
            memcpy(_buffer, data, size);
        }

        QNN_LOG_DEBUG("alloc buffer: %p, size: %ld\n", (void *) _buffer, (long) size);
    }

    explicit qnn_mem_buffer(size_t size) : qnn_mem_buffer(nullptr, size) {}

    ~qnn_mem_buffer() {
        QNN_LOG_DEBUG("free buffer: %p, size: %ld\n", (void *) _buffer, (long) _size);
        // the free will do nothing if the _buffer is nullptr
        qnn::align_free(_buffer);
    }

    bool is_valid() const override { return _buffer != nullptr; }

    uint8_t * get_buffer() override { return _buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

  private:
    size_t    _size   = 0;
    uint8_t * _buffer = nullptr;

    DISABLE_COPY(qnn_mem_buffer);
    DISABLE_MOVE(qnn_mem_buffer);
};

class qnn_mem_buffer_slice : public qnn_buffer_interface {
  public:
    qnn_mem_buffer_slice(const uint8_t * buffer, size_t size) : _buffer(const_cast<uint8_t *>(buffer)), _size(size) {}

    bool is_valid() const override { return _buffer && _size; }

    uint8_t * get_buffer() override { return _buffer; }

    size_t get_size() const override { return _size; }

    Qnn_MemHandle_t get_mem_handle() const override { return nullptr; }

  private:
    uint8_t * _buffer = nullptr;
    size_t    _size   = 0;

    DISABLE_COPY(qnn_mem_buffer_slice);
    DISABLE_MOVE(qnn_mem_buffer_slice);
};

}  // namespace qnn
