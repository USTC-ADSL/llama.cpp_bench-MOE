#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "buffer.hpp"
#include "ggml.h"
#include "ggml-backend.h"
#include "qnn-lib.hpp"
#include "utils.hpp"

namespace qnn {

struct qnn_aot_graph_config {
    std::string type;
    std::string graph_name;
    std::string model_path;
    std::string x_name;
    std::string out_name;
    std::string q_name;
    std::string k_name;
    std::string v_name;
    std::string cache_k_name;
    std::string cache_v_name;
    std::string attn_bias_name;

    size_t batch_size     = 0;
    size_t cache_size     = 0;
    size_t context_size   = 0;
    size_t start_layer_id = 0;
    size_t end_layer_id   = 0;
    size_t kv_size        = 0;

    std::string kv_path_format;
    size_t      n_hvx_threads = 4;
};

struct qnn_aot_model_params {
    size_t n_layers       = 0;
    size_t vocab_size     = 0;
    size_t embed_dim      = 0;
    size_t ffn_hidden_dim = 0;
    size_t head_dim       = 0;
    size_t n_kv_heads     = 0;

    float rope_theta           = 1000000.0f;
    float rms_norm_eps         = 1e-6f;
    float attention_mask_value = -50000.0f;

    bool tie_embedding = true;
};

struct qnn_aot_config {
    qnn_aot_model_params model;
    size_t               n_hvx_threads = 4;

    std::vector<qnn_aot_graph_config> transformer_graphs;
    std::vector<qnn_aot_graph_config> attention_graphs;
    std::vector<qnn_aot_graph_config> attn_proj_graphs;
    std::vector<qnn_aot_graph_config> attn_core_graphs;
    std::vector<qnn_aot_graph_config> ffn_graphs;
    std::vector<qnn_aot_graph_config> lm_head_graphs;

    bool load(const std::string & config_path);
};

struct qnn_aot_context {
    qnn_instance_ptr                      instance;
    std::string                           binary_path;
    Qnn_ContextHandle_t                   context_handle = nullptr;
    QnnSystemContext_Handle_t             system_context = nullptr;
    const QnnSystemContext_BinaryInfo_t * binary_info    = nullptr;
    Qnn_ContextBinarySize_t               binary_info_size = 0;

    qnn_aot_context(qnn_instance_ptr instance, const std::string & binary_path);
    ~qnn_aot_context();

    bool is_valid() const { return context_handle != nullptr && system_context != nullptr && binary_info != nullptr; }
};

class qnn_aot_graph {
  public:
    qnn_aot_graph(qnn_instance_ptr instance,
                  std::shared_ptr<qnn_aot_context> context,
                  qnn_aot_graph_config config,
                  const qnn_aot_graph * sibling = nullptr);
    ~qnn_aot_graph();

    bool is_valid() const { return _valid; }
    bool execute();

    void *       buffer_data(const std::string & name);
    const void * buffer_data(const std::string & name) const;
    size_t       buffer_size(const std::string & name) const;
    bool         has_buffer(const std::string & name) const;
    bool         bind_external_tensor(const std::string & name, ggml_tensor * tensor);
    void         clear_external_tensor_bindings();
    // Returns QNN_DATATYPE_UNDEFINED if the tensor name is not found.
    Qnn_DataType_t tensor_data_type(const std::string & name) const;
    size_t       batch_size() const { return _config.batch_size; }
    const qnn_aot_graph_config & config() const { return _config; }

  private:
    bool retrieve_graph_metadata();
    bool allocate_tensor_buffers();
    bool set_hvx_threads(size_t n_threads);

    qnn_instance_ptr                 _instance;
    qnn_interface_ptr                _qnn_interface;
    std::shared_ptr<qnn_aot_context> _context;
    qnn_aot_graph_config             _config;
    const qnn_aot_graph *            _sibling = nullptr;

    Qnn_GraphHandle_t _graph_handle = nullptr;
    bool              _valid        = false;

    std::vector<Qnn_Tensor_t> _inputs;
    std::vector<Qnn_Tensor_t> _outputs;

    std::unordered_map<std::string, size_t> _input_index;
    std::unordered_map<std::string, size_t> _output_index;
    std::unordered_map<std::string, qnn_buffer_ptr> _buffers;
    std::unordered_map<std::string, qnn_buffer_ptr> _external_buffers;
    std::shared_ptr<qnn_shared_buffer_allocator> _shared_allocator;
};

class qnn_aot_runtime {
  public:
    explicit qnn_aot_runtime(qnn_instance_ptr instance, backend_index_type device);
    ~qnn_aot_runtime();

    bool initialize(const std::string & config_path, const std::string & model_dir);
    bool supports_op(const ggml_tensor * op) const;
    bool supports_fragment_op(const ggml_tensor * op) const;
    bool prefers_cpu_op(const ggml_tensor * op) const;
    bool maybe_execute(ggml_cgraph * cgraph);
    bool has_pending_generic_kv_writeback() const;
    bool flush_pending_generic_kv_writeback();
    void reset_state();

    bool is_enabled() const { return _enabled; }
    const std::string & config_path() const { return _config_path; }

  private:
    using graph_bucket = std::vector<std::unique_ptr<qnn_aot_graph>>;
    using graph_family = std::map<size_t, graph_bucket>;

    struct rope_embedding {
        std::vector<float> cos_values;
        std::vector<float> sin_values;
    };

    struct aot_match_result {
        ggml_tensor * embd = nullptr;
        ggml_tensor * aux  = nullptr;
        ggml_tensor * out  = nullptr;
        ggml_tensor * q_out = nullptr;
        ggml_tensor * k_out = nullptr;
        ggml_tensor * v_out = nullptr;
        ggml_tensor * cache_k = nullptr;
        ggml_tensor * cache_v = nullptr;
        ggml_tensor * kq_mask = nullptr;
        ggml_tensor * k_idxs = nullptr;
        ggml_tensor * v_idxs = nullptr;
        std::map<size_t, ggml_tensor *> cache_k_layers;
        std::map<size_t, ggml_tensor *> cache_v_layers;
        size_t        n_tokens = 0;
        size_t        inferred_pos = 0;
        size_t        start_layer_id = 0;
        size_t        end_layer_id   = 0;
        size_t        layer_id       = std::numeric_limits<size_t>::max();
        bool          is_attention   = false;
        bool          is_attn_proj   = false;
        bool          is_attn_core   = false;
        bool          is_transformer = false;
        bool          is_ffn         = false;
        bool          is_lm_head     = false;
    };

    struct pending_generic_kv_writeback_layer {
        ggml_tensor * cache_k = nullptr;
        ggml_tensor * cache_v = nullptr;
        std::vector<int64_t> k_idxs;
        std::vector<int64_t> v_idxs;
        std::vector<float>   key_rows;
        std::vector<float>   value_rows;
        size_t               key_token_values = 0;
        size_t               value_token_values = 0;
        size_t               n_tokens = 0;
    };

    static bool has_prefix(const char * name, const char * prefix);
    static size_t parse_layer_id_from_name(const char * name);
    static bool is_attention_proj_stage_name(const char * name);
    static bool is_attention_core_stage_name(const char * name);
    static bool is_attention_output_stage_name(const char * name);
    static bool is_attention_stage_name(const char * name);
    static bool is_ffn_stage_name(const char * name);
    static bool is_transformer_stage_name(const char * name);
    static bool is_cpu_stage_name(const char * name);
    static bool is_lm_head_stage_name(const char * name);
    static bool is_embedding_lookup(const ggml_tensor * op);
    bool is_transformer_output_candidate(const ggml_tensor * tensor) const;

    aot_match_result match_attention_graph(ggml_cgraph * cgraph) const;
    aot_match_result match_attn_proj_graph(ggml_cgraph * cgraph) const;
    aot_match_result match_attn_core_graph(ggml_cgraph * cgraph) const;
    aot_match_result match_transformer_graph(ggml_cgraph * cgraph) const;
    aot_match_result match_ffn_graph(ggml_cgraph * cgraph) const;
    aot_match_result match_lm_head_graph(ggml_cgraph * cgraph) const;
    bool execute_fragment_view(ggml_cgraph * cgraph, int i0, int i1);
    bool try_execute_fragmented_transformer(ggml_cgraph * cgraph);
    bool execute_attention(ggml_cgraph * cgraph, const aot_match_result & match);
    bool execute_attn_proj(ggml_cgraph * cgraph, const aot_match_result & match);
    bool execute_attn_core(ggml_cgraph * cgraph, const aot_match_result & match);
    bool execute_transformer(ggml_cgraph * cgraph, const aot_match_result & match, ggml_tensor * last_row_out = nullptr);
    bool execute_ffn(ggml_cgraph * cgraph, const aot_match_result & match);
    bool execute_lm_head(ggml_cgraph * cgraph, const aot_match_result & match);
    bool execute_tail_replay_fragment(ggml_cgraph * cgraph, size_t begin, size_t end);
    ggml_backend_t ensure_cpu_backend();
    qnn_aot_graph * select_attention_graph(size_t start_layer_id, size_t end_layer_id, size_t n_tokens) const;
    qnn_aot_graph * select_attn_proj_graph(size_t n_tokens, size_t layer_id);
    qnn_aot_graph * select_attn_core_graph(size_t n_tokens, size_t layer_id);
    graph_bucket * select_transformer_graphs(size_t n_tokens);
    graph_bucket * ensure_transformer_graph_bucket_loaded(size_t batch_size);
    qnn_aot_graph * select_ffn_graph(size_t n_tokens, size_t layer_id);
    qnn_aot_graph * select_lm_head_graph(size_t n_tokens);
    qnn_aot_graph * select_graph(const std::vector<qnn_aot_graph_config> & configs,
                                 graph_family &                              family,
                                 size_t                                      n_tokens,
                                 size_t                                      layer_id);
    qnn_aot_graph * ensure_graph_loaded(const qnn_aot_graph_config & graph_config,
                                        graph_family &                family);

    void compute_rope_embeds();
    void fill_rope_embeds(qnn_aot_graph & graph, size_t start_pos, size_t n_tokens);
    void fill_attention_bias(qnn_aot_graph & graph, size_t n_tokens);
    void save_kv(qnn_aot_graph & graph, size_t n_tokens);
    bool import_generic_kv_prefix_to_graph(qnn_aot_graph & graph,
                                           const aot_match_result & match,
                                           size_t source_token_offset,
                                           size_t dest_token_offset,
                                           size_t n_tokens);
    bool should_write_generic_kv(const aot_match_result & match) const;
    bool should_defer_generic_kv_writeback(const aot_match_result & match) const;
    bool collect_generic_kv_from_graph(qnn_aot_graph & graph,
                                       const aot_match_result & match,
                                       size_t token_offset,
                                       size_t n_tokens,
                                       std::vector<pending_generic_kv_writeback_layer> & payloads) const;
    bool stage_generic_kv_from_graph(qnn_aot_graph & graph, const aot_match_result & match, size_t token_offset, size_t n_tokens);
    bool write_generic_kv_from_graph(qnn_aot_graph & graph, const aot_match_result & match, size_t token_offset, size_t n_tokens);
    bool load_seed_kv_into_graph(qnn_aot_graph & graph);
    bool load_seed_kv();
    std::string resolve_model_path(const std::string & relative_path) const;

    void zero_transformer_state();
    size_t infer_start_pos(const std::vector<ggml_tensor *> & inputs, size_t n_tokens) const;
    std::string format_kv_path(const qnn_aot_graph_config & config, size_t layer_id, const char * kv_type, size_t head_id) const;

    qnn_instance_ptr   _instance;
    backend_index_type _device;
    ggml_backend_t     _cpu_backend = nullptr;

    bool        _enabled        = false;
    bool        _seed_kv_loaded = false;
    std::string _config_path;
    std::string _model_dir;

    qnn_aot_config _config;
    std::vector<rope_embedding> _rope_embeds;
    graph_family _transformer_graphs;
    std::vector<std::unique_ptr<qnn_aot_graph>> _attention_graphs;
    graph_family _attn_proj_graphs;
    graph_family _attn_core_graphs;
    graph_family _ffn_graphs;
    graph_family _lm_head_graphs;
    std::unordered_map<std::string, std::shared_ptr<qnn_aot_context>> _contexts;
    std::mutex _lazy_graph_mutex;

    size_t _kv_position = 0;
    size_t _seed_kv_size = 0;
    bool _seed_kv_missing_warned = false;
    std::vector<pending_generic_kv_writeback_layer> _pending_generic_kv_writeback;
};

}  // namespace qnn
