#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-dyn-route.h"
#include "llama-hetero-route.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <string>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;
struct llama_opencl_external_host_sync_timing;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_sched_reserve_timing {
    int64_t sched_new_us = 0;
    int64_t memory_init_us = 0;
    int64_t feature_probe_us = 0;
    int64_t plan_reserve_us = 0;
    int64_t finalize_us = 0;

    void clear() {
        *this = {};
    }

    void accumulate(const llama_sched_reserve_timing & other) {
        sched_new_us += other.sched_new_us;
        memory_init_us += other.memory_init_us;
        feature_probe_us += other.feature_probe_us;
        plan_reserve_us += other.plan_reserve_us;
        finalize_us += other.finalize_us;
    }

    int64_t accounted_us() const {
        return sched_new_us + memory_init_us + feature_probe_us + plan_reserve_us + finalize_us;
    }
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;
    const std::vector<ggml_backend_t> & get_backend_ptrs() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    float * get_embeddings_pre_norm();
    float * get_embeddings_pre_norm_ith(int32_t i);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_embeddings_pre_norm(bool value, bool masked);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);
    ggml_status graph_compute(ggml_cgraph * gf, const llama_ubatch & ubatch, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

    // Internal workflow2 / dynamic-stage-scheduling hook.
    // The current implementation only updates routing plans that are compatible
    // with the already-allocated KV contract; incompatible plans require
    // context rebuild or future KV migration support.
    bool set_hetero_plan(llama_hetero_execution_plan plan);
    const llama_hetero_execution_plan & get_hetero_plan() const;
    bool set_dynamic_route_config(const llama_dynamic_route_config & config);
    std::string get_dynamic_route_mode() const;

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    bool apply_hetero_plan(llama_hetero_execution_plan plan, bool update_base_plan, const char * source);
    bool ensure_hetero_backend_ready(const std::string & backend_name, const char * route_name);
    bool ensure_hetero_backends_for_route(const llama_hetero_route_spec & route, const char * label_prefix);
    bool ensure_dynamic_route_backends_ready(const llama_dynamic_route_runtime_config & config);
    bool backend_available_for_route(const std::string & backend_name) const;
    ggml_backend_t find_backend_for_route(const std::string & backend_name) const;
    void maybe_prewarm_dynamic_qnn_opencl_kv_aliases();
    bool sync_dynamic_cpu_opencl_kv(
            bool host_to_device,
            llama_opencl_external_host_sync_timing * timing = nullptr);
    bool rebuild_dynamic_consumer_kv_from_state(
            const std::string & producer_backend,
            const std::string & consumer_backend,
            const char * reason);
    void maybe_debug_dump_powerserve_prefix_before_qnn_switch();
    bool migrate_dynamic_cpu_opencl_kv(const std::string & producer_backend, const std::string & consumer_backend);
    void validate_dynamic_seq0_token_history();
    void record_dynamic_seq0_token_history(const llama_batch & batch_inp, size_t prefix_tokens_before_decode);
    bool replay_dynamic_qnn_prefix();
    void maybe_apply_dynamic_route(uint32_t n_tokens);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;
    ggml_type kv_type_k = GGML_TYPE_F16;
    ggml_type kv_type_v = GGML_TYPE_F16;
    bool kv_swa_full = false;
    bool kv_attn_v_trans = true;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    // hidden state before the final output norm (2-dimensional array: [n_outputs][n_embd])
    // populated only when cparams.embeddings_pre_norm is enabled and the model graph
    // sets llm_graph_result::t_h_pre_norm
    buffer_view<float> embd_pre_norm = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;
    ggml_backend_sched_ptr aot_saved_sched;

    bool sched_need_reserve = true;
    uint32_t sched_reserve_request_tokens = 0;
    std::vector<llama_hetero_execution_plan> hetero_dynamic_pre_reserved_plans;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // Force the next graph build onto the non-QNN fallback path used by the
    // AoT bootstrap correction after seeding the initial token. When all model
    // weights remain on CPU this becomes a CPU-only correction graph; when
    // weights are already pre-allocated on another backend, the correction
    // keeps those backends alive and only reroutes QNN-owned stages.
    bool aot_force_cpu_graph = false;
    bool aot_bootstrap_cpu_sched_active = false;
    bool aot_active_route_requests_qnn = false;
    bool aot_skip_bootstrap_for_next_decode = false;
    std::vector<llama_token> dynamic_seq0_token_history;
    llama_hetero_execution_plan qnn_prefix_replay_restore_plan;
    bool qnn_prefix_replay_restore_plan_valid = false;
    bool qnn_prefix_replay_pending = false;
    bool qnn_prefix_replay_active = false;
    bool qnn_prefix_replay_rebuild_live_memory = false;

    llama_hetero_execution_plan hetero_plan;
    llama_hetero_execution_plan hetero_plan_base;
    llama_hetero_kv_contract    hetero_kv_contract_allocated;
    llama_dynamic_route_runtime_config dynamic_route_config;
    llama_dynamic_route_runtime_state  dynamic_route_state;

    struct hetero_phase_timing_trace {
        bool active = false;
        bool route_applied = false;
        bool route_noop = false;
        bool bootstrap_ran = false;

        uint32_t n_tokens = 0;

        int64_t batch_start_us = 0;
        int64_t route_decide_us = 0;
        int64_t route_apply_us = 0;
        int64_t reserve_us = 0;
        int64_t reserve_sched_new_us = 0;
        int64_t reserve_memory_init_us = 0;
        int64_t reserve_feature_probe_us = 0;
        int64_t reserve_plan_reserve_us = 0;
        int64_t reserve_finalize_us = 0;
        int64_t memory_update_us = 0;
        int64_t kv_migration_us = 0;
        int64_t kv_alias_us = 0;
        int64_t kv_backend_sync_us = 0;
        int64_t kv_transfer_us = 0;
        int64_t process_ubatch_us = 0;
        int64_t bootstrap_sync_us = 0;
        int64_t bootstrap_sched_rebuild_us = 0;

        int32_t process_ubatches = 0;
        int32_t graph_runs_reused = 0;
        int32_t graph_runs_rebuilt = 0;

        std::string route_label;
        std::string route_reason;
        std::string target_route;

        void reset() {
            *this = {};
        }
    };

    hetero_phase_timing_trace hetero_phase_trace;
    bool hetero_phase_trace_suppress_sync_log = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
