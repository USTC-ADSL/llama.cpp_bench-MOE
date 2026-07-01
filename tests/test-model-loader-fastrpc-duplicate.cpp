#include "../src/llama-model-loader.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"
#include "testing.h"

#include <cstring>

ggml_backend_buffer_type_t llama_model_loader_select_weight_backend_buft_from_list(
        const llama_hparams & hparams,
        ggml_tensor * tensor,
        ggml_op op,
        const buft_list_t * buft_list,
        const char * backend_name);

ggml_backend_buffer_type_t llama_model_loader_select_weight_device_buft(
        const llama_hparams & hparams,
        ggml_tensor * tensor,
        ggml_op op,
        const char * device_name,
        bool use_host_buft);

ggml_backend_buffer_type_t llama_model_loader_select_fastrpc_weight_duplicate_buft(
        const llama_hparams & hparams,
        ggml_tensor * tensor,
        ggml_op op,
        const buft_list_t * buft_list_layer,
        const char * device_name,
        bool allow_cpu_fallback);

bool llama_model_loader_should_replace_fastrpc_opencl_weight_duplicate(
        const char * backend,
        ggml_backend_buffer_type_t existing_buft,
        ggml_backend_buffer_type_t candidate_buft);

bool llama_model_loader_should_enable_fastrpc_opencl_weight_duplicate(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route);

bool llama_model_loader_should_enable_fastrpc_opencl_dual_residency(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route);

bool llama_model_loader_should_enable_cpu_fastrpc_dual_residency(
        const llama_hetero_route_spec & dynamic_prefill_route,
        const llama_hetero_route_spec & dynamic_decode_route);

bool llama_model_loader_weight_route_stage(
        llm_tensor tn_tensor,
        llm_tensor_layer layer,
        llama_hetero_route_stage & stage);

bool llama_model_loader_fastrpc_opencl_dual_residency_op_stage(
        llm_tensor tn_tensor,
        const char * suffix,
        int flags,
        ggml_op & op,
        llama_hetero_route_stage & stage);

bool llama_model_loader_opencl_cpu_extra_cpu_copy_op_stage(
        llm_tensor tn_tensor,
        const char * suffix,
        int flags,
        ggml_op & op,
        llama_hetero_route_stage & stage);

const ggml_tensor * llama_model_resolve_weight_for_fastrpc_duplicate(
        const ggml_tensor * original,
        const ggml_tensor * fastrpc_copy,
        llama_hetero_route_stage stage,
        const llama_hetero_route_spec & route);

const ggml_tensor * llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
        const ggml_tensor * original,
        const ggml_tensor * opencl_copy,
        const ggml_tensor * fastrpc_copy,
        llama_hetero_route_stage stage,
        const llama_hetero_route_spec & route);

const ggml_tensor * llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
        const ggml_tensor * original,
        const ggml_tensor * cpu_copy,
        const ggml_tensor * fastrpc_copy,
        llama_hetero_route_stage stage,
        const llama_hetero_route_spec & route);

namespace {

const char * test_fastrpc_dev_name = "TestHTP0";
bool test_fastrpc_repack_support_enabled = true;

const char * test_fastrpc_buft_name(ggml_backend_buffer_type_t buft);
size_t test_fastrpc_buft_alignment(ggml_backend_buffer_type_t buft);
ggml_backend_dev_t test_fastrpc_dev();
ggml_backend_buffer_type_t test_fastrpc_default_buft();
ggml_backend_buffer_type_t test_fastrpc_repack_buft();
ggml_backend_buffer_type_t * test_fastrpc_extra_bufts(ggml_backend_dev_t dev);
ggml_backend_buffer_type_t test_opencl_host_buft();

const char * test_fastrpc_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return test_fastrpc_dev_name;
}

const char * test_fastrpc_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "test FastRPC device";
}

enum ggml_backend_dev_type test_fastrpc_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

ggml_backend_buffer_type_t test_fastrpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return test_fastrpc_default_buft();
}

bool test_fastrpc_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    const ggml_tensor * src0 = op != nullptr ? op->src[0] : nullptr;
    return op != nullptr &&
           op->op == GGML_OP_MUL_MAT &&
           src0 != nullptr &&
           src0->buffer != nullptr &&
           test_fastrpc_repack_support_enabled &&
           ggml_backend_buffer_get_type(src0->buffer) == test_fastrpc_repack_buft();
}

bool test_fastrpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return buft == test_fastrpc_default_buft() || buft == test_fastrpc_repack_buft();
}

const char * test_fastrpc_buft_name(ggml_backend_buffer_type_t buft) {
    if (buft == test_fastrpc_repack_buft()) {
        return "TestHTP0-Repack";
    }
    if (buft == test_fastrpc_default_buft()) {
        return "TestHTP0";
    }
    return "TestHTP0-Unknown";
}

size_t test_fastrpc_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

ggml_backend_buffer_type_t test_fastrpc_default_buft() {
    static ggml_backend_buffer_type_i iface = {
        /* .get_name       = */ test_fastrpc_buft_name,
        /* .alloc_buffer   = */ nullptr,
        /* .get_alignment  = */ test_fastrpc_buft_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ nullptr,
    };
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ iface,
        /* .device  = */ test_fastrpc_dev(),
        /* .context = */ nullptr,
    };
    return &buft;
}

ggml_backend_buffer_type_t test_fastrpc_repack_buft() {
    static ggml_backend_buffer_type_i iface = {
        /* .get_name       = */ test_fastrpc_buft_name,
        /* .alloc_buffer   = */ nullptr,
        /* .get_alignment  = */ test_fastrpc_buft_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ nullptr,
    };
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ iface,
        /* .device  = */ test_fastrpc_dev(),
        /* .context = */ nullptr,
    };
    return &buft;
}

ggml_backend_buffer_type_t * test_fastrpc_extra_bufts(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    static ggml_backend_buffer_type_t bufts[] = {
        test_fastrpc_repack_buft(),
        nullptr,
    };
    return bufts;
}

ggml_backend_dev_t test_fastrpc_dev() {
    static ggml_backend_device_i iface = {
        /* .get_name              = */ test_fastrpc_device_get_name,
        /* .get_description       = */ test_fastrpc_device_get_description,
        /* .get_memory            = */ nullptr,
        /* .get_type              = */ test_fastrpc_device_get_type,
        /* .get_props             = */ nullptr,
        /* .init_backend          = */ nullptr,
        /* .get_buffer_type       = */ test_fastrpc_device_get_buffer_type,
        /* .get_host_buffer_type  = */ nullptr,
        /* .buffer_from_host_ptr  = */ nullptr,
        /* .supports_op           = */ test_fastrpc_device_supports_op,
        /* .supports_buft         = */ test_fastrpc_device_supports_buft,
        /* .offload_op            = */ nullptr,
        /* .event_new             = */ nullptr,
        /* .event_free            = */ nullptr,
        /* .event_synchronize     = */ nullptr,
    };
    static ggml_backend_device dev = {
        /* .iface   = */ iface,
        /* .reg     = */ nullptr,
        /* .context = */ nullptr,
    };
    return &dev;
}

const char * test_fastrpc_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "TestFastRPC";
}

size_t test_fastrpc_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

ggml_backend_dev_t test_fastrpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    return index == 0 ? test_fastrpc_dev() : nullptr;
}

void * test_fastrpc_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (std::strcmp(name, "ggml_backend_dev_get_extra_bufts") == 0) {
        return reinterpret_cast<void *>(test_fastrpc_extra_bufts);
    }
    return nullptr;
}

void register_test_fastrpc_backend() {
    static ggml_backend_reg_i iface = {
        /* .get_name         = */ test_fastrpc_reg_get_name,
        /* .get_device_count = */ test_fastrpc_reg_get_device_count,
        /* .get_device       = */ test_fastrpc_reg_get_device,
        /* .get_proc_address = */ test_fastrpc_reg_get_proc_address,
    };
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ iface,
        /* .context     = */ nullptr,
    };
    static bool registered = false;
    if (!registered) {
        test_fastrpc_dev()->reg = &reg;
        ggml_backend_register(&reg);
        registered = true;
    }
}

ggml_tensor * new_test_fastrpc_weight(ggml_context * ctx) {
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 32, 16);
    ggml_set_name(weight, "blk.0.attn_q.weight");
    return weight;
}

const char * test_opencl_host_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "OpenCL_Host";
}

size_t test_opencl_host_buft_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 128;
}

ggml_backend_buffer_type_t test_opencl_host_buft() {
    static ggml_backend_buffer_type_i iface = {
        /* .get_name       = */ test_opencl_host_buft_name,
        /* .alloc_buffer   = */ nullptr,
        /* .get_alignment  = */ test_opencl_host_buft_alignment,
        /* .get_max_size   = */ nullptr,
        /* .get_alloc_size = */ nullptr,
        /* .is_host        = */ nullptr,
    };
    static ggml_backend_buffer_type buft = {
        /* .iface   = */ iface,
        /* .device  = */ nullptr,
        /* .context = */ nullptr,
    };
    return &buft;
}

}

int main() {
    testing t;

    t.test("FastRPC duplicate selection scans same-device repack bufts", [](testing & t) {
        register_test_fastrpc_backend();

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*4,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weight = new_test_fastrpc_weight(ctx);
        llama_hparams hparams = {};

        buft_list_t bufts = {
            { test_fastrpc_dev(), test_fastrpc_default_buft() },
            { test_fastrpc_dev(), test_fastrpc_repack_buft()  },
        };

        t.assert_true(
                "selector should continue past unsupported default HTP0 buft and choose the repack buft",
                llama_model_loader_select_weight_backend_buft_from_list(
                        hparams, weight, GGML_OP_MUL_MAT, &bufts, test_fastrpc_dev_name) == test_fastrpc_repack_buft());

        ggml_free(ctx);
    });

    t.test("FastRPC duplicate fallback tries device extra bufts", [](testing & t) {
        register_test_fastrpc_backend();
        test_fastrpc_repack_support_enabled = true;

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*4,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weight = new_test_fastrpc_weight(ctx);
        llama_hparams hparams = {};

        t.assert_true(
                "named-device fallback should choose the HTP0 repack extra buft when the default buft rejects MUL_MAT",
                llama_model_loader_select_weight_device_buft(
                        hparams, weight, GGML_OP_MUL_MAT, test_fastrpc_dev_name, /* use_host_buft = */ false) == test_fastrpc_repack_buft());

        ggml_free(ctx);
    });

    t.test("FastRPC duplicate fallback accepts canonical backend name", [](testing & t) {
        register_test_fastrpc_backend();
        test_fastrpc_repack_support_enabled = true;

        const char * saved_dev_name = test_fastrpc_dev_name;
        test_fastrpc_dev_name = "HTP0";

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*4,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weight = new_test_fastrpc_weight(ctx);
        llama_hparams hparams = {};

        t.assert_true(
                "named-device fallback should find HTP0 when the caller asks for canonical FastRPC",
                llama_model_loader_select_weight_device_buft(
                        hparams, weight, GGML_OP_MUL_MAT, "fastrpc", /* use_host_buft = */ false) == test_fastrpc_repack_buft());

        test_fastrpc_dev_name = saved_dev_name;
        ggml_free(ctx);
    });

    t.test("FastRPC output duplicate can fall back to CPU when HTP0 rejects lm-head", [](testing & t) {
        register_test_fastrpc_backend();
        test_fastrpc_repack_support_enabled = false;

        ggml_init_params params = {
            /* .mem_size   = */ ggml_tensor_overhead()*4,
            /* .mem_buffer = */ nullptr,
            /* .no_alloc   = */ true,
        };
        ggml_context * ctx = ggml_init(params);
        ggml_tensor * weight = new_test_fastrpc_weight(ctx);
        llama_hparams hparams = {};

        buft_list_t bufts = {
            { test_fastrpc_dev(), test_fastrpc_default_buft() },
            { test_fastrpc_dev(), test_fastrpc_repack_buft()  },
        };

        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev) : nullptr;

        t.assert_true(
                "when HTP0 has no supported duplicate buft, output-stage FastRPC duplicate should use CPU instead of leaking OpenCL_Host",
                llama_model_loader_select_fastrpc_weight_duplicate_buft(
                        hparams, weight, GGML_OP_MUL_MAT, &bufts, test_fastrpc_dev_name, /* allow_cpu_fallback = */ true) == cpu_buft);

        test_fastrpc_repack_support_enabled = true;
        ggml_free(ctx);
    });

    t.test("OpenCL duplicate replaces earlier CPU fallback for tied token embedding", [](testing & t) {
        ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_buffer_type_t cpu_buft = cpu_dev != nullptr ? ggml_backend_dev_buffer_type(cpu_dev) : nullptr;

        t.assert_true(
                "a later OpenCL-capable output duplicate should replace an earlier CPU fallback stored under the same tied tensor name",
                llama_model_loader_should_replace_fastrpc_opencl_weight_duplicate(
                        "opencl", cpu_buft, test_opencl_host_buft()));

        t.assert_true(
                "an existing OpenCL duplicate should not be replaced by a CPU fallback",
                !llama_model_loader_should_replace_fastrpc_opencl_weight_duplicate(
                        "opencl", test_opencl_host_buft(), cpu_buft));
    });

    t.test("OpenCL and FastRPC dynamic routes request dual weight residency", [](testing & t) {
        t.assert_true(
                "OpenCL prefill -> FastRPC decode should keep FastRPC duplicates beside OpenCL primary weights",
                llama_model_loader_should_enable_fastrpc_opencl_weight_duplicate(
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("fastrpc")));
        t.assert_true(
                "OpenCL prefill -> FastRPC decode should request full OpenCL/FastRPC dual residency",
                llama_model_loader_should_enable_fastrpc_opencl_dual_residency(
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("fastrpc")));

        t.assert_true(
                "FastRPC prefill -> OpenCL decode should keep FastRPC duplicates even when OpenCL primary weights are preferred",
                llama_model_loader_should_enable_fastrpc_opencl_weight_duplicate(
                        llama_hetero_parse_route_spec("fastrpc"),
                        llama_hetero_parse_route_spec("opencl")));
        t.assert_true(
                "FastRPC prefill -> OpenCL decode should request full OpenCL/FastRPC dual residency",
                llama_model_loader_should_enable_fastrpc_opencl_dual_residency(
                        llama_hetero_parse_route_spec("fastrpc"),
                        llama_hetero_parse_route_spec("opencl")));
    });

    t.test("CPU and FastRPC dynamic routes request separate CPU FastRPC residency", [](testing & t) {
        t.assert_true(
                "CPU -> FastRPC switching should not be treated as an OpenCL/FastRPC dual-residency request",
                !llama_model_loader_should_enable_fastrpc_opencl_weight_duplicate(
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("fastrpc")));
        t.assert_true(
                "CPU -> FastRPC switching should request CPU/FastRPC dual residency",
                llama_model_loader_should_enable_cpu_fastrpc_dual_residency(
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("fastrpc")));
        t.assert_true(
                "FastRPC -> CPU switching should request CPU/FastRPC dual residency",
                llama_model_loader_should_enable_cpu_fastrpc_dual_residency(
                        llama_hetero_parse_route_spec("fastrpc"),
                        llama_hetero_parse_route_spec("cpu")));
        t.assert_true(
                "OpenCL -> QNN switching should keep using the QNN/OpenCL residency path, not FastRPC dual residency",
                !llama_model_loader_should_enable_fastrpc_opencl_dual_residency(
                        llama_hetero_parse_route_spec("opencl"),
                        llama_hetero_parse_route_spec("qnn-npu")));
        t.assert_true(
                "CPU -> OpenCL switching should keep using the CPU/OpenCL extra-copy path, not FastRPC dual residency",
                !llama_model_loader_should_enable_fastrpc_opencl_dual_residency(
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("opencl")));
        t.assert_true(
                "CPU -> OpenCL switching should not request CPU/FastRPC residency",
                !llama_model_loader_should_enable_cpu_fastrpc_dual_residency(
                        llama_hetero_parse_route_spec("cpu"),
                        llama_hetero_parse_route_spec("opencl")));
    });

    t.test("FastRPC route stages resolve to the FastRPC duplicate", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor fastrpc_copy = {};

        t.assert_true(
                "FastRPC FFN stages should consume the FastRPC-resident duplicate when it exists",
                llama_model_resolve_weight_for_fastrpc_duplicate(
                        &original,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("HTP0")) == &fastrpc_copy);
    });

    t.test("OpenCL route stages keep the OpenCL primary weight", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor fastrpc_copy = {};

        t.assert_true(
                "OpenCL FFN stages should keep the primary OpenCL-resident weight",
                llama_model_resolve_weight_for_fastrpc_duplicate(
                        &original,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("opencl")) == &original);
    });

    t.test("per-stage routes only switch stages that run on FastRPC", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor fastrpc_copy = {};
        llama_hetero_route_spec route = {};
        route.attn = "opencl";
        route.ffn = "fastrpc";
        route.output = "opencl";

        t.assert_true(
                "FFN should use the FastRPC duplicate when only FFN is routed to FastRPC",
                llama_model_resolve_weight_for_fastrpc_duplicate(
                        &original,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        route) == &fastrpc_copy);
        t.assert_true(
                "output should keep the original weight when output remains on OpenCL",
                llama_model_resolve_weight_for_fastrpc_duplicate(
                        &original,
                        &fastrpc_copy,
                        llama_hetero_route_stage::OUTPUT,
                        route) == &original);
    });

    t.test("dual residency resolver selects OpenCL and FastRPC copies by route stage", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor opencl_copy = {};
        ggml_tensor fastrpc_copy = {};

        t.assert_true(
                "OpenCL FFN stages should consume the OpenCL-resident copy even when the original is not OpenCL",
                llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
                        &original,
                        &opencl_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("opencl")) == &opencl_copy);

        t.assert_true(
                "FastRPC FFN stages should consume the FastRPC-resident copy",
                llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
                        &original,
                        &opencl_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("fastrpc")) == &fastrpc_copy);

        t.assert_true(
                "CPU stages should keep the original weight so CPU/OpenCL fallback logic can handle them separately",
                llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
                        &original,
                        &opencl_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("cpu")) == &original);

        llama_hetero_route_spec mixed_route = {};
        mixed_route.attn = "opencl";
        mixed_route.ffn = "fastrpc";
        mixed_route.output = "opencl";
        t.assert_true(
                "per-stage dual resolver should choose the FastRPC copy only for FastRPC-owned stages",
                llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
                        &original,
                        &opencl_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        mixed_route) == &fastrpc_copy);
        t.assert_true(
                "per-stage dual resolver should choose the OpenCL copy for OpenCL-owned stages",
                llama_model_resolve_weight_for_fastrpc_opencl_dual_residency(
                        &original,
                        &opencl_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::OUTPUT,
                        mixed_route) == &opencl_copy);
    });

    t.test("CPU FastRPC dual residency resolver selects CPU and FastRPC copies by route stage", [](testing & t) {
        ggml_tensor original = {};
        ggml_tensor cpu_copy = {};
        ggml_tensor fastrpc_copy = {};

        t.assert_true(
                "CPU FFN stages should consume the CPU-resident copy",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("cpu")) == &cpu_copy);

        t.assert_true(
                "FastRPC FFN stages should consume the FastRPC-resident copy",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("HTP0")) == &fastrpc_copy);

        t.assert_true(
                "OpenCL stages should not accidentally consume CPU/FastRPC dual-residency copies",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        llama_hetero_parse_route_spec("opencl")) == &original);

        llama_hetero_route_spec mixed_route = {};
        mixed_route.attn = "cpu";
        mixed_route.ffn = "fastrpc";
        mixed_route.output = "opencl";
        t.assert_true(
                "per-stage CPU/FastRPC resolver should choose the FastRPC copy only for FastRPC-owned stages",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::FFN,
                        mixed_route) == &fastrpc_copy);
        t.assert_true(
                "per-stage CPU/FastRPC resolver should choose the CPU copy only for CPU-owned stages",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::ATTN_PROJ,
                        mixed_route) == &cpu_copy);
        t.assert_true(
                "per-stage CPU/FastRPC resolver should keep the original for unrelated OpenCL stages",
                llama_model_resolve_weight_for_cpu_fastrpc_dual_residency(
                        &original,
                        &cpu_copy,
                        &fastrpc_copy,
                        llama_hetero_route_stage::OUTPUT,
                        mixed_route) == &original);
    });

    t.test("FastRPC OpenCL dual residency includes elementwise route weights", [](testing & t) {
        llama_hetero_route_stage stage = llama_hetero_route_stage::OUTPUT;

        t.assert_true(
                "attention norm weights should belong to the attention projection stage",
                llama_model_loader_weight_route_stage(
                        LLM_TENSOR_ATTN_NORM,
                        LLM_TENSOR_LAYER_REPEATING,
                        stage));
        t.assert_equal(
                "attention norm stage",
                int(llama_hetero_route_stage::ATTN_PROJ),
                int(stage));

        t.assert_true(
                "FFN norm weights should belong to the FFN stage",
                llama_model_loader_weight_route_stage(
                        LLM_TENSOR_FFN_NORM,
                        LLM_TENSOR_LAYER_REPEATING,
                        stage));
        t.assert_equal(
                "FFN norm stage",
                int(llama_hetero_route_stage::FFN),
                int(stage));
    });

    t.test("FastRPC OpenCL dual residency prepares norm and bias ops", [](testing & t) {
        ggml_op op = GGML_OP_NONE;
        llama_hetero_route_stage stage = llama_hetero_route_stage::OUTPUT;

        t.assert_true(
                "attention norm scale should get dual residency as an HTP/OpenCL MUL input",
                llama_model_loader_fastrpc_opencl_dual_residency_op_stage(
                        LLM_TENSOR_ATTN_NORM,
                        "weight",
                        0,
                        op,
                        stage));
        t.assert_equal("attention norm op", int(GGML_OP_MUL), int(op));
        t.assert_equal("attention norm stage", int(llama_hetero_route_stage::ATTN_PROJ), int(stage));

        t.assert_true(
                "attention Q bias should get dual residency as an HTP/OpenCL ADD input",
                llama_model_loader_fastrpc_opencl_dual_residency_op_stage(
                        LLM_TENSOR_ATTN_Q,
                        "bias",
                        0,
                        op,
                        stage));
        t.assert_equal("attention Q bias op", int(GGML_OP_ADD), int(op));
        t.assert_equal("attention Q bias stage", int(llama_hetero_route_stage::ATTN_PROJ), int(stage));

        t.assert_true(
                "FFN norm scale should get dual residency as an HTP/OpenCL MUL input",
                llama_model_loader_fastrpc_opencl_dual_residency_op_stage(
                        LLM_TENSOR_FFN_NORM,
                        "weight",
                        0,
                        op,
                        stage));
        t.assert_equal("FFN norm op", int(GGML_OP_MUL), int(op));
        t.assert_equal("FFN norm stage", int(llama_hetero_route_stage::FFN), int(stage));

        t.assert_true(
                "output norm scale should get dual residency as an HTP/OpenCL MUL input",
                llama_model_loader_fastrpc_opencl_dual_residency_op_stage(
                        LLM_TENSOR_OUTPUT_NORM,
                        "weight",
                        0,
                        op,
                        stage));
        t.assert_equal("output norm op", int(GGML_OP_MUL), int(op));
        t.assert_equal("output norm stage", int(llama_hetero_route_stage::OUTPUT), int(stage));
    });

    t.test("CPU OpenCL extra CPU copy prepares norm and bias ops", [](testing & t) {
        ggml_op op = GGML_OP_NONE;
        llama_hetero_route_stage stage = llama_hetero_route_stage::OUTPUT;

        t.assert_true(
                "attention norm scale should get a CPU duplicate for CPU prefill",
                llama_model_loader_opencl_cpu_extra_cpu_copy_op_stage(
                        LLM_TENSOR_ATTN_NORM,
                        "weight",
                        0,
                        op,
                        stage));
        t.assert_equal("attention norm CPU copy op", int(GGML_OP_MUL), int(op));
        t.assert_equal("attention norm CPU copy stage", int(llama_hetero_route_stage::ATTN_PROJ), int(stage));

        t.assert_true(
                "attention Q bias should get a CPU duplicate for CPU prefill",
                llama_model_loader_opencl_cpu_extra_cpu_copy_op_stage(
                        LLM_TENSOR_ATTN_Q,
                        "bias",
                        0,
                        op,
                        stage));
        t.assert_equal("attention Q bias CPU copy op", int(GGML_OP_ADD), int(op));
        t.assert_equal("attention Q bias CPU copy stage", int(llama_hetero_route_stage::ATTN_PROJ), int(stage));

        t.assert_true(
                "output norm scale should get a CPU duplicate for CPU prefill",
                llama_model_loader_opencl_cpu_extra_cpu_copy_op_stage(
                        LLM_TENSOR_OUTPUT_NORM,
                        "weight",
                        0,
                        op,
                        stage));
        t.assert_equal("output norm CPU copy op", int(GGML_OP_MUL), int(op));
        t.assert_equal("output norm CPU copy stage", int(llama_hetero_route_stage::OUTPUT), int(stage));
    });

    return t.summary();
}
