#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-hexagon.h"
#include "ggml-opencl.h"

#include "bench-common.h"
#include "op-interface.h"
#include "op-mul-mat.h"
#include "op-rms-norm.h"
#include "op-ffn.h"
#include "op-swiglu.h"
#include "op-soft-max.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>

// #define GGML_OPENCL_COMPUTE_STAT

// Parse type string to ggml_type
// Supports: fp32/f32, fp16/f16, q8_0/int8/i8
static ggml_type parse_type(const std::string& type_str) {
    std::string s = type_str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    
    if (s == "fp32" || s == "f32" || s == "float32") {
        return GGML_TYPE_F32;
    } else if (s == "fp16" || s == "f16" || s == "float16") {
        return GGML_TYPE_F16;
    } else if (s == "q8_0" || s == "int8" || s == "i8") {
        return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_COUNT;  // Invalid type
}

// Get type name string
static const char* type_name(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q8_0: return "Q8_0";
        default:             return "UNKNOWN";
    }
}

struct options {
    std::string op = "mul_mat";
    int64_t m = 2048;
    int64_t k = 2048;
    int64_t n = 1;
    int     runs = 50;
    int     n_threads = 4;  // CPU backend thread count
    std::vector<std::string> backends = {"HTP0", "GPUOpenCL", "CPU"};
    ggml_type weight_type = GGML_TYPE_Q8_0;  // Weight matrix type (for mul_mat, ffn)
    ggml_type input_type = GGML_TYPE_F32;    // Input tensor type
};

static void print_usage(const char * argv0) {
    std::printf("Usage: %s [--op NAME] [--backend NAME] [--m M] [--k K] [--n N] [--runs R] [--threads T] [--wtype TYPE] [--itype TYPE]\n", argv0);
    std::printf("Operators:\n");
    std::printf("  mul_mat   - Matrix multiplication (weight × input)\n");
    std::printf("  rms_norm  - RMS normalization\n");
    std::printf("  swiglu    - SiLU activation: y = x * sigmoid(x)\n");
    std::printf("  ffn       - FFN fused: MUL_MAT(Gate/Up) → SWIGLU → MUL_MAT(Down)\n");
    std::printf("  soft_max  - Soft max with optional mask (attention weights)\n");
    std::printf("Type options:\n");
    std::printf("  --wtype TYPE - Weight matrix type (default: q8_0)\n");
    std::printf("  --itype TYPE - Input tensor type (default: f32)\n");
    std::printf("  --type TYPE  - Set both weight and input type to the same value\n");
    std::printf("  Supported types: fp32/f32, fp16/f16, q8_0/int8/i8\n");
    std::printf("Other options:\n");
    std::printf("  --threads T  - Number of threads for CPU backend (default: 4)\n");
    std::printf("Defaults: --op mul_mat --backend HTP0 --backend GPUOpenCL --backend CPU --m 2048 --k 2048 --n 1 --runs 50 --threads 4 --wtype q8_0 --itype f32\n");
    std::printf("Examples:\n");
    std::printf("  %s --op mul_mat --wtype fp16 --itype fp16\n", argv0);
    std::printf("  %s --op mul_mat --wtype q8_0 --itype f32 --backend HTP0\n", argv0);
    std::printf("  %s --op ffn --type fp32 --backend CPU\n", argv0);
}

static std::unique_ptr<OpInterface> create_operator(const options& opt) {
    if (opt.op == "mul_mat") {
        return std::make_unique<OpMulMat>(opt.m, opt.k, opt.n, opt.weight_type, opt.input_type);
    } else if (opt.op == "rms_norm") {
        return std::make_unique<OpRmsNorm>(opt.m, opt.n);
    } else if (opt.op == "swiglu") {
        // Pure SiLU: y = x * sigmoid(x)
        // Tests only the SiLU activation function
        return std::make_unique<OpSwiGLU>(opt.m, opt.n);
    } else if (opt.op == "ffn") {
        // FFN: up to m=6144, then down to 2048
        // For standard Llama-3.2-1B: k=2048, m=6144, down_m=2048
        int64_t down_m = 2048;  // down projection output dimension
        return std::make_unique<OpFFN>(opt.m, opt.k, opt.n, down_m);
    } else if (opt.op == "soft_max") {
        // Soft max with mask for attention mechanism
        // Default config based on kq_soft_max-5 from profiling:
        // input: [256, 1, 16, 1], mask: [256, 64, 1, 1]
        // Using m for ne0 (seq_len), k for mask_ne1 (kv_len), n for ne2 (n_heads)
        int64_t ne0 = opt.m;       // seq_len (default 256)
        int64_t ne1 = 1;           // always 1 for attention
        int64_t ne2 = opt.n;       // n_heads (default 16)
        int64_t ne3 = 1;           // batch size
        int64_t mask_ne0 = opt.m;  // seq_len
        int64_t mask_ne1 = opt.k;  // kv_len (default 64)
        float scale = 1.0f;        // scale factor
        return std::make_unique<OpSoftMax>(ne0, ne1, ne2, ne3, mask_ne0, mask_ne1, scale);
    }
    return std::make_unique<OpMulMat>(opt.m, opt.k, opt.n, opt.weight_type, opt.input_type);  // default
}

static std::string canonical_backend_name(std::string backend_name) {
    std::transform(backend_name.begin(), backend_name.end(), backend_name.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });

    if (backend_name == "htp0" || backend_name == "htp" || backend_name == "npu" || backend_name == "qnn" || backend_name == "qnn-npu") {
        return "qnn-npu";
    }
    if (backend_name == "opencl0" || backend_name == "opencl" || backend_name == "gpuopencl" || backend_name == "gpu") {
        return "GPUOpenCL";
    }
    if (backend_name == "cpu") {
        return "CPU";
    }

    return backend_name;
}

static bench_result run_once(const options & opt, const std::string & backend_name) {
    bench_result r;
    r.backend = backend_name;

    const std::string canonical_backend = canonical_backend_name(backend_name);
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(canonical_backend.c_str());
    if (!dev) {
        r.note = "device not found";
        return r;
    }

    ResourceGuard guard;
    guard.backend = ggml_backend_dev_init(dev, nullptr);
    if (!guard.backend) {
        r.note = "backend init failed";
        return r;
    }

    // Set thread count for CPU backend
    if (ggml_backend_is_cpu(guard.backend)) {
        ggml_backend_cpu_set_n_threads(guard.backend, opt.n_threads);
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 32u * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    guard.ctx = ggml_init(params);
    if (!guard.ctx) {
        r.note = "ggml_init failed";
        return r;
    }

    // Create operator
    auto op = create_operator(opt);
    op->create_tensors(guard.ctx);
    ggml_tensor* y = op->create_graph(guard.ctx);

    if (!y) {
        r.note = "op creation failed";
        return r;
    }

    ggml_cgraph * graph = ggml_new_graph(guard.ctx);
    ggml_build_forward_expand(graph, y);

    // Check if all operations in the graph are supported by the backend
    int n_nodes = ggml_graph_n_nodes(graph);
    for (int i = 0; i < n_nodes; ++i) {
        ggml_tensor* node = ggml_graph_node(graph, i);
        if (!ggml_backend_dev_supports_op(dev, node)) {
            r.note = std::string("op ") + ggml_op_name(node->op) + " not supported";
            return r;
        }
    }

    // ggml_backend_sched_new requires CPU backend as last element
    ggml_backend_t backends[2] = { nullptr, nullptr };
    int n_backends = 0;

    if (ggml_backend_is_cpu(guard.backend)) {
        backends[n_backends++] = guard.backend;
    } else {
        guard.cpu_backend = ggml_backend_cpu_init();
        if (!guard.cpu_backend) {
            r.note = "cpu backend init failed";
            return r;
        }
        backends[n_backends++] = guard.backend;
        backends[n_backends++] = guard.cpu_backend;
    }

    guard.sched = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, /*parallel=*/false, /*op_offload=*/true);

    if (!ggml_backend_sched_alloc_graph(guard.sched, graph)) {
        r.note = "allocation failed";
        return r;
    }

    // Fill input tensors
    op->fill_inputs();

    // Measure first run (cold start)
    ggml_time_init();
    int64_t t0 = ggml_time_us();
    if (ggml_backend_sched_graph_compute(guard.sched, graph) != GGML_STATUS_SUCCESS) {
        r.note = "compute failed";
        return r;
    }
    int64_t t1 = ggml_time_us();
    r.first_us = double(t1 - t0);

    // Verify result
    if (!verify_result(op->get_output())) {
        r.note = "result contains NaN/Inf";
        return r;
    }

    // Measure stable state (remaining runs)
    double total_us = 0.0;
    int stable_runs = 0;

    for (int i = 1; i < opt.runs; ++i) {
        t0 = ggml_time_us();
        if (ggml_backend_sched_graph_compute(guard.sched, graph) != GGML_STATUS_SUCCESS) {
            r.note = "compute failed";
            return r;
        }
        int64_t ti = ggml_time_us();
        double elapsed = double(ti - t0);
        total_us += elapsed;
        stable_runs++;
        r.min_us = std::min(r.min_us, elapsed);
        r.max_us = std::max(r.max_us, elapsed);
    }

    if (stable_runs > 0) {
        r.avg_us = total_us / stable_runs;
    } else {
        r.avg_us = r.first_us;
        r.min_us = r.first_us;
        r.max_us = r.first_us;
    }

    r.ok = true;

    return r;
}

static options parse(int argc, char ** argv) {
    options opt;
    opt.backends.clear();

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (a == "--op" && i + 1 < argc) {
            opt.op = argv[++i];
        } else if (a == "--backend" && i + 1 < argc) {
            opt.backends.push_back(argv[++i]);
        } else if (a == "--m" && i + 1 < argc) {
            opt.m = std::atoll(argv[++i]);
        } else if (a == "--k" && i + 1 < argc) {
            opt.k = std::atoll(argv[++i]);
        } else if (a == "--n" && i + 1 < argc) {
            opt.n = std::atoll(argv[++i]);
        } else if (a == "--runs" && i + 1 < argc) {
            opt.runs = std::atoi(argv[++i]);
        } else if ((a == "--threads" || a == "-t") && i + 1 < argc) {
            int threads = std::atoi(argv[++i]);
            if (threads <= 0) {
                std::fprintf(stderr, "Error: Thread count must be a positive integer, got '%s'\n", argv[i]);
                print_usage(argv[0]);
                std::exit(1);
            }
            opt.n_threads = threads;
        } else if ((a == "--wtype" || a == "--weight-type") && i + 1 < argc) {
            ggml_type t = parse_type(argv[++i]);
            if (t == GGML_TYPE_COUNT) {
                std::fprintf(stderr, "Error: Invalid weight type '%s'\n", argv[i]);
                print_usage(argv[0]);
                std::exit(1);
            }
            opt.weight_type = t;
        } else if ((a == "--itype" || a == "--input-type") && i + 1 < argc) {
            ggml_type t = parse_type(argv[++i]);
            if (t == GGML_TYPE_COUNT) {
                std::fprintf(stderr, "Error: Invalid input type '%s'\n", argv[i]);
                print_usage(argv[0]);
                std::exit(1);
            }
            opt.input_type = t;
        } else if (a == "--type" && i + 1 < argc) {
            // Shorthand: set both weight and input type to the same value
            ggml_type t = parse_type(argv[++i]);
            if (t == GGML_TYPE_COUNT) {
                std::fprintf(stderr, "Error: Invalid type '%s'\n", argv[i]);
                print_usage(argv[0]);
                std::exit(1);
            }
            opt.weight_type = t;
            opt.input_type = t;
        } else {
            print_usage(argv[0]);
            std::exit(1);
        }
    }

    if (opt.backends.empty()) {
        opt.backends = {"HTP0", "GPUOpenCL", "CPU"};
    }
    return opt;
}

int main(int argc, char ** argv) {
    const auto opt = parse(argc, argv);
    auto op = create_operator(opt);

    std::printf("%s (threads=%d, wtype=%s, itype=%s)\n",
                op->description().c_str(), opt.n_threads,
                type_name(opt.weight_type), type_name(opt.input_type));
    std::printf("%-12s %-12s %-12s %-12s %-12s %s\n", "backend", "cold us", "avg us", "min us", "max us", "note");

    for (const auto & name : opt.backends) {
        auto r = run_once(opt, name);
        if (r.ok) {
            std::printf("%-12s %-12.2f %-12.2f %-12.2f %-12.2f %s\n",
                       name.c_str(), r.first_us, r.avg_us, r.min_us, r.max_us, "");
        } else {
            std::printf("%-12s %-12s %-12s %-12s %-12s %s\n",
                       name.c_str(), "-", "-", "-", "-", r.note.c_str());
        }
    }

    return 0;
}
