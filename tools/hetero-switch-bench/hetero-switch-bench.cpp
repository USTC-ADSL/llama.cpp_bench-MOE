#include <CL/cl.h>

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef RPCMEM_HEAP_ID_SYSTEM
#define RPCMEM_HEAP_ID_SYSTEM 25
#endif
#ifndef RPCMEM_DEFAULT_FLAGS
#define RPCMEM_DEFAULT_FLAGS 1
#endif

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
    int                 warmup = 5;
    int                 iters  = 50;
    std::vector<size_t> sizes  = {1024ull, 64ull * 1024ull, 1ull * 1024ull * 1024ull, 16ull * 1024ull * 1024ull};
    std::string         csv_path;
};

struct bench_row {
    std::string mode;
    std::string flow;
    size_t      size_bytes;
    int         iter;
    double      latency_us;
    double      throughput_gbps;
    int         valid;
};

static void die(const std::string & msg) {
    throw std::runtime_error(msg);
}

static std::string cl_err_to_string(cl_int err) {
    switch (err) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
        default: {
            std::ostringstream oss;
            oss << "Unknown OpenCL error (" << err << ")";
            return oss.str();
        }
    }
}

static void check_cl(cl_int err, const char * what) {
    if (err != CL_SUCCESS) {
        std::ostringstream oss;
        oss << what << " failed: " << cl_err_to_string(err) << " (" << err << ")";
        die(oss.str());
    }
}

class rpcmem_loader {
  public:
    using rpcmem_init_t   = void (*)();
    using rpcmem_deinit_t = void (*)();
    using rpcmem_alloc_t  = void * (*)(int heapid, uint32_t flags, int size);
    using rpcmem_alloc2_t = void * (*)(int heapid, uint32_t flags, size_t size);
    using rpcmem_free_t   = void (*)(void * p);
    using rpcmem_to_fd_t  = int (*)(void * p);

    rpcmem_loader() {
        const char * candidates[] = {
            "libcdsprpc.so",
            "libadsprpc.so",
        };

        for (const char * lib : candidates) {
            handle_ = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
            if (handle_) {
                lib_name_ = lib;
                break;
            }
        }

        if (!handle_) {
            die("Failed to dlopen libcdsprpc.so/libadsprpc.so; ensure ADSP rpc libs are in LD_LIBRARY_PATH");
        }

        init_    = reinterpret_cast<rpcmem_init_t>(dlsym(handle_, "rpcmem_init"));
        deinit_  = reinterpret_cast<rpcmem_deinit_t>(dlsym(handle_, "rpcmem_deinit"));
        alloc_   = reinterpret_cast<rpcmem_alloc_t>(dlsym(handle_, "rpcmem_alloc"));
        alloc2_  = reinterpret_cast<rpcmem_alloc2_t>(dlsym(handle_, "rpcmem_alloc2"));
        free_    = reinterpret_cast<rpcmem_free_t>(dlsym(handle_, "rpcmem_free"));
        to_fd_   = reinterpret_cast<rpcmem_to_fd_t>(dlsym(handle_, "rpcmem_to_fd"));

        if (!init_ || !deinit_ || !alloc_ || !free_ || !to_fd_) {
            die("Failed to resolve rpcmem symbols from " + lib_name_);
        }

        init_();
    }

    ~rpcmem_loader() {
        if (deinit_) {
            deinit_();
        }
        if (handle_) {
            dlclose(handle_);
        }
    }

    void * alloc(size_t bytes) const {
        if (alloc2_) {
            return alloc2_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, bytes);
        }

        if (bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
            return nullptr;
        }
        return alloc_(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, static_cast<int>(bytes));
    }

    void free(void * p) const {
        free_(p);
    }

    int to_fd(void * p) const {
        return to_fd_(p);
    }

    const std::string & lib_name() const {
        return lib_name_;
    }

  private:
    void *          handle_ = nullptr;
    std::string     lib_name_;
    rpcmem_init_t   init_   = nullptr;
    rpcmem_deinit_t deinit_ = nullptr;
    rpcmem_alloc_t  alloc_  = nullptr;
    rpcmem_alloc2_t alloc2_ = nullptr;
    rpcmem_free_t   free_   = nullptr;
    rpcmem_to_fd_t  to_fd_  = nullptr;
};

class opencl_env {
  public:
    opencl_env() {
        cl_int err;

        cl_uint num_platforms = 0;
        check_cl(clGetPlatformIDs(0, nullptr, &num_platforms), "clGetPlatformIDs(count)");
        if (num_platforms == 0) {
            die("No OpenCL platform found");
        }

        std::vector<cl_platform_id> platforms(num_platforms);
        check_cl(clGetPlatformIDs(num_platforms, platforms.data(), nullptr), "clGetPlatformIDs(list)");

        // Prefer GPU; fallback to any available device.
        for (cl_platform_id p : platforms) {
            if (try_pick_device(p, CL_DEVICE_TYPE_GPU)) {
                break;
            }
        }
        if (!device_) {
            for (cl_platform_id p : platforms) {
                if (try_pick_device(p, CL_DEVICE_TYPE_ALL)) {
                    break;
                }
            }
        }

        if (!device_) {
            die("No OpenCL device found");
        }

        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
        check_cl(err, "clCreateContext");

        queue_ = clCreateCommandQueue(context_, device_, 0, &err);
        check_cl(err, "clCreateCommandQueue");

        static const char * kSource = R"CLC(
__kernel void reduce_u8_sum(__global const uchar * in, ulong n, __global ulong * out) {
    ulong s = 0;
    for (ulong i = 0; i < n; ++i) {
        s += (ulong)in[i];
    }
    out[0] = s;
}

__kernel void fill_u8(__global uchar * out, ulong n, uchar seed) {
    ulong gid = get_global_id(0);
    if (gid < n) {
        out[gid] = (uchar)((gid ^ (ulong)seed) & 0xFF);
    }
}
)CLC";

        const char * srcs[] = {kSource};
        program_ = clCreateProgramWithSource(context_, 1, srcs, nullptr, &err);
        check_cl(err, "clCreateProgramWithSource");

        err = clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t log_size = 0;
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
            std::string log(log_size, '\0');
            clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
            std::ostringstream oss;
            oss << "clBuildProgram failed: " << cl_err_to_string(err) << "\n" << log;
            die(oss.str());
        }

        kernel_reduce_ = clCreateKernel(program_, "reduce_u8_sum", &err);
        check_cl(err, "clCreateKernel(reduce_u8_sum)");
        kernel_fill_ = clCreateKernel(program_, "fill_u8", &err);
        check_cl(err, "clCreateKernel(fill_u8)");

        // Device info for visibility.
        char dev_name[256] = {};
        check_cl(clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(dev_name), dev_name, nullptr), "clGetDeviceInfo(name)");
        std::fprintf(stderr, "[hetero-switch-bench] OpenCL device: %s\n", dev_name);
    }

    ~opencl_env() {
        if (kernel_fill_) {
            clReleaseKernel(kernel_fill_);
        }
        if (kernel_reduce_) {
            clReleaseKernel(kernel_reduce_);
        }
        if (program_) {
            clReleaseProgram(program_);
        }
        if (queue_) {
            clReleaseCommandQueue(queue_);
        }
        if (context_) {
            clReleaseContext(context_);
        }
    }

    cl_context       context() const { return context_; }
    cl_command_queue queue() const { return queue_; }
    cl_kernel        kernel_reduce() const { return kernel_reduce_; }
    cl_kernel        kernel_fill() const { return kernel_fill_; }

  private:
    bool try_pick_device(cl_platform_id platform, cl_device_type type) {
        cl_uint num_devices = 0;
        cl_int  err         = clGetDeviceIDs(platform, type, 0, nullptr, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) {
            return false;
        }
        std::vector<cl_device_id> devices(num_devices);
        check_cl(clGetDeviceIDs(platform, type, num_devices, devices.data(), nullptr), "clGetDeviceIDs");
        device_ = devices[0];
        return true;
    }

  private:
    cl_device_id     device_        = nullptr;
    cl_context       context_       = nullptr;
    cl_command_queue queue_         = nullptr;
    cl_program       program_       = nullptr;
    cl_kernel        kernel_reduce_ = nullptr;
    cl_kernel        kernel_fill_   = nullptr;
};

static void fill_host_pattern(uint8_t * p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; ++i) {
        p[i] = static_cast<uint8_t>((i ^ static_cast<size_t>(seed)) & 0xFFu);
    }
}

static uint64_t sum_host_u8(const uint8_t * p, size_t n) {
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) {
        s += p[i];
    }
    return s;
}

static bool check_host_pattern(const uint8_t * p, size_t n, uint8_t seed) {
    for (size_t i = 0; i < n; ++i) {
        const uint8_t expect = static_cast<uint8_t>((i ^ static_cast<size_t>(seed)) & 0xFFu);
        if (p[i] != expect) {
            return false;
        }
    }
    return true;
}

static double bytes_and_us_to_gbps(size_t bytes, double us) {
    if (us <= 0.0) {
        return 0.0;
    }
    const double sec = us / 1e6;
    return (static_cast<double>(bytes) / sec) / 1e9;
}

static bench_row run_h2cl_case(opencl_env & env,
                               cl_mem cl_buf,
                               cl_mem checksum_buf,
                               uint8_t * host_buf,
                               size_t size,
                               int iter,
                               bool shared_mode,
                               bool measure_only) {
    fill_host_pattern(host_buf, size, static_cast<uint8_t>(0xA5 + (iter % 31)));
    const uint64_t expected_sum = sum_host_u8(host_buf, size);

    const auto t0 = clock_type::now();

    if (!shared_mode) {
        check_cl(clEnqueueWriteBuffer(env.queue(), cl_buf, CL_TRUE, 0, size, host_buf, 0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(h2cl)");
    }

    cl_ulong n = static_cast<cl_ulong>(size);
    check_cl(clSetKernelArg(env.kernel_reduce(), 0, sizeof(cl_mem), &cl_buf), "clSetKernelArg(reduce,arg0)");
    check_cl(clSetKernelArg(env.kernel_reduce(), 1, sizeof(cl_ulong), &n), "clSetKernelArg(reduce,arg1)");
    check_cl(clSetKernelArg(env.kernel_reduce(), 2, sizeof(cl_mem), &checksum_buf), "clSetKernelArg(reduce,arg2)");

    const size_t gws = 1;
    check_cl(clEnqueueNDRangeKernel(env.queue(), env.kernel_reduce(), 1, nullptr, &gws, nullptr, 0, nullptr, nullptr),
             "clEnqueueNDRangeKernel(reduce)");

    cl_ulong gpu_sum = 0;
    check_cl(clEnqueueReadBuffer(env.queue(), checksum_buf, CL_TRUE, 0, sizeof(cl_ulong), &gpu_sum, 0, nullptr, nullptr),
             "clEnqueueReadBuffer(checksum)");

    const auto   t1 = clock_type::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    bool valid = true;
    if (!measure_only) {
        valid = (expected_sum == static_cast<uint64_t>(gpu_sum));
    }

    bench_row row;
    row.mode            = shared_mode ? "shared_host_ptr" : "memcpy";
    row.flow            = "host_write_to_opencl_read";
    row.size_bytes      = size;
    row.iter            = iter;
    row.latency_us      = us;
    row.throughput_gbps = bytes_and_us_to_gbps(size, us);
    row.valid           = valid ? 1 : 0;
    return row;
}

static bench_row run_cl2h_case(opencl_env & env,
                               cl_mem cl_buf,
                               uint8_t * host_buf,
                               size_t size,
                               int iter,
                               bool shared_mode,
                               bool measure_only) {
    const uint8_t seed = static_cast<uint8_t>(0x5A + (iter % 29));
    cl_ulong      n    = static_cast<cl_ulong>(size);

    const auto t0 = clock_type::now();

    check_cl(clSetKernelArg(env.kernel_fill(), 0, sizeof(cl_mem), &cl_buf), "clSetKernelArg(fill,arg0)");
    check_cl(clSetKernelArg(env.kernel_fill(), 1, sizeof(cl_ulong), &n), "clSetKernelArg(fill,arg1)");
    check_cl(clSetKernelArg(env.kernel_fill(), 2, sizeof(uint8_t), &seed), "clSetKernelArg(fill,arg2)");

    const size_t local = 256;
    const size_t gws   = ((size + local - 1) / local) * local;
    check_cl(clEnqueueNDRangeKernel(env.queue(), env.kernel_fill(), 1, nullptr, &gws, &local, 0, nullptr, nullptr),
             "clEnqueueNDRangeKernel(fill)");

    if (shared_mode) {
        check_cl(clFinish(env.queue()), "clFinish(shared)");
    } else {
        check_cl(clEnqueueReadBuffer(env.queue(), cl_buf, CL_TRUE, 0, size, host_buf, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(cl2h)");
    }

    const auto   t1 = clock_type::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    bool valid = true;
    if (!measure_only) {
        valid = check_host_pattern(host_buf, size, seed);
    }

    bench_row row;
    row.mode            = shared_mode ? "shared_host_ptr" : "memcpy";
    row.flow            = "opencl_write_to_host_read";
    row.size_bytes      = size;
    row.iter            = iter;
    row.latency_us      = us;
    row.throughput_gbps = bytes_and_us_to_gbps(size, us);
    row.valid           = valid ? 1 : 0;
    return row;
}

static options parse_options(int argc, char ** argv) {
    options opt;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        auto need_value = [&](const char * flag) -> std::string {
            if (i + 1 >= argc) {
                die(std::string("missing value for ") + flag);
            }
            ++i;
            return argv[i];
        };

        if (a == "--warmup") {
            opt.warmup = std::stoi(need_value("--warmup"));
        } else if (a == "--iters") {
            opt.iters = std::stoi(need_value("--iters"));
        } else if (a == "--sizes") {
            const std::string spec = need_value("--sizes");
            std::vector<size_t> parsed;
            std::stringstream   ss(spec);
            std::string         tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.empty()) {
                    continue;
                }
                parsed.push_back(static_cast<size_t>(std::stoull(tok)));
            }
            if (parsed.empty()) {
                die("--sizes produced an empty list");
            }
            opt.sizes = std::move(parsed);
        } else if (a == "--csv") {
            opt.csv_path = need_value("--csv");
        } else if (a == "--help" || a == "-h") {
            std::cout
                << "Usage: hetero-switch-bench [--warmup N] [--iters N] [--sizes b1,b2,...] [--csv path]\n"
                << "Default sizes: 1024,65536,1048576,16777216\n";
            std::exit(0);
        } else {
            die("unknown arg: " + a);
        }
    }

    if (opt.warmup < 0 || opt.iters <= 0) {
        die("invalid --warmup/--iters");
    }

    return opt;
}

static void write_csv(std::ostream & os, const std::vector<bench_row> & rows) {
    os << "mode,flow,size_bytes,iter,latency_us,throughput_gbps,valid\n";
    for (const auto & r : rows) {
        os << r.mode << ',' << r.flow << ',' << r.size_bytes << ',' << r.iter << ',' << r.latency_us << ','
           << r.throughput_gbps << ',' << r.valid << '\n';
    }
}

static void print_summary(const std::vector<bench_row> & rows) {
    struct key {
        std::string mode;
        std::string flow;
        size_t      size = 0;
    };

    std::vector<key> keys;
    keys.reserve(rows.size());
    for (const auto & r : rows) {
        key k{r.mode, r.flow, r.size_bytes};
        bool exists = false;
        for (const auto & x : keys) {
            if (x.mode == k.mode && x.flow == k.flow && x.size == k.size) {
                exists = true;
                break;
            }
        }
        if (!exists) {
            keys.push_back(std::move(k));
        }
    }

    std::fprintf(stderr, "[hetero-switch-bench] summary (avg over measured iterations):\n");
    for (const auto & k : keys) {
        double sum_us = 0.0;
        double sum_gb = 0.0;
        int    n      = 0;
        int    valid  = 1;
        for (const auto & r : rows) {
            if (r.mode == k.mode && r.flow == k.flow && r.size_bytes == k.size) {
                sum_us += r.latency_us;
                sum_gb += r.throughput_gbps;
                n++;
                valid = valid && r.valid;
            }
        }

        if (n > 0) {
            std::fprintf(stderr,
                         "  mode=%s flow=%s size=%zu avg_latency_us=%.3f avg_throughput_gbps=%.3f valid=%d\n",
                         k.mode.c_str(),
                         k.flow.c_str(),
                         k.size,
                         sum_us / n,
                         sum_gb / n,
                         valid);
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const options opt = parse_options(argc, argv);

        std::fprintf(stderr,
                     "[hetero-switch-bench] warmup=%d, iters=%d, sizes=%zu\n",
                     opt.warmup,
                     opt.iters,
                     opt.sizes.size());

        rpcmem_loader rpc;
        std::fprintf(stderr, "[hetero-switch-bench] rpcmem lib: %s\n", rpc.lib_name().c_str());

        opencl_env env;

        std::vector<bench_row> rows;
        rows.reserve(opt.sizes.size() * static_cast<size_t>(opt.iters) * 4);

        for (size_t size : opt.sizes) {
            std::fprintf(stderr, "[hetero-switch-bench] size=%zu bytes\n", size);

            void * memcpy_raw = nullptr;
            if (posix_memalign(&memcpy_raw, 64, size) != 0 || !memcpy_raw) {
                die("posix_memalign failed for memcpy host buffer");
            }
            std::unique_ptr<uint8_t, decltype(&std::free)> memcpy_host(static_cast<uint8_t *>(memcpy_raw), &std::free);

            void * shared_raw = rpc.alloc(size);
            if (!shared_raw) {
                die("rpcmem allocation failed for shared host buffer");
            }
            std::unique_ptr<void, std::function<void(void *)>> shared_guard(
                shared_raw,
                [&](void * p) {
                    if (p) {
                        rpc.free(p);
                    }
                });

            int fd = rpc.to_fd(shared_raw);
            if (fd < 0) {
                die("rpcmem_to_fd failed; shared allocation is not valid");
            }

            auto * shared_host = static_cast<uint8_t *>(shared_raw);

            cl_int err = CL_SUCCESS;
            cl_mem cl_shared_buf = clCreateBuffer(env.context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, size, shared_host, &err);
            check_cl(err, "clCreateBuffer(shared host ptr)");

            cl_mem cl_memcpy_buf = clCreateBuffer(env.context(), CL_MEM_READ_WRITE, size, nullptr, &err);
            check_cl(err, "clCreateBuffer(memcpy)");

            cl_mem checksum_buf = clCreateBuffer(env.context(), CL_MEM_READ_WRITE, sizeof(cl_ulong), nullptr, &err);
            check_cl(err, "clCreateBuffer(checksum)");

            auto cleanup = [&]() {
                if (checksum_buf) {
                    clReleaseMemObject(checksum_buf);
                    checksum_buf = nullptr;
                }
                if (cl_memcpy_buf) {
                    clReleaseMemObject(cl_memcpy_buf);
                    cl_memcpy_buf = nullptr;
                }
                if (cl_shared_buf) {
                    clReleaseMemObject(cl_shared_buf);
                    cl_shared_buf = nullptr;
                }
            };

            try {
                const int total = opt.warmup + opt.iters;
                for (int i = 0; i < total; ++i) {
                    const bool warmup = (i < opt.warmup);
                    const int  iter   = warmup ? -1 : (i - opt.warmup);

                    auto r1 = run_h2cl_case(env, cl_shared_buf, checksum_buf, shared_host, size, i, true, warmup);
                    auto r2 = run_h2cl_case(env,
                                            cl_memcpy_buf,
                                            checksum_buf,
                                            memcpy_host.get(),
                                            size,
                                            i,
                                            false,
                                            warmup);
                    auto r3 = run_cl2h_case(env, cl_shared_buf, shared_host, size, i, true, warmup);
                    auto r4 = run_cl2h_case(env, cl_memcpy_buf, memcpy_host.get(), size, i, false, warmup);

                    if (!warmup) {
                        r1.iter = iter;
                        r2.iter = iter;
                        r3.iter = iter;
                        r4.iter = iter;

                        rows.push_back(std::move(r1));
                        rows.push_back(std::move(r2));
                        rows.push_back(std::move(r3));
                        rows.push_back(std::move(r4));
                    }
                }
            } catch (...) {
                cleanup();
                throw;
            }

            cleanup();
        }

        if (opt.csv_path.empty()) {
            write_csv(std::cout, rows);
        } else {
            std::ofstream ofs(opt.csv_path);
            if (!ofs) {
                die("failed to open CSV output: " + opt.csv_path);
            }
            write_csv(ofs, rows);
            std::fprintf(stderr, "[hetero-switch-bench] wrote CSV: %s\n", opt.csv_path.c_str());
        }

        print_summary(rows);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[hetero-switch-bench] ERROR: %s\n", e.what());
        return 1;
    }
}
