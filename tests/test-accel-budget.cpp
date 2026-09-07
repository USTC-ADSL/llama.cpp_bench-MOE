// Android regression/charge probe. Run only in an isolated experiment memory cgroup.
#define CL_TARGET_OPENCL_VERSION 300
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include "../ggml/src/ggml-opencl/ggml-opencl-budget.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <dlfcn.h>
#include <sys/stat.h>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <fcntl.h>
#include <climits>

extern "C" int htpdrv_init();
extern "C" void * rpcmem_alloc2(int, uint32_t, size_t);
extern "C" void rpcmem_free(void *);
extern "C" int rpcmem_to_fd(void *);
extern "C" void * (*rpcmem_alloc2_pfn)(int,uint32_t,size_t);
extern "C" void * (*rpcmem_alloc_pfn)(int,uint32_t,int);
static int injected_driver_calls;
static constexpr size_t block = 64ULL << 20;
static const char * test_name;
static int failures;
static long long counter(const std::string & path) { std::ifstream f(path); long long n = -1; f >> n; return n; }
static void row(const char * phase, const char * status = "success", uint64_t requested = 0, double ms = 0, const char * detail = "") {
    auto s = ggml_budget_snapshot();
    const char * cg = std::getenv("GGML_BUDGET_TEST_CGROUP");
    auto usage = cg ? counter(std::string(cg) + "/memory.usage_in_bytes") : -1;
    auto kgsl = counter("/sys/class/kgsl/kgsl/proc/" + std::to_string(getpid()) + "/gpumem_mapped");
    std::printf("GGML_BUDGET_TEST,%s,%d,%s,%s,%llu,%llu,%llu,%llu,%llu,%lld,%lld,%llu,%llu,%.6f,%s\n",
        test_name, s.mode, phase, status, (unsigned long long)requested, (unsigned long long)s.used,
        (unsigned long long)s.reserved, (unsigned long long)s.cap, (unsigned long long)s.denied, usage, kgsl,
        (unsigned long long)s.unknown, (unsigned long long)s.peak, ms, detail);
    std::fflush(stdout);
}
static void check(bool ok, const char * phase) { if (!ok) ++failures; row(phase, ok ? "PASS" : "FAIL"); }
static void settle(uint64_t used = 0) {
    for (int i = 0; i < 100 && ggml_budget_snapshot().used != used; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
struct CL {
    cl_device_id device; cl_context ctx; cl_command_queue queue;
    CL() {
        cl_platform_id platform; cl_uint n; cl_int err;
        if (clGetPlatformIDs(1, &platform, &n) || !n || clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &n)) std::exit(3);
        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err); if (err) std::exit(3);
        queue = clCreateCommandQueue(ctx, device, 0, &err); if (err) std::exit(3);
    }
    ~CL() { clReleaseCommandQueue(queue); clReleaseContext(ctx); }
    void touch(cl_mem mem, size_t bytes) {
        uint32_t pattern = 0x12345678;
        cl_event done;
        if (clEnqueueFillBuffer(queue, mem, &pattern, sizeof(pattern), 0, bytes, 0, nullptr, &done)) std::exit(4);
        clFlush(queue); clWaitForEvents(1, &done); clReleaseEvent(done);
    }
};
static void ledger() {
    void * opencl = dlopen("libggml-opencl.so",RTLD_NOW|RTLD_LOCAL);
    void * hexagon = dlopen("libggml-hexagon.so",RTLD_NOW|RTLD_LOCAL);
    check(opencl && hexagon && dlsym(opencl,"ggml_budget_snapshot")==dlsym(hexagon,"ggml_budget_snapshot") &&
          dlsym(hexagon,"ggml_budget_snapshot")==reinterpret_cast<void *>(&ggml_budget_snapshot),"two_dso_one_manager");
    if(opencl) dlclose(opencl); if(hexagon) dlclose(hexagon);
    check(ggml_budget_round(UINT64_MAX, 4096) == UINT64_MAX, "overflow_round");
    std::atomic<int> ready{0}, acquired{0}; std::atomic<bool> release{false};
    std::vector<std::thread> threads;
    for (int i=0; i<16; ++i) threads.emplace_back([&,i] {
        auto ticket = ggml_budget_reserve(i%2, block, GGML_BUDGET_EXTRA, "concurrent_reserve");
        if (ticket) ++acquired;
        ++ready; while (!release.load()) std::this_thread::yield();
        if (ticket) ggml_budget_cancel(ticket);
    });
    while (ready != 16) std::this_thread::yield();
    check(acquired == (ggml_budget_mode() == 2 ? 2 : 16), "concurrent_shared_cap");
    release = true; for (auto & t : threads) t.join();
    check(!ggml_budget_snapshot().used && !ggml_budget_snapshot().reserved, "reservation_rollback");
    auto a = ggml_budget_reserve(2, block, GGML_BUDGET_CHARGED, "charged_owner"); ggml_budget_commit(a, 77);
    auto alias = ggml_budget_alias(2, 77, 88);
    check(alias && ggml_budget_backing_generation(alias)==a && !ggml_budget_snapshot().used, "charged_alias_no_extra");
    ggml_budget_release(alias); ggml_budget_release(a);
    auto b = ggml_budget_reserve(2, block, GGML_BUDGET_EXTRA, "native_reuse"); ggml_budget_commit(b, 77);
    ggml_budget_release(a); // stale destruction must not release the reused identity.
    check(a != b && ggml_budget_find(2, 77) == b && ggml_budget_snapshot().used == block, "generation_reuse");
    ggml_budget_release(b);
}
static void allocations(const std::string & kind) {
    std::unique_ptr<CL> cl;
    bool use_cl = kind == "opencl" || kind == "mixed";
    bool use_rpc = kind == "rpcmem" || kind == "mixed";
    row("before_init");
    if (use_cl) cl.reset(new CL);
    if (use_rpc && htpdrv_init()) std::exit(3);
    row("initialized");
    std::vector<cl_mem> cls; std::vector<void *> rpcs, pages;
    int fd = -1;
    if (kind == "file") {
        fd = open("charge-probe.tmp", O_RDWR|O_CREAT|O_EXCL,0600);
        if (fd < 0) std::exit(3);
        unlink("charge-probe.tmp"); if (ftruncate(fd, 3*block)) std::exit(3);
    }
    int accepted = 0;
    uint64_t rpc_capacity = 0;
    for (int i=0;i<3;++i) {
        bool success = false;
        if (use_cl && (kind != "mixed" || i%2==0)) {
            cl_int err; auto p = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, block, nullptr, &err);
            if (p) { cls.push_back(p); cl->touch(p, block); success = true; }
        } else if (use_rpc) {
            auto p = rpcmem_alloc2(25, 1, block);
            if (p) {
                rpcs.push_back(p); std::memset(p, 0x5a, block); success = true;
                struct stat st{};
                if(!fstat(rpcmem_to_fd(p),&st)) rpc_capacity += st.st_size;
            }
        } else {
            auto p = mmap(nullptr, block, PROT_READ|PROT_WRITE, kind=="file" ? MAP_SHARED : MAP_PRIVATE|MAP_ANONYMOUS,
                          fd, kind=="file" ? i*block : 0);
            if (p != MAP_FAILED) { pages.push_back(p); std::memset(p, 0x3c, block); success = true; }
        }
        accepted += success;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        row(("block_"+std::to_string(i+1)).c_str(), success ? "success" : "allocation_failed",block,0,
            ("rpcmem_fd_capacity="+std::to_string(rpc_capacity)).c_str());
    }
    if (use_cl || use_rpc) check(accepted == (ggml_budget_mode()==2 ? 2:3), "boundary");
    for (auto p : cls) clReleaseMemObject(p);
    for (auto p : rpcs) rpcmem_free(p);
    for (auto p : pages) munmap(p,block);
    if (fd>=0) close(fd);
    settle(); row("released"); check(ggml_budget_snapshot().used==0,"balance");
}
static void lifecycle() {
    CL cl; cl_int err;
    auto root = clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE, block, nullptr, &err);
    if (!root) std::exit(3);
    auto generation = ggml_budget_find(0,reinterpret_cast<uintptr_t>(root));
    cl_buffer_region region{0,block}; auto sub = clCreateSubBuffer(root, 0, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
    cl_image_format fmt{CL_RGBA,CL_HALF_FLOAT}; cl_image_desc desc{};
    desc.image_type=CL_MEM_OBJECT_IMAGE1D_BUFFER; desc.image_width=block/8; desc.buffer=sub;
    auto img = clCreateImage(cl.ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &err);
    check(sub && img && ggml_budget_backing_generation(ggml_budget_find(0,reinterpret_cast<uintptr_t>(img)))==generation &&
          ggml_budget_snapshot().used == block,"subbuffer_image_alias");
    clReleaseMemObject(root);
    check(ggml_budget_snapshot().used == block,"parent_release_retains_storage");
    clReleaseMemObject(sub);
    if (img) clReleaseMemObject(img);
    settle(); check(ggml_budget_snapshot().used==0,"alias_final_release");

    auto before=ggml_budget_snapshot();
    auto bad=clCreateBuffer(cl.ctx, CL_MEM_READ_WRITE|CL_MEM_READ_ONLY, block, nullptr,&err);
    check(!bad && ggml_budget_snapshot().used == before.used && !ggml_budget_snapshot().reserved,"driver_failure_rollback");

    root=clCreateBuffer(cl.ctx,CL_MEM_READ_WRITE,block,nullptr,&err);
    clRetainMemObject(root); clReleaseMemObject(root);
    check(ggml_budget_snapshot().used==block,"retained_pool");
    cl.touch(root,block); check(ggml_budget_snapshot().used==block,"pool_reuse");
    auto gate=clCreateUserEvent(cl.ctx,&err); cl_event done; uint32_t pattern=42;
    err=clEnqueueFillBuffer(cl.queue,root,&pattern,4,0,block,1,&gate,&done);
    if(err) std::exit(4);
    clFlush(cl.queue); clReleaseMemObject(root);
    check(ggml_budget_snapshot().used==block,"inflight_release");
    clSetUserEventStatus(gate,CL_COMPLETE); clWaitForEvents(1,&done); clReleaseEvent(done); clReleaseEvent(gate);
    settle(); check(!ggml_budget_snapshot().used,"async_completion_release");
    void * host=nullptr;
    if(posix_memalign(&host,4096,1<<20)) std::exit(3);
    memset(host,0x5a,1<<20);
    auto imported=clCreateBuffer(cl.ctx,CL_MEM_READ_WRITE|CL_MEM_USE_HOST_PTR,1<<20,host,&err);
    check(imported && ggml_budget_snapshot().unknown==(1<<20) && ggml_budget_snapshot().used==(1<<20),
          "use_host_ptr_unknown_not_zero_copy");
    if(imported) {cl.touch(imported,1<<20);clReleaseMemObject(imported);}
    settle(); free(host);
    cl_image_format image_format{CL_RGBA,CL_HALF_FLOAT}; cl_image_desc image_desc{};
    image_desc.image_type=CL_MEM_OBJECT_IMAGE2D;image_desc.image_width=512;image_desc.image_height=512;
    auto independent=clCreateImage(cl.ctx,CL_MEM_READ_WRITE,&image_format,&image_desc,nullptr,&err);
    size_t mem_size=0,row_pitch=0;
    if(independent) {
        clGetMemObjectInfo(independent,CL_MEM_SIZE,sizeof(mem_size),&mem_size,nullptr);
        clGetImageInfo(independent,CL_IMAGE_ROW_PITCH,sizeof(row_pitch),&row_pitch,nullptr);
    }
    row("independent_image_layout",ggml_budget_mode()==2?"BLOCKED":"UNKNOWN",2<<20,0,
        ("cl_mem_size="+std::to_string(mem_size)+";row_pitch="+std::to_string(row_pitch)).c_str());
    check(ggml_budget_mode()==2 ? !independent && err==CL_INVALID_OPERATION && !ggml_budget_snapshot().used :
          independent && ggml_budget_snapshot().unknown==ggml_cl_budget_bytes(mem_size),"independent_image_coverage_guard");
    if(independent) clReleaseMemObject(independent);
    settle();
    const cl_mem_properties properties[]={0x41A6,1,0}; // Upstream CL_LARGE_BUFFER_QCOM fallback.
    auto large=clCreateBufferWithProperties(cl.ctx,properties,CL_MEM_READ_WRITE,block,nullptr,&err);
    size_t large_capacity=0;
    if(large) {
        clGetMemObjectInfo(large,CL_MEM_SIZE,sizeof(large_capacity),&large_capacity,nullptr);
        cl.touch(large,block);
    }
    row("large_buffer_properties",large?"PASS":"BLOCKED",block,0,
        ("cl_mem_size="+std::to_string(large_capacity)+";error="+std::to_string(err)).c_str());
    if(large) {
        check(ggml_budget_snapshot().used==ggml_cl_budget_bytes(large_capacity),"large_buffer_capacity");
        clReleaseMemObject(large);settle();
    }
    bool fresh_generations=true;
    for (int i=0;i<32;++i) {
        root=clCreateBuffer(cl.ctx,CL_MEM_READ_WRITE,1<<20,nullptr,&err);
        if (!root) std::exit(3);
        fresh_generations &= ggml_budget_find(0,reinterpret_cast<uintptr_t>(root)) != generation;
        cl.touch(root,1<<20); clReleaseMemObject(root);
    }
    check(fresh_generations,"new_generations");
    settle(); check(!ggml_budget_snapshot().used,"repeated_create_destroy");
    if (htpdrv_init()) std::exit(3);
    auto original_alloc2=rpcmem_alloc2_pfn;auto original_alloc=rpcmem_alloc_pfn;
    rpcmem_alloc2_pfn=[](int,uint32_t,size_t)->void * {return nullptr;};
    auto failed_before=ggml_budget_snapshot().failed;
    check(!rpcmem_alloc2(25,1,block) && ggml_budget_snapshot().failed==failed_before+1 &&
          !ggml_budget_snapshot().reserved && !ggml_budget_snapshot().used,"rpcmem_injected_failure_rollback");
    rpcmem_alloc2_pfn=nullptr;
    rpcmem_alloc_pfn=[](int,uint32_t,int)->void * {++injected_driver_calls;return nullptr;};
    check(!rpcmem_alloc2(25,1,uint64_t(INT_MAX)+1) && !injected_driver_calls,"legacy_size_narrowing_guard");
    check(!rpcmem_alloc2(25,1,4096) && injected_driver_calls==1 && !ggml_budget_snapshot().reserved,
          "legacy_allocator_failure_rollback");
    rpcmem_alloc2_pfn=original_alloc2;rpcmem_alloc_pfn=original_alloc;
    int oldfd=-1, reused=0; uint64_t previous=0;
    for(int i=0;i<32;++i) {
        auto p=rpcmem_alloc2(25,1,1<<20); if(!p) std::exit(3);
        auto ticket=ggml_budget_find(1,reinterpret_cast<uintptr_t>(p)); int fd=rpcmem_to_fd(p);
        reused += fd==oldfd; oldfd=fd;
        if (ticket==previous) ++failures; previous=ticket;
        memset(p,0x5a,1<<20); rpcmem_free(p);
    }
    check(reused>0 && !ggml_budget_snapshot().used,"rpcmem_fd_reuse");
}
static void operator_test() {
    CL cl; cl_int err; constexpr size_t n=4096;
    const char * source="__kernel void op(__global float *x) { int i=get_global_id(0); x[i]=x[i]*2.0f+1.0f; }";
    auto program=clCreateProgramWithSource(cl.ctx,1,&source,nullptr,&err);
    if(err || clBuildProgram(program,1,&cl.device,nullptr,nullptr,nullptr)) std::exit(4);
    auto kernel=clCreateKernel(program,"op",&err);
    std::vector<float> values(n); for(size_t i=0;i<n;++i) values[i]=float(i);
    auto start=std::chrono::steady_clock::now();
    bool valid=true;
    const char * it_env=std::getenv("GGML_BUDGET_TEST_ITERATIONS");
    const char * warm_env=std::getenv("GGML_BUDGET_TEST_WARMUP");
    const int iterations=it_env?std::atoi(it_env):30;
    const int warmup=warm_env?std::atoi(warm_env):0;
    if(iterations<1 || iterations>10000 || warmup<0 || warmup>10000) std::exit(2);
    for(int k=-warmup;k<iterations;++k) {
        if(k==0) start=std::chrono::steady_clock::now();
        auto mem=clCreateBuffer(cl.ctx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,n*4,values.data(),&err);
        if (!mem) std::exit(3);
        clSetKernelArg(kernel,0,sizeof(mem),&mem);
        clEnqueueNDRangeKernel(cl.queue,kernel,1,nullptr,&n,nullptr,0,nullptr,nullptr);
        std::vector<float> out(n); clEnqueueReadBuffer(cl.queue,mem,CL_TRUE,0,n*4,out.data(),0,nullptr,nullptr);
        for(size_t i=0;i<n;++i) valid &= out[i]==float(i)*2+1;
        clReleaseMemObject(mem);
    }
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    row((std::to_string(iterations)+"_operator_iterations").c_str(),valid?"PASS":"FAIL",n*4,ms,"exact_float_affine");
    if(!valid) ++failures;
    clReleaseKernel(kernel); clReleaseProgram(program);
}
int main(int argc,char ** argv) {
    if(argc!=2) return 2;
    test_name=argv[1]; std::string test=test_name;
    if(test=="ledger") ledger();
    else if(test=="lifecycle") lifecycle();
    else if(test=="operator") operator_test();
    else allocations(test);
    ggml_budget_denial d{};
    for(uint64_t i=0;ggml_budget_denial_at(i,&d);++i)
        std::printf("DENIAL,%s,%s,%d,%llu,%llu,%llu,%llu\n",test_name,d.stage,d.backend,
            (unsigned long long)d.requested,(unsigned long long)d.used,(unsigned long long)d.cap,(unsigned long long)d.count);
    return failures?1:0;
}
