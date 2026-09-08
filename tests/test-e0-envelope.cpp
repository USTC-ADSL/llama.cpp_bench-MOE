// Reuse the existing real-allocation helpers; do not change their old tests.
#define main accel_budget_small_test_main
#include "test-accel-budget.cpp"
#undef main

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::string kind = argv[1];
    if (kind != "opencl" && kind != "rpcmem" && kind != "mixed") return 2;
    test_name = argv[1];
    if (ggml_budget_mode() != 2 || ggml_budget_snapshot().cap != 3072ULL * (1 << 20)) return 2;
    row("before_init");
    std::unique_ptr<CL> cl;
    if (kind != "rpcmem") cl.reset(new CL);
    if (kind != "opencl" && htpdrv_init()) return 3;
    row("initialized");
    const size_t host_bytes = 512ULL << 20;
    void * host = mmap(nullptr, host_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host == MAP_FAILED) return 3;
    std::memset(host, 0x3c, host_bytes);
    row("host_512MiB_touched");
    std::vector<cl_mem> cls;
    std::vector<void *> rpcs;
    uint64_t rpc_capacity = 0;
    for (int i = 0; i < 49; ++i) {
        bool success = false;
        if (kind == "opencl" || (kind == "mixed" && i % 2 == 0)) {
            cl_int err = CL_SUCCESS;
            cl_mem p = clCreateBuffer(cl->ctx, CL_MEM_READ_WRITE, block, nullptr, &err);
            if (p) { cls.push_back(p); cl->touch(p, block); success = true; }
        } else {
            void * p = rpcmem_alloc2(25, 1, block);
            if (p) {
                rpcs.push_back(p);
                std::memset(p, 0x5a, block);
                struct stat st{};
                check(!fstat(rpcmem_to_fd(p), &st), "rpcmem_capacity_readable");
                rpc_capacity += st.st_size;
                success = true;
            }
        }
        row(("block_" + std::to_string(i + 1)).c_str(), success ? "success" : "allocation_failed", block, 0,
            ("rpcmem_fd_capacity=" + std::to_string(rpc_capacity)).c_str());
        check(success == (i < 48), "expected_boundary");
        if (!success && i < 48) break;
    }
    auto s = ggml_budget_snapshot();
    check(s.used == 3072ULL * (1 << 20) && !s.reserved && s.denied == 1, "3GiB_live_next64MiB_denied");
    // Keep live storage long enough for the existing independent safety sampler.
    std::this_thread::sleep_for(std::chrono::seconds(2));
    row("held_peak");
    for (auto p : cls) clReleaseMemObject(p);
    for (auto p : rpcs) rpcmem_free(p);
    munmap(host, host_bytes);
    settle();
    check(!ggml_budget_snapshot().used && !ggml_budget_snapshot().reserved, "final_balance");
    return failures ? 1 : 0;
}
