#include <cstdlib>
#include <cstdio>
#include <unordered_map>
#include <unistd.h>

namespace {
int mode = 0, initialized = 0, allocations = 0, releases = 0, bad_frees = 0;
std::unordered_map<void *, int> live;
}
extern "C" {
void rpcmem_init() { ++initialized; }
void rpcmem_deinit() {}
void test_fail_mode(int value) { mode = value; }
int test_count(int which) {
    switch (which) { case 0: return initialized; case 1: return allocations;
        case 2: return releases; case 3: return live.size(); default: return bad_frees; }
}
#ifdef MOE_TEST_LEGACY_RPCMEM
void * rpcmem_alloc(int, unsigned, int bytes) {
#else
void * rpcmem_alloc2(int, unsigned, size_t bytes) {
#endif
    if (mode == 1) return nullptr;
    void * pointer = nullptr;
    if (posix_memalign(&pointer, 4096, bytes)) return nullptr;
    char path[] = "/tmp/moe-fake-rpcmem-XXXXXX";
    const int fd = mkstemp(path); unlink(path);
    if (fd < 0) { free(pointer); return nullptr; }
    live.emplace(pointer, fd); ++allocations; return pointer;
}
int rpcmem_to_fd(void * pointer) { return mode == 2 ? -1 : live.at(pointer); }
void rpcmem_free(void * pointer) {
    auto found = live.find(pointer);
    if (found == live.end()) { ++bad_frees; return; }
    close(found->second); live.erase(found); free(pointer); ++releases;
}
}
