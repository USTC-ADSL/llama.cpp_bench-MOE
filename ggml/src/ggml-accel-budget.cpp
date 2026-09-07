#include "ggml-accel-budget.h"
#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#if defined(__unix__) || defined(__ANDROID__)
#include <dlfcn.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace {
void exit_report();
struct entry {
    uint64_t bytes;
    uintptr_t native = 0;
    int backend;
    bool pending = true;
    bool unknown;
    bool alias = false;
    uint64_t backing_generation = 0;
};
struct BudgetManager {
    std::mutex mutex;
    ggml_budget_stats stats{};
    uint64_t generation = 0;
    int coverage[3] = {GGML_BUDGET_UNKNOWN, GGML_BUDGET_UNKNOWN, GGML_BUDGET_EXTRA};
    std::map<uint64_t, entry> entries;
    std::map<std::pair<int, uintptr_t>, uint64_t> identities;
    std::map<std::tuple<int, std::string, uint64_t, uint64_t>, ggml_budget_denial> denials;
    BudgetManager() {
        const char * report = std::getenv("GGML_ACCEL_BUDGET_REPORT");
        if (report && !std::strcmp(report, "1")) std::atexit(exit_report);
        const char * mode = std::getenv("GGML_ACCEL_BUDGET_MODE");
        if (!mode || !std::strcmp(mode, "off")) return;
        stats.mode = !std::strcmp(mode, "observe") ? 1 : !std::strcmp(mode, "enforce") ? 2 : -1;
        const char * cap = std::getenv("GGML_ACCEL_BUDGET_EXTRA_BYTES");
        if (cap && *cap && *cap != '-') {
            char * end = nullptr; errno = 0;
            stats.cap = std::strtoull(cap, &end, 10);
            if (errno || *end) stats.mode = -1;
        } else if (stats.mode == 2) stats.mode = -1;
        else stats.cap = UINT64_MAX;
        const char * names[] = {"GGML_ACCEL_BUDGET_OPENCL_COVERAGE", "GGML_ACCEL_BUDGET_RPCMEM_COVERAGE"};
        for (int i = 0; i < 2; ++i) {
            const char * value = std::getenv(names[i]);
            if (!value || !std::strcmp(value, "unknown")) continue;
            if (!std::strcmp(value, "extra")) coverage[i] = GGML_BUDGET_EXTRA;
            else if (!std::strcmp(value, "charged")) coverage[i] = GGML_BUDGET_CHARGED;
            else stats.mode = -1;
        }
    }
};
BudgetManager & manager() { static BudgetManager * m = new BudgetManager; return *m; }
void exit_report() {
    const auto s = ggml_budget_snapshot();
    std::printf("GGML_BUDGET_SUMMARY,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",s.mode,
        (unsigned long long)s.used,(unsigned long long)s.reserved,(unsigned long long)s.peak,
        (unsigned long long)s.cap,(unsigned long long)s.denied,(unsigned long long)s.unknown,
        (unsigned long long)s.coverage_gaps);
    ggml_budget_denial d{};
    for (uint64_t i=0;ggml_budget_denial_at(i,&d);++i)
        std::printf("GGML_BUDGET_DENIAL,%s,%d,%llu,%llu,%llu,%llu\n",d.stage,d.backend,
            (unsigned long long)d.requested,(unsigned long long)d.used,(unsigned long long)d.cap,(unsigned long long)d.count);
}
void erase_entry(BudgetManager & m, uint64_t ticket, bool failed) {
    auto it = m.entries.find(ticket);
    if (it == m.entries.end()) return;
    const auto & e = it->second;
    if (e.pending) m.stats.reserved -= e.bytes;
    else { m.stats.used -= e.bytes; --m.stats.live; }
    if (e.unknown) m.stats.unknown -= e.bytes;
    if (e.alias) --m.stats.aliases;
    if (e.native) {
        auto id = m.identities.find({e.backend, e.native});
        if (id != m.identities.end() && id->second == ticket) m.identities.erase(id);
    }
    m.stats.failed += failed;
    m.entries.erase(it);
}
}
int ggml_budget_mode() { return manager().stats.mode; }
uint64_t ggml_budget_round(uint64_t size, uint64_t alignment) {
    if (!alignment || size > UINT64_MAX - (alignment - 1)) return UINT64_MAX;
    return ((size + alignment - 1) / alignment) * alignment;
}
uint64_t ggml_budget_reserve(int backend, uint64_t bytes, int coverage, const char * stage) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    if (m.stats.mode <= 0 || backend < 0 || backend > GGML_BUDGET_TEST) return 0;
    if (coverage == GGML_BUDGET_DEFAULT) coverage = m.coverage[backend];
    uint64_t debit = coverage == GGML_BUDGET_CHARGED ? 0 : bytes;
    const uint64_t total = m.stats.used + m.stats.reserved;
    if (debit > UINT64_MAX - total || (m.stats.mode == 2 && (total > m.stats.cap || debit > m.stats.cap - total))) {
        ++m.stats.denied;
        try {
            auto key = std::make_tuple(backend, std::string(stage ? stage : "allocation").substr(0,47), bytes, total);
            auto & d = m.denials[key]; d.backend = backend; d.requested = bytes; d.used = total; d.cap = m.stats.cap; ++d.count;
            std::strncpy(d.stage, std::get<1>(key).c_str(), sizeof(d.stage)-1);
        } catch (...) { ++m.stats.coverage_gaps; }
        return 0;
    }
    if (m.generation == UINTPTR_MAX) return 0;
    uint64_t ticket = ++m.generation;
    try { m.entries.emplace(ticket, entry{debit, 0, backend, true, coverage == GGML_BUDGET_UNKNOWN, false, ticket}); }
    catch (...) { ++m.stats.failed; return 0; }
    m.stats.reserved += debit;
    if (coverage == GGML_BUDGET_UNKNOWN) m.stats.unknown += debit;
    m.stats.peak = std::max(m.stats.peak, total + debit);
    return ticket;
}
int ggml_budget_observe_size(uint64_t ticket, uint64_t bytes) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    auto it=m.entries.find(ticket);
    if(m.stats.mode!=1 || it==m.entries.end() || !it->second.pending) return 0;
    auto & e=it->second;
    const auto total=m.stats.used+m.stats.reserved-e.bytes;
    if(bytes>UINT64_MAX-total) return 0;
    m.stats.reserved=m.stats.reserved-e.bytes+bytes;
    if(e.unknown) m.stats.unknown=m.stats.unknown-e.bytes+bytes;
    e.bytes=bytes;m.stats.peak=std::max(m.stats.peak,total+bytes);
    return 1;
}
int ggml_budget_commit(uint64_t ticket, uintptr_t native) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    auto it = m.entries.find(ticket); if (it == m.entries.end()) return 0;
    if (!it->second.pending) return 1;
    auto & e = it->second;
    // Allocate identity metadata before changing reservation state.
    try { m.identities[{e.backend, native}] = ticket; }
    catch (...) { return 0; }
    e.native = native; e.pending = false;
    m.stats.reserved -= e.bytes; m.stats.used += e.bytes; ++m.stats.live;
    return 1;
}
void ggml_budget_cancel(uint64_t ticket) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex); erase_entry(m, ticket, true);
}
void ggml_budget_release(uint64_t ticket) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex); erase_entry(m, ticket, false);
}
uint64_t ggml_budget_find(int backend, uintptr_t native) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    auto it = m.identities.find({backend, native}); return it == m.identities.end() ? 0 : it->second;
}
uint64_t ggml_budget_alias(int backend, uintptr_t parent, uintptr_t alias) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    auto parent_id=m.identities.find({backend,parent});
    if (parent_id==m.identities.end()) { ++m.stats.coverage_gaps; return 0; }
    const auto backing=m.entries.at(parent_id->second).backing_generation;
    if (m.generation == UINTPTR_MAX) return 0;
    const uint64_t ticket = ++m.generation;
    try {
        m.entries.emplace(ticket, entry{0, alias, backend, false, false, true, backing});
        m.identities[{backend, alias}] = ticket;
    } catch (...) { m.entries.erase(ticket); ++m.stats.failed; return 0; }
    ++m.stats.live; ++m.stats.aliases;
    return ticket;
}
uint64_t ggml_budget_backing_generation(uint64_t ticket) {
    auto & m=manager();std::lock_guard<std::mutex> lock(m.mutex);
    auto it=m.entries.find(ticket);return it==m.entries.end()?0:it->second.backing_generation;
}
void ggml_budget_gap() { auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex); ++m.stats.coverage_gaps; }
ggml_budget_stats ggml_budget_snapshot() { auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex); return m.stats; }
int ggml_budget_denial_at(uint64_t index, ggml_budget_denial * out) {
    auto & m = manager(); std::lock_guard<std::mutex> lock(m.mutex);
    if (!out || index >= m.denials.size()) return 0;
    auto it = m.denials.begin(); std::advance(it, index); *out = it->second; return 1;
}
#ifdef _WIN32
void __stdcall ggml_budget_cl_destroy(struct _cl_mem *, void * ticket) {
#else
void ggml_budget_cl_destroy(struct _cl_mem *, void * ticket) {
#endif
    ggml_budget_release(reinterpret_cast<uintptr_t>(ticket));
}
int ggml_budget_pin_callback() {
#if defined(__unix__) || defined(__ANDROID__)
    static bool pinned = [] {
        Dl_info info{};
        if (!dladdr(reinterpret_cast<void *>(&ggml_budget_cl_destroy), &info)) return false;
        // Deliberately retain a loader reference for asynchronous driver callbacks.
        return dlopen(info.dli_fname, RTLD_NOW | RTLD_LOCAL) != nullptr;
    }();
    return pinned;
#elif defined(_WIN32)
    static bool pinned = [] {
        HMODULE module;
        return GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
               reinterpret_cast<LPCSTR>(&ggml_budget_cl_destroy), &module) != 0;
    }();
    return pinned;
#else
    return 0;
#endif
}
