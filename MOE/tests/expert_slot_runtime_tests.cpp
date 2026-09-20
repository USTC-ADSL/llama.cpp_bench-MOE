#include "expert_slot_arena.h"
#include <iostream>
#include <thread>
#include <vector>
#include <cstring>

using namespace shared_expert;
static void require(bool value) { if (!value) throw std::runtime_error("assertion failed"); }
template<class F> static void rejects(F && f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    require(rejected);
}
static ExpertHandle publish(ExpertSlotArena & arena, uint32_t slot, uint32_t expert) {
    auto write = arena.begin_load(slot);
    return write.publish({0, expert}, BackendId::cpu, ExpertSlotLayout::cpu_canonical_q4);
}
int main() {
    try {
        auto buffer = std::make_shared<moe::CpuPrivateBuffer>(4096);
        ExpertSlotArena arena(buffer, 4, 1024);
        auto old = publish(arena, 0, 1000);
        { auto read = arena.acquire(old); rejects([&] { arena.begin_load(0); });
          rejects([&] { arena.acquire(old); }); }
        { auto read = arena.acquire(old); require(read.size() == 1024); }
        { auto write = arena.begin_load(0); require(!old.valid()); }
        require(arena.snapshot(0).state == ExpertSlotState::empty);
        require(!arena.lookup({0, 1000}));
        rejects([&] { arena.acquire(old); });
        auto current = publish(arena, 0, 1000);
        ExpertSlotArena other(buffer, 4, 1024);
        rejects([&] { other.acquire(current); });
        { auto write = arena.begin_load(1);
          rejects([&] { write.publish({0, 1000}, BackendId::cpu, ExpertSlotLayout::cpu_canonical_q4); }); }
        require(arena.snapshot(1).state == ExpertSlotState::empty);
        { auto write = arena.begin_load(1); auto moved = std::move(write);
          rejects([&] { write.data(); });
          rejects([&] { moved.write(1000, "x", 30); });
          rejects([&] { moved.publish({0, 1}, BackendId::gpu, ExpertSlotLayout::gpu_q4_soa_trans4); }); }
        auto second = publish(arena, 1, 1);
        int waits = 0;
        auto completion = std::make_shared<moe::Completion>([&] { ++waits; });
        { auto a = arena.acquire(current); auto b = arena.acquire(second);
          a.complete_after(completion); b.complete_after(completion);
          a.finish(); b.finish(); a.finish(); }
        require(waits == 1);
        require(arena.snapshot(0).state == ExpertSlotState::ready);
        // RAII owners keep storage alive after the Arena facade is destroyed.
        std::weak_ptr<moe::Buffer> weak;
        ReadLease outstanding;
        { auto storage = std::make_shared<moe::CpuPrivateBuffer>(128); weak = storage;
          ExpertSlotArena local(storage, 1, 128);
          outstanding = local.acquire(publish(local, 0, 0)); }
        require(!weak.expired()); outstanding.finish(); require(weak.expired());
        // Completion failure is sticky: no retry masquerades as success.
        int failures = 0;
        moe::Completion failed([&] { ++failures; throw std::runtime_error("device failed"); });
        rejects([&] { failed.wait(); }); rejects([&] { failed.wait(); });
        require(failures == 1 && !failed.completed());
        // Scatter placement protects disjoint physical ranges without a second allocation.
        ExpertSlotArena scatter(buffer, {{{0, 256, 4}, {4, 512, 4}}, {{0, 768, 8}}}, 8);
        { auto write = scatter.begin_load(0);
          require(write.data() == nullptr); write.write(0, "abcdefgh", 8);
          auto handle = write.publish({1, 0}, BackendId::cpu, ExpertSlotLayout::cpu_canonical_q4);
          auto read = scatter.acquire(handle); require(read.ranges().size() == 2 && !read.data()); }
        require(std::memcmp(static_cast<char *>(buffer->host_data()) + 256, "abcd", 4) == 0);
        require(std::memcmp(static_cast<char *>(buffer->host_data()) + 512, "efgh", 4) == 0);
        rejects([&] { ExpertSlotArena bad(buffer, {{{0, 0, 8}}, {{0, 4, 8}}}, 8); });
        rejects([&] { ExpertSlotArena bad(buffer, {{{0, 1, 8}}}, 8, 4); });
        // Failed completion preserves allocation AND imported view, even after facade destruction.
        std::weak_ptr<moe::Buffer> failed_storage;
        std::weak_ptr<int> failed_view;
        { auto storage = std::make_shared<moe::CpuPrivateBuffer>(128); failed_storage = storage;
          ExpertSlotArena local(storage, 1, 128);
          auto view = std::make_shared<int>(42); failed_view = view;
          auto lease = local.acquire(publish(local, 0, 0)); lease.retain_backend_view(view);
          lease.complete_after(std::make_shared<moe::Completion>([] { throw std::runtime_error("lost completion"); }));
          rejects([&] { lease.finish(); }); rejects([&] { local.begin_load(0); }); }
        require(!failed_storage.expired() && !failed_view.expired());
        // The one-shot completion is shared safely across backend worker joins.
        waits = 0;
        auto batch = std::make_shared<moe::Completion>([&] { ++waits; });
        std::thread first([&] { batch->wait(); }), last([&] { batch->wait(); });
        first.join(); last.join(); require(waits == 1);
        std::cout << "runtime ownership/state/completion tests passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
