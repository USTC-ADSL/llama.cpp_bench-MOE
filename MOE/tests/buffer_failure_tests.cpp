#include "buffer.h"
#include <iostream>
#include <limits>
static void require(bool value) { if (!value) throw std::runtime_error("resource failure invariant"); }
template<class F> void rejects(F && f) {
    bool failed = false; try { f(); } catch (const std::exception &) { failed = true; } require(failed);
}
int main() {
    try {
        auto session = moe::RpcmemSession::acquire();
        require(session->symbol("moe_nonexistent_symbol", false) == nullptr);
        rejects([&] { session->symbol("moe_nonexistent_symbol"); });
        if (!session->symbol("rpcmem_alloc2", false))
            require(session->allocate(static_cast<size_t>(std::numeric_limits<int>::max()) + 1) == nullptr);
        auto mode = reinterpret_cast<void (*)(int)>(session->symbol("test_fail_mode"));
        auto count = reinterpret_cast<int (*)(int)>(session->symbol("test_count"));
        mode(1); rejects([] { moe::SharedDmaBuffer buffer(4096); });
        require(count(1) == 0 && count(2) == 0);
        mode(2); rejects([] { moe::HtpPrivateBuffer buffer(4096); });
        require(count(1) == 1 && count(2) == 1 && count(3) == 0);
        mode(0);
        { auto shared = std::make_shared<moe::SharedDmaBuffer>(4096);
          auto private_htp = std::make_shared<moe::HtpPrivateBuffer>(4096);
          require(shared->allocation_id() != private_htp->allocation_id());
          require(shared->supports(moe::Backend::gpu) && !private_htp->supports(moe::Backend::gpu));
          auto view_owner = shared; shared.reset(); require(count(3) == 2);
          view_owner.reset(); require(count(3) == 1); }
        require(count(0) == 1 && count(1) == count(2) && count(3) == 0 && count(4) == 0);
        std::cout << "allocation/export failure rollback and owner lifetime passed\n";
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
