#pragma once

// This header is for dedicated profile targets. Production translation units
// compile MOE_OBSERVE to nothing, including evaluation of the stage name.
#ifdef MOE_RUNTIME_PROFILE
#include <chrono>
#include <functional>
namespace moe::profile {
using Observer = std::function<void(const char *, double)>;
inline thread_local Observer observer;
class Scope {
public:
    explicit Scope(const char * stage) : stage_(stage), start_(std::chrono::steady_clock::now()) {}
    ~Scope() noexcept {
        if (observer) {
            try { observer(stage_, std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start_).count()); } catch (...) {}
        }
    }
private:
    const char * stage_;
    std::chrono::steady_clock::time_point start_;
};
}
#define MOE_OBSERVE(stage) ::moe::profile::Scope moe_profile_scope(stage)
#else
#define MOE_OBSERVE(stage) ((void) 0)
#endif
