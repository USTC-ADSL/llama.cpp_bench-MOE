#include "moe-token-workers.h"
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>

int main() {
    for (int threads : {2, 3, 4, 8}) {
        moe_token_workers workers(threads);
        for (int tokens : {1, 2, 3, 4, 5, 7, 8, 9, 17, 128, 768}) {
            std::vector<float> serial(tokens * 2048), parallel(serial.size());
            std::vector<std::atomic<int>> visits(tokens);
            auto calculate = [](int t, int c) {
                float sum = 0;
                for (int k = 0; k < 4; ++k) { sum = std::fma(float(k + 1) / 7, float(t * 2048 + c - k) / 13, sum); }
                return sum + float(c) / 11;
            };
            for (int t = 0; t < tokens; ++t) {
                visits[t] = 0;
                for (int c = 0; c < 2048; ++c) { serial[t * 2048 + c] = calculate(t, c); }
            }
            for (int round = 0; round < 8; ++round) {
                workers.run(tokens, [&](int begin, int end) {
                    for (int t = begin; t < end; ++t) {
                        ++visits[t];
                        for (int c = 0; c < 2048; ++c) { parallel[t * 2048 + c] = calculate(t, c); }
                    }
                });
                if (std::memcmp(serial.data(), parallel.data(), serial.size() * sizeof(float))) { return 1; }
            }
            for (auto & count : visits) { if (count != 8) { return 2; } }
        }
        bool caught = false;
        try { workers.run(16, [](int begin, int) { if (begin > 0) { throw std::runtime_error("worker exception"); } }); }
        catch (const std::runtime_error &) { caught = true; }
        if (!caught) { return 3; }
        workers.run(16, [](int, int) {});
        // The caller owns the inline accelerator branch. Its failure must still
        // drain workers before per-call buffers/closures can leave scope.
        std::atomic<int> completed{0}; caught=false;
        try {
            workers.run(threads,[&](int begin,int) {
                if(begin==0) { throw std::runtime_error("inline branch exception"); }
                ++completed;
            });
        } catch(const std::runtime_error &) { caught=true; }
        if(!caught || completed!=threads-1) { return 4; }
        workers.run(threads,[](int,int) {});
    }
    std::cout << "PASS: token coverage, repeated dispatch, bitwise FMA order, exception recovery\n";
}
