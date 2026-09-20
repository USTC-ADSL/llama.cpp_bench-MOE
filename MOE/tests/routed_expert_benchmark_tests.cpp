#include "routed_expert_benchmark.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace routed_expert;
using namespace shared_expert;

static void check(bool condition) { if (!condition) throw std::runtime_error("routed contract failed"); }

int main() {
    try {
        check(parse_tokens("all") == std::vector<unsigned>({1, 3, 32}));
        check(parse_tokens("2,7,64,128,256") == std::vector<unsigned>({2, 7, 64, 128, 256}));
        check(parse_tokens("100000") == std::vector<unsigned>({100000}));
        for (const char * value : {"", "0", "1,0", "1,", ",1", "1,,2", "-1", "1.5", "1,1", "1,01",
                                   "100001", "999999999999999999999999", "all,1"}) {
            bool rejected = false;
            try { parse_tokens(value); } catch (const std::exception &) { rejected = true; }
            check(rejected);
        }
        for (Mode mode : { Mode::gpu, Mode::htp, Mode::serial, Mode::parallel }) {
            size_t end = 0;
            unsigned seen[4][3][16] = {};
            for (const auto & s : layout(mode)) {
                check(s.offset == end && s.offset % 4096 == 0);
                end += s.bytes;
                for (unsigned e : s.expert_ids) {
                    ++seen[s.layer][s.kind][e];
                    if (heterogeneous(mode)) check(s.gpu == (e % 2 == 0));
                }
                const auto type = s.kind == 2 ? QuantType::q4_1 : QuantType::q4_0;
                auto native = gpu_native_layout(type, s.kind == 2 ? 960 : 4096,
                                                s.kind == 2 ? 4096 : 960, s.count);
                (void) native;
                check(tensor_canonical_bytes(type, s.kind == 2 ? 960 : 4096,
                                            s.kind == 2 ? 4096 : 960, s.count) == s.bytes);
            }
            check(end == 440401920);
            for (auto & l : seen) for (auto & k : l) for (unsigned n : k) check(n == 1);
        }
        for (unsigned t : {1u, 2u, 3u, 7u, 32u, 64u, 128u, 256u}) for (unsigned l = 0; l < 4; ++l) {
            bool seen[16] = {};
            for (unsigned s = 0; s < 8; ++s) {
                auto r = routes(t, l, s);
                for (unsigned i = 0; i < t; ++i) {
                    check(r.ids[2*i] != r.ids[2*i+1]);
                    check(r.weights[2*i] + r.weights[2*i+1] == 1.0f);
                    for (unsigned k = 0; k < 2; ++k) {
                        unsigned e = r.ids[2*i+k], b = e % 2;
                        seen[e] = true;
                        check(r.local_ids[b][i] == int(e / 2));
                        check(r.local_weights[b][i] == r.weights[2*i+k]);
                    }
                }
            }
            for (bool value : seen) check(value);
        }
        // Multi-expert conversion must preserve expert order and all scale planes.
        for (unsigned n : {8u, 16u}) for (auto type : {QuantType::q4_0, QuantType::q4_1}) {
            size_t bytes = tensor_canonical_bytes(type, 64, 32, n);
            std::vector<uint8_t> canonical(bytes), native(bytes), restored(bytes);
            for (size_t i = 0; i < bytes; ++i) canonical[i] = (i * 31 + i / 4096) % 251;
            std::string error;
            check(canonical_to_gpu_native(type, 64, 32, n, canonical.data(), bytes, native.data(), bytes, error));
            check(gpu_native_to_canonical(type, 64, 32, n, native.data(), bytes, restored.data(), bytes, error));
            check(restored == canonical);
            check(canonical_to_htp_tiled(type, 64, 32, n, canonical.data(), bytes, native.data(), bytes, error));
            check(htp_tiled_to_canonical(type, 64, 32, n, native.data(), bytes, restored.data(), bytes, error));
            check(restored == canonical);
        }
        std::cout << "routed layout, routes, and multi-expert native round trips passed\n";
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
