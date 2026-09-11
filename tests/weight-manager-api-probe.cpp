#include "llama-weight-manager.h"
#include "ggml-cpp.h"
#include "../vendor/nlohmann/json.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

template<class F> static void rejects(F f) {
    try { f(); } catch (const std::runtime_error &) { return; }
    throw std::runtime_error("expected request rejection");
}

int main(int argc, char ** argv) {
    try {
        require(argc == 2, "usage: weight-manager-api-probe OUTPUT_PREFIX");
        const std::string prefix = argv[1];
        const std::string source = prefix + ".source.bin";
        const std::string config = prefix + ".config.json";
        const std::string report = prefix + ".weights.json";
        ggml_backend_load_all();
        auto cpu = ggml_backend_dev_by_name("CPU");
        require(cpu != nullptr, "CPU missing");
        ggml_backend_ptr backend(ggml_backend_dev_init(cpu, nullptr));
        ggml_context_ptr ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}));
        require(backend && ctx, "initialization failed");
        auto * expert = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 32, 32, 3);
        auto * whole = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 32);
        auto * expert_view = ggml_view_3d(ctx.get(), expert, 32, 32, 3, expert->nb[1], expert->nb[2], 0);
        auto * whole_view = ggml_view_2d(ctx.get(), whole, 32, 32, whole->nb[1], 0);
        ggml_set_name(expert, "expert");
        ggml_set_name(whole, "whole");
        const json policy = {
            {"version", 1}, {"budgets", {{"CPU", {{"total_mib", 8}, {"expert_slots", 2}}}}},
            {"weights", {{{"pattern", "^expert$"}, {"residency", "swap"}, {"pin_experts", {0}}},
                         {{"pattern", "^whole$"}, {"residency", "swap"}}}}, {"report", report},
        };
        { std::ofstream out(config); out << policy.dump(2); require(bool(out), "config write failed"); }
        std::vector<float> data(32 * 32 * 4, 0.5f);
        { std::ofstream out(source, std::ios::binary);
          out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
          require(bool(out), "source write failed"); }
        llama_weight_manager manager(config.c_str());
        manager.register_weight(expert, 0, 0, cpu);
        manager.register_weight(whole, 0, ggml_nbytes(expert), cpu);
        llama_files files;
        files.emplace_back(new llama_file(source.c_str(), "rb"));
        manager.adopt_files(std::move(files));
        manager.prefetch(backend.get(), expert_view, {0, 1});
        manager.prefetch(backend.get(), expert, {0, 2});
        rejects([&] { manager.evict(expert_view); });
        rejects([&] { manager.prefetch(backend.get(), expert); });
        rejects([&] { manager.prefetch(backend.get(), whole, {0}); });
        manager.prefetch(backend.get(), whole_view);
        manager.evict(whole_view);
        manager.prefetch(backend.get(), whole);
        manager.write_report();
        std::ifstream in(report);
        const auto result = json::parse(in).at("devices").at("CPU");
        require(result.at("misses") == 5 && result.at("hits") == 1 && result.at("evictions") == 2,
                "cache control counters differ");
        require(result.at("cache_bytes").get<size_t>() < 16384, "unbounded cache allocation");
        printf("prefetch/view-evict/pinning/invalid-requests PASS\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    return 0;
}
