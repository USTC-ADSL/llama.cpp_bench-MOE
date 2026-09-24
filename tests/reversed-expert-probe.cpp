#define FULLMODEL_ORACLE_NO_MAIN
#include "fullmodel-oracle.cpp"

int main(int argc, char ** argv) {
    try {
        require(argc == 8, "usage: reversed-expert-probe MODEL CAPTURE_BIN PLAN LAYER OUTPUT REPS ORDER");
        const int il = std::stoi(argv[4]), reps = std::stoi(argv[6]), order = std::stoi(argv[7]);
        require(il >= 0 && il < 24 && reps > 0 && (order == 0 || order == 1), "invalid arguments");
        ggml_backend_load_all();
        std::ifstream pin(argv[3]); json plans; pin >> plans;
        const auto & layer = plans.at("layers").at(il);
        const int nt = layer.at("tokens");
        std::ifstream bin(argv[2], std::ios::binary);
        auto x = binary_read<float>(bin, 2048*nt);
        auto probs = binary_read<float>(bin, 60*nt);
        auto ids = binary_read<int32_t>(bin, 4*nt);
        auto reference = binary_read<float>(bin, 2048*nt);
        require(ids == layer.at("ids").get<std::vector<int32_t>>(), "capture/plan route mismatch");
        placement original = parse_plan(layer.at("selected"));
        require(original.shared == 2, "expected original GPU shared placement");
        placement reversed = original; reversed.name = "cpu_fixed_gpu_remainder_shared_npu"; reversed.shared = 1;
        for (int & owner : reversed.owner) {
            require(owner == 0 || owner == 1, "expected CPU/NPU routed partition");
            if (owner == 1) { owner = 2; }
        }
        ggml_context * raw = nullptr;
        gguf_context_ptr gguf(gguf_init_from_file(argv[1], {true, &raw})); ggml_context_ptr meta(raw);
        std::unique_ptr<FILE, decltype(&fclose)> file(fopen(argv[1], "rb"), fclose);
        require(gguf && meta && file, "model open failed");
        source src{gguf.get(), meta.get(), file.get()};
        device cpu("CPU"), npu("HTP0"), gpu("GPUOpenCL"); device * devs[] = {&cpu, &npu, &gpu};
        std::unique_ptr<partitioned_ffn> variants[2];
        const placement policies[] = {original, reversed};
        // Reverse initialization order across processes as well as measured execution order.
        for (int i = 0; i < 2; ++i) {
            int p = (i + order) % 2;
            variants[p] = std::make_unique<partitioned_ffn>(devs, src, il, nt, policies[p], true);
            variants[p]->expected_ids = ids;
        }
        json report = {{"tokens", nt}, {"layer", il}, {"order", order}, {"plans", {plan_json(original), plan_json(reversed)}},
                       {"scope", "paired complete FFN replay; fixed CPU Expert IDs and real captured routes; compact resident banks; all transfers/join included; attention excluded"},
                       {"samples", json::array()}};
        std::vector<double> samples[2];
        size_t warmed[2]{};
        for (int r = -3; r < reps; ++r) {
            for (int j = 0; j < 2; ++j) {
                int p = (j + order + r + 4) % 2;
                rusage before{}, after{}; getrusage(RUSAGE_SELF, &before);
                auto start = timer::now();
                auto output = variants[p]->run(x, probs, ids, true);
                double us = elapsed(start); getrusage(RUSAGE_SELF, &after);
                auto check = chain_difference(output, reference, nt);
                if (r == -1) { warmed[p] = variants[p]->allocations; }
                if (r >= 0) {
                    require(warmed[p] == variants[p]->allocations, "timed graph allocation");
                    samples[p].push_back(us);
                    report["samples"].push_back({{"round", r}, {"policy", p}, {"total_us", us}, {"phases", variants[p]->last},
                        {"check", check}, {"major_faults", after.ru_majflt-before.ru_majflt}, {"minor_faults", after.ru_minflt-before.ru_minflt}});
                }
            }
        }
        report["timing"] = {stats(samples[0], nt), stats(samples[1], nt)};
        std::ifstream status("/proc/self/status"); report["process_status"] = std::string((std::istreambuf_iterator<char>(status)), {});
        std::ofstream out(argv[5]); out << report.dump(2) << '\n'; require(out.good(), "result write failed");
        fprintf(stderr, "REVERSED_RESULT original=%.3f ms reversed=%.3f ms\n",
                report["timing"][0]["median_us"].get<double>()/1000, report["timing"][1]["median_us"].get<double>()/1000);
        return 0;
    } catch (const std::exception & e) { fprintf(stderr, "ERROR: %s\n", e.what()); return 1; }
}
