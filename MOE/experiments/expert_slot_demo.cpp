#include "expert_graph.h"
#include "ggml_buffer_view.h"
#include <iostream>

using namespace shared_expert;
int main(int argc, char ** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cout << "expert-slot-demo PACK [expert-id]\nLoad one Expert, run CPU/GPU/HTP views, and reuse the Slot.\n";
        return argc < 2 ? 1 : 0;
    }
    try {
        ExpertLoader loader;
        std::string error;
        if (!loader.open(argv[1], error)) throw std::runtime_error(error);
        validate_profile_graph_contract(loader.plans());
        const auto expert = argc > 2 ? static_cast<uint32_t>(std::stoul(argv[2])) : 0;
        ExpertStorage storage(loader.slot_stride());
        for (auto backend : {BackendId::cpu, BackendId::gpu, BackendId::htp}) {
            const auto kind = backend == BackendId::cpu ? moe::Backend::cpu :
                backend == BackendId::gpu ? moe::Backend::gpu : moe::Backend::htp;
            auto device = std::make_shared<moe::GgmlDevice>(kind);
            auto view = std::make_shared<moe::GgmlBufferView>(device, storage.storage());
            auto handle = loader.load(storage.slots(), 0, expert, backend);
            auto lease = storage.slots().acquire(handle);
            lease.retain_backend_view(view);
            {
                ExpertGraph graph(device->get(), view->get(), storage, 1);
                graph.compute_blocking();
                auto output = graph.output();
                std::cout << backend_name(backend) << " expert=" << expert
                          << " output[0]=" << output.at(0) << '\n';
            }
            lease.finish();
        }
        return 0;
    } catch (const std::exception & error) { std::cerr << error.what() << '\n'; return 1; }
}
