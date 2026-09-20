#include "expert_loader.h"
#include "expert_layout.h"
#include "runtime_observer.h"

namespace shared_expert {

bool ExpertLoader::open(const std::string & path, std::string & error) {
    close();
    if (!expert_pack::load_metadata(path, metadata_, error)) return false;
    if (!source_.open(path, error)) { metadata_ = {}; return false; }
    pack_path_ = path;
    try { plans_ = std::make_shared<expert_native_plans>(build_expert_native_plans(*this)); }
    catch (const std::exception & e) { error = e.what(); close(); return false; }
    return true;
}

void ExpertLoader::close() {
    source_.close(); metadata_ = {}; plans_.reset(); pack_path_.clear(); gguf_source_ = false;
}

bool ExpertLoader::use_gguf_source(const std::string & path, std::string & error) {
    if (!plans_) { error = "open pack metadata before selecting a source"; return false; }
    expert_source::Reader source;
    if (!source.open(path, error)) return false;
    for (const auto & entry : metadata_.entries) {
        if (entry.source_tensor_offset > source.file_size() ||
            entry.payload_bytes > source.file_size() - entry.source_tensor_offset) {
            error = "GGUF source range exceeds file bounds"; return false;
        }
    }
    source_ = std::move(source); gguf_source_ = true;
    return true;
}

const expert_native_plans & ExpertLoader::plans() const {
    if (!plans_) throw std::logic_error("Expert source is not open");
    return *plans_;
}
size_t ExpertLoader::slot_stride() const { return expert_slot_stride(plans()); }

ExpertHandle ExpertLoader::load(ExpertSlotArena & arena, uint32_t slot, uint32_t expert,
                               BackendId backend, bool verify) {
    auto lease = arena.begin_load(slot);
    return load(lease, expert, backend, verify);
}

ExpertHandle ExpertLoader::load(WriteLease & lease, uint32_t expert, BackendId backend, bool verify) try {
    const auto layout = prepare(lease, expert, backend, verify);
    return lease.publish({source_layer(), expert}, backend, layout);
} catch (...) {
    lease.cancel();
    throw;
}

ExpertSlotLayout ExpertLoader::prepare(WriteLease & lease, uint32_t expert, BackendId backend, bool verify) try {
    if (lease.size() < slot_stride()) throw std::invalid_argument("expert exceeds slot capacity");
    ExpertSlotLayout layout;
    switch (backend) {
        case BackendId::cpu: layout = ExpertSlotLayout::cpu_canonical_q4; break;
        case BackendId::gpu: layout = ExpertSlotLayout::gpu_q4_soa_trans4; break;
        case BackendId::htp: layout = ExpertSlotLayout::htp_q4_tiled32; break;
        default: throw std::invalid_argument("backend is required");
    }
    const std::array<const tensor_native_plan *, 3> tensors{&plans().gate, &plans().up, &plans().down};
    std::array<std::vector<uint8_t> *, 3> workspaces{&workspace_.gate, &workspace_.up, &workspace_.down};
    std::vector<expert_source::TensorDestination> destinations;
    bool direct = backend == BackendId::cpu;
    for (const auto * tensor : tensors)
        direct = direct && lease.data(tensor->slot_offset, tensor->slot_bytes);
    for (size_t i = 0; i < tensors.size(); ++i) {
        const auto * entry = tensor_entry(expert, tensors[i]->kind);
        if (!entry) throw std::out_of_range("expert ID");
        void * destination;
        if (direct) destination = lease.data(tensors[i]->slot_offset, tensors[i]->slot_bytes);
        else { workspaces[i]->resize(entry->payload_bytes); destination = workspaces[i]->data(); }
        destinations.push_back({gguf_source_ ? entry->source_tensor_offset : entry->payload_offset,
                                static_cast<size_t>(entry->payload_bytes), destination});
    }
    std::string error;
    { MOE_OBSERVE("expert_read");
      if (!source_.read(destinations, error)) throw std::runtime_error(error);
    }
    if (verify) {
        for (size_t i = 0; i < tensors.size(); ++i)
            if (crc32(destinations[i].destination, destinations[i].size) !=
                tensor_entry(expert, tensors[i]->kind)->payload_crc32)
                throw std::runtime_error("Expert payload CRC mismatch");
    }
    if (!direct) {
        for (size_t i = 0; i < tensors.size(); ++i) {
            const auto & plan = *tensors[i];
            void * destination = lease.data(plan.slot_offset, plan.slot_bytes);
            const bool upload = !destination;
            if (upload) { native_workspace_.resize(plan.slot_bytes); destination = native_workspace_.data(); }
            { MOE_OBSERVE("expert_repack");
              convert_expert_tensor(layout, plan, *workspaces[i], destination, plan.slot_bytes); }
            if (upload) lease.write(plan.slot_offset, destination, plan.slot_bytes);
        }
    }
    return layout;
} catch (...) {
    lease.cancel();
    throw;
}

const PackTensorEntry * ExpertLoader::tensor_entry(uint32_t expert, PackTensorKind kind) const {
    return expert_pack::find_entry(metadata_, expert, kind);
}

bool ExpertLoader::read_expert(uint32_t expert, CanonicalExpert & output, std::string & error) const {
    CanonicalExpert result;
    if (!read_expert_reuse(expert, result, error)) return false;
    output = std::move(result);
    return true;
}

bool ExpertLoader::read_expert_reuse(uint32_t expert, CanonicalExpert & output, std::string & error) const {
    std::array<std::vector<uint8_t> *, 3> payloads{ &output.gate, &output.up, &output.down };
    std::vector<expert_source::TensorDestination> destinations;
    for (uint32_t i = 0; i < payloads.size(); ++i) {
        const auto * entry = tensor_entry(expert, static_cast<PackTensorKind>(i));
        if (!entry) { error = "Expert source is closed or Expert ID is out of range"; return false; }
        payloads[i]->resize(entry->payload_bytes);
        destinations.push_back({gguf_source_ ? entry->source_tensor_offset : entry->payload_offset,
                                payloads[i]->size(), payloads[i]->data()});
    }
    if (!source_.read(destinations, error)) return false;
    for (uint32_t i = 0; i < payloads.size(); ++i) {
        if (crc32(payloads[i]->data(), payloads[i]->size()) !=
                tensor_entry(expert, static_cast<PackTensorKind>(i))->payload_crc32) {
            error = "Expert payload CRC mismatch";
            return false;
        }
    }
    return true;
}
} // namespace shared_expert
