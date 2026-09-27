#pragma once
#include "ggml-hexagon.h"

struct moe_device_timing {
    ggml_backend_hexagon_get_timing_fn read = nullptr;
    explicit moe_device_timing(bool enabled) {
        if (!enabled) { return; }
        auto reg = ggml_backend_reg_by_name("HTP");
        require(reg != nullptr, "HTP registry missing for timing");
        read = reinterpret_cast<ggml_backend_hexagon_get_timing_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_timing"));
        require(read != nullptr, "HTP timing interface missing");
        require(snapshot().enabled, "HTP timing must be enabled before backend loading");
    }
    ggml_hexagon_timing snapshot() const {
        ggml_hexagon_timing value{};
        if (read) { require(read(&value,sizeof(value)) && value.version == 1, "HTP timing snapshot failed"); }
        return value;
    }
    json delta(const ggml_hexagon_timing & before, const ggml_hexagon_timing & after) const {
        static const char * names[] = {"dsp_batches", "dsp_batch_us", "dsp_batch_cycles", "dsp_ops", "dsp_op_us", "dsp_op_cycles",
            "host_queue_writes", "host_queue_write_us", "host_queue_reads", "host_queue_read_us"};
        static_assert(sizeof(names)/sizeof(*names) == GGML_HEXAGON_TIMING_COUNT, "timing schema mismatch");
        json result={{"profile_mode",after.profile_mode},{"devices",after.devices}};
        for(size_t i=0;i<GGML_HEXAGON_TIMING_COUNT;++i) {
            require(after.counters[i] >= before.counters[i], "HTP counter decreased");
            result[names[i]]=after.counters[i]-before.counters[i];
        }
        return result;
    }
};
