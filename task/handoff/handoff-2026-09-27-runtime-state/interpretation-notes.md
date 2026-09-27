# Interpretation constraints checked during collection

- On fd8657d6, `adb -s fd8657d6 shell 'cat /sys/devices/system/cpu/cpufreq/policy0/related_cpus /sys/devices/system/cpu/cpufreq/policy6/related_cpus'` returned `0 1 2 3 4 5` and `6 7`. Fields `cpu0_mhz`/`cpu6_mhz` refer to these two frequency policies, not independently clocked individual cores. This topology is device-specific.
- `skin_c` reads the `quiet_therm` sensor, not the thermal HAL's composite skin temperature. Missing GPU/NPU clock observations stay null. No measured causal attribution of prior drift to accelerator DVFS.
- Telemetry time includes sensor discovery and preload sampling in decision control, and before/after sampling in each observation-inclusive request. It excludes report serialization, surrounding `getrusage` calls, route/model file I/O, model loading/preparation and warmup. Do not call it full process latency.
- Each preload state produces one process-median calibration observation. Four A cycles and three B cycles provide only eight fitting/six validation points per execution path; empirical residual bounds are not statistical confidence bounds.
- Independent sequential process controls cannot establish the simultaneously optimal backend choice for an individual request. The harbor prompt must not be used for subsequent retuning presented as held-out validation.

## Post-study correction: NPU votes and GPU access (2026-09-27)

User questioned the clock assumptions. Source inspection confirms `ggml/src/ggml-hexagon/htp/main.c:452` requests DCVS disabled (`set_dcvs_enable=TRUE`, `dcvs_enable=FALSE`), core/bus min/max/target corners all MAX, sleep disabled, and v79 protected bus corners. At line488, v75+ HMX uses min/max/target MAX with `HAP_CLK_PERF_HIGH`. Failed power requests terminate session initialization. These are fixed performance requests, not proof of measured constant Hz or constant wall time. The existing native timer (`tests/fullmodel-oracle.cpp:648`) includes decode, synchronization and copying logits; whole-request variation cannot by itself identify HMX clock variation.

The earlier GPU-unavailable statement only described ordinary shell access. On the same fd8657d6, shell `id` is uid2000/context `u:r:shell:s0`, SELinux Enforcing; direct reads of gpuclk, clock_mhz and devfreq/cur_freq return Permission denied, as do the physical-device aliases. However existing `/system/bin/su` works (`su -c id`: uid0/context `u:r:ksu:s0`). Read-only root commands succeed:

```sh
adb -s fd8657d6 shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/gpuclk'"
# 222000000 (Hz)
adb -s fd8657d6 shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/clock_mhz'"
# 222 (MHz)
adb -s fd8657d6 shell "su -c 'cat /sys/class/kgsl/kgsl-3d0/devfreq/cur_freq'"
# 222000000 (Hz)
```

The available-frequency list is1100000000,1050000000,967000000,900000000,832000000,734000000,660000000,607000000,525000000,443000000,389000000,342000000,222000000,160000000Hz. `gpu_busy_percentage` returned0%; `gpubusy` returned581361/1022913. These are one-time snapshots without inference, not load-frequency measurements. Utilization is distinct from clock. No permission/governor/clock setting changed and no inference was rerun.

GPU clock is therefore readable with available root privileges; the previous collection did not use those privileges. Archived null observations and frozen model remain unchanged, and no GPU-clock predictive benefit has been tested. Future sampling should use a persistent read-only privileged collector while retaining the original inference privilege, measure overhead and align timestamps. Do not launch `su` per Expert or silently rerun the inference itself as root. First separate DSP compute from host/RPC/synchronization and shared-memory effects before attributing native latency drift to NPU DVFS.
