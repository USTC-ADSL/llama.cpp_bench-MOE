# Interpretation constraints checked during collection

- On fd8657d6, `adb -s fd8657d6 shell 'cat /sys/devices/system/cpu/cpufreq/policy0/related_cpus /sys/devices/system/cpu/cpufreq/policy6/related_cpus'` returned `0 1 2 3 4 5` and `6 7`. Fields `cpu0_mhz`/`cpu6_mhz` refer to these two frequency policies, not independently clocked individual cores. This topology is device-specific.
- `skin_c` reads the `quiet_therm` sensor, not the thermal HAL's composite skin temperature. Missing GPU/NPU clock observations stay null. No measured causal attribution of prior drift to accelerator DVFS.
- Telemetry time includes sensor discovery and preload sampling in decision control, and before/after sampling in each observation-inclusive request. It excludes report serialization, surrounding `getrusage` calls, route/model file I/O, model loading/preparation and warmup. Do not call it full process latency.
- Each preload state produces one process-median calibration observation. Four A cycles and three B cycles provide only eight fitting/six validation points per execution path; empirical residual bounds are not statistical confidence bounds.
- Independent sequential process controls cannot establish the simultaneously optimal backend choice for an individual request. The harbor prompt must not be used for subsequent retuning presented as held-out validation.
