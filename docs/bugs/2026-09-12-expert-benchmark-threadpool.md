# Expert Benchmark Included Disposable CPU Threadpool Work

## Symptom And Trigger

The initial standalone expert-compute probe showed large single-token CPU latency
and run-to-run variation despite correct expert outputs. run01..03 CPU full-FFN
medians at T1 were 608.750, 414.479 and 482.760 microseconds on fd8657d6.

The probe initialized the CPU backend and set four threads, but did not attach a
persistent pool. Repeated small graph calls therefore measured worker lifecycle
overhead in addition to expert computation.

## Root Cause

ggml/src/ggml-cpu/ggml-cpu.cpp:227 initializes the backend threadpool to NULL.
ggml/src/ggml-cpu/ggml-cpu.c:3403 creates a disposable pool when it is NULL,
and :3461 frees that pool after the graph. Setting n_threads alone does not create
a reusable pool. tools/llama-bench/llama-bench.cpp:2433 instead creates and attaches
a persistent pool outside inference measurement.

The initial probe's results were valid raw backend call times, but were not the
intended steady CPU comparison against other initialized backends. This was a
benchmark setup defect, not a production CPU correctness regression.

## Change

tests/expert-compute-probe.cpp obtains ggml_threadpool_new/free and
ggml_backend_cpu_set_threadpool from the CPU registry. It creates and attaches one
default four-thread pool per CPU backend, outside all timers, and frees it after
the backend. JSON records cpu_persistent_threadpool. No backend source changes.

## Verification

The probe is rebuilt with the Android NDK command recorded in
task/handoff/handoff-2026-09-12-expert-compute/handoff-01-results.md and deployed
with binary SHA256 verification. steady01..03 rerun CPU/OpenCL/HTP HMX/HVX with
100 repetitions per supported token shape, ten warmups and before/after numerical
checks. Final numerical tables, exact commands and evidence are in that handoff.

steady01/02 CPU T1 full-FFN medians are 77.891/77.969 microseconds. The aggregate
script requires persistent pools in every final CPU result and excludes run01..03
from final timing. Separate profile03 verifies final HTP paths with the same binary.

## Limits

Phone clocks and affinity remain uncontrolled. The benchmark repeatedly computes
the same resident expert; it does not measure CPU cache misses after swapping in
a different expert. Pure HVX T128 is separately reported unsupported due to the
current VTCM sizing; no backend fallback or VTCM policy is changed.
