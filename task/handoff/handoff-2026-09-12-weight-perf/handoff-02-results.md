# Weight Capacity Throughput: Final Results

## Outcome And Scope

On Android fd8657d6 (Redmi K80 Pro), the all-fixed manager initially lost CPU
REPACK optimizations. Commit d08ea1ad6 fixes this and skips unused report work.
CPU/OpenCL/HTP all-fixed throughput is now within 3% of normal defaults. Mixed
placement shows no slowdown, but the positive TG difference is not fully isolated.
The 6/5/4 GiB capacity sweeps are complete. One CPU4GiB semantic limitation remains;
the controlled layout comparison below is essential when interpreting correctness.

Model: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf.
Baseline/build/root-cause details: handoff-01.md and
../../../docs/bugs/2026-09-12-fixed-weight-cpu-performance.md.

## Protocol

Four CPU threads, batch128, ubatch4, standard llama-bench warmup. PP128 and TG32
are separate synthetic workloads, not a single119-input/32-output chat request.
Each baseline has five repetitions; each capacity point has three. Timers include
backend synchronization and demand reads/uploads, exclude model load. Reports are
disabled in timed configs; separate PP0/TG1 diagnostics establish capacity.
Weight cache state persists across repetitions and PP/TG contexts, so these are
warm-run throughput samples. There is no strict fixed-clock or simultaneous A/B
control. Temperature/process memory snapshots and native sample times are retained.

Mixed placement uses -dev GPUOpenCL/HTP0 -ngl17 -ts8/9: layers0..7 CPU,
8..15 OpenCL,16..23 HTP, with corresponding manager placement rules.

## Throughput

Mean tok/s, default versus all-fixed after the fix:

| Backend | Default PP128 | Fixed PP128 | Default TG32 | Fixed TG32 |
| --- | ---: | ---: | ---: | ---: |
| CPU | 65.699 | 67.287 | 31.892 | 32.149 |
| OpenCL | 19.196 | 19.183 | 28.338 | 28.229 |
| HTP | 48.248 | 49.366 | 15.789 | 16.057 |
| Mixed | 31.530 | 31.884 | 21.301 | 23.179 |

Mixed no-op-offload control: default/fixed PP33.815/33.244, TG21.698/22.776;
fixed TG standard deviation1.357. Do not interpret its positive gap as a proven
manager speedup. All-fixed CPU capacity is8143910912bytes (7.584608GiB), with
zero swap operations in the semantic regression report.

Shrinking weight allocation capacity, each cell PP128 / TG32 tok/s:

| Backend | 6 GiB | 5 GiB | 4 GiB |
| --- | ---: | ---: | ---: |
| CPU | 32.574 / 19.079 | 23.393 / 15.116 | 18.444 / 10.654 |
| OpenCL | 8.608 / 6.536 | 6.030 / 4.753 | 4.532 / 3.426 |
| HTP | 14.424 / 6.487 | 9.659 / 4.420 | 7.832 / 3.261 |
| Mixed | 14.569 / 8.706 | 8.029 / 6.517 | 6.584 / 4.425 |

evidence/throughput-summary.json contains all samples and standard deviations.
Some mixed results have10..19% relative standard deviation; displayed precision
does not imply corresponding measurement accuracy. Shrinking incurs synchronous
miss handling, upload/layout conversion and isolated-node scheduling; CPU streamed
experts also use canonical kernels. No stage-specific cost attribution was measured.

## Allocation Policy And Limits

GiB means2^30bytes. The target is fixed buffers plus allocated weight cache;
runtime, reserve, staging, driver/private buffers and OS page cache are separate.
Each configured backend receives256MiB allowance above its weight allocation,
including128MiB reserve. Fixed buffers are application-owned, not OS mlocked.
Actual VmSwap was nonzero in several runs; device shell locked-memory soft/hard
limits were65536bytes. These experiments do not demonstrate physical residency
of exactly6/5/4GiB, nor do they disable Android swap.

Nonexpert weights are fixed. Some entire expert banks are fixed and remaining
banks share small LRU pools. Single-backend fixed selection follows layer order;
mixed rotates backend layer groups. This is a specific static policy, not an
optimized expert-selection curve. Exact rules are saved in every .config.json.

| Target | Single-backend actual GiB | Fixed expert banks / slots per pool | Mixed actual GiB | Fixed banks / slots |
| --- | --- | --- | ---: | --- |
| 6 | 5.998457..5.998564 | 53 / 45 | 5.998493 | 51 / 35 |
| 5 | 4.998426..4.998528 | 42 / 44 | 4.992418 | 40 / 34 |
| 4 | 3.998396..3.998492 | 31 / 43 | 3.995407 | 29 / 34 |

Directly expanding one expert pool was rejected by the OpenCL1GiB allocation
limit before inference (1536-base test requested2388.38MiB). Source inspection
also found potential intermediate32-bit offset overflow at huge slot IDs. Kernels
were not changed; those unsupported sizes are not claimed numerically validated.

## Numerical And Semantic Validation

tests/weight-slice-probe.cpp now accepts DEVICE [TOKENS] [HIGH_SLOT]. It compares
Qwen-sized2048x1408Q4_0 matrices at slots0..7 versus128..135. CPU/OpenCL/HTP each
passed tokens1 and4 with max_abs=0. The original OpenCL tokens4 replacement test
also passed12rounds, including reset at round6. Commands and verified probe hash
are in evidence/slice-results.json; the selected logs are retained.

The actual4GiB configs were tested with a119-token delivery-note prompt, greedy
generation, natural EOG enabled, maximum32output tokens, default CPU layouts.
All four executions exited0 and every checked logit was finite:

| Placement | Output tokens | Assessment |
| --- | ---: | --- |
| CPU | 32 | Four facts correct, but appended incorrect claim that delivery was cancelled on Monday; no EOG suppression |
| OpenCL | 23 | Correct facts/explanation, natural EOG |
| HTP | 32 | Correct facts and explanation so far; sentence truncated at token cap |
| Mixed | 32 | Correct facts and continuation; token cap reached |

This suite did not pass general semantic correctness: keyword checks would miss
the CPU contradiction. Evidence/s4semantic01-results.json retains exact commands,
outputs and probe hash; per-case prompts/tokens/summaries are also retained.

CPU layout control canon01 used the same libs, prompt and actual4GiB config, but
omitted WEIGHT_PROBE_DEFAULT_LAYOUT in both all-fixed and swap runs. Both produced
the same correct18-token labeled answer and natural EOG. All2734848saved float32
logits were bit-identical (RMSE/max_abs0), SHA256:
667a7ea306744a75ef4c9c974dcb56e802db05f4e526c6eb12275148d2a80dbe.
Saved logits exclude the terminating EOG decision; both summaries record it.

With optimized CPU layouts, all-fixed perfreg01 versus s4semantic01-cpu already
differs in the first logit vector (max_abs0.437613); over the first18steps,
max_abs1.909296 and RMSE0.132362. Both share the first18generated tokens before
termination diverges. This control supports layout-dependent numerical behavior,
not a demonstrated source/slice corruption on this case. It does not isolate the
first differing ARM arithmetic instruction or guarantee every prompt is sound.

CPU_REPACK set_tensor requires offset0/full tensor size
(ggml/src/ggml-cpu/repack.cpp:4733), so using it for partial expert uploads needs
a per-expert repack implementation. This remains future work; no semantic workaround
or disabling of optimized CPU kernels was applied to the throughput configurations.

## Commands And Evidence

Build/deployment commands are in handoff-01.md. The slice probe was rebuilt with:

```sh
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include tests/weight-slice-probe.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-slice-probe
```

Additional sweeps, following the recorded6GiB sweep:

```sh
python3 tests/run-weight-perf.py --tag s5cap01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 5 --reps 1 --prompt 0 --gen 1 --report
python3 tests/run-weight-perf.py --tag s5run01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 5 --reps 3
python3 tests/run-weight-perf.py --tag s4cap01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 4 --reps 1 --prompt 0 --gen 1 --report
python3 tests/run-weight-perf.py --tag s4run01 --case cpu-swap gpu-swap htp-swap mixed-swap --weight-gib 4 --reps 3
```

Every benchmark .json records its complete adb command, parsed native samples and
device snapshots. Capacity reports are .weights.json. Timed configs match their
diagnostic configs except report. Deployment identity uses cap01-hashes.json:
binary build metadata predates d08ea1ad6 and is not itself the final source identity.
Exact semantic/control/slice commands are in their respective *results.json files.
Raw logits are retained locally and ignored by Git; compact outputs/comparisons
are committed. Large routine stdout logs are similarly ignored.

## Closure

Production fix checkpoint: d08ea1ad6. This stage adds the supported slice probe,
capacity/semantic evidence and final documentation. All requested throughput
points and the CPU layout control completed. CPU optimized swap numerical
consistency remains a known limitation, not a completed fix. No additional
throughput rerun is needed absent a code or protocol change. Device artifact
cleanup and final validation are recorded in tasklist.md before the test commit.

Final checks passed: independently recomputed all 40 PP/TG means and standard
deviations from samples_ns, checked all 12 capacity/config pairs, parsed JSON
artifacts, compared canonical logits/tokens and optimized first18 token prefixes.
Commands: `python3 -m py_compile tests/run-weight-perf.py tests/run-weight-manager.py`
and `git diff --check`. Device probe SHA256 values were rechecked before cleanup.
After confirming no benchmark/probe processes remained, removed only the two owned
directories with `adb -s fd8657d6 shell 'rm -rf /data/local/tmp/llama-weight-perf /data/local/tmp/llama-weight-manager'`.
Both directories are absent; the original GGUF remains (8149865792 bytes).
No active test sessions or required device tests remain. Reproduction starts with
the existing runner's --deploy option and the saved configs.
