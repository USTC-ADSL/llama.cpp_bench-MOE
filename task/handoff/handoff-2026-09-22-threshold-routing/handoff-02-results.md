# Full Transformer Layer Routing Results

## Conclusion

For this 256-token text prefill on Redmi K80 Pro, assigning experts with fewer
than 16 tokens to CPU and the rest to NPU does not improve complete layer latency
over keeping every routed expert on NPU. Parallel mixed execution is faster than
all-CPU experts, but slower than all-NPU experts in both the one- and two-layer
experiments. This is a result for one prompt and threshold, not a rejection of
all heterogeneous scheduling policies.

The user requested resident weights and complete Transformer layers. The experiment
includes attention, RoPE, normalization, router/top-4, routed and shared experts,
residuals, activation transfers, synchronization and aggregation. CPU handles all
non-routed operations identically across policies. Both CPU and NPU expert weight
representations are ready before timing. Layer 1 consumes layer 0's actual output.
See handoff-01.md for implementation, timing boundaries and initial diagnostics.

## Measured Latency

Three independent process runs, three warmups per policy/depth per run and twenty
measured rounds per run. Policy order rotates each round. Each cell below pools
60 directly measured totals, not a sum of component medians. There are 720 total
formal timing records across six policies and two depths.

| Expert policy | One layer median / P95, ms | Two layers median / P95, ms |
| --- | ---: | ---: |
| All CPU, original top-k layout | 220.866 / 270.398 | 494.193 / 585.950 |
| All NPU, original top-k layout | 146.175 / 178.275 | 312.070 / 354.643 |
| All CPU, flattened pair layout | 226.914 / 275.806 | 503.459 / 587.540 |
| All NPU, flattened pair layout | 146.849 / 179.270 | 314.813 / 357.980 |
| Threshold 16, serial CPU/NPU | 163.406 / 203.384 | 364.235 / 417.800 |
| Threshold 16, parallel CPU/NPU | 153.526 / 191.278 | 341.916 / 400.092 |

By pooled medians, mixed parallel increases total latency relative to all-NPU by
5.03% for one layer and 9.56% for two. Relative to all-CPU it reduces latency by
30.49% and 30.81%, respectively. The flattened all-NPU control reaches the same
conclusion, so the loss is not solely the mixed layout's gather overhead.

Within-round paired mixed-parallel minus all-NPU median latency increases are
8.935 ms for one layer and 28.244 ms for two. Mixed wins only 1/60 pairs in each
depth. Every run's median paired difference favors all-NPU:

| Run | One-layer all-NPU / mixed parallel, ms | Two-layer all-NPU / mixed parallel, ms |
| --- | ---: | ---: |
| run01 | 139.646 / 144.772 | 288.688 / 312.876 |
| run02 | 146.175 / 153.271 | 314.575 / 342.302 |
| run03 | 166.262 / 179.337 | 347.963 / 381.697 |

Clocks and CPU affinity are uncontrolled. Absolute times rise across the runs;
thermal snapshots are preserved, and the OS-reported thermal status is not a clock
lock. The consistent paired direction is more useful than treating a pooled time
as a universal device latency. Repeated warm runs on one phone are not independent
samples of the entire deployment population.

## Why the Threshold Loses

The first layer sends 268 token-expert pairs to CPU and 756 to NPU. The mixed
second layer sends 318/706. Each layer has 256*4=1024 selected pairs among 60
experts. Complete experts move together; count 16 is assigned to NPU.

One-layer pooled phase medians:

| Measurement | All NPU | Mixed parallel |
| --- | ---: | ---: |
| CPU attention/shared expert/router, ms | 125.264 | 124.136 |
| Host partition/gather, ms | 0.121 | 0.401 |
| Expert stage wall time including transfers/waits, ms | 18.531 | 27.079 |
| CPU expert compute plus backend sync, ms | 0 | 26.447 |
| NPU expert compute plus backend sync, ms | 18.017 | 11.157 |
| NPU input set plus sync, ms | 0.108 | 0.498 |
| NPU output get plus sync, ms | 0.437 | 0.357 |
| Host aggregation/residual, ms | 1.941 | 2.248 |

Phase medians are independently computed and need not add to the total median.
The small experts collectively leave a slower CPU branch, even after overlap.
Two-layer summed expert-stage wall medians are 37.485 ms for all-NPU versus
67.011 ms for mixed parallel; mixed CPU compute/sync alone sums to 65.680 ms.
Host launch/join and cross-backend synchronization are included, but data transfer
is not the dominant measured cause of the loss.

The earlier single-expert crossover includes a separate backend invocation for
that one expert. Here all selected experts assigned to one backend share a graph
submission, and NPU gate/up fusion remains active. Therefore independent-expert
latency comparisons cannot directly determine the fastest whole-layer partition.
A future policy would need to consider total CPU/NPU branch work and batched
backend execution cost, rather than only a local threshold per expert.

## Correctness and Synchronization

The experimental CPU graph matches native llama.cpp layer outputs: one-layer
global NRMSE 1.996e-8, two-layer 6.445e-5 (worst token 0.334%). All per-layer
same-input MoE comparisons satisfy the original 4% per-token NRMSE criterion.
CPU native/grouped and NPU native/grouped produce equivalent results in this
fixture. Every measured result and selected ID sequence is bit-identical to its
policy's preflight result. CPU future completion and NPU synchronization occur
before aggregation; the next layer begins only after that aggregation.

There is a material numerical limit: dynamic two-layer NPU/mixed execution changes
second-layer expert sets compared with the CPU chain. All-NPU changes 5 token
sets; mixed changes 6. The mixed global NRMSE is 0.248%, but its worst token is
7.391%; tokens 74, 129 and 253 exceed 4%. The all-NPU worst token is 7.392%, with
tokens 74 and 129 exceeding 4%. Thus these dynamic chains do not pass strict
4%-per-token equivalence, and summary.json explicitly retains false flags.

An untimed diagnostic fixes IDs to the CPU chain while preserving each backend's
computed activations and router weights. Worst-token error then falls to 1.158%
for all-NPU and 1.058% for mixed, passing the unchanged 4% requirement. This
supports routing discontinuity as the amplification mechanism. It is not a
model-quality evaluation or a reason to declare the dynamic-chain flags passed.
Formal timings always use each policy's actual dynamically recomputed routes.

profile01 uses the exact same binaries, libraries, model and reference as formal
runs. Its log shows HMX_COMP and hmx-tiled MUL_MAT_ID/MUL_MAT_ID_NX for both original
top-k and mixed 756/706-pair shapes. For example, original log line 24 is the
256-token top-4 fused gate/up operation; lines 31158/31160 are mixed layer-0
gate/up and down. Full profile evidence is losslessly compressed. Profiling
timings are excluded because instrumentation changes execution cost.

## Artifacts and Reproduction

Implementation checkpoint: 9cf503276. Formal data: evidence/run01..03.json.gz;
commands, hashes and thermal snapshots: corresponding .run.json files. Original
llama outputs and token IDs: evidence/reference.json.gz. Readable aggregate data:
evidence/summary.json. Failing smoke01/02 logs and successful smoke03 are retained
separately. All source/library/model/reference hashes match across formal runs.
Model SHA256: 8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.

Additional commands after handoff-01:

```sh
python3 tests/run-expert-routing.py --tag run03 --reps 20 --warmup 3
python3 tests/summarize-expert-routing.py
python3 tests/run-expert-routing.py --profile --tag profile01 --reps 1 --warmup 1
python3 -m py_compile tests/run-expert-routing.py tests/summarize-expert-routing.py
git diff --check
```

For a new reproduction after device cleanup, rebuild with the NDK commands in
handoff-01, then use --deploy --reference and a fresh tag. The default summary
script intentionally summarizes the fixed run01..03 experiment; adapt its tag
list when conducting a new dataset. Summary regeneration checks provenance,
1024 pairs per layer, threshold accounting and phase containment in total time.

Only one prompt, threshold 16, four CPU threads and one hardware configuration
were tested. This is fresh 256-token causal prefill with F32 K/V held in graph
tensors, not persistent decode KV cache management. No embedding lookup, LM head,
token sampling, model initialization or weight preparation is timed. CPU dense
attention/shared-expert graphs are fixed across comparisons; this is not a search
for the fastest placement of all operations or a claim about full-model latency.

Final cleanup confirmed no running probe, removed only the owned device directory,
and verified its absence and the original model's unchanged 8149865792-byte size:

```sh
adb -s fd8657d6 shell pidof expert-routing-probe expert-routing-reference
adb -s fd8657d6 shell rm -rf /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell test ! -e /data/local/tmp/llama-expert-routing
adb -s fd8657d6 shell stat -c '%n:%s' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
```

Summary regeneration from compressed evidence, Python syntax checks and final
diff review passed. The final Git checkpoint is the commit containing this report.
No original model or unrelated device artifacts were removed.
