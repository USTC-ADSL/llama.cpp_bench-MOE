# Fixed CPU, Reversed Routed/Shared Backends

## Scope

User requested the previously untested comparison: keep the same CPU Expert IDs, place remaining routed Experts on GPU and shared on NPU. Measured on fd8657d6, Qwen1.5-MoE Q4_0,128 tokens, layer0. Replayed the archived native CPU FFN input, probabilities and top-k IDs from capture128-v2. This is a complete shared+routed FFN comparison, with gather/upload/compute/readback/synchronization/weighted aggregation included. It excludes attention, router computation and other layers; it is not a new whole-model throughput measurement.

CPU IDs remain `[2,4,5,6,7,8,9,10,12,16,17,22,24,27,29,30,33,35,38,40,45,50,57,58]`,24 active Experts and119 token-Expert pairs. The remaining35 active Experts account for393 pairs; one additional inactive Expert remains in the non-CPU resident bank. Total512 pairs, unchanged between plans. Shared processes all128 tokens. Remaining routed Experts change NPU -> GPU; shared changes GPU -> NPU.

Both compact layouts are loaded before measurement and retained simultaneously for paired testing. Each layout separately owns one copy of each Expert; holding two layouts during the experiment duplicates this layer's banks, not the whole model. Initialization, weight conversion and graph construction are outside timing. Three warmup rounds then32 measured rounds in each of two independent processes. Policy order alternates every round, and initial order plus bank construction order reverse between processes. Both policies use identical input/routes/reference and backend/library versions. Search is not repeated.

An unrelated model transfer to fd8657d6 was observed during deployment. Our runner was paused before benchmark execution until that transfer ended. A targeted device process check found no other llama/benchmark/expert process before resuming. No other task was interrupted. Frequency/affinity are not fixed; manifests preserve thermal state and hashes.

## Results

| Metric | Original: CPU tail, NPU remainder, GPU shared | Reversed: CPU tail, GPU remainder, NPU shared |
| --- | ---: | ---: |
| Complete FFN median |11.5406ms|18.8713ms|
| Process a median |11.5691ms|18.8713ms|
| Process b median |11.5207ms|18.8808ms|
| CPU upload+compute+readback |10.3954ms|10.1320ms|
| NPU upload+compute+readback |10.8639ms (routed)|2.3586ms (shared)|
| GPU upload+compute+readback |7.1378ms (shared)|18.2624ms (routed)|
| Joined Expert wall |10.9870ms|18.3287ms|
| Gather |0.2583ms|0.2612ms|
| Combine |0.2554ms|0.2546ms|
| Worst-token FFN NRMSE vs native CPU |1.5994%|1.0043%|
| Major faults in timed samples |0|0|

Each policy has64 measured samples. Reversed latency rises63.521%. It wins0/64 matched rounds; paired median original-minus-reversed saved time is-7.3508ms, round-bootstrap95% interval[-7.4249,-7.2491]ms. This interval characterizes these rounds, not cross-device/prompt generalization. All128 timed outputs pass4% per-token FFN NRMSE. Graph allocation counts remain unchanged after warmup; pair counts and backend interval containment checks pass. Independent phase medians do not add exactly to total median.

All three synchronized backend-call intervals overlap in every sample. Their common overlap median is7.1074ms for the original plan and2.3588ms for reversed (minimum6.8227/2.2739ms). These are host intervals enclosing completed backend work, not hardware execution traces.

Pure backend graph-compute medians (still whole subgraphs, not isolated HMX/OpenCL kernels): shared NPU2.1501ms vs GPU6.9235ms; remaining routed NPU10.4025ms vs GPU17.7627ms. Both jobs are faster on NPU in this measurement. The original assignment nevertheless completes earlier because GPU shared fits below the~11ms CPU/NPU critical path. Reversal accelerates a previously hidden shared branch but increases the routed critical path to~18ms. It does not demonstrate GPU/NPU equal speed, nor an impossibility of reversed placement on other layers/input lengths.

## Implementation and Reproduction

- tests/fullmodel-oracle.cpp: added a main guard, preserving default executable behavior, to reuse partitioned_ffn without copying the execution implementation.
- tests/reversed-expert-probe.cpp: compact resident paired layouts, fixed CPU ownership, alternating execution, per-backend phases, numerical checks and fault accounting.
- tests/run-reversed-expert-probe.py: target-device deployment, archived input restoration, SHA256 validation, two-process protocol and summary generation.

Commands:

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/reversed-expert-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/reversed-expert-probe
python3 tests/run-reversed-expert-probe.py
python3 -m py_compile tests/run-reversed-expert-probe.py
git diff --check
```

Evidence: a.json/b.json preserve every paired sample, full owner arrays and checks; a.run.json/b.run.json preserve commands, input/binary hashes, model stat and thermal snapshots; summary.json has aggregated results. Remote outputs matched host SHA256 after pull; compressed logs passed gzip integrity checks. The original fullmodel-oracle.cpp default-main compilation also passed clang++ -fsyntax-only with the same include paths, checking the reuse guard. No library rebuild was required.

After confirming no probe process remained, removed only /data/local/tmp/llama-reversed-placement and verified it absent. The model remains at8149865792bytes; unrelated device files were preserved. The runner refuses to overwrite existing a/b results, so use a fresh output directory/tag design for any future repeat. No required process or experiment remains running.
