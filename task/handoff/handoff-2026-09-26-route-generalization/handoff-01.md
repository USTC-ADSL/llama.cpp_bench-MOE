# Placement Transfer Screening: Verified Checkpoint

## Goal and Contract

User authorized starting multi-input experiments and asked whether per-backend models are needed. Target fd8657d6 only. Branch feat/dynamic-moe-placement, initial clean parent a98d212b4. Implementation checkpoint40360df19. All48 screening processes and six re-search confirmation processes completed; results.md contains findings and limitations. Scope is whole-model resident Prefill (attention/KV/router/FFN/output), not dynamic migration or generation. Two authored prompts not used in historical placement search, lengths64/128/256/512; four timed calls after two warmups per screening process. Source128 placement applies to64/128 and source256 to256/512. These choices are fixed in the runner. NPU/custom order reverses across independent cohorts. CPU/GPU provide one screening process each. All custom paths use shared activations, direct callback I/O, arena, fused host reduction. Single NPU/GPU baselines use native offload with ordinary unsupported-op fallbacks.

## Implementation

- tests/run-fullmodel-oracle.py: explicit --prompt-file works on every run, not only deployment; hash check and source metadata. Reference plan hashes archived. Add gpu and --transfer-placement.
- tests/fullmodel-oracle.cpp: native GPU target; explicit transfer flag permits source-prompt/source-length mismatch while retaining exact frozen ownership. Runtime tokenization/router unchanged; no route replay. Target FFN sizes use T on layers0-22 and1 on layer23. No stale source routes treated as reference. Reports actual60-entry route histograms per layer, transfer state and source length.
- tests/run-route-generalization.py: sequential resume-capable screen, failures retained, no device switching. Existing successful tags skipped; failed tags not overwritten.
- tests/summarize-route-generalization.py: checks executable/library hashes, target inputs, finite logits, per-backend pair counts against ownership and route histograms, fixed warmed allocations and bitwise repeated process outputs. Reports CPU-relative logit NRMSE, top1, timing, faults and per-layer backend costs. Four-sample estimates are screening results, not tail-latency claims.
- Prompts under this task's prompts/ are authored database-design English and historical-investigation Chinese passages; prefixes are used, not padded. They are not a representative application dataset.

## Commands

Built successfully:
```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 -m py_compile tests/run-fullmodel-oracle.py tests/run-route-generalization.py tests/summarize-route-generalization.py
```

First deployment/native process:
```sh
python3 tests/run-fullmodel-oracle.py --mode npu --tokens 128 --tag technical-128-npu-a --reps 4 --deploy --prompt-file task/handoff/handoff-2026-09-26-route-generalization/prompts/technical.txt --output-dir task/handoff/handoff-2026-09-26-route-generalization/evidence
```
Exact device command and hashes are in each manifest.

Completed screen commands (runner resumes plain pre-archive output; use a fresh task directory for a new cohort):
```sh
python3 tests/run-route-generalization.py task/handoff/handoff-2026-09-26-route-generalization > task/handoff/handoff-2026-09-26-route-generalization/screen.log 2>&1
python3 tests/summarize-route-generalization.py task/handoff/handoff-2026-09-26-route-generalization
```

## Target-Input Re-search

Searched narrative128 after transferred placement failed. Capture and search used the same runner and prompt with --tokens 128, tags narrative-128-capture /narrative-128-search; search references narrative-128-capture. Capture uses --mode capture --reps 1, search uses --mode search --reps 3. Complete actual commands are in .run.json.gz. CPU capture matches screen inputs and logits bitwise.

Confirmation uses --reps 8 and order NPU-A, frozen-A, new-A, new-B, frozen-B, NPU-B, tags narrative-128-research-{npu,frozen,new}-{a,b}. NPU uses --mode npu. Frozen uses --mode custom --plan-file historical search128.json.gz --transfer-placement; new uses --mode custom --plan-file this task's narrative-128-search.json, without transfer. Both custom policies enable --shared-activations --direct-callback-io --activation-arena --fused-host-reduce. Same executable and shared libraries throughout. Existing search uses noncompact copy-path banks and a4% per-token local FFN filter; it is a restricted candidate search, not a global optimum or a validated model for the optimized execution path.

Results: frozen narrative128 is slower than NPU, but new placement restores15.22% /14.74% throughput gain in independent confirmation processes. Both policies remain resident and selected before timing; no claim of online decisions or free weight migration. The extended summarizer audits all six confirmations and numeric comparisons. All56 device processes (48 screening + capture + search +6 confirmation) exit0.

## Archive and Cleanup

All runner outputs were pulled and SHA256 verified.247 gzip artifacts (including24 large captured layer arrays) pass sha256sum --quiet -c SHA256SUMS. Captured layer arrays remain locally archived but excluded from Git by evidence/.gitignore; regenerate from capture when cloning without those files. Compression is deterministic to deduplicate identical logits. screen.log.gz/research.log.gz retain host progress; manifests contain full commands/hashes/thermals. summary.json regenerates from gzip evidence and audits successfully. All sixteen transferred ownership maps match historical source plans exactly. screen-failures.json is empty.

No fullmodel-oracle process remained. Removed only owned /data/local/tmp/llama-fullmodel-oracle and verified absence; model remains8,149,865,792 bytes. Other devices and global cache untouched. No device tests remain running. git diff --check passes. Evidence/documentation checkpoint follows the implementation commit.

## Next Work

Design per-backend grouped-work profiles and concurrency corrections using the measured opposite outcomes at the same sequence length. Expand from two authored inputs to independent workload samples. Check ranking accuracy and decision overhead, then constrain choices by resident weight availability. The failed512 transfer is not proof of no possible512 heterogeneous gain: no512 re-search was run. Its240.49 MiB arena also violates a strict200 MiB allocation budget, requiring separate memory planning before claiming that constraint.

## Model Direction

Use backend-specific lookup/interpolation for routed groups and shared workloads, then jointly measured concurrency corrections plus gather/join costs. Need route histogram, active Expert count, backend grouping/kernel alignment, representation/residency and co-running shared placement. Do not infer marginal cost by summing isolated Expert timings, or assume independent times stay unchanged in concurrency. First screen supplies observations, not a trained or validated predictor.
