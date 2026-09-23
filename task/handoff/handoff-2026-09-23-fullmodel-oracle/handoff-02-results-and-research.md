# Full-Model Results and Research Assessment

## Experimental Contract

Device fd8657d6 (Redmi K80 Pro), one request, Qwen1.5-MoE-A2.7B-Chat Q4_0, 24 complete Transformer layers. Fixed prompt is tests/expert-routing-prompt.txt, truncated to the first128 or256 native tokenizer tokens. Token IDs and artifact hashes are archived. This measures whole-model Prefill through final last-token logits, including real attention, KV writes, normalization, routing, complete shared/routed FFNs, residuals and vocabulary projection. It is not generated-sequence Decode throughput or tokenizer/model-loading-inclusive request latency.

Weights are loaded before timing. Each custom Expert has one assigned backend; shared weights have one assigned backend. No timed weight loading, migration or bank construction occurs. Native attention/dense operations use the normal NPU offload graph, with normal unsupported-op fallbacks. The last layer computes FFN only for the requested final token, exactly as native llama does. Per-layer backend calls execute concurrently, then synchronize and aggregate before the next layer. Cross-layer dependency is preserved.

Offline CPU capture supplies actual per-layer input, router probabilities and top-k IDs. Candidates include all9 whole-routed/shared backend combinations and sparse-tail NPU splits with1/4/8/16/24/32 active Experts assigned to CPU or GPU; split candidates place shared on NPU/GPU. Tail order is ascending observed token count, then Expert ID. Each layer selects the fastest candidate passing4% worst-token FFN NRMSE from three profiling samples. This is a bounded candidate search, not a globally optimal partition or an exhaustive search over full-model plans. Search banks contain all60 Experts per backend; final runtime banks are compact, so search ranking is not guaranteed to hold under final bank sizes/cache pressure. Compact winners are revalidated numerically.

The chosen placements are frozen before independent full-model measurements. Runtime computes real dynamic routes; recorded CPU route IDs are never forced. Search samples do not enter reported throughput. Two warmups precede12 measured samples/process for128 and8/process for256. Policy order is reversed across process groups. This is not an interleaved same-process paired experiment. Frequencies/affinity are not fixed; retain process-level results and thermal manifests.

## Why the Earlier256-Token Gain Was Small

The old experiment measured complete first1/2 layers but kept attention/router on CPU. Its two-layer baseline/split medians were41.176/26.155ms at32 tokens,84.560/75.432ms at128, and137.233/129.432ms at256. Relative latency reduction dropped36.48% ->10.79% ->5.68%.

At256 the baseline attention/router median was92.437ms, about67% of total time, while Expert wall was41.900ms. Split Expert wall was31.260ms: the Expert-stage saving remained substantial. Gather rose0.411 ->1.150ms; split attention/router was94.179ms and combine2.049 ->2.178ms. These independent phase medians must not be added as an exact decomposition of median total time.

For an additive attribution, use the mean of24 matched NPU-minus-split round differences:

| Old two-layer256 component | Mean saved ms |
| --- | ---: |
| Expert wall | +11.1234 |
| Attention/router | -1.4305 |
| Partition/gather | -0.7517 |
| Combine | -0.2629 |
| Remaining measured overhead | -0.0420 |
| Complete-layer wall | +8.6364 |

Positive means faster. Therefore the small percentage is largely an Amdahl bottleneck in the old CPU attention/router path plus growing partition/join costs. It does not show that the Expert work ceased to benefit, or establish256 as a universal heterogeneity cutoff. NPU grouping efficiency also changes with workload: whole-op kernel selection, active Expert count and per-group padding make the marginal cost of removing an Expert different from its isolated latency. Relative Expert-stage gain can shrink as useful work per group grows, but the phase data do not isolate that mechanism causally.

## Exact Earlier Allocations

Layer numbering starts at0. Counts are active routed Experts, not all60 resident weights. Pair counts are token-Expert assignments. All unspecified routed Experts remain on NPU; inactive ones contribute no pairs. Exact Expert IDs are in ../handoff-2026-09-23-oracle-routing/evidence/final-prefill-summary.json under split_placements.

| Tokens | Layer | Sparse-tail assignment | Active CPU/NPU/GPU Experts | CPU/NPU/GPU pairs | Shared |
| ---: | ---: | --- | --- | --- | --- |
|16|0|lowest24 to CPU|24/15/0|26/38/0|GPU|
|16|1|lowest24 to CPU|24/13/0|28/36/0|GPU|
|32|0|lowest24 to CPU|24/26/0|36/92/0|GPU|
|32|1|lowest32 to CPU|32/19/0|48/80/0|GPU|
|64|0|lowest32 to CPU|32/24/0|97/159/0|NPU|
|64|1|lowest24 to GPU|0/31/24|0/197/59|NPU|
|128|0|lowest16 to CPU|16/43/0|67/445/0|GPU|
|128|1|lowest16 to CPU|16/44/0|77/435/0|GPU|
|256|0|lowest24 to GPU|0/36/24|0/771/253|NPU|
|256|1|lowest24 to GPU|0/36/24|0/748/276|NPU|

These are frozen assignments based on each layer's observed histogram, not a runtime rule sending every Expert with fewer than8/16 tokens to CPU.

## Whole-Model128 Results

Final instrumented confirmation uses b/c processes only,24 samples per policy, identical binary/library/prompt hashes. Initial a-process smoke results are excluded.

| Policy | Median ms | Prefill token/s | Latency vs native NPU |
| --- | ---: | ---: | ---: |
| Native CPU |1929.922|66.32|+234.09%|
| Native NPU offload |577.661|221.58|baseline|
| Callback, all shared+routed NPU |710.639|180.12|+23.02%|
| Best retained unsplit shared/routed plan |667.245|191.83|+15.51%|
| Fine-grained frozen plan |451.288|283.63|-21.88%|

Throughput gain is28.00%, not21.88%: latency reduction and throughput improvement use different denominators. Per-process native NPU589.227/573.708ms versus fine-grained455.860/443.245ms, reductions22.63%/22.74%. CPU drifts1544.876 ->2240.917ms; its pooled median is not a stable operating point. Native GPU whole-model performance was not measured, so the baseline claim is restricted to native CPU/NPU.

The callback all-NPU control is slower than native by132.978ms. Thus transfer/scheduling/host aggregation overhead is material; the fine-grained result is not obtained by comparing only against a weakened callback baseline. The unsplit shared/routed plan also fails to beat native NPU. Fine-grained splitting is required for the retained winning plan, although this does not prove all conceivable branch-only implementations are inferior.

Final128 assignment:

- Layers0,1,3,4,6-22:24 least-used active Experts on CPU; remaining Experts on NPU; shared on GPU.
- Layer2: all routed on NPU, shared on CPU.
- Layer5:24 least-used active Experts on CPU; remainder and shared on NPU.
- Layer23: routed and shared on CPU; only the last token enters this FFN.

For the common CPU-tail layers, offline CPU pair count ranges78-132 out of512. For example layer0 has119 CPU/393 NPU pairs and layer1 has132/380. This differs from the older two-layer128 plan: new search candidates are evaluated with a native full-model capture, different context/kernel configuration and a per-FFN numerical constraint. Expert-count constants are not universal.

Exact60-entry owners, shared owner, active Expert IDs, offline counts and actual runtime pair counts are archived in search128.json, summary128.json and each confirmation JSON. Backend IDs0/1/2 mean CPU/NPU/GPU. Selected Expert+shared logical weight bytes total7128.094MiB: CPU2765.820, NPU3972.379, GPU389.895. Dense/model metadata and activation buffers are additional.

## Whole-Model256 Results

Two independent8-sample processes per policy,16 samples pooled. Same fixed prompt prefix and final binary/library versions as128. Search and capture are separate from confirmation.

| Policy | Median ms | Prefill token/s | Latency vs native NPU |
| --- | ---: | ---: | ---: |
| Native CPU |5732.410|44.66|+649.65%|
| Native NPU offload |764.677|334.78|baseline|
| Callback, all shared+routed NPU |1081.342|236.74|+41.41%|
| Best retained unsplit shared/routed plan |1292.460|198.07|+69.02%|
| Fine-grained frozen plan |843.781|303.40|+10.34%|

The fine-grained plan is slower in both process comparisons:824.994 versus770.088ms and850.666 versus760.532ms. It loses9.37% throughput in the pooled table. Native NPU is therefore the measured preference at256 for this implementation. This is a negative result for the retained plan, not proof that every possible heterogeneous placement is slower. CPU drifts4914.947 ->5780.028ms; do not interpret its pooled number as a controlled-frequency result.

Exact256 plan:

- Layer0:32 least-used active routed Experts on GPU, remainder and shared on NPU.
- Layer1:24 least-used active routed Experts on GPU, remainder and shared on NPU.
- Layer2: all routed on GPU, shared on CPU (numerical screening excludes faster retained candidates).
- Layers3,4,6-22:16 least-used active routed Experts on CPU, remainder on NPU, shared on GPU.
- Layer5:16 least-used active routed Experts on CPU, remainder and shared on NPU.
- Layer23: all routed on NPU, shared on CPU; one FFN token.

All exact Expert IDs/counts and actual runtime pair counts are in search256.json, summary256.json and run JSON. Expert+shared logical weights: CPU1522.133MiB, NPU4714.887MiB, GPU891.074MiB; total7128.094MiB, the same logical total as128. Dynamic routing changes1516/23556 top-k slots, stable across both processes. All policies have next-token top1=279. Final-logit CPU-relative NRMSE: native NPU4.896%, callback all-NPU4.632%, branch-only3.659%, fine-grained3.950%.

This new256 failure has a different boundary from the old two-layer CPU-attention experiment. In the clean b-process all-NPU callback, Expert wall635.042ms falls to479.966ms for fine-grained execution. However the native integrated baseline is770.088ms while the all-NPU callback is1094.489ms: there is a large integration/host-boundary penalty. Fine-grained gather/combine medians at256 are28.660/52.659ms, versus7.785/8.337ms at128 in the b-process. Native graph optimization, transfers, scalar aggregation and CPU operating state are not fully isolated, so do not attribute the whole difference to one cause or subtract independent medians as a causal decomposition.

Current custom output aggregation materializes all four routed Expert outputs in host memory: `4 * B * 2048 * sizeof(float)` =8MiB per full-token layer at256, before reduction. For23 such layers that is184MiB of routed output values passing through host aggregation, excluding shared outputs, inputs and additional copies; it is not all necessarily device-to-host DMA because CPU also owns pairs. Splitting also explicitly gathers/duplicates input token rows. Native integrated execution has no equivalent mandatory host round trip for every Expert output. A route-aware backend-local weighted reduction with only per-token partial sums crossing devices is a concrete optimization target, but is not implemented or measured here.

The c-process all-NPU callback control records8408 major faults, while b records0. It cannot be treated as a pure synchronization microbenchmark. Main native NPU b/c both record0; fine-grained b/c record1/0. The slow256 fine-grained conclusion therefore does not depend on that anomalous control. The separate custom-npu256-paging-check run measures1071.973ms over8samples with0major faults in every sample, confirming that the large integration cost persists without this anomaly. The supplemental run is archived separately and does not silently replace c in the pooled table.

## Numerical and Residency Limits

All selected compact FFNs pass the4% worst-token NRMSE check against native CPU captured FFN output. Layer2 rejects faster candidates: at128 the sparse-tail24 CPU/shared-GPU candidate is11.489ms but token0 FFN NRMSE4.995%; the accepted NPU-routed/CPU-shared choice is slower. At256 the fastest sparse-tail16 CPU/shared-GPU candidate is17.722ms with the same4.995% token0 error; the selected GPU-routed/CPU-shared candidate passes at3.771%. This is a numerical screening constraint, not proof of a downstream language-quality threshold.

At128 all measured policies produce top1 token537. Final-logit CPU-relative NRMSE is4.036% for native NPU,4.060% for callback all-NPU,2.800% for branch-only and2.407% for fine-grained. These logits are deterministic across the two processes. The4% per-token FFN criterion is different from final-logit NRMSE and must not be conflated with it. One prompt's next-token agreement cannot establish perplexity or generation-quality equivalence.

The fine-grained runtime changes778 top-k slots out of11780 relative to offline CPU capture; these can include order changes as well as Expert-set changes. Actual route values, probabilities and ownership determine execution. Cached-graph allocation counts remain unchanged after warmup. The plan's benefit therefore survives these observed route changes, but this is not evaluation on an unseen prompt.

Weights are logically resident, with no hot-path load/migration. Absolute physical residency is not enforced: custom VmSwap is approximately348MiB, and its two128 processes show3/1 major faults over12 measured samples each. Other policies also have swap or occasional faults. Do not infer those pages are necessarily active weights, but do not claim zero OS paging. Native NPU RSS does not include all DMA-backed accelerator weights, so RSS alone cannot compare complete physical model footprints. A stricter future study needs device-wide DMA accounting and a controlled lock/residency mechanism.

## Research Assessment and HeteroLLM

There is measured engineering value and a plausible research problem: on this device/model/fixed prompt, a complete-model resident Expert partition beats native NPU even after including integration overhead. The current experiment alone is not sufficient evidence of a publishable general scheduler.

The comparison target is **HeteroLLM: Accelerating Large Language Model Inference on Mobile SoCs with Heterogeneous AI Accelerators**, Le Chen et al.,2025-01-13, DOI10.20944/preprints202501.0901.v1. It is not the unrelated2026 arXiv2603.29002 work with the same short name. Primary metadata/abstract: https://api.crossref.org/works/10.20944/preprints202501.0901.v1 . Full-text landing/PDF access returnsHTTP403, independently retried; no claim of absence from its body is established.

The abstract explicitly covers different Prefill/Decode tensor partition strategies, mobile GPU/NPU heterogeneity, unified-memory-address fast synchronization, and layer/tensor-level heterogeneous execution. Therefore merely combining CPU/GPU/NPU, splitting matrices, treating Prefill/Decode differently, or reducing copies is not a defensible new contribution by itself. The abstract does not discuss MoE routes/shared Experts, but this is not proof that the full paper omits them. Broader MoE scheduling literature must also be reviewed before novelty claims.

Promising hypotheses and designs, requiring further validation:

1. **Route-conditioned marginal cost.** Schedule the change in a grouped kernel's critical path when removing an Expert, not independent Expert FLOPs or isolated latency. Features include each Expert's token count, active Expert count, kernel selection/padding, backend bank layout and transfer volume. Candidate selection should fit the CPU aggregate workload under the remaining accelerator work. Compare against token thresholds, count/FLOP balance, and isolated-Expert cost models. The current threshold failures and successful sparse-tail splits motivate this hypothesis; they do not yet validate a predictive cost model.
2. **Joint shared/routed scheduling under contention.** Treat shared as a dense job consuming backend compute/memory bandwidth, not a free branch. Optimize its owner together with routed partitions and include concurrent slowdown. The128 fine-grained plan beats the retained branch-only plan, and256 layers choose different shared owners. Ablate shared-only, routed-only and joint placement with the same integration path.
3. **Residency-constrained online decisions.** The offline plan can place weights before timing because its prompt is known. An online policy cannot reassign arbitrary Experts for free when each has only one resident copy. Use a memory-budgeted static partition plus selective replicas, then choose only among resident alternatives after routing. Account for backend-specific packed layouts and replica bytes; compare throughput against memory use and include any migration cost. This is a core constraint missing from a simple max(CPU,NPU) argument.
4. **Whole-graph gain prediction and fallback.** Predict the complete-layer/model benefit, including attention, host join and graph split costs. Select native single-backend execution when dispatch savings do not clear overhead and a noise margin. With one-copy static placement, this decision must precede bank placement: an instantaneous online fallback to all-NPU requires resident NPU replicas or must pay migration costs. A candidate design retains an NPU backbone and selectively replicates only useful CPU/GPU Experts under a memory budget; its extra residency must be measured. Retain CPU Decode as the measured scheduling preference: prior independent single-request Decode tests did not establish reliable Expert-splitting gains, but full-model phase-switch weight costs are not measured here. The earlier256 Amdahl effect and the slow callback all-NPU control provide concrete motivation.

A useful objective is `T_layer = T_dense + T_dispatch + max_b T_b(routed_b, shared_b | simultaneous contention) + T_join`, subject to numeric tolerance, backend support and resident-memory constraints. `T_b` must be measured/modelled during concurrency; standalone20ms NPU and10ms CPU do not guarantee20ms concurrent execution if they compete for memory or CPU orchestration resources.

The research question worth pursuing is: **Can a small route-aware, residency-constrained online scheduler retain most of the offline full-model gain across prompts, lengths, layers, devices and thermal states, without sacrificing quality?** Measure oracle gap, scheduler overhead, tail latency, energy and model quality; compare with the best native CPU/NPU/GPU baseline, a HeteroLLM-style tensor partition baseline when reproducible, branch-only scheduling, naive thresholds, and resident-memory-matched alternatives. Until then the defensible claim is a successful offline design and a set of measured failure modes, not a universal heterogeneous-MoE speedup.

## Practical Decision

For a known fixed prompt and these measured configurations, choose the frozen128 Expert partition before loading resident banks; choose native NPU for256. The retained design has a real whole-model benefit in one regime and a demonstrated failure in another. An online switch between those layouts is not free without additional replicas. Keeping this residency distinction explicit is necessary when turning the experiment into a scheduler.

## Reproduction and Archive

Implementation checkpoint:458fb3483. Exact commands, binary hashes, thermal snapshots and process results are in evidence/*.run.json and compressed logs. Core build/Android commands are recorded in handoff-01-implementation.md. Summaries regenerate with tests/summarize-fullmodel-oracle.py for --tokens128/256 using b/c process sets. Large search/run JSON files referenced above are archived with an additional .gz suffix; the summarizer reads either form. Large per-layer capture arrays are compressed and retained on the host, excluded from Git by evidence/.gitignore; their hashes are recorded alongside tracked logits/plans/results. They can also be regenerated from the recorded model/prompt/binary configuration.

All104 device-side JSON/binary result files matched host SHA256 before archival. Logs, logits and48 large per-layer capture arrays are gzip-compressed; gzip integrity checks passed. Both summaries regenerate successfully from compressed logits. evidence/SHA256SUMS covers all result JSON and compressed artifacts; the ignored capture archives remain on the host. Independent source review found no routing/index/weighting/timing-boundary bug, while confirming bounded-search, noncompact profiling and process-state limitations described above.

After confirming no fullmodel-oracle process remained, removed only /data/local/tmp/llama-fullmodel-oracle and verified it absent. Model remains /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf at8149865792bytes. No other device or unrelated directory was used or cleaned. A new run needs --deploy and fresh tags; after cleanup, capture/search must be regenerated or archived files pushed back explicitly. No required experiment remains running.
