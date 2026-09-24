# Shared Activations at 256 Tokens

## Conclusion

Shared activation buffers work and remove the Expert jobs' explicit tensor_set/get staging. They save a measurable part of the256-token critical path, but do not remove host gather/reduction or native graph callback boundaries. The earlier statement that copying alone caused the256-token regression was stronger than the evidence: the historical report explicitly listed multiple unisolated integration costs.

An additional bottleneck was identified and experimentally reduced: repeated callback input/result arrays at256 tokens produce about24,000 minor faults per model invocation. Optional direct callback I/O removes those arrays and copies. In a same-process ABBA experiment, all six blocks improve, mean saving41.754ms, with byte-identical logits. At128 this additional change has no consistent timing benefit. This is a concrete shape-dependent host allocation cost, not simply a linear CPU/NPU Expert-work imbalance.

## Scope

fd8657d6, Qwen1.5-MoE Q4_0, one fixed-prompt request,24 complete Transformer layers including attention/KV/router/final logits. Expert weights resident before timing. Context reset and initial weight load/conversion excluded. Historical `search256.json.gz` fixes ownership, while real router IDs are recomputed. Each standalone process has2 warmups plus12 samples. Both256 ABBA and128 regression have2 warmups plus24 alternating samples (12 per mode). Do not use the alternating run's mixed overall timing as either mode's throughput.

The256 plan places32 routed Experts on GPU at L0,24 on GPU at L1 (remaining routed and shared NPU); L2 routed GPU/shared CPU; most later layers CPU16/NPU44/shared GPU; L5 shared NPU; L23 routed NPU/shared CPU, only the last FFN token. Exact owners are preserved in every result. No policy changes or weight replication were introduced.

## Shared Buffer Verification

`oracle_job::run()` returns after backend compute in shared mode; its tensor_set/get branch is not executed. Gather writes shared tensor data and aggregation reads it after completion. This is an application-level no-staging path; backend kernels may still pack tensors or perform cache operations. The zero upload/readback phase fields are a branch marker, not an independent driver trace. The prior physical-alias64-round test remains valid; this task additionally verifies complete256-token model outputs against the copy control and historical CPU reference.

Original-code cohort, same executable/libraries throughout, order native-a, copy-a, shared-a, shared-b, copy-b, native-b:

| Mode | A median ms | B median ms | Pooled token/s |
|---|---:|---:|---:|
| Native NPU |676.414|688.183|376.42|
| Heterogeneous copy |647.521|700.588|380.48|
| Heterogeneous shared |624.414|659.001|402.23|

Shared saves23.107 /41.587ms versus copy (3.57% /5.94% latency), and beats the corresponding native runs. However copy-a already beats native without shared buffers. It is invalid to attribute the entire difference from historical843.781ms to shared buffers: operating state and implementation changed. All results in this cohort are retained in summary-original.json.

For the cleaner A process comparison, additive mean decomposition is:

| Whole-model component | Copy ms | Shared ms |
|---|---:|---:|
| Gather |11.859|11.855|
| Parallel Expert wall |402.358|375.810|
| Combine |45.707|46.711|
| Other model work and callback boundaries |188.670|189.949|
| Total mean |648.594|624.325|

NPU compute sums remain331.482 /332.732ms; its explicit upload/readback sums16.938 /10.605ms disappear from the shared path. This is consistent with the26.548ms reduction in Expert wall. Backend phase sums overlap and cannot all be added to total latency.

## Integration Overhead Remains

Same binary, all routed/shared Experts on NPU through the callback:

| Control | Median ms | Mean Expert wall ms | Mean combine ms |
|---|---:|---:|---:|
| Callback all-NPU, copy |1007.623|570.362|157.245|
| Callback all-NPU, shared |979.352|521.442|163.606|

Native NPU is676-688ms in the bounding cohort. Thus shared buffers do not erase the large integration penalty. The roughly300ms difference is not a pure synchronization measurement: graph structure, host aggregation, backend work and operating state all differ. The all-NPU controls have0/1 total major faults, so their large penalty is not explained by extensive measured major faults.

## New Optimization: Direct Callback I/O

The old callback copies the normalized input and probabilities into vectors, builds a separate result vector, then memcpy's that result into the scheduler's destination tensor. For256 tokens, the input and result are each2MiB per full-token layer. Across23 such layers, the two arrays account for92MiB of allocation volume per invocation, plus the last-token layer and small probability arrays. This is allocation volume, not92MiB of simultaneously live extra RAM; the two large temporaries peak around4MiB.

The optional path borrows the callback's contiguous F32 input/probability pointers and aggregates into its existing F32 destination. Small route-ID packing remains. All backend jobs finish before the callback returns; the scheduler retains these tensors for the callback lifetime. Expert kernels, routing, weight placement and accumulation order are unchanged.

Independent new-binary cohort,2 warmups +12 samples/process:

| Mode | A median ms | B median ms | Pooled token/s |
|---|---:|---:|---:|
| Native NPU |704.704|758.075|349.37|
| Copy |704.747|779.249|343.82|
| Shared only |687.207|750.941|364.05|
| Shared + direct callback I/O |647.062|655.737|393.21|

Each mode uses the same executable/libraries. Endpoint native latency drifts7.6%; copy-final-b has632 major faults across12 samples and is not a clean copy-cost microbenchmark. The results show measured full-model acceleration, but pooled12.55% direct-vs-native throughput gain is not a controlled-frequency universal guarantee. Shared-only margins versus native become small (about0.94% latency in B), so a scheduler should retain a noise margin. See summary-direct.json for all faults and phases.

To separate direct I/O from process drift, a final same-process ABBA experiment alternates temporary-array/shared mode (A) and direct-I/O/shared mode (B). Weights, graph jobs, shared buffers and policy are unchanged; no warmed graph allocations occur. Every round's logits are compared byte-for-byte in the executable after timing. Final logits also match the independently archived copy-path output.

|256-token ABBA result|Temporary arrays|Direct callback I/O|
|---|---:|---:|
|Median ms|690.357|653.322|
|Mean ms|690.642|648.888|
|Median-based token/s|370.82|391.84|
|Minor faults per call|23969.92|370.83|
|Total major faults,12 calls|4|0|

All6 blocks save25.317-55.313ms, mean41.754ms. Mean latency falls6.05%; median-based throughput rises5.67%. These are different statistics. Minor faults fall98.45%; they do not imply UFS reads. The measured intervention removes allocation, initialization and copies together, so it does not isolate page-fault service cost alone or identify the allocator's internal size threshold.

Additive same-process mean savings: combine22.477ms, other model/boundary17.678ms, Expert wall1.969ms, gather -0.370ms =41.754ms. Direct mode still spends393.443ms in Expert wall,37.221ms in combine,12.792ms in gather, and205.432ms in other model/boundary work. The last category includes real attention/dense work and must not be labeled entirely removable overhead.

##128-token Regression

Same-process ABBA128 results: temporary-array mean470.442ms, direct470.562ms; paired mean saving -0.120ms, mixed signs across blocks. Both have about371-372 minor faults per invocation. No consistent performance benefit is demonstrated at128. All rounds' logits match each other and the prior128 copy result byte-for-byte. This supports focusing the new optimization on the large-activation regime rather than assuming a universal percentage gain. Frequencies were not locked, and these numbers must not be compared directly to earlier128 runs under different operating conditions.

At256, native NPU CPU-relative logit NRMSE is4.896%, heterogeneous copy/shared/direct3.950%, top1=279. All heterogeneous logits are byte-identical across both cohorts and both direct modes. At128 the historical top1=537 and byte-identical output are retained. This is fixed-prompt validation, not general model-quality equivalence.

## Next Priorities

1. **Direct callback I/O is already implemented as an opt-in prototype.** CLI `--direct-callback-io` / environment `MOE_DIRECT_CALLBACK_IO=1`. It applies to the resident fullmodel-oracle executor; it does not silently replace the bounded dynamic-moe runtime or llama-cli. Default behavior is retained.
2. **Reuse shared activation arenas across sequential layers.** Current256 graphs retain1,320,655,744 bytes (~1259.48MiB) of Expert activation buffers, essentially the same as copy control. All per-layer futures finish before the next layer, so per-backend arenas could be sized to maximum simultaneous layer requirements rather than summing24 cached graphs. This targets memory pressure and paging; exact arena sizes/performance remain unimplemented and unmeasured.
3. **Reduce host aggregation traffic.** Four routed outputs require8MiB per full256-token layer,184MiB across23 layers, before shared output and other traffic. Direct mode still spends37.2ms in host combine. Backend-local weighted partial sums, or a properly vectorized/shared reduction kernel, are candidates. Changing accumulation order can alter quantization/routes; preserve rounding/order or revalidate numerical tolerance. No speedup is promised before implementation.
4. **Revisit placement using the remaining critical path.** Shared256 A typically finishes last on NPU; GPU-routed L2 alone takes about38ms and was chosen partly by numerical constraints. Optimize marginal grouped-kernel time under concurrent contention and resident-weight limits, rather than isolated Expert latency. Measured launch/join envelope beyond the longest backend job is only about2.9ms over the full model in the direct ABBA run, so persistent host workers are a smaller measured opportunity than aggregation or allocation.

## Artifacts

summary-original.json, summary-direct.json, summary-paired256.json and summary-paired128.json contain auditable statistics. Raw outputs, final logits and manifests are in evidence/. Detailed reproduction, build hashes and cleanup are in handoff-01.md. This task changes only experimental execution/measurement code, not the core shared DMA import implementation.
