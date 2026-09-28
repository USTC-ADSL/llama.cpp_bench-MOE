# Causal next-layer preparation

## Measured motivation

Same development prompt,128 tokens, complete model, identical24 layer routes/placement/logits: instrumented routed DSP op time is358.920ms canonical vs168.371ms packed. Main-thread phase sums canonical/packed: activation28.79/27.08ms; exposed DMA-pop wait66.09/61.83ms; weight decode/reorder INCLUDING worker dispatch230.73/45.40ms; HMX queue wait19.12/19.13ms; output4.74/4.76ms. This is profiling attribution, not an end-to-end speedup: packed reference prepares weights synchronously and slowly on CPU. DMA submission lifetime can overlap; only the added ID_DMA_WAIT interval measures exposed pop waiting. Per-worker trace sums overlap and are never added to main-thread phases. No trace buffer reached capacity; all matched phases belong to their enclosing operation.

## Prediction target and protocol

Predict layer L+1's useful NPU Expert set from information available at layer L. Expert IDs are layer-local; prefill activation-set precision is misleading because almost all60Experts activate. Corpus: six distinct development prompts, two separately declared validation prompts, each actual first online128-token request. Initial classroom text was shorter than128tokens and failed before inference; its evidence is retained, and extended inputs use new names before fitting. Final layer23 has only one FFN token and goes CPU, so predict destinations1..22 only.

Candidates are same-ID high-count heuristic, layer-specific NPU ownership prior, and low-rank ridge correction conditioned on current-layer counts. Choose method/ridge by leave-one-prompt-out within the six training prompts, with a fixed0.5percentage-point improvement requirement over prior. Validation is never fitted. Prior wins (LOO precision@4 96.40%); two validation cases each98.86% precision@4, versus same-ID50.0%/40.91%. Same-ID adjacent count correlations0.0136/0.0173. The current evidence supports stable per-layer hotspots, not adjacent-layer index similarity. Frozen model226KB includes unused ridge parameters for comparison; runtime prior is just22×60scores.

## Bounded prototype

Two packed NPU target slots, each K complete Expert triples. Canonical DMA remains the sole complete source. K4/8/12 implies37.125/74.25/111.375MiB for both slots, on top of the previous scoped72.26MiB. Start with K4. Never cache all layers or decompress full weights to FP16 DRAM.

After layer L GPU shared compute finishes, launch GPU canonical→packed-HTP preparation for predicted L+1 IDs into the next slot. It overlaps remaining CPU/NPU work and possibly next attention. OpenCL uses one in-order queue and mutable kernel arguments: only one GPU API caller at a time. Before next GPU shared work, join any earlier pack future; report that wait separately. At L+1 NPU entry, use cache only if preparation is already complete; otherwise compute all actual NPU routes from canonical bytes without waiting for a speculative hit. Predicted IDs not assigned to NPU are unused, never substituted for actual routing.

Cache hits use a compact packed graph; misses use the original canonical graph. Both run serially on NPU, using the same activation arena after completion, and scatter into original token/top-k positions. Preserve exact weighted aggregation order. Slot index, layer, sorted IDs and completion state determine validity; do not mix formats in one tensor. Target slot cannot be overwritten while in use; shared/NPU stages join before that slot is reused two layers later. Source DMA import and target views must outlive queued work. Future destruction joins on every exit path.

Record requested IDs, useful hits, readiness, preparation start/end, GPU queue wait, NPU begin/end, bytes and wasted preparation. IO-coherent OpenCL imports alias existing DMA: raw budget observer may charge source imports again, so disclose raw counts and known aliased source bytes separately; do not mistake imported source capacity for another physical weight allocation. Total driver memory remains outside a strict200MiB proof.

Acceptance: exact logits versus unprefetched online execution, actual timeline overlap, bounded slot memory, and fresh-process first-prefill comparisons against both unprefetched online and symmetric resident-output native. Profiled runs are excluded from speed claims. Speculative predictions never read future counts or use a target-prompt prepass.

## Follow-up mixed-layout design

Single ID op carries canonical weights/activation/actual IDs/packed subset/immutable compact map in src0..4. Map[e]=-1 means canonical, otherwise packed slot; no original weight writes. DSP checks shapes, layout, map type and bounds before reading any selected weight. HMX-only. Gate/up pair fusion preserves both alternatives and the same map in a seven-input NX op; mixed/ordinary fusion is rejected. Per-expert HMX helper chooses the backing and layout flag but scatters with original Expert/token indices. Weight/DMA/map lifetimes span the complete op; no partial-ready reads. The benchmark keeps old split-graph mode behind an explicit option for fixed-placement comparison.

Conversion alternatives to measure next: enqueue gate/up/down conversions together and wait once, preserving serialized OpenCL ownership; consider source-import metadata lifetime only after distinguishing import overhead from kernel time. Keeping all22 source imports does not copy canonical bytes but enlarges driver mappings and observer alias coverage, so do not assume it is free under the memory target. Full FP16 expansion is excluded. Existing tile-local canonical conversion remains the exact fallback. Incremental publication requires per-group fences and immutable address mapping and is not implemented by simply relaxing the layer-ready test.
