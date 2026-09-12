# Explicit Weight Management

`--weight-config FILE` enables model-owned weight placement and bounded demand
loading on CPU, GPUOpenCL and HTP0. `LLAMA_ARG_WEIGHT_CONFIG` is the environment
equivalent; library callers set `llama_model_params::weight_config` before loading.
Without a configuration the existing loading and execution path is retained.

## Configuration

See [the Qwen example](../examples/weight-manager-qwen.json). It assigns the 24
Qwen1.5-MoE layers in groups of eight and swaps only expert weights. All other
weights default to `fixed`. Example Android invocation from a writable directory,
with backend libraries and the matching HTP skeleton available in the library paths:

```sh
./llama-server \
  -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf \
  --device GPUOpenCL,HTP0 -ngl 99 \
  --weight-config weight-manager-qwen.json \
  --load-mode mmap -c 256 -b 64 -ub 4
```

- `version`: currently 1.
- `budgets`: device names as reported by the installed backends; CPU is required.
  Each device has `total_mib`, optional `reserve_mib` (default 0), and
  `expert_slots` (default 8). MiB means 1,048,576 bytes.
- `layers`: inclusive, nonoverlapping layer ranges and their default backend.
  Device names must also be selected by `--device`. CPU needs no explicit selection.
- `weights`: regular expressions matched with `std::regex_search`; the last
  matching rule supplies the complete policy, not a merge with earlier rules.
  `residency` is `fixed` or `swap`. Optional `backend` overrides placement for that
  weight. Optional `pin_experts` lists expert IDs that cannot be evicted after
  their first load. Unmatched rules and invalid configurations are errors.
- `report`: optional JSON output path, overwritten with current counters after
  loading, graph computation and manager destruction. Its parent must exist.

For ordinary-weight swap, add a rule such as
`{"pattern":"blk\\.[01]\\.attn_q\\.weight$","residency":"swap"}`.
Shared and routed experts are not classified by the manager: the matching tensor
name and policy determine residency. The example matches routed expert banks;
extend its patterns to include shared-expert weights when needed.

Layer placement also guides KV placement through the existing llama.cpp path.
An explicit per-weight backend is strict. Default fixed-weight placement may fall
back to CPU for unsupported operations, as in the existing compatibility path;
the report records actual weight placement. Unsupported swapped operations fail.
Automatic memory fitting, opportunistic op offload and pipeline parallelism are
disabled while the manager is enabled. The policy overrides legacy lazy loading.

## Ownership And Execution

`src/llama-model-loader.cpp` continues to parse GGUF metadata, validate tensors and
load fixed data. It registers each weight's source shard and offset with
`src/llama-weight-manager.cpp`. The model owns the manager, source file handles,
source mappings and fixed buffers for their entire required lifetime.

The manager owns the policy, budgets and cache pools. The scheduler identifies an
operation using swapped weights, copies its normal inputs, then invokes the
manager synchronously. The manager reads missing data from the source file,
updates backend cache slots and executes a local copy of the operation. Expert
IDs are remapped to slots; the model graph's weight and ID references are retained.

Pools are shared across weights with the same device, dtype and matrix shape.
Expert pools hold `expert_slots` matrices and use LRU replacement. Ordinary-weight
pools hold one tensor per compatible shape. A call protects all of its requested
experts while selecting victims. If the distinct requested IDs exceed capacity,
or pinning leaves too few replaceable slots, execution fails with a diagnostic.
Larger microbatches can require more slots than the model's per-token top-k.

Layout conversion remains in backend `set_tensor` implementations: OpenCL converts
complete Q4_0 expert slices to its SoA/Adreno layout; HTP repacks them to its tiled
layout. Repeated expert uploads reuse the bank's storage and metadata. Whole-tensor
replacement recreates its backend buffer to release derived layout objects.
Fixed CPU weights retain the normal optimized CPU buffer types, including weight
repacking when enabled. CPU swap sources and cache slots keep the canonical
layout required by partial loading.

Internal interfaces are in `src/llama-weight-manager.h`: `prefetch()` loads a whole
ordinary weight or specified experts; `evict()` invalidates the weight's cache
entries and refuses pinned entries. Both resolve views to their registered source.
These controls use the same cache as demand execution. Eviction releases slot
contents for reuse, not the pool allocation. There is no public C acquire/release
handle or runtime placement change API in this version.

## Budget Meaning

The accounting equation per backend is fixed buffers + runtime buffers + cache
buffers + configured reserve, with temporary staging/upload allowances checked
when used. Runtime includes KV, scheduler compute buffers and output buffers.
Cache capacity is charged even when slots are empty. Reports include actual
backend buffer sizes, peak accounted bytes, hits, misses, evictions, reads and
uploads. CPU host staging is reported separately. Model memory breakdown excludes
the virtual address size of lazy source mappings.
`cached_source_bytes` counts source-format bytes in occupied cache slots, excluding
unused slots and backend layout overhead. It is distinct from allocated cache
capacity and does not measure physical page residency. Omitting `report` disables
report construction and file I/O during inference.

These are application allocation budgets, not separate physical RAM banks or an
OS-enforced process limit. CPU, GPU and HTP share phone DRAM. File page cache,
driver allocations and backend-private scratch are not exhaustively intercepted;
`reserve_mib` leaves explicit allowance. Fixed buffers remain owned by the model
but are not automatically mlocked. Runtime allocations are checked after context
allocation and before execution, so this is not a preallocation hard cap for every
allocation. The 4 GiB example is a sum of logical device budgets; it does not imply
a measured 4 GiB physical-memory guarantee or disable Android's system swap.

## Current Limits

Swap supports Q4_0, F16 and F32, subject to backend operation support. Device tests
cover Qwen1.5-MoE Q4_0; other architectures and quantization formats need their own
validation. One context per model, synchronous reads/uploads, fixed execution
backend per operation, no KV/activation swap, no automatic cross-device migration.
Device weight views and multiple swapped inputs sharing one pool in one operation
are rejected. Direct contiguous CPU views are supported. New pool shapes require
additional capacity; the manager does not reclaim other pools to make room.
Direct I/O, no-allocation model loading, tensor split and legacy tensor-buffer
overrides cannot be combined with this manager.

This is a correctness-oriented foundation. It does not yet overlap source I/O,
conversion and computation as a FlexGen-style throughput pipeline would.

## Validation

`tests/run-weight-manager.py` builds configurations, deploys supplied Android
binaries and compares all logits on default device `fd8657d6`. Run a same-tag
reference before its swap case. `--case` accepts multiple sequential cases;
`--ubatch` and `--slots` control prefill coverage. `weight-slice-probe.cpp` compares
12 rounds of partial Q4_0 uploads against complete uploads, including buffer reset.
`weight-manager-api-probe.cpp` checks explicit prefetch, view eviction and pinning.
Build commands and device evidence are recorded in the task handoff directory.

`--semantic` selects a complete delivery-note question rendered through the
model's chat template (119 input tokens on the tested Qwen model). It generates
32 tokens freely, without reference-token input, saves `.prompt.txt`,
`.prompt.tokens.txt` and `.output.txt`, and checks the recipient, contents,
delivery day and cancelled earlier date. EOG is suppressed when necessary to
reach the fixed output count; `.summary.json` records how often this happened.
The readable answers also need inspection: keyword checks alone do not establish
general semantic quality, and the 32-token limit can truncate a sentence.
The result field `semantic_checks` contains only keyword-presence checks; it does
not detect contradictions in an appended explanation.

Add `--stop-eog` for semantic assessment with natural termination and a maximum of
32 output tokens. This requires `--semantic`. The summary records `max_steps`,
actual `steps`, `stop_eog` and `stopped_eog`. Saved logits cover generated text
tokens; the terminating EOG decision is not included. Use fixed-length mode when
an equal decode workload is needed, and inspect EOG suppression before interpreting
the continued text as a natural answer.

```sh
python3 tests/run-weight-manager.py --deploy --semantic --tag semantic01 --ubatch 4 --slots 32 --case cpu-ref cpu-swap gpu-ref gpu-swap htp-ref htp-swap mixed-fixed mixed-swap-4g
python3 tests/run-weight-manager.py --semantic --stop-eog --tag natural01 --ubatch 4 --slots 32 --case mixed-fixed mixed-swap-4g
```

On the tested 119-token question, all four reference/swap pairs produced identical
32-token sequences and saved logits. CPU, OpenCL and HTP answers retrieved the
correct facts, with the explanation truncated. Both mixed runs appended an
incorrect explanation after EOG was suppressed. With natural termination, both
mixed runs stopped after 18 tokens containing the four correct facts and remained
identical. The model used labeled lines rather than the requested two sentences.
See the [semantic test record](../task/handoff/handoff-2026-09-11-weight-manager/handoff-02-semantic.md)
for evidence and limitations.

`llama-bench --weight-config FILE` uses the normal synchronized benchmark timer
with the configured weight manager. The scalar option applies to all tests in an
invocation. `--no-repack` provides a diagnostic baseline without extra CPU buffer
types; omit it when comparing to normal default performance. Machine-readable
results include both settings. Model loading is outside the timing interval;
weight reads/uploads on cache misses remain inside it. Report files are updated
during compute when enabled, so capacity diagnostics and report-free throughput
runs should be kept separate.
