# Fixed-Route Resident Expert Demo

`routed-expert-bench` measures four strictly serial MoE layers with fixed
cross-backend routes. It does not run a prompt, router, attention, residual path,
or complete model inference. Each layer uses its own deterministic activation;
all modes share the same activation, IDs and router weights.

## Layout and Execution

The four Phi-mini-MoE packs contain 16 experts per layer. Gate/up are Q4_0
`[4096,960,E]`, down is Q4_1 `[960,4096,E]`. The shared weight arena is exactly
440,401,920 bytes (420 MiB). Existing expert-slot layouts and benchmarks are
unchanged.

| Mode | Tensor layout per layer | Expert graphs per layer |
| --- | --- | --- |
| `routed_single_backend --backend gpu` | gate[16], up[16], down[16], GPU native | 1 |
| `routed_single_backend --backend htp` | gate[16], up[16], down[16], HTP native | 1 |
| `routed_hetero_serial` | gate{GPU[8],HTP[8]}, up{GPU[8],HTP[8]}, down{GPU[8],HTP[8]} | 2, sequential |
| `routed_hetero_parallel` | same as hetero serial | 2, concurrent |

Full multi-expert canonical tensors are converted to native layout before
binding. Concatenating single-expert native buffers would interleave SoA scale
planes incorrectly. Each resident expert has exactly one native copy.

For layer `l`, token `t`, fixture `s`, GPU expert is `2*((t+l+s)%8)` and HTP
expert is `2*((3*t+l+s)%8)+1`. Their weights are 0.6 and 0.4. Odd token indices
swap the top-k slots and weights. Local expert IDs are `global_id/2`.
Formal fixtures are 0..2; preflight fixtures 0..7 cover all experts.

Each expert graph performs gate/up, SwiGLU, down and router weighting. A single
backend also reduces both selected experts. Heterogeneous results are combined
by a separate prebuilt GPU ADD graph. HTP output uses explicit host staging
and backend get/set; Shared DMA weights do not imply activation coherence.

The logical IDs shapes are `[2,T]` and `[1,T]`. Their preallocated I32 backing
tensors are `[16,T]` and `[8,T]`, respectively. This matches inference's top-k
view: the existing OpenCL MoE reorder derives expert count from IDs row stride.
A compact IDs tensor is not supported by that optimized path.

## Buffer and Timing Contract

Every graph context is allocated before execution with
`ggml_backend_alloc_ctx_tensors`: activation, IDs backing, router weights,
gate/up, SwiGLU, down, weighted outputs, aggregation inputs and outputs.
Each layer/backend graph persists across fixtures and repetitions for a token
shape. CPU transfer staging and two host workers are also created in setup.
Both workers receive jobs through a persistent generation barrier.

Preflight executes all eight routing fixtures for each shape before timing.
This also initializes OpenCL workspace capacities and HTP execution state.
OpenCL workspace allocation grows only when a larger capacity is requested;
the formal loop reuses the warmed shape. OpenCL still creates subbuffer/image
descriptors and synchronization events during execution. These are handles
over existing storage, not newly allocated intermediate tensor payloads;
their driver costs remain in the measured graph time. HTP may rebuild host
operator descriptors as layers change; its tensor and queue payload buffers
are preallocated. Reported compute bytes exclude backend-private workspaces.

| Metric | Boundary |
| --- | --- |
| `expert_core_us` | Backend inputs are ready. Starts before worker dispatch, ends when selected workers complete synchronized graph execution. Four-layer value sums the four measured intervals. |
| `moe_dispatch_total_us` | Activation starts on GPU. Includes activation GPU-to-host-to-HTP handoff when needed, worker compute, output HTP-to-host-to-GPU handoff and GPU aggregation. Ends with GPU output ready. |

Core and total are separate passes. In a core pass the layer join/aggregation
still runs **outside** each measured core interval so the next layer starts
only after completion. `four_layer_wall_us` retains that outer duration;
it must not be relabeled as core time. Single-backend core includes its own
aggregation, whereas heterogeneous core excludes the cross-backend join.
Compare serial/parallel core only for the identical heterogeneous graphs.
Use total for comparison with the best single backend.

Fixed IDs/weights are uploaded outside timing. Total repeats the activation
handoff every iteration even though the values are fixed. Allocation, weight
I/O/conversion, CPU reference, validation, JSON serialization and teardown are
outside both measured intervals. Worker scheduling is included. Host start/end
timestamps are diagnostic evidence, not proof of overlapping device kernels.

## Build and Run

From the repository root, using the existing Android OpenCL/Hexagon cache:

```sh
cmake -S . -B build-android-pd-final
cmake --build build-android-pd-final --target routed-expert-bench -j 12
cmake --build MOE/experiments/build/host -j 8
ctest --test-dir MOE/experiments/build/host --output-on-failure
```

The runner uses normal `adb -s fd8657d6`, the root AGENTS.md recovery policy,
an independent runtime directory, and existing packs from
`/data/local/tmp/shared-expert-resident-20260915/layer-{0,1,2,3}.pack`.
Override `--remote-pack-dir` when packs reside elsewhere. It pushes the built
executable, host libraries and V79 skel, and retains binary hashes, commands,
environment, raw output, device/thermal snapshots and failures locally.

```sh
python3 MOE/experiments/run_routed_experts.py \
  --output-dir MOE/results/tmp/20260915-1356-routed-formal \
  --sessions 3 --warmup 5 --repeat 30
python3 MOE/experiments/analyze_routed_experts.py \
  MOE/results/tmp/20260915-1356-routed-formal/session-*/raw.jsonl \
  --output-dir MOE/results/routed-expert-compute
```

Choose a fresh output directory for each run. A reduced `--sessions 1
--warmup 1 --repeat 1 --tokens 1` run is a smoke test only; analyze it with
`--allow-smoke`. The analyzer otherwise requires at least three distinct
sessions, 30 samples, five warmups, all modes and token shapes, complete
pre/post validations, matching fixtures/environment/weight fingerprints,
unchanged arena CRC and correct graph/active-pair accounting. It reports
per-layer and four-layer p50/p95/p99, serial/parallel speedup, and best-single
versus parallel total. A slowdown is a valid result.
