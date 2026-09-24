# Dynamic Placement Implementation

## Scope and Branch

Branch: `feat/dynamic-moe-placement`, based on `moe-mobile-opt` at the start of this task. Single-request Qwen1.5-MoE Q4_0 on fd8657d6. Experimental standalone executable, not a new llama-cli flag. The existing public MoE graph callback keeps native attention/router/KV/residual/output operations; `tests/dynamic-moe-runtime.h` supplies the replacement FFN executor.

## Implemented Behavior

- Runtime `moe_placement_policy` receives phase, step, layer, actual token count and all60 Expert token counts after the native router runs. It returns all60 routed owners plus an independent shared owner (CPU0/NPU1/GPU2).
- `tests/dynamic-moe-policy.json` demonstrates default, per-layer and per-step decisions. User groups identify Expert IDs; group_size bounds each backend's converted working set and may be1.
- One immutable read-only GGUF mmap source. The model loader also maps unused Expert tensors from the same file; these are file-backed page aliases, not another allocated full converted CPU bank. Logs confirm CPU_Mapped. mmap residency/page faults remain OS-controlled.
- Three cache slots, one per backend, hold at most one routed group and an optional shared bank. Graph references are freed before weight eviction. All conversions finish before concurrent backend jobs start; all jobs finish before eviction. Empty slots are released. No whole-model bank exists on each backend.
- Every backend job in a wave runs concurrently; later waves depend on reuse of the bounded slots. There is no promise of optimal pipeline overlap.
- Shared output and weighted routed outputs are retained until final top-k-order reduction. Separate multiplication and ordered addition match native CPU rounding; backend-order/FMA accumulation was found to cause large downstream route/logit drift.
- Grouped routed jobs use MUL_MAT_ID. CPU single-Expert jobs use ordinary MUL_MAT because CPU_REPACK rejects effectively2D MUL_MAT_ID weights. GPU/NPU retain MUL_MAT_ID. OpenCL Expert-named tensors use MoE packing, incompatible with ordinary dense MUL_MAT.
- Non-Expert weights use the configured phase backend (native unsupported-op fallbacks still apply). Prefill-to-Decode backend switch serializes state, frees old context/model, loads the new phase, restores KV/logits, and checks a byte-identical state round trip. Transition cost is reported separately. This is functional state migration, not a fast in-place backend switch.

## Memory and Shared Buffer Boundary

Initial200MiB goal was relaxed by the user. Report actual converted ggml buffer sizes, including intermediate slot replacement peaks; these exclude source page residency, dense weights, KV, conversion staging, driver scratch and CPU vectors. Never present this as a physical RAM cap.

Activations still use explicit set/get and host aggregation. No OpenCL external-fd import exists in the current backend. HTP internally uses host-accessible rpcmem, but true CPU/GPU/NPU activation zero-copy is not implemented. Backend conversion reuses current set_tensor paths: CPU_REPACK, Hexagon tiled repack and OpenCL SoA conversion. No new fused compute-and-convert kernel is claimed.

## Build and Run

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/dynamic-moe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/dynamic-moe
python3 tests/run-dynamic-moe.py
```

The runner deploys and checks hashes in `/data/local/tmp/llama-dynamic-moe`; never touches other devices or models. Use `DYNAMIC_EVIDENCE=evidence-NEW` to avoid overwriting prior results. `DYNAMIC_VERIFY_ALL=1` independently checks every mixed FFN against an unsplit CPU FFN on identical inputs; validation temporarily loads an additional one-layer CPU reference and affects timing/memory. `verify_layer=0` only checks layer0; default -1 disables validation. No timing containing reference validation is a performance benchmark.

Device executable arguments:

```text
dynamic-moe MODEL PROMPT TOKENS DECODE_STEPS POLICY_JSON OUTPUT dynamic
```

Use `native` as the last argument for the corresponding phase-backend native baseline. `decode_tokens` can teacher-force identical decode inputs across policies. Each report includes real placements, route counts, wave conversion/compute times, state migration, faults and logits in OUTPUT.logits.bin.

## Validation History

- evidence: initial single-Expert CPU unsupported-op failure, preserved.
- evidence-v2/v3/v4: full-model paths ran; CPU logit discrepancy7.37% exposed aggregation-order drift. v4 captures show identical first-layer inputs and rounding-scale initial output differences, amplified through quantization and routing.
- evidence-v5: preserving top-k reduction order reduced CPU final-logit NRMSE to approximately8e-8 for Prefill and both Decode steps. Mixed error19.9% exposed GPU single-Expert layout mismatch.
- evidence-v6: same-input FFN checks isolated mixed layer2 failure (worst-token51.2%).
- check-layer2.json versus check-layer2-fixed.json: retaining GPU MUL_MAT_ID reduces single-Expert GPU whole-FFN NRMSE92.8% to1.93%; worst-token3.77%. Groups8/60 GPU also1.93%; CPU all sizes exact0. NPU groups8/60 worst-token3.82%; single-Expert5.66%, a remaining kernel/numeric behavior, not claimed to pass the4% screen.
- evidence-v7: all-layer mixed FFN screen after layout fix; layer2 now passes (worst-token0.91%), layer4 rejects at5.92%. This remains an explicit numerical limitation of the arbitrary preset.
- evidence-v8/v9: final16/128-token full-model smoke suites pass strict CPU equivalence (1e-5 bound, observed approximately1e-7), mixed/native accelerator full-logit smoke bound10%, finite outputs, route accounting, buffer-accounting check and exact state round trip. Results do not establish the separate4% FFN threshold everywhere. See results.md.

## Final Checkpoint and Cleanup

Implementation:4622cdc87. Manual Android builds of dynamic-moe, dynamic-moe-check and the pre-existing fullmodel-oracle (output fullmodel-oracle-regression) pass. The latter confirms helper API source compatibility, not a new old-policy device benchmark. Python py_compile and git diff --check pass. The checker uses the same compiler/flags as the main command above with tests/dynamic-moe-check.cpp; the compatibility build substitutes tests/fullmodel-oracle.cpp. Exact device commands/binary hashes for full-model runs are in *.run.json.

Diagnostic invocation (after deploying libraries, checker and capture):

```sh
adb -s fd8657d6 shell 'cd /data/local/tmp/llama-dynamic-moe && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 GGML_HEXAGON_NHMX=1 GGML_HEXAGON_MM_SELECT=3 GGML_HEXAGON_PROFILE=0 ./dynamic-moe-check /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf capture.bin 16 2 check-layer2-fixed.json'
```

Main JSON, binary outputs and logs are gzip-archived; gzip integrity passes. SHA256SUMS records archived evidence including ignored large captures/failed logits retained on the host. Runner verifies hashes for its remote outputs before pulling. No original experiment is overwritten.

No dynamic-moe or dynamic-moe-check process remained before cleanup. Removed only /data/local/tmp/llama-dynamic-moe, verified absence, and checked the original model size8149865792bytes. Future runs need deployment again; the runner performs it. No required work is pending for this functional scope. Future true cross-backend shared activations, fused conversion and faster phase migration are not implemented.

Final runner guard rejects every nonempty evidence directory, including failed and gzip-archived runs. Verified with DYNAMIC_EVIDENCE=evidence-v9: expected refusal occurs before deployment and existing evidence is preserved. Python py_compile passes after this host-only guard change; the device executable is unchanged.
