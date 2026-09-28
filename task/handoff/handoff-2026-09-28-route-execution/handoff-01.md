# Controlled audit and materialized GET_ROWS checkpoint

Base638a00822, branch feat/dynamic-moe-placement. User requests continued execution optimization and questions earlier unit-cost interpretation. TODO.md is user-owned and must remain unstaged. Device fd8657d6 only, no concurrent experiments. This stage is experimental, not completion of the execution optimization.

## Completed

The previous arithmetic used the correct effective token counts and top4 work. However average FFN latency per token is not an intrinsic unit-work cost, and its decrease alone does not establish the cause of reduced heterogeneous speedup. Layer16 old NPU reference checks fail the old4% worst-token criterion; do not claim all old checks passed. Old major faults are zero, not83/84/83 (those are minor faults).

Added opt-in MOE_NPU_DEVICE_GATHER / --npu-device-gather: copy each input token once; GET_ROWS on NPU expands routed input; shared expert aliases original input. Normal GGML semantics. New per-layer host-copy byte counters. Both full-model paths still schedule current routes inside timing.

Added tests/moe-route-unit-probe.cpp and tests/run-moe-route-execution-study.py. The similarly named pre-existing tests/run-moe-execution-study.py is unchanged. Probe uses one resident layer0, repeats the exact128-token capture6x/12x, alternates forward/reverse case order,2 warmups and6 samples, and validates repeated/equivalent outputs.

Native backend-compute medians (ms),128/768/1536: NPU17.317/31.514/46.536; CPU49.713/296.558/594.878; GPU31.027/123.139/233.446. Each backend has its own process with all sizes resident, so within-backend scaling is controlled; between-backend device state is not simultaneous. Native/split/GET_ROWS outputs are exactly equal within each backend. Original NPU DSP rate ~2112cycles/us, one batch, zero major faults.

GET_ROWS lowers host gather1.323→0.280,5.219→1.102,6.533→1.573ms but raises NPU compute17.345→17.567,31.044→31.897,46.671→48.028ms. Copied activation bytes5/30/60MiB→1/6/12MiB. Scalar probe reduction is diagnostic, not comparable with fullmodel NEON reduction.

Fullmodel A/B counterbalanced sequence native/base/gather/gather/base/native,2 warmups+6 timed each:768 base958.491/953.035ms vs gather956.429/958.258ms;1536 base1885.621/1977.753 vs gather1872.225/1904.595ms.768 no stable benefit;1536 gains0.715%/3.841% with drift. Full-model base/gather logits bitwise equal; repeated logits and allocations stable.768 input-copy537.695→138.008MiB; NPU gather37.481→12.995ms but compute522.943→542.380ms; expert wall583.252→578.254ms. Branch intervals overlap; never sum them as E2E.

Resident probe12 native/shared/threeway cases passed1e-4 reference and bitwise reverse cache revisits.

## Commands and evidence

Compiled fullmodel-oracle, route-resident-probe, moe-route-unit-probe using NDK r27d aarch64-linux-android34-clang++, -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann, -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base. Baseline binary saved as bin/fullmodel-oracle-route-base.

python3 tests/run-moe-route-execution-study.py deploy|unit|unit-cpu|unit-gpu|resident and full --tokens768/1536 (tokens passed as separate CLI argument). All passed. New evidence/* reports contain samples and .run.json manifests. first-pass.json summarizes fullmodel. deployment.json is latest deploy; per-run embedded manifests identify actual artifact versions. Raw files ignored by local .gitignore, must archive before final cleanup.

## Next

Implement true backend GET_ROWS→RESHAPE→HMX MUL_MAT_ID fusion, opt-in and with fallback. HMX already gathers rows; separate activation row from output row in mmid_row_mapping. Preserve route slot i1 and output row i2; down projection must not use original-token map. Descriptor src9 can carry optional index tensor, but must preserve ordinary/mixed gate-up NX fusion and descriptor capacity/clone accounting.

Do not skip GET_ROWS without proving source/index lifetime: current shared_activation_layout only tracks public src dependencies and does not honor output flags. Original input/index can be reused after GET_ROWS before later projections. Need explicit retained lifetimes and backend eligibility/fallback checks. Read exact code before editing. Incremental targets ggml-hexagon and htp-v79, deploy matching host/DSP libraries. Then numerical regression, primitive and counterbalanced fullmodel tests, honest report and cleanup.
