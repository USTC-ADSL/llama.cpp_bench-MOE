# Implementation and initial device validation

Work is active; acceptance/generalization and final cleanup are not complete.

## Changes

- tests/moe-route-scheduler.h: bounded deterministic multistart whole-expert move/swap search, including joint shared/expert moves; actual current counts and explicit availability, immutable physical bank widths; no timing lookup. Old candidate families are seeds, not admissibility restrictions. Default 2048 evaluations/layer.
- tests/moe-placement-model.h/.py: physical bank width features; validate coefficients at load; fixed-size prediction arrays. Old choose() remains unchanged for historical controls.
- tests/fullmodel-oracle.cpp: independent resident union banks/shared variants; current-route callback schedules before partition, decision included in gather and llama_decode; shared-aware graph cache/reservations; physical/logical memory and preparation reports; external calibration placements and updated lifetime/overlap profile contract.
- tests/moe-decode-branch.h: adapt fixed decode cache/reservation indexing; behavior unchanged.
- tests/run-fullmodel-oracle.py: --route-model and --profile-candidates with hashes.
- tests/moe-route-plan.cpp: identical host planner and explicit per-request provision manifest. Prepares compact selected ownership plus six promising alternate expert layouts and all shared layouts. This is limited, route-conditioned prepared residency, not universal/cold-request execution. Timed search sees only actual current route and this manifest.
- tests/fit-moe-route-model.py: nonnegative, relative-weighted Ridge. Nonnegative coefficients prevent exploiting negative marginal work outside calibration. No identity features. 128/768/1536 development profiles and earlier technical 768 full-model samples. model-dev-v1.json is a development checkpoint, not yet a final freeze.
- tests/test-moe-route-scheduler.cpp and route-resident-probe.cpp: meaningful host exhaustive/availability/current-route checks and device graph/mapping migration checks.

## Validation and measurements

Host: `g++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-route-scheduler.cpp -o /tmp/test-moe-route-scheduler && /tmp/test-moe-route-scheduler` passed (small exhaustive optimum 22; 144 evaluations). Includes inactive-expert constrained availability. `python3 tests/test-moe-placement-model.py` passed 4 existing contracts.

Android compilation: `/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle`. Same command with route-resident-probe.cpp/output route-resident-probe compiled and ran on fd8657d6. No library rebuild needed.

Device contract: evidence/resident-contract.json/log; 12 native/shared/three-way cases, 15 cached graphs, exact zero union-vs-compact difference; reverse-order revisit bitwise equal and no allocations.

Profiles in evidence/dev-{128,768,1536}-profile.json. Four layers 0/8/16/23, 28 ordinary candidates and 19 terminal candidates, three repetitions each after two warmups. Captures use real CPU routes/activations, not synthetic IDs. Native and accelerated layer16 checks both slightly exceed previous 4% worst-token threshold at token13 (~4.09%); do not report all candidate checks passed. CPU-only matches reference; native/heterogeneous use reduced precision.

Model training: 501 rows, alpha=.03, fit total MAPE9.44%; CPU/NPU/GPU MAPE25.80%/6.22%/5.19%. Held-length total MAPE128/768/1536 =8.5%/10.7%/10.3%, candidate regret8.1%/8.8%/4.4%. These are development CV, not independent confirmation.

First real current-route 768 run evidence/dev-768-route-v1.json: median940.309ms vs initial native1077.799ms (+14.62% throughput). Sum backend windows CPU505.81/NPU507.51/GPU472.72ms; FFN including planning600.67ms; planning11.89ms; model579.43ms excludes planning. Actual owner selection differs from provision choice in5/24 layers; real IDs differ from CPU capture in4541 slots. Logits relative L2 to native3.1415%, same top1; all rounds bitwise stable. These remain development results; fixed controls/native B are being collected.

Resident logical and reported allocated weights8.484GiB, activation arena288.30MiB, preparation9.326s. Preparation includes backend/bank import and arena construction outside timing; no cold-start gain is claimed.

## Evidence and execution notes

All run manifests contain exact commands, source/model hashes and device metadata. Remote fixed directory /data/local/tmp/llama-fullmodel-oracle. Deployed libc++_shared.so separately from build-android-opencl-hexagon/ggml/src/ggml-hexagon/ship because previous task cleaned the directory. No other experiments overlapped.

Initial dev-768-capture failed because default prompt is shorter than768. dev-1536-capture similarly failed because old technical prompt is shorter than1536. Successful replacements are dev-768-capture-v2 and dev-1536-capture-v2. Long development prompt is technical-extended.txt (explicitly repeated technical cases); do not reuse the short prompt at1536/2048. Failed logs retained as evidence for now.

Search scout alleged inactive projection skips inactive experts, but exact line61 uses `|| !counts[e]` and therefore projects them; allegation rejected after source check and a passing added contract. The NPU preference loop was clarified to {1,0,2}+break (same semantics as prior last-write {2,0,1}). Equal-count swaps leave all identity-free features unchanged, so skipping them is valid. Runtime scout found no blocker; repeated expert IDs within one token are outside the real top-k contract.

## Next

Finish 768 fixed c8/g15 + old CPU4/sharedGPU + native B controls; run current model at128 and1536 development lengths. Refit only if data warrants, then freeze code/coefficients/provision protocol before independent prompts and unseen lengths (1024/2048). Preserve TODO.md untracked. Commit stage after final source review and required validation; final independent acceptance still pending.
