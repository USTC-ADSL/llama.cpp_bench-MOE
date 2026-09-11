# Natural-Prompt Semantic Regression

## Request And Setup

User requested roughly 128 input tokens and 32 output tokens. They explicitly
relaxed the input length requirement in favor of a natural complete question.
Final prompt is 119 tokens including the model chat template. Device is
fd8657d6 (Redmi K80 Pro), using the existing
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf. Working tree was clean at start.

The prompt is a self-contained delivery note: Alice sends Bob three books,
delivery is Friday, and an earlier Monday proposal was cancelled. It asks for
the recipient, contents, delivery day and cancelled date. Additional context
describes packaging, the confirmed address and payment; no artificial padding
or truncated prompt is used.

Generation is greedy and feeds back its own generated tokens. Semantic mode
does not accept teacher tokens. EOG is suppressed if necessary to obtain a
fixed 32 generated tokens; summary.json records the suppression count. The
32-token cutoff can truncate the final explanation. A supplementary --stop-eog
mode allows natural termination with a maximum of 32 tokens. All inputs, generated
IDs, readable output and complete vocabulary logits for generated text tokens are
saved. The terminating EOG decision's logits are not saved in natural-stop mode.

Runtime settings: context=256, batch=256, ubatch=4, CPU threads=4. Cache banks
have 32 expert slots. CPU/OpenCL/HTP each have a corresponding no-swap reference.
The mixed configuration uses CPU layers 0-7, GPUOpenCL 8-15 and HTP0 16-23;
its swap budgets are 1 GiB / 1 GiB / 2 GiB, with 128 MiB reserve per device.
These remain logical allocation budgets, not physical RAM limits.

## Changes

- tests/weight-manager-probe.cpp: model chat template, natural semantic prompt,
  32-step free generation, explicit EOG handling, prompt/output artifacts and
  actual token counts. Existing short-prompt mode retains its eight-step behavior.
- tests/run-weight-manager.py: --semantic and --stop-eog, disable reference-token
  input in this mode, verify token counts and four expected keywords, compare prompts, output IDs
  and complete logits against corresponding references. Deployment hashes are
  now recorded per tag rather than overwriting prior deployment evidence.
- docs/weight-manager.md: semantic mode usage and interpretation limits.

No inference/manager/backend production code changed in this follow-up.
A read-only independent review of the two probe/runner files found no execution
correctness issues. Keyword assertions alone are not sufficient for semantic
assessment; readable outputs are also inspected.

## Commands

```sh
adb devices -l
/mnt/sda1/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iinclude -Iggml/include tests/weight-manager-probe.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/weight-manager-probe
python3 tests/run-weight-manager.py --deploy --semantic --tag sem128c --ubatch 4 --slots 32 --case cpu-ref cpu-swap gpu-ref gpu-swap htp-ref htp-swap mixed-fixed mixed-swap-4g
# After adding natural EOG handling, repeat the probe compilation above, then:
adb -s fd8657d6 push build-android-opencl-hexagon/bin/weight-manager-probe /data/local/tmp/llama-weight-manager/weight-manager-probe
sha256sum build-android-opencl-hexagon/bin/weight-manager-probe
adb -s fd8657d6 shell sha256sum /data/local/tmp/llama-weight-manager/weight-manager-probe
python3 tests/run-weight-manager.py --semantic --stop-eog --tag semnatural --ubatch 4 --slots 32 --case mixed-fixed mixed-swap-4g
```

The valid Android build and existing libraries were reused; only the probe was
recompiled. Deployment verifies hashes and uses the v79 skeleton from
build-android-opencl-hexagon/ggml/src/ggml-hexagon/. Evidence is stored as
evidence/sem128c-*; raw logits and full logs stay local under existing ignore
rules. Reports/configs/tokens and readable prompt/output are retained.

Initial tags sem128a and sem128b failed before inference because the earlier
exact-token sentence-combination search could not form exactly 128 tokens.
The diagnostic candidate range was 79..132 tokens. Following the user's length
clarification, that search was removed. These were test-input construction
failures, not model/backend failures. Their small result/config/hash records
remain as evidence; do not use those tags as inference references.

The natural-stop probe's host/device SHA256 matched:
`294c637b88f0d2cce5194ca6ffc2160f54dffe51c70f5aa02760849862fe33c2`.
Libraries were unchanged from evidence/sem128c-hashes.json.

## Final Results

All ten runs completed on the Android device. Inputs were identical (119 tokens),
generation was free greedy decoding, and all checked logits were finite. Each of
the following pairs had identical generated token IDs and saved float32 logits:
RMSE=0 and max_abs=0. This compares swap to its corresponding backend reference,
not logits across different backends.

| Tag / pair | Output tokens | Manual semantic assessment | Reference / swap seconds |
| --- | ---: | --- | ---: |
| sem128c CPU | 32 | Correct facts, explanation truncated | 9.679 / 20.409 |
| sem128c OpenCL | 32 | Correct facts, explanation truncated | 33.902 / 74.470 |
| sem128c HTP | 32 | Correct facts, explanation truncated | 14.042 / 44.744 |
| sem128c mixed | 32 | Correct labeled facts, incorrect appended explanation | 29.702 / 53.192 |
| semnatural mixed | 18 | Correct labeled facts, natural termination | 28.781 / 49.094 |

Times cover the whole probe process, including model load and test artifact I/O;
they are single runs, not prefill/decode throughput measurements.

All three single-backend pairs produced:

```text
Recipient: Bob
Contents: Three books
Delivery day: Friday
Cancelled date: Monday

The delivery was initially proposed for Monday but was cancelled and rescheduled
```

Both mixed fixed-32 runs instead appended:

```text
The delivery was scheduled for Friday, but it was cancelled on Monday
```

This is incorrect/unsupported: the note says an earlier Monday delivery proposal
was cancelled, not that Friday delivery was cancelled on Monday. The runner's
`semantic_checks` are keyword-presence checks and all passed despite this error.
The fixed-32 mixed answer therefore does not pass full semantic assessment.

CPU and mixed fixed-32 runs each suppressed EOG once; OpenCL and HTP suppressed
none. Inspection of the raw fixed-32 logits found the first greedy EOG at step 18
(zero-based) for CPU and mixed, after the four correct labeled facts. EOG IDs in
the model log are 128247, 151643 and 151645. The incorrect mixed explanation was
generated after forced continuation and is also present in the no-swap reference;
it is not a swap regression.

The supplementary mixed --stop-eog pair confirmed natural termination after 18
text tokens, with no suppressed EOG and the following identical outputs:

```text
Recipient: Bob
Contents: Three books
Delivery day: Friday
Cancelled date: Monday
```

Both modes use labeled lines instead of the requested two sentences. This is one
fact-retrieval regression example; it does not establish general language quality,
perfect instruction following or why mixed-backend continuation differs from the
single-backend runs. No production behavior was changed to improve the answer.

## Replacement And Budgets

The 32-slot caches exercised actual replacement, not just resident cache hits:

| sem128c swap | Misses | Evictions | Source bytes read |
| --- | ---: | ---: | ---: |
| CPU | 36756 | 36692 | 59618820096 |
| OpenCL | 36750 | 36686 | 59609088000 |
| HTP | 36774 | 36710 | 59648016384 |

Mixed fixed-32 swap evictions were CPU=12567, GPU=12389, HTP=11341. Natural-stop
swap evictions were CPU=11366, GPU=11207, HTP=10131. Both mixed swap runs reported
peak accounted bytes CPU=926938656, GPU=490749568, HTP=490731392, including each
backend's 128 MiB reserve. These fit the configured logical budgets, but do not
establish a 4 GiB physical RAM limit. Android system swap remains independent;
the natural mixed swap run sampled nonzero VmSwap (32256 kB).

## Closure

All inference tests are complete. Compact evidence is retained under sem128c-*
and semnatural-*; full logs, raw logits and most weight registries remain local
under existing ignore rules. No additional inference run is needed for the
documentation-only closure. Final `python3 -m py_compile tests/run-weight-manager.py`
and `git diff --check` passed. A final independent reread of saved artifacts
confirmed all five pairs have identical token files and finite, identical logits:
4,861,952 float32 values per fixed-32 run and 2,734,848 per natural-stop run.

After confirming no task probes were running, device cleanup completed with:

```sh
adb -s fd8657d6 shell 'test -f /data/local/tmp/llama-weight-manager/weight-manager-probe && ! pidof weight-manager-probe weight-slice-probe weight-manager-api-probe && rm -rf /data/local/tmp/llama-weight-manager'
adb -s fd8657d6 shell 'test ! -e /data/local/tmp/llama-weight-manager && test -r /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
```

The model was preserved. No other device or system-swap setting was changed.
This test/probe/documentation checkpoint follows fd7c7070f and f2af031a5; identify
its commit with `git log -1 -- tests/weight-manager-probe.cpp`. No production
manager/backend/model changes or uncompleted inference tests remain in this task.
