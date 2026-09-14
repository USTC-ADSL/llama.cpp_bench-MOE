# Redmi K80 Pro Qwen1.5-MoE CPU / HTP Retest

Date: 2026-09-14. Device: fd8657d6, Redmi K80 Pro (24122RKC7C), SM8750.
Goal: recheck the reported CPU prefill advantage. No APK was built or installed.

## Results

The reported CPU prefill advantage was not reproduced. HTP prefill is faster;
CPU decode is faster. Values below are mean tokens/s over three measured
repetitions per configuration after llama-bench's default warmup.

| Batch / ubatch | Workload | CPU, 4 threads | CPU, 8 threads | HTP, 4 host threads |
| --- | --- | ---: | ---: | ---: |
| 512 / 128 | pp128 | 74.429 | 113.905 | 289.544 |
| 512 / 128 | pp512 | 85.155 | 112.059 | 287.732 |
| 512 / 128 | tg128 | 31.993 | 31.868 | 15.910 |
| 1024 / 512 | pp128 | 74.314 | 103.443 | 269.677 |
| 1024 / 512 | pp512 | 81.141 | 100.757 | 637.081 |
| 1024 / 512 | pp1024 | 76.025 | 95.176 | 627.866 |

pp means prompt processing; tg means token generation. This is synthetic-token
llama-bench throughput, not a real-prompt TTFT measurement. The 18 formal
configurations contain 54 samples. Raw samples and sample standard deviations
are retained in the four formal JSON files and evidence/summary.json.

For pp512, HTP is 2.57x CPU8 at ub128 and 6.32x CPU8 at ub512. HTP pp512
throughput increases 2.21x between these settings. Both batch and ubatch changed,
so this is not an isolated single-parameter experiment, although both batch
limits admit the entire 512-token prompt. Larger microbatches allow more tokens
to share expert matrix work. The pp128 results do not show a similar increase.

## Commands And Configuration

The model already existed on the selected device and was reused:

```text
/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
size: 8149865792 bytes
sha256: 8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473
```

Reused the valid build-android-opencl-hexagon binaries. All 11 deployed file
hashes matched the host (evidence/host.sha256 and evidence/device.sha256).
There are no inference-source changes between their source revision
832338aa99d057f619444e4ff8528da199349f7a and pre-report HEAD 3fd392cf0.
This retest required no rebuild or production-source changes.

Commands ran through adb -s fd8657d6 shell in the task-owned directory
/data/local/tmp/llama-moe-prefill-retest. Common arguments and environment:

```sh
cd /data/local/tmp/llama-moe-prefill-retest
export LD_LIBRARY_PATH=.:/vendor/lib64
export ADSP_LIBRARY_PATH=.
./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf \
  -fa on -lm mmap -lzm off -r 3 -o json -v [backend options] [workload options]
```

The bracketed arguments above are expanded as follows:

| Run | Backend options | Workload options |
| --- | --- | --- |
| cpu-ub128 | -dev none -ngl 0 -nopo 1 -t 4,8 | -p 128,512 -n 128 -b 512 -ub 128 |
| htp-ub128 | -dev HTP0 -ngl 99 -t 4 | -p 128,512 -n 128 -b 512 -ub 128 |
| htp-ub512 | -dev HTP0 -ngl 99 -t 4 | -p 128,512,1024 -n 0 -b 1024 -ub 512 |
| cpu-ub512 | -dev none -ngl 0 -nopo 1 -t 4,8 | -p 128,512,1024 -n 0 -b 1024 -ub 512 |

HTP runs additionally set GGML_HEXAGON_NHMX=1 GGML_HEXAGON_MM_SELECT=3.
The table order is execution order. Current code reads GGML_HEXAGON_NHMX;
GGML_HEXAGON_USE_HMX is not an effective knob in this revision.

The separate diagnostic run used the HTP environment plus
GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_VERBOSE=1, common model/FA/mmap options,
and -dev HTP0 -ngl 99 -t 4 -p 128,512 -n 4 -b 512 -ub 512 -r 1 --no-warmup.
Its timings are excluded from performance comparisons.

Expert numerical validation ran with the same library paths:

```sh
./expert-compute-probe /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf \
  htp-numeric.json HTP0 hmx 3 2 1,128
```

It passed at T=1 and T=128; T128 full-FFN worst-token NRMSE was 0.000582735.
This validates expert numerical behavior, not complete-model output quality.

## HTP Execution Evidence

The device auto-detects Hexagon v79 and successfully loads the matching v79
skeleton. Hardware reports 6 HVX, 1 HMX and 8 MB VTCM. Standard adb shell
execution works. No architecture mismatch, whole-model CPU fallback or
HMX/VTCM failure was observed.

The completed profile records show, for each of pp128 and pp512:

- 46 hmx-tiled expert operations: fused gate/up and down in layers 0 through 22.
- 2 hvx-tiled expert operations in layer 23. The final-layer graph selects only
  the required output token before FFN (src/models/qwen2moe.cpp:118).
- tg4 has 192 hvx-tiled expert operations, eight per layer, and no HMX expert ops.

Thus HTP execution does not mean every operation uses its matrix accelerator.
Single-token decode uses HVX here. Embedding, Q6_K output projection and router
softmax remain on CPU; softmax introduces readback, synchronization and upload.
These are plausible contributors to weaker HTP decode performance, not a
quantitative attribution of all elapsed time.

evidence/profile-excerpts.txt preserves representative original line numbers.
The complete profile is losslessly compressed as evidence/htp-profile.log.gz.
These are backend kernel completion records, not independent hardware PMU traces.

## Earlier Results And Limits

docs/Motivation_Exp/E1/report.md contains an earlier same-device 512-token
real-prompt result with t/tb4, b512/ub128: median CPU TTFT 11873.722 ms versus HTP
2224.106 ms. That result also favors HTP. It is context, not directly comparable
to the present synthetic-token throughput measurement.

docs/Motivation_Exp/E0_followup/report.md explicitly warns that its CPU perf
records belong to another phone, while later HTP records use fd8657d6. Those
numbers cannot establish a same-device CPU/HTP speed ratio. The user's specific
older CPU-faster result has not been uniquely identified, so its cause remains
unproven; configuration and metric differences require checking against its logs.

Initial battery temperature was 23.1 C, and post-formal-run snapshots were about
27.4 to 31.8 C. Reported CPU frequency caps remained 3.5328/4.32 GHz. No affinity,
frequency or security settings were changed. Battery temperature and frequency
caps do not fully characterize SoC thermals or instantaneous clocks. Run order
was not randomized; CPU pp128 had slower first samples, which were retained.

## Validation, Cleanup And Checkpoint

Ran python3 task/handoff/handoff-2026-09-14-k80-prefill/analyze.py successfully
after compressing the profile. It verifies all 54 positive samples, reported
means and standard deviations, CPU/HTP offload settings, and exact expert kernel
counts including the last-layer exception, then generates evidence/summary.json.

After confirming no llama-bench or expert-compute-probe process was running,
removed only /data/local/tmp/llama-moe-prefill-retest using rm -r. Verified the
directory no longer exists and the original model remains 8149865792 bytes;
see evidence/cleanup-before.txt and evidence/cleanup-after.txt. All relevant
logs were retained on the host. One trailing space in the file-size heading of
each formal text log was removed for Git whitespace validation; numeric JSON
and the compressed profile are unchanged. No APK or device security changes
were made.

Before the results checkpoint, git status contained only this new task directory
on branch moe-mobile-opt, based on 3fd392cf0. The scoped checkpoint contains
this report, checklist, analysis script and evidence; no production code changes.
No additional test is required to support the results above. A future exact
reproduction of the user's historical discrepancy needs the specific old command
and log. Re-deploy the hashed binaries before running the documented commands,
because this task's temporary device directory has been cleaned.
