# OnePlus 15T Qwen1.5-MoE Benchmark

Target: 3B162U00GZY00000 (PLZ110, SM8850). Model: Qwen1.5-MoE-A2.7B-Chat Q4_0.
Source revision: 832338aa99d057f619444e4ff8528da199349f7a.

- [x] Confirm target online, locate local model and existing Android Release build.
- [x] Transfer model and verify SHA256; deploy matching benchmark libraries.
- [x] Run CPU benchmarks at 4 and 8 threads, three repetitions per test.
- [x] Attempt HTP execution with three sessions and a single session; capture failure evidence.
- [x] Compare device vendor DSP ELF and deployed skeleton: both v81; direct capability query remains denied.
- [ ] Obtain HTP timings (blocked by shell access denial; no APK per user instruction).
- [x] Review results, preserve commands/logs and clean temporary artifacts; retain target model and benchmark for retry.
- [x] Save completed results and explicitly blocked HTP status in a dedicated Git checkpoint.

Device model path: /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf.
Device test directory: /data/local/tmp/llama-moe-bench (absent before this task).
Build command: `cmake --build build-android-opencl-hexagon --target llama-bench -j 8`.
Evidence directory: `evidence/` alongside this file. Model is retained on device.

Transfer finished through source-device HTTP over ADB forward/reverse in 195.664 s
(41.65 MB/s). Source: fd8657d6, original model unchanged. Target size 8149865792;
SHA256: 8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473.
All 10 deployed executable/library hashes match host. Temporary source server and
ADB port mappings have been removed. HTP autodetection returns error 114 on this
device; tests explicitly set GGML_HEXAGON_ARCH=81 and use three virtual sessions.
