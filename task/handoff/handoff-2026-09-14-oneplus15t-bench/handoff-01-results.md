# OnePlus 15T Qwen1.5-MoE Results

## Status

Model transfer and CPU benchmarks completed on 3B162U00GZY00000 (PLZ110,
OnePlus 15T, SM8850). HTP inference is blocked by the production device's
SELinux policy for the ADB shell domain. No HTP throughput was measured.
User explicitly requested no APK. No APK was installed; temporary APK sources,
tools and build artifacts were removed. No production code, SELinux policy,
device permissions or clock settings were changed.

Source revision: 832338aa99d057f619444e4ff8528da199349f7a. Existing Android Release
build was incrementally rebuilt with:

```sh
cmake --build build-android-opencl-hexagon --target llama-bench -j 8
```

## Model And Deployment

Model retained at `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`.
Size: 8149865792 bytes. SHA256:
`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.
The target hash matches the host model under
`models/Qwen1.5-MoE-A2.7B-Chat-Q4_0-GGUF/`.

Host ADB upload was slow and was interrupted before switching to device-to-device
HTTP transfer. Source fd8657d6 already held the same model. The successful
transfer took 195.663518 seconds, 41652454 bytes/s. No inference was run on the
source device. Source model was only read and was not modified.

Transfer commands (temporary handler is preserved in evidence/serve-model.sh):

```sh
adb devices -l
adb -s fd8657d6 forward --no-rebind tcp:18766 tcp:18766
adb -s 3B162U00GZY00000 reverse --no-rebind tcp:18766 tcp:18766
adb -s fd8657d6 shell mkdir -p /data/local/tmp/llama-moe-transfer
adb -s fd8657d6 push task/handoff/handoff-2026-09-14-oneplus15t-bench/evidence/serve-model.sh /data/local/tmp/llama-moe-transfer/serve-model.sh
adb -s fd8657d6 shell 'toybox nc -l -s 127.0.0.1 -p 18766 sh /data/local/tmp/llama-moe-transfer/serve-model.sh'
# In another host terminal while the server is listening:
adb -s 3B162U00GZY00000 shell 'curl -fSs --noproxy "*" --connect-timeout 10 --max-time 900 -o /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -w "bytes=%{size_download} seconds=%{time_total} bytes_per_second=%{speed_download}\n" http://127.0.0.1:18766/model.gguf'
adb -s 3B162U00GZY00000 shell 'stat -c %s /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf; sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
adb -s fd8657d6 forward --remove tcp:18766
adb -s 3B162U00GZY00000 reverse --remove tcp:18766
adb -s fd8657d6 shell 'rm /data/local/tmp/llama-moe-transfer/serve-model.sh && rmdir /data/local/tmp/llama-moe-transfer'
```

Benchmark files are retained together at `/data/local/tmp/llama-moe-bench` for
resuming HTP diagnosis. All ten deployed files match host SHA256 values, recorded
in evidence/host-libraries.sha256 and evidence/device-libraries.sha256. Deployment:

```sh
adb -s 3B162U00GZY00000 shell mkdir -p /data/local/tmp/llama-moe-bench
adb -s 3B162U00GZY00000 push build-android-opencl-hexagon/bin/llama-bench build-android-opencl-hexagon/bin/libllama-bench-impl.so build-android-opencl-hexagon/bin/libllama-common.so build-android-opencl-hexagon/bin/libllama.so build-android-opencl-hexagon/bin/libggml.so build-android-opencl-hexagon/bin/libggml-base.so build-android-opencl-hexagon/bin/libggml-cpu.so build-android-opencl-hexagon/bin/libggml-opencl.so build-android-opencl-hexagon/bin/libggml-hexagon.so build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v81.so /data/local/tmp/llama-moe-bench/
adb -s 3B162U00GZY00000 shell chmod 755 /data/local/tmp/llama-moe-bench/llama-bench
```

## CPU Results

Mean +/- sample standard deviation, tokens/s, three repetitions after the tool's
default warmup. CPU repacking enabled, all 25 layers on CPU, flash attention on,
batch/ubatch 512, mmap, lazy mode off, F16 KV, context depth zero. Tests are
synthetic llama-bench pp/tg workloads. Loading, tokenization and sampling are not
included in the reported throughput. These are not chat completion timings.

| CPU threads | pp128 | pp512 | tg128 |
| ---: | ---: | ---: | ---: |
| 4 | 19.26 +/- 3.47 | 36.94 +/- 1.47 | 17.68 +/- 2.74 |
| 8 | 49.45 +/- 10.89 | 32.98 +/- 1.84 | 19.69 +/- 9.67 |

Eight-thread tg128 samples: 8.54507, 25.9037, 24.616 tokens/s. Large variation
prevents treating the fastest sample as a stable device performance estimate.
CPU frequency caps were dynamic: initial policy0/policy6 maxima were
1996800/2438400 kHz, and the post-CPU snapshot showed 1785600/1497600 kHz.
Battery temperature was 34.5 C during transfer and 37.1 C after CPU testing.
Phone background activity, DVFS, memory residency and cooling were uncontrolled.
These measurements establish current-condition throughput, not peak performance
or a definitive best thread count.

Exact CPU command, stdout in evidence/cpu.json and stderr in evidence/cpu.log:

```sh
adb -s 3B162U00GZY00000 shell 'cd /data/local/tmp/llama-moe-bench && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. GGML_HEXAGON_ARCH=81 ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -dev none -ngl 0 -nopo 1 -t 4,8 -p 128,512 -n 128 -b 512 -ub 512 -fa on -lm mmap -lzm off -r 3 -o json -v'
```

CPU JSON validation passed: six configurations, three positive samples each,
positive mean throughput, zero offloaded layers. Logs confirm CPU_Mapped and
CPU_REPACK model buffers and CPU compute buffers. Merely registering HTP/OpenCL
backends does not mean this CPU run executed on those devices.

Final `git diff --cached --check` passed. Only trailing whitespace was normalized
in four text logs and the two readelf header outputs; numeric JSON samples were
not edited. The checkpoint contains only this task's report and evidence.

## HTP Version And Failure

The deployed `libggml-htp-v81.so` has ELF machine QUALCOMM DSP6 and flags `0x81`.
The phone's own `/vendor/lib64/rfs/dsp/libbitml_nsp_81na_skel.so` has the same
machine and flags `0x81`. Header outputs are preserved in evidence/deployed-htp-elf.txt
and evidence/vendor-htp-elf.txt. The temporary host copy of the vendor binary was
deleted after inspection; the device vendor file was only read.

FastRPC sysfs reports nsp1000, domain_id 1000, instance_id 0, legacy_id 3,
legacy_name cdsp, status 1. This is consistent with discovery in the benchmark.
Hardware capability query from shell cannot complete: it returns 114 (0x72).
The printed v73 is the software fallback at ggml-hexagon.cpp:6350, not a successful
hardware version query. Setting GGML_HEXAGON_ARCH=81 selects the matching v81
skeleton but does not grant access to the FastRPC device node.

Both three-session and single-session attempts fail at htp_iface_open, before
weight allocation or graph execution. Qualcomm AEEStdErr.h:196 defines 0x72 as
AEE_ECONNREFUSED, "Connection refused to DSP". The decisive logcat evidence is:

```text
open_device_node: no access to default device of domain 1000, open thru HAL
open_device_node failed ... (errno 13, Permission denied)
avc: denied { open } for path="/dev/fastrpc-nsp1000"
scontext=u:r:shell:s0 tcontext=u:object_r:vendor_qdsp_device:s0
tclass=chr_file permissive=0
```

The HAL fallback also encounters a shell-domain vndbinder denial. These records
are in evidence/htp-fastrpc.log:28 and subsequent lines. Version mismatch is not
the cause of the observed failure: the connection is rejected before the DSP
module can load. Direct runtime hardware-version confirmation and actual v81
execution remain unverified until the access restriction is resolved.

Device reports ro.debuggable=0, build type user, SELinux Enforcing, no `su`.
No permission changes, root installation, policy changes or APK deployment were
performed. Do not retry other skeleton versions to address this access denial.

Three-session attempt (evidence/htp.log and htp-attempt.stdout), exit 1:

```sh
adb -s 3B162U00GZY00000 shell 'cd /data/local/tmp/llama-moe-bench && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=. GGML_HEXAGON_ARCH=81 GGML_HEXAGON_DEVICES=HTP0:0,HTP0:1,HTP0:2 GGML_HEXAGON_NHMX=1 GGML_HEXAGON_MM_SELECT=3 ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -dev HTP0:0/HTP0:1/HTP0:2 -ngl 99 -ts 1/1/1 -t 4 -p 128,512 -n 128 -b 512 -ub 512 -fa on -lm mmap -lzm off -r 3 -o json -v'
```

Single-session diagnostic (evidence/htp-single.log and htp-single-attempt.stdout),
exit 1. This fails immediately at opening the base session; it does not test
whether the complete model fits in a single session's virtual address space.

```sh
adb -s 3B162U00GZY00000 shell 'cd /data/local/tmp/llama-moe-bench && LD_LIBRARY_PATH=.:/vendor/lib64 ADSP_LIBRARY_PATH=/data/local/tmp/llama-moe-bench GGML_HEXAGON_ARCH=81 GGML_HEXAGON_DEVICES=HTP0:0 ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -dev HTP0:0 -ngl 99 -t 4 -p 16 -n 4 -b 128 -ub 128 -fa on -lm mmap -lzm off -r 1 -o json -v'
adb -s 3B162U00GZY00000 logcat -d -s llama-bench:V adsprpc:V fastrpc:V
```

## Resume And Cleanup

No benchmark process remains. Original model and the fixed benchmark directory
are intentionally retained on the target; there are no device log files or
extra model copies from this task. Transfer servers and port mappings are gone.
All evidence is on the host. Broad unrelated logcat output was removed after
collecting the targeted FastRPC evidence.

Remaining work is HTP timing, blocked on a supported execution context with
FastRPC access. User disallows APK. Once such an environment is available, confirm
`adb devices -l`, retry architecture detection without an override, then run the
three-session command above. Do not change device security policy or unlock/root
the device as part of this benchmark without a separate user instruction.

This checkpoint records completed transfer/CPU results and an explicitly blocked
HTP test; it does not claim an end-to-end HTP inference success.
