# Expert E2E Repeat on OnePlus Ace 5 Pro

## Scope

User requests the same benchmark as the fd8657d6 experiment, after transferring
the model directly from host to db6c02cf. No other Android device may supply the
model. Use 64 tokens per Expert, real layer 0 Expert IDs 7/8, one or two Experts,
resident weights, CPU 4 threads/OpenCL/HTP HMX/HTP HVX. Time upload, complete
gate/up/SwiGLU/down graph, synchronization and output readback. No attention,
router or aggregation in this Expert-level benchmark.

## Host Source and Device

Host source (regular local file, 8149865792 bytes):
`models/Qwen1.5-MoE-A2.7B-Chat-Q4_0-GGUF/qwen1.5-moe-a2.7b-chat-q4_0.gguf`.
Host SHA256 was computed and matches the prior experiment:
`8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`.
Destination: `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf` on db6c02cf.
Device PKR110, SM8750, Android 15, MemTotal 15469100 kB. Initially 285 GB free.
Destination model and `/data/local/tmp/llama-expert-e2e` did not exist initially.

Existing host probe and all six libraries have the same SHA256 hashes as the
prior experiment (implementation d3dcc4290, results 72020f546). No rebuild or
production code change is needed. No operations transfer data from fd8657d6.

## Transfer Progress and Recovery

Direct `adb push` and single `adb exec-in dd` were slow. Both were stopped and
their processes ended before starting the partitioned transfer. A temporary
host HTTP server on 192.168.77.3:18767 was unreachable from the device and was
stopped. No temporary HTTP service remains.

`transfer-model.py` copies eight disjoint byte ranges from the host file to the
same device file with `dd ... oflag=seek_bytes conv=notrunc`; it verifies full
device SHA256 and length and only then creates `transfer.json`. It starts with
eight streams, but concurrency congested the ADB connection, causing control
queries to time out. Pausing senders restored ADB responses: device remained
online and had 283 GB free. The run was then paced at two active senders;
the rest were temporarily SIGSTOP'ed and automatically resumed in pairs as
active senders finish. Existing data is preserved. This operational adjustment
does not change the model bytes. Logical file size is not completion evidence.

Transfer is BLOCKED and stopped at this checkpoint. After approximately 47
minutes, submitted bytes reached 4809818112/8149865792, then progress was nearly
absent for several minutes. Submitted bytes include buffering and are NOT a
verified count of device bytes. Repeated `timeout 12 adb devices -l` and
`timeout 45 adb devices -l` returned exit 124. Pausing all active senders for
45 seconds did not restore responses. The scheduler was interrupted (resuming
all stopped children in finally), all remaining sender clients terminated, and
the transfer parent exited with BrokenPipeError as expected from cancellation.
No local transfer/scheduler/HTTP processes remain. A final 45-second ADB query
after stopping all senders also timed out. The user was asked to check the
device/ADB connection; no response had arrived at this checkpoint.

No `transfer.json` was produced, no full device SHA256 verified, and no benchmark
deployed or executed. The destination is an unverified sparse partial model,
possibly with full logical length. DO NOT use it for inference or measurements.
Device-side sender termination and cleanup cannot be confirmed while ADB is
unreachable. Do not blindly rerun `transfer-model.py`: it truncates the partial
destination and would discard completed data.

## Resume After ADB Recovers

1. Confirm `adb devices -l` responds and db6c02cf is in `device` state. Check for
   leftover task-owned `dd` processes, identifying their command lines before
   stopping them. Do not affect other devices or unrelated processes.
2. The first two ranges completed with successful sender exits before the next
   pair started. Their combined prefix is 2038431744 bytes (1944 MiB). Verify
   that prefix on device before reusing it; successful transfer exit alone does
   not establish integrity.
3. Host prefix SHA256 (computed independently after stopping):
   `624ef20f740956f283cad353d6e6c8b7b750d2a753de7ebc05e4daf89ff99ab5`.
4. If the device prefix matches, resume from offset 2038431744 and overwrite
   the unverified remainder using host data. Use a stable connection; the prior
   link was slow and intermittently stopped responding. Otherwise retransmit
   the whole file from host.
5. Verify complete device length and SHA256 against the host values above, then
   update tasklist and run the tests below. Preserve the model after testing.

Prefix verification and conditional resume commands (only after steps 1-3):

```sh
adb -s db6c02cf shell 'dd if=/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf bs=1048576 count=1944 status=none | sha256sum'
dd if=models/Qwen1.5-MoE-A2.7B-Chat-Q4_0-GGUF/qwen1.5-moe-a2.7b-chat-q4_0.gguf bs=1048576 skip=1944 status=none | adb -s db6c02cf exec-in 'dd of=/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf bs=1048576 seek=1944 conv=notrunc'
adb -s db6c02cf shell 'stat -c %s /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf; sha256sum /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf'
```

## Next Commands

All evidence uses the existing runner directory
`task/handoff/handoff-2026-09-23-expert-e2e/evidence` and distinct `ace5pro_` tags.
Do not overwrite the original run01/run02/run03 evidence or summary.json.

```sh
python3 tests/run-expert-compute.py --serial db6c02cf --e2e --deploy --tag ace5pro_smoke01 --reps 3 --warmup 2
python3 tests/run-expert-compute.py --serial db6c02cf --e2e --tag ace5pro_run01 --modes cpu,opencl,hmx,hvx --reps 100 --warmup 10
python3 tests/run-expert-compute.py --serial db6c02cf --e2e --tag ace5pro_run02 --modes hvx,hmx,opencl,cpu --reps 100 --warmup 10
python3 tests/run-expert-compute.py --serial db6c02cf --e2e --tag ace5pro_run03 --modes opencl,cpu,hvx,hmx --reps 100 --warmup 10
python3 tests/run-expert-compute.py --serial db6c02cf --e2e --tag ace5pro_profile01 --modes hmx,hvx --reps 1 --warmup 0 --profile
python3 tests/summarize-expert-e2e.py --tags ace5pro_run01 ace5pro_run02 ace5pro_run03 --output ace5pro-summary.json
```

After measurements: audit provenance, sample counts and numerical checks; inspect
profile MUL_MAT_ID tile modes and HMX_COMP/HVX_COMP events for both Expert counts.
Archive only new raw JSON/log files with gzip; keep manifests and summary readable.
Verify regenerated summary equality, remove only the owned device probe directory,
retain the transferred model, update tasklist/results, check diff and commit.
