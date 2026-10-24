# HTP 映射复用位图未覆盖高 32 个槽位

## 现象与触发条件

K80 Pro（fd8657d6），单份 CPU4x8 Qwen1.5-MoE 权重、公共 shared scratch、128 token prefill、CPU router/routed + NPU shared。32 步 decode 能完成，但重新 prefill 到第 8 层时退出 134，host 报 `dspqueue_read 0x2e`。原 async 和新常驻 worker 都能复现，CPU 专家控制组通过。

DSP logcat 为 TLBMISS RW，PC 相对 libggml-htp-v79.so 的偏移 0x93644，经 llvm-addr2line 定位为 `get_rows_thread_st_int32_t`，Bad VA 0x117A4000。host 侧 gather 索引检查通过。

## 根因与修改

`htp-ctx.h` 的 `HTP_MAX_MMAPS=64`，但 `main.c` 中 `reuse_buf` / `prep_op_bufs` 使用 32 位 `m_reuse` 和 `1<<i`。槽位 32–63 的位运算超出类型宽度，不能可靠记录当前批次仍使用的映射。地址空间压力触发 drop unused mappings 后，仍在使用的映射可能被解除，batch descriptor 保留旧 base，GET_ROWS 解引用时 DSP 崩溃。

最小修复：位图改为 uint64_t，两个移位改为 UINT64_C(1)<<i，并静态断言容量不超过 64。计算内核、gather 算法和阶段切换协议保持原逻辑。

## 验证与备份

- `cmake --build build-android-opencl-hexagon --target htp-v79 -j 8` 成功。
- 使用同一个 v4 executor，只替换修复后的 DSP 库，`diagnostic-v5-late-mapfix` 成功完成 quality 32 步以及两次 warmup / 一次计时序列，每次均正常重新 prefill。
- 失败证据：`task/handoff/handoff-2026-10-24-early-shared/evidence/diagnostic-v4-late-trace.{log,logcat,run.json}`。
- 成功证据：同目录 `diagnostic-v5-late-mapfix.{log,run.json,json}`。实际 adb 命令完整记录在 run.json。
- 原 main.c、旧 DSP 库及所有失败版本保留在 evidence/backup 与 frozen-v*-failed；修复版本在 frozen-v5-mapfix。

## 限制

此时验证的是 K80 Pro / v79 的原始失败路径；其他 DSP 架构未进行设备验证。带临时阶段 trace 的成功样本只作为回归证据，不用于性能结论。完整提前/晚启动正确性与正式性能对比继续在同任务中执行。

后续完成：去掉临时分支 trace 后，五组 pilot 和 24 个正式进程全部通过（128/1024 上下文，3072 个计时 decode token，包含反复 decode → prefill 切换）。源码/二进制冻结为任务目录的 frozen.json。early/late 的输出文件经 SHA256 核对完全相同；性能结果与精度限制见该任务 results.md。修复前后 v4/v5 冻结文件对比确认唯一变化为 libggml-htp-v79.so，详见 mapping-regression.json。
