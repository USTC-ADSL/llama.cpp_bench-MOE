# Decode → prefill 映射故障定位

接续 handoff-01，v2/v3/v4 均未通过，不得使用其失败样本报告性能。所有旧 frozen 源码/二进制已保留。

- v3 CPU 控制成功；late trace 明确完成 32 步 quality decode 后在 DECODE_BEGIN 的 prefix rebuild 崩溃。
- v3 regular 原有 async 路径同样失败，排除仅由新增常驻 worker 导致。
- v4 在 moe-shared-weight.h 临时加入 BRANCH_BEGIN/END 和 host gather 索引检查。prefix rebuild 第 8 层 HTP 分支失败，host 索引检查通过。
- logcat 的 DSP TLBMISS PC 0xFDECD644、load 0xFDE3A000，llvm-addr2line 对 0x93644 解出 get_rows_thread_st_int32_t，LR 0x93248 是 op_get_rows。Bad VA 0x117A4000。
- 发现 ggml/src/ggml-hexagon/htp/main.c 的 prep_op_bufs 使用 uint32_t m_reuse / 1<<i，而 htp-ctx.h HTP_MAX_MMAPS=64。槽位 >=32 的移位越界，清理映射时可能丢失仍被当前 batch 使用的映射。已改 uint64_t/UINT64_C(1)，增加容量静态断言；待同设备验证，不能提前宣称已修复。
- 正在增量构建 htp-v79，日志 evidence/build-htp-v79-map-mask.log。随后只替换 DSP lib，复跑 v4 late trace，可单独验证底层修复效果。

## 修复后

仅替换 v79 DSP 库，v5 late-mapfix 通过 quality32 + warmup2 + timed1 的完整循环。修复独立提交 74087cdc6；Bug 简报 docs/bugs/2026-10-24-htp-mapping-reuse-bitmap.md。v5 冻结证据在 frozen-v5-mapfix.json / evidence/frozen-v5-mapfix。

已移除 moe-shared-weight.h 临时 trace（恢复原文件），补齐 engine 正常退出时 early_shared_branches.clear()。重新编译执行器，日志 build-executor-v6-clean.log；最新 frozen.json 为干净 v6。run.py/analyze.py 的 pilot tag 已改 pilot-v6-*。pilot 五组在运行，driver 日志 evidence/pilot-v6-driver.log。通过后再执行 analyzer 和 formal，禁止用 v5 trace 数据报告性能。

pilot-v6 五组已全部通过，`python3 .../analyze.py pilot` 成功，结果 pilot-validation.json。96 个路由节点（24×projection/softmax/argsort/topk）均 CPU；五组 prefill 逐位相同，early/late/regular decode 相同，cpu/off decode 相同。NPU shared 相对 CPU shared logits 非逐位相同：平均 NRMSE 3.35%，最大 9.03%，33 个位置 top1 全相同。这里只证明启动时机不改变异构输出，不证明 NPU/CPU 数值等价或生成质量无损。

formal 已启动：24 进程，wetland/foundry ×128/1024，上下文后续 N=1 decode；early/late/cpu 各两进程，4×32 计时 token/进程，2 warmup，20 秒进程间冷却。日志 evidence/formal-driver.log，进度 runs.json。测试进程期间不要改冻结源码/二进制；不测 2K。

代码和方案由主代理处理。只读探子提出 graph UID / arena 风险，但未找到实际 UID 冲突或 arena 重分配，不能把推测当根因。无需 reset session、关闭 gather 或跳过 prefix rebuild。
