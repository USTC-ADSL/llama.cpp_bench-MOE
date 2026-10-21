# 1K 基线完成，继续分配与 router 复测

代码检查点 `469104b85`，分支 `feat/shared-weight-layout`。用户 TODO.md 未动。用户明确允许 4K 异常时先完成 1K，并确认无额外性能调节/并行压力测试。设备 fd8657d6 当前在线。

## 已完成

- 协议见 protocol.md。新增 bounded prefill、GPU router override/trace、独立 router probe、长度/计划 runner 和汇总脚本；只改 tests，后端生产 kernel 未改。
- 4K 整批运行首次出现 kernel_panic；4×1024 连续 KV 分块能完成，但 wetland ABBA major faults 与耗时严重漂移，作为诊断保留，不发布稳定吞吐结论。
- 第一轮 1K 第四个 native 进程权重加载时设备重启（boot reason reboot）。前三个结果与失败日志移至 evidence/interrupted-first-1k/。之后每进程空闲 30 秒，重新完整运行两输入 ABBA。
- 正式 1K 各条件 2 进程×6 requests，2 warmups/进程，共 48 timed requests。wetland native 1252.272747 ms → shared 1182.671992 ms，吞吐 +5.885%；foundry 1261.601224 → 1165.903854 ms，吞吐 +8.208%。summary-1024.json 可重生成。
- 固定对照 token IDs / owner / shared placement / binary hashes 一致；同布局跨进程 logits bitwise 一致；跨布局 top1 相同、top10 overlap=10，NRMSE wetland 0.01573375 / foundry 0.01086934。
- Router GPU 模型配置经 v3 smoke 成功：24 层 probs/top-k buffers 为 OpenCL，11.25 MiB router 权重迁移，其他主层权重/KV 保留 HTP。早期 v1/v2 诊断失败的根因是显式 tensor_split 绕过 OpenCL 内存发现，loader 查询 alignment 时 OpenCL 尚未初始化；已在测试 harness 提前初始化 GPU。no_host=true 防止 CPU override 被重选为 GPU host buffer。
- 独立 router 初测显示 N=1024 GPU event ~0.761–0.764 ms，HTP DSP op ~1.878–1.879 ms。F32 M=60 不满足 HMX 32 对齐，HTP 当前走 HVX；softmax 对 60 也被支持检查拒绝。初版 top4 probe 错把非连续 view 当连续读，已修正为按 nb[1] 提取，必须用 final probe 结果，不引用 profile 初稿的 top4 mismatch。

## 当前工具/进程

主线程 exec session `60876` 正按顺序执行：

1. tests/analyze-moe-shared-weight-4k.py --tokens 1024（已成功）；
2. tests/run-moe-router-probe.py（部署修正 stride 的 probe，layer 0/12，每长度64次；输出 router-summary.json）；
3. 三个 wetland 候选 c8-g30 / c8-g45 / c4-g30，分别 3 reps、每进程前 30 秒空闲，tag `wetland-<candidate>-explore`。

不要与它并发执行设备负载。设备目录 `/data/local/tmp/llama-shared-weight-4k-20261021`。

## 接下来

1. 等 session 60876 完成，检查 final router top4/reference 与 candidate 全请求 median / 各端阶段。
2. 在 wetland 选择候选后，在 wetland/foundry 对 shared 原分配 vs 候选做独立完整请求确认，不用探索样本充当确认。
3. 在原 shared c8-g15 下对 HTP/default router vs GPU router 做两输入交替重复；trace 仅诊断，不放入正式计时。要补一次 default router trace 验证 CPU fallback。
4. 汇总阶段/总耗时/内存/数值差异；保存 results.md、summary 与 archive manifest。证据包括全部失败日志与两次 pstore，不能删除。
5. 最后清理本任务设备目录，模型和公共 OpenCL cache 不动。

## 构建与复现

fullmodel 与 probe 均用 protocol.md 中 NDK clang++ 增量编译。只有已有 deprecated ggml_mul_mat_set_prec warning。最终待归档二进制哈希见 binaries.json，注意 HTP 库来自 build/ggml/src/ggml-hexagon/libggml-htp-v79.so。host/source与device命令由每个 *.run.json 记录。

实际正式基线命令：`python3 tests/run-moe-shared-weight-4k.py baseline --tokens 1024 --chunk 1024`。

`python3 tests/analyze-moe-shared-weight-4k.py --tokens 1024` 已通过；该脚本比较 ownership 字段，不错误要求 native/shared 不同内存记账字段完全相等。输入末尾多余空行在测试期间清理，对首1024/4096 token 无影响；所有正式 token IDs 逐条件核验一致，实际文件 hash 以各 run.json 为准。
