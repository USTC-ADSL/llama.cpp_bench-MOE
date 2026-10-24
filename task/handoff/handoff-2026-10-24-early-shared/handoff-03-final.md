# 提前 shared 任务完成

## 目标与状态

用户要求 router 及相关算子留在 CPU，在 CPU router/routed 工作期间提前启动 NPU shared。已实现、完成设备校验和正式实验、归档、清理。此文取代 handoff-01/02 的在途状态。分支 feat/shared-weight-layout；原用户 TODO.md、paper/ 未修改。

## 结果

24 个正式进程、3072 个计时 token 全通过，覆盖 wetland/foundry ×128/1024 前缀，每步 decode N=1，不测 2K。每条件每策略两进程，4×32 计时 token/进程，2 warmup。router 投影/softmax/argsort/top-k 全部 CPU。CPU 四线程，无绑核或锁频。

提前相对晚启动吞吐：wetland/128 +2.11%，foundry/128 +0.58%，wetland/1024 −0.74%，foundry/1024 −0.05%。相对同 custom dense/attention 路径下 CPU 全专家参考，四组均降低约 20%–22%。实际 NPU 启动领先 router 就绪仅 14–19 µs/层，shared 图 1.07–1.18 ms，CPU routed 完成后仍等待 shared 0.45–0.62 ms。因此保留实验开关，默认关闭，不强制启用 NPU shared。

early/late 输出文件经 SHA256 核对逐字节相同，跨进程也相同；CPU/off pilot 输出一致；所有 prefill 一致，正式计时内无专家图新增。NPU shared 相对 CPU shared 有数值差异，foundry/128 有 1/33 位置 top-1 不同；不宣称精度等价或生成质量无损。所有缺页和慢样本保留，计时 token 合计 10 次 major faults。

详细方法、逐条件数据、分层时间、内存容量、精度限制均在 results.md 和 summary.json。CPU 参考是 CPU 执行 shared+4 routed，未在本轮复测整图原生 CPU。

## 代码与检查点

- `74087cdc6`：HTP 映射表 64 槽却使用 32 位复用位图，导致 decode→prefill 后解除仍被当前 batch 使用的高位映射、GET_ROWS TLBMISS。改 uint64_t/UINT64_C(1) 并静态断言。v4→v5 唯一二进制变化为 libggml-htp-v79.so，同失败输入通过。
- `e7cb57beb`：可选 pre-router input hook、完整 CPU router 放置、共享 worker 提交/等待、缓存 job 和生命周期，五组 pilot。核心 tests/moe-early-shared.h，入口 tests/run-fullmodel-oracle.py 的 --decode-shared-launch early|late|cpu|regular，默认 off。
- 最终数据/归档元信息由本阶段结果提交保存。没有修改算术 kernel，也没有关闭 gather、跳过 prefix rebuild 或重置 session 来绕过问题。

## 实际执行

库和执行器命令见 build-commands.json；另外运行 `cmake --build build-android-opencl-hexagon --target htp-v79 -j 8`。构建日志全部在 evidence。

`run.py pilot`、`analyze.py pilot`、`run.py formal`、`analyze.py formal` 均完成。实际完整 adb 命令、设备/model stat、库 hash、thermal/memory 快照、退出状态记录在每个 `<tag>.run.json`。正式冻结版本为 frozen.json；旧版/失败版保存 frozen-v1～v5 和 backup。不要使用现有 tag 重跑或覆盖证据。

## 归档与清理

归档 `/mnt/sda1/pzw/moe/experiment-artifacts/early-shared-20261024.tar.gz`，1,490,376,977 bytes；SHA256 `c908f04779e6a773fbfbc3c17b289510af7904b3d752c75f362be702efb9e51e`。

406 个原始证据文件按 manifest 逐一验证归档内长度和 SHA256；设备目录中所有文件均找到 hash 匹配的本地/冻结备份，包括失败实验未完整生成的 logits 文件。输入文本、teacher tokens 额外保存在 evidence/inputs。archive.json 保存最终核验状态；归档本身生成于此最终交接和清理元信息之前。

已确认无运行中的 fullmodel-oracle，再删除 `/data/local/tmp/llama-early-shared-20261024` 并确认不存在。模型文件未删除。设备始终在线且 boot ID 不变。主机 evidence 保留，Git 忽略大产物，旧二进制可从备份/归档恢复。

## 后续范围

本任务没有待执行测试。若继续优化，应先针对 N=1 NPU shared 完整图延迟及数值差异定位，或改变工作负载后重新标定；不要把小于进程间波动的早启动差异写成稳定 OSDI 加速结果。单/双小核、2K、其他 DSP 架构及自由生成质量均未在此任务验证。
