# 收尾：默认维持 async，局部线程开销改善未转化为稳定 E2E 收益

用户当前任务为检查并减少 gather / 输出合并等实现开销，不再追求固定收益。代码阶段提交 **bc07dd807**，分支 **feat/shared-weight-layout**。本文件与最终结果由后续文档提交保存；用户的未跟踪 TODO.md 从未修改。

## 最终结果与当前实现

- 8 进程、48 次完整 1K 计时请求全部成功，期间设备 fd8657d6 boot ID 未变。两个输入各两次进程重复，2 warmups + 6 timed，每进程间隔 30s，wetland ABBA / foundry BAAB。
- 同一输入的 logits 跨进程/方案逐位一致，route counts 和专家分配一致。单份物理权重 7,474,348,032 B；三端 activation arena 合计 483.031 MiB；重复请求缓存图数量 70 且不增长。
- V1 async → pool：wetland 1038.517→1095.669ms，吞吐 **-5.216%**；foundry 1024.948→1024.288ms，吞吐 **+0.064%**，视为持平。以上为两进程中位数均值，全部样本保留，不能声称稳定加速。
- 单层最终版本启动延迟约 88–130→12–20us；但完整模型 pool 的主机合并及 FFN 外时间反而增加，抵消专家 wall 改善。CPU 调度/频率/cache 的具体原因尚未通过核驻留和硬件 counter 定位，不能把推测当作根因。
- 默认 **async** 未变。仅显式 `--shared-dispatch pool` / `MOE_SHARED_DISPATCH=pool` 才启用两条跨层复用 worker；移除参数或设置 async 回退。kernel、物理权重格式、gather/合并算法未修改。当前构建 fullmodel-oracle 为含实验开关的最终源码版本；旧 V1 executable 有独立备份。
- 连续行 gather 合并和 NEON16 合并未稳定减少耗时，已从执行代码撤出，仅存 experiments/ 与原始证据。
- 仍为 expert CPU 4 线程、outer 4 线程、默认 affinity；未达到早先只用 1–2 小核的长期目标。pool 有线程栈/TLS 开销，不是总内存零开销。

## 主要发现

原路径已经使用输出指针表 + fused weighted reduction 直接写 dst，没有大块拼接副本。每个 1K 请求的 gather 目标写入量约 920 MiB，合并逻辑读写约 1104 MiB（非实测 DRAM 流量）。任意 callback host pointer 不能直接作为 HTP DMA tensor 导入。因此若要结构性减小开销，需要设计设备索引读取 / gather 与矩阵乘融合，而不是改 memcpy 或 tensor->data。本轮没有实现这类 kernel 改动。

## 验证与复现

构建命令、单层设备测试命令和代码范围在 handoff-01.md。八组合 probe 288 次、最终 async/pool probe 72 次计时均逐位相同，参考 FFN 每 token NRMSE <4%。设备 test-moe-token-workers 通过，覆盖 3-way、worker 和 inline 异常后的 drain/reuse。已检查最终代码 diff、Python 编译与 git diff --check。

- 完整 runner：`python3 task/handoff/handoff-2026-10-21-gather-merge/confirm.py`；会拒绝覆盖冻结候选/同名证据，重新实验需使用新 cohort 目录，不要直接删除当前记录再跑。
- 原始统计重放：`python3 task/handoff/handoff-2026-10-21-gather-merge/analyze.py`。读取 evidence/，核验 hash、输入、分配、内存与 logits 后生成 summary.json；不需要设备。
- frozen.json、confirmation-runs.json、summary.json 与 results.md 描述所有结果。每进程环境/二进制 hash/命令在 evidence/*.run.json。

## 归档和清理

归档：`/mnt/sda1/pzw/moe/experiment-artifacts/gather-merge-1k-20261021.tar.gz`。

- 大小 **55,737,655 B**，**79** 个成员，经 SHA256 逐项验证。
- SHA256：`f79db37accfeba0bd5489788657d9aff9994ee4059989587be08bd5de6916881`。
- 包含原始捕获输入、单层/完整日志、结果、旧/新 executable、库、源码快照和未采用的实验；不重复打包 GGUF 模型。
- archive-record.json、artifact-manifest.json 记录归档；归档是在设备清理前生成，最终状态以本交接和 Git tasklist 为准。
- 已确认无本任务进程，删除自有设备目录 `/data/local/tmp/llama-gather-merge-20261021`。模型 stat 核验前后一致，公共 cache 和其他目录未清理。详细清理记录见 cleanup-record.json。

当前任务全部完成，无正在运行的测试。后续不应将 pool 当成默认加速方案，也不应重复已经没有稳定收益的连续行 memcpy / NEON16 实验来追求指定百分比。
