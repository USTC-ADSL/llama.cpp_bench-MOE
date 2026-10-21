# 阶段收尾：正式收益21–23%，25%目标尚未达成

分支 `feat/shared-weight-layout`。已验证实现为 `5cab4e301`，依赖导入资源修复 `2898132f6`。最终文档提交见后续Git。用户 `TODO.md` 保持未跟踪、未修改。

## 当前事实

- 全部正式V1数据完整，8进程、48请求。wetland NPU1294.742→shared1055.629ms，吞吐+22.651%；foundry1292.547→1065.375ms，+21.323%。结果、正确性及阶段时间见 `results.md`/`summary.json`，不能将约28%的探索结果表述成确认收益。
- 当前workspace代码、已构建fullmodel及库已恢复V1。`frozen.json`内22个文件SHA256全部重新核验通过。未采用任何CPU affinity改动，也未采用后续OpenCL view elision补丁。
- 保留4个expert CPU线程和outer4线程、默认affinity；单份CPU4x8 payload 7,474,348,032B，arena483.031MiB，正式重复图缓存70个且稳定。1–2小核长期目标、4K、任意新输入流图缓存上界未完成。
- `tests/test-moe-route-scheduler.cpp`设备测试通过；Python语法检查、统计核验及git diff --check通过。完整基线与共享的二进制/库hash、token IDs和模型文件stat一致；同方案跨进程logits逐位一致。

## 后续失败与回退

全程pin7探索约1554ms，callback内pin7约1215ms；均比默认方案差。二进制及patch保存在evidence，源码已移除这些尝试。没有将慢NPU配置引入任何正式对照。

GPU shared transpose存在不用input image却仍创建view的路径。实验补丁`experiments/view-elision.patch`通过N1/17/32×三后端共9项算子回归，但独立technical全模型新校准在权重加载阶段失败，未得出端到端收益。新旧.so及源补丁保留，当前库已恢复。它没有改kernel算术，也没有在失败时进入计算图，不能据此确认它导致或解决了重启。

设备发生了一次确认的kernel panic：原boot ID `f8c81ca4-027e-4197-bfc3-12433b0d31fc`，恢复后`f3665dc6-f63f-4aec-8777-8fd2ef58f997`。默认串号fd8657d6自动重新上线，`sys.boot_completed=1`；boot reason前缀为kernel_panic，后缀及ramoops存在乱码/控制字节。`evidence/panic-console.txt`是二进制原始ramoops，勿按纯文本覆盖；`panic-printable-fragments.json`只提取可读片段，不是修复后的日志，也不足以可靠断言硬件损坏或OOM根因。

失败记录为`technical-context-v2.{log,run.json}`，exit255、最后完成resident layer14，没有计时样本。`device-disconnect-recovery.json`记录重启后只读状态。runner在读取after_thermal时发现设备消失，失败manifest已经保留exit字段。不要将它计入吞吐统计，也不要立即重复大模型加载而忽略先前4K/1K已有的重启风险。

## 归档与清理

归档：`/mnt/sda1/pzw/moe/experiment-artifacts/cost-placement-1k-20261021.tar.gz`。

- 大小110,594,463B，共198成员，逐项SHA256验证通过。
- SHA256：`0a14f56f300f757fc45b6230e33dfc662322e61f094af819e5f5c3fa68fad4de`。
- 包含全部原始profile/完整模型数据、logits、失败日志、ramoops、旧/新二进制库、输入、模型参数、源文件与实验patch。模型GGUF未重复打包。
- `artifact-manifest.json`记录成员hash。归档在最终清理/交接状态更新前生成；最终状态以Git中的tasklist、cleanup-record和本交接为准。
- 已删除仅属于本任务的设备目录 `/data/local/tmp/llama-cost-placement-20261021`，约128.9MiB。确认无fullmodel-oracle运行，模型stat未变，pstore和公共CL cache保留。见`cleanup-record.json`。

## 恢复与下一步

统计可以直接重跑：`python3 task/handoff/handoff-2026-10-21-cost-placement/analyze.py`。分析检查原始run manifest与冻结hash，不要求未来live源码永久不变。profile和正式runner分别为`profile.py`与`confirm.py`；runner会核验当前workspace全部冻结hash，并拒绝覆盖同名evidence。

正式候选：`runtime-fast-cpu150.json`、`inputs/{wetland,foundry}-fast-cpu150.json.gz`、NEON合并、4线程、默认router、单份CPU4x8、shared activations、direct callback IO、arena lifetime reuse、overlap gather。`frozen.json`已保存完整配置，设备实际命令见每个`.run.json`。

下一次继续应先确认默认设备上线并调查加载时的物理内存/不可回收DMA/系统panic条件，不要先重跑4K或改系统参数来美化性能。稳定后，才重新用独立technical输入校准更接近实测的branch完成时间；V1的HTP gather仍比GPU多约26ms，请勿只匹配compute。若进一步修改GPU为batched shared MUL_MAT_ID，需单独设计和逐expert/token回归；现有shared2D kernel不能直接当native SoA MMID使用。新候选必须再次冻结并独立对照，保留本轮未达25%的正式数据。
