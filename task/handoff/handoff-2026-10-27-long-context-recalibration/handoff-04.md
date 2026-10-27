# 最终状态：2K 收益已验证，4K 明确未通过

本轮已完成，不存在仍运行的测试/代理任务。最终结论与完整限制见 report.md；机器可读统计见 summary.json。2K wetland 新模型相对旧异构/NPU 加速 1.0413×/1.1300×，foundry 为 1.0269×/1.1379×。每条件 2 个进程、12 计时样本，共 72 次计时。同一 boot 27fd07ca-ca1f-4134-8c0a-1e1a8ce46c6d 内完成全部对照；12 个结果通过完整性、输入 token、同版 binary、route-model hash、数值有限性、进程内及进程间 logits 一致性检查，无并发/监控缺口。top-1 与 NPU 一致；最终 logits NRMSE 新模型 wetland 4.2326%、foundry 1.6096%，不声称全面质量等价。

校准改善分支预测，但并行专家 wall 没变快；结果合并时间缩短是可观测收益来源之一，详见 report.md 的分解。foundry 旧/新进程耗时范围重叠，额外校准收益较小，不声称统计显著。2K 全部推理完成，未观察到新增 kernel/GPU OOM；后台 LMKD、swap 和 major faults 仍存在，新 foundry 的 major faults 尤其明显。pooled 443.094 MiB 保留容量不变，当前调度仍没有内存准入。

4K full-model 运行发生设备重启，未取得结果，未重试。failure-4096.json 固定失败证据与边界，不能把之前污染 1536 的 OOM 消息当成这次 4K 根因。基础单层模型覆盖到 4096；完整模型 context scale 只由 1024/2048 生成。1536/3072 无完整模型收益对照。后续若继续，应先处理全模型保留/启动内存和准入，再考虑 3K/4K 的完整验证。

原始 evidence/ 共 470 个文件、793885783 bytes，保留在本机、被 Git ignore；每文件 SHA256/大小见 evidence-manifest.json。设备上 63 个文件全部核对 SHA256，已有本地证据/构建核验，6 个末轮辅助文件额外拉回 evidence/final-device/。随后仅删除 /data/local/tmp/llama-long-calibration-20261027，已验证目录不存在、无推理进程、boot 和模型 stat 不变；见 device-archive.json / device-cleanup.json。后续复测需要新建独立目录/新 tag 重新部署，不能原样重跑本次 tag。

构建/采样/拟合/对照实际命令见 handoff-01.md、handoff-02.md、handoff-03.md 及 report.md。最终运行 `python3 .../summarize.py > .../evidence/analysis-final.log 2>&1` 成功；`python3 -m py_compile .../*.py` 与 `git diff --check` 通过；额外断言完成 12 进程/72 样本、相同 eval boot/model/inputs、old/new route hashes、两次进程输出一致，全部通过。最终修改范围仅本任务目录，未动用户 TODO.md、paper/、results/。

此前检查点 bc239b171、74de00a48、17b00f27f；本 handoff、report、汇总、归档清理记录作为最终独立结果提交。没有未提交的后端/内核修改。未解决事项仅上述明确边界，不将本次测试描述为修复了 4K 重启或长期内存稳定性。
