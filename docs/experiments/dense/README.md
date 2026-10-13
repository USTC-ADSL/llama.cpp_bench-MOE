# Dense 模型实验归档

归档日期：2026-10-13。收录 2026-10-01 至 2026-10-03 的 11 组 Qwen2.5-3B Attention / FFN 实验，保留报告、历史交接记录、任务清单、原始结果、诊断日志和图表。

## 实验索引

| 日期 | 实验与报告 | 内容 |
|---|---|---|
| 10-01 | [dense-af](2026-10-01-dense-af/report.md) | 完整 Attention / FFN，CPU、GPU、NPU，prefill 与 decode microbench |
| 10-01 | [dense-af-power](2026-10-01-dense-af-power/report.md) | 单层 A/F 三后端功耗与能耗，18 个有效窗口 |
| 10-01 | [af-dvfs](2026-10-01-af-dvfs/report.md) | 独立 A/F 工作点扫描；另见 [六点 Pareto](2026-10-01-af-dvfs/report-pareto-six-points.md) |
| 10-01 | [af-multilayer](2026-10-01-af-multilayer/report.md) | 36 层权重常驻的 A/F 分块实验 |
| 10-01 | [af-handoff](2026-10-01-af-handoff/report.md) | 分块之间实际激活传递及 mixed-chain 功率轨迹 |
| 10-01 | [af-real-decode](2026-10-01-af-real-decode/report.md) | 完整模型真实自回归 decode 与全 NPU 对照 |
| 10-01 | [af-real-energy](2026-10-01-af-real-energy/report.md) | 真实生成的整机能量与吞吐 |
| 10-01 | [af-decode-diagnostic](2026-10-01-af-decode-diagnostic/report.md) | 混合 decode 的轻量诊断 |
| 10-02 | [attn-core](2026-10-02-attn-core/report.md) | Attention 投影在 NPU、注意力核心在 CPU 的真实生成 |
| 10-03 | [layer-energy](2026-10-03-layer-energy/report.md) | 完整 Attention+FFN 层能耗，5 个层实测及其余层插值 |
| 10-03 | [same-backend-split](2026-10-03-same-backend-split/report.md) | 同后端单图、逐层同步、逐层同步加拷贝对照 |

## 阅读与复现

先读各实验的 `report.md`；历史构建、部署和验证过程见同目录 `handoff-*.md`。实验工具仍位于仓库根目录下的 `tools/dense-af-bench/`。

独立块、36 层分块链与完整模型生成的计时边界不同，不能混合排名。基础 microbench 中 prefill 的 NPU 优势、decode 的 CPU 优势，不能直接外推真实生成；真实 decode 实验中混合方案反而慢于全 NPU。设备、CPU 工作点、KV 长度、电池倍率和测量窗口以各报告为准。层能耗的未测层属于插值。

本次采用原件归档：迁移文件内容逐字节保留，包括历史命令、日志内的输出路径和设备 serial。重用历史命令时，在仓库根目录执行，并将其中路径按下列规则转换：

```text
task/handoff/handoff-<日期>-<实验名>/<相对路径>
→ docs/experiments/dense/<日期>-<实验名>/<相对路径>
```

该规则仅适用于本索引列出的 11 组实验。分析既有数据使用新归档路径；重新采集时使用新的输出目录，避免覆盖历史结果，并重新确认 ADB serial。历史交接文档中的旧绝对路径和 Markdown 链接也按此规则定位，不作为当前可直接运行的命令承诺。

## 完整性与版本状态

`manifest.json` 记录每个文件的原路径、新路径、大小、SHA-256 及归档前是否受 Git 跟踪。迁移包含 569 个文件、13,564,485 字节；原有 551 个受跟踪文件继续纳入 Git，15 个被忽略的日志和 3 个未跟踪的 mixed-chain-power-trace 图表/数据文件保持原有版本状态。因此完整本地归档比仅 Git clone 多 18 个文件，清单明确标记了这一差异。

- [x] 确定 11 组实验范围与归档路径
- [x] 完整迁移并核验文件哈希、索引链接和 Git 重命名
- [x] 保存独立 Git 归档提交（本索引随归档一同提交）

本次仅整理既有实验资料，不构建、不部署、不重新测量设备；历史结论仍受各报告所列限制约束。
