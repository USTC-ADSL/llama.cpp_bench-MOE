# 128-token历史收益核对与补充对照

用户在长度扫描期间指出历史128收益约40%。保留原始冻结扫描协议与所有结果；不重拟合代价模型，不修改binary/kernel。主扫描完成后再执行独立补充对照。

## 已核实

- 不能说用户把64记成128。09-26跨输入表确实只有64约40%、Technical128约28%，但09-24原始输入shared activation cohort在128时有A/B吞吐+29.803%/+41.379%，合并24样本中位数+36.236%。原始JSON已由analyze128.py核对，并连同run manifests拷贝到evidence/historical128。
- 原始文档：`task/handoff/handoff-2026-09-24-offline-shared-activation/results.md`。NPU A/B 519.9398435/573.988567 ms，异构400.560599/405.992734 ms；NPU端点漂移10.4%，不能把其中最佳41.38%当稳定跨输入收益。
- 历史原始prompt为`tests/expert-routing-prompt.txt`，plan为09-23-fullmodel-oracle/evidence/search128.json.gz。大多数层CPU处理24个低负载routed专家、其余NPU、shared在GPU；这是对原始输入离线搜索后的固定owner，实时重算route。旧执行按owner预放各端布局。
- 当前wetland/foundry是不同输入；当前route动态调度、单份CPU4x8，前23层shared仅允许NPU，GPU承担约36%的routed pairs。这不是单变量历史性能回归对照。
- 当前128在前23层×12请求中，GPU是wetland的275/276层、foundry的276/276层最长分支（按backend host gather+compute）。预测CPU/NPU/GPU累计约292–305 ms，实测约185–190/167–168/322–324 ms。模型高估CPU/NPU，相对低估GPU，预测均衡不等于实际均衡。
- 亲读`tests/moe-shared-weight.h:171–209`确认分支阶段计时和并发执行；`tests/fullmodel-oracle.cpp:153–169`确认decision计入gather，不能重复相加。约2.25–2.29 ms host partition字段包含约1.22 ms决策；combine约4.45–4.76 ms；分支gather包含在expert_wall内。
- 当前模型来源technical-profile-complete.json的tokens=1024。runtime-fast-cpu150.json的shared系数是constant，shared×nt/tiles系数为0；CPU系数乘1.5。不要直接把这些代码事实当成已分离的各项因果贡献，但它们与128实测预测偏差一致。

## 补充实验

`python3 task/handoff/handoff-2026-10-22-length-sweep/attribute128.py`只能在主扫描40进程全部成功且不再运行后启动。使用当前同一remote与binary，原始历史prompt，128 tokens、2 warmup+6timed、30s间隔，8进程镜像顺序：

1. native NPU
2. current dynamic + CPU4x8
3. historical fixed owner + CPU4x8
4. historical fixed owner + backend layouts
5. historical fixed owner + backend layouts
6. historical fixed owner + CPU4x8
7. current dynamic + CPU4x8
8. native NPU

前三个custom特性共同包含shared activations/direct callback/arena/lifetime/fused reduce/overlap；CPU4x8路径额外用NEON+HTP indexed gather。fixed-native是整个旧布局执行路径对照，不是只改一种kernel packing的纯消融。每进程token IDs必须等于历史search128的128 IDs；冻存新增输入和plan，逐次检查版本/hash/boot ID，失败保留证据。

证据tag为original128-*，单独attribution-runs.json，不混入主扫描40进程。`analyze128.py`生成attribution-summary.json，检查输入、版本、同mode重复输出、跨NPU logits、全部样本，并复算历史数字和当前预测偏差。主扫描未完成前该脚本可只生成历史与当前128 profile；补充实验开始但未完成时它会拒绝产生完整attribution。

## 恢复与收尾

代码未修改；本阶段新增attribute128.py/analyze128.py，render.py/archive.py待最终使用。主扫描统一exec session曾为58422；最新进度以runs.json为准。无需重建。全部完成后依次analyze.py、analyze128.py、render.py，写入解释报告并更新tasklist，归档全部证据（包含补充8进程），验证后清理仅本任务remote，保存cleanup-record并提交。用户TODO.md不动。
