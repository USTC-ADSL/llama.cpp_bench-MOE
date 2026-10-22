# 联合调度 128–1K：最终交接

## 已完成与结论

实现检查点 a6dad6aee、0a0685477；分支 feat/shared-weight-layout。新调度支持 shared length/tile 项、三端 shared/routed 联合有界搜索、optional 按长度插值的后端 context 倍率。旧模型无新字段时行为兼容。没有改计算 kernel；8 个 host/DSP 库哈希与原备份一致。

独立 FFN profiling：192 候选，每候选2 warmup +5 timed，全通过逐token4%数值检查；原始 FFN fit 在8组留出层排序中选中已测候选最快者。technical 完整模型校准4进程24 timed，用于一次离线 context 倍率校正。随后冻结模型，完整验证48进程288 timed，全部退出0，boot ID f3665dc6-f63f-4aec-8777-8fd2ef58f997 不变，无降级chunk、修改频率、换设备或删除慢样本。

新调度相对原生NPU吞吐收益（wetland/foundry）：128为63.52%/54.59%，256为45.42%/46.12%，512为38.47%/38.53%，1024为23.33%/28.84%。相对同批旧调度：128为49.39%/35.59%，256为20.84%/20.64%，512为12.01%/6.91%，1024为4.00%/3.49%。完整表和边界见 results.md。指标为权重预放置后的单chunk完整prefill，不含冷启动加载。

128前23层 shared 都选GPU；平均每层CPU/NPU/GPU gather+compute 从8.422/7.282/15.241 ms变为8.583/8.795/8.707 ms。当前改动同时更新模型、候选和context校正，不能把总收益归因于单个开关。256–1K shared以NPU为主；分配仍根据当前route动态计算，模型参数离线冻结，无专家ID查表、无在线反馈。

## 内存与数值边界

新旧单份专家 unique weight payload/allocation 都为7,474,348,032 B。开放三端shared容量后，arena总容量在128/256/512/1024增加22.508/45.016/90.031/180.062 MiB；不是总内存零开销，也不是物理驻留测量。新调度每完整请求决策平均0.736/0.879/1.093/1.786 ms，已包含端到端时间。

新调度vsNPU logits NRMSE为1.484%–5.205%，8组top1相同且top10集合全部重合；各模式同条件跨进程logits逐位一致。不能代替生成质量验证。1K/wetland新调度单请求最多439 major faults，全部保留。CPU expert仍四线程，未固定小核，最终1–2小核目标尚未满足/验证。2K按用户要求未运行。

## 执行、产物、清理

构建与调度测试见 handoff-01/02。最终执行：

```sh
python3 task/handoff/handoff-2026-10-22-joint-scheduler/analyze.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/report.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/archive.py precleanup
python3 task/handoff/handoff-2026-10-22-joint-scheduler/cleanup.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/archive.py final
python3 task/handoff/handoff-2026-10-22-joint-scheduler/archive.py final-v2
git diff --check
```

report.py再次生成时仅补充了冷启动范围说明，未改变测量数据。summary.json、comparison.csv、branch128-summary.json、memory-summary.json、PNG/SVG图为最终展示；所有原始JSON、logits、日志、运行manifest和冻结文件在 evidence/。

precleanup归档逐文件hash已校验，随后仅清理 /data/local/tmp/llama-joint-scheduler-20261022，无遗留fullmodel-oracle进程，GGUF的路径/大小/mtime不变；未重新计算完整模型SHA。cleanup-record.json记录清理。Git封板检查发现CSV默认CRLF和Matplotlib SVG行尾空格，已修正生成器并重新生成展示文件，测量样本和数值不变。第一次final归档记录保存为preformat-archive-record.json。最终归档为 /mnt/sda1/pzw/moe/experiment-artifacts/joint-scheduler-128-1024-20261022-final-v2.tar.gz，完整SHA/成员核验见 archive-record.json。

原版本回退材料在 evidence/backup，当前候选为 runtime-joint-context.json + joint-plan.json.gz；校准与验证各自冻结的binary/源码见 evidence/freeze、evidence/freeze-validation。旧runtime-fast-cpu150与旧plan未覆盖。TODO.md未修改。结果提交只包含本任务目录，不包含用户TODO。

## 后续边界

本轮已封板，不再根据当前验证文本修改模型。后续优先缩小额外shared激活容量、单独校准1–2小核CPU配置；新输入、非训练长度和模型族需独立验证。2K继续暂缓。所有新实验应使用新tag/任务目录；不要直接重跑本目录run.py或profile.py覆盖当前证据。设备目录已清理，继续测试需重新显式部署。当前实现位于实验执行器与调度模型中，没有自动替换其他工具的默认运行配置。
