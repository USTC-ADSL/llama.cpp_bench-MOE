# 冻结候选并开始留出比较

192 个 FFN 候选（4 长度 × 4 层 × 12 配置，每候选 2 warmup + 5 timed）全部通过逐 token 4% 误差检查。fit-report.json 保留所有数值检查、预测误差和排序。只用层 0/12 拟合，在层 6/18 的 8 组验证中全部选中已测候选最快者；不是全空间最优证明。读取路径与特征映射经过独立只读核验。

technical 完整模型校准 4 进程 × 6 timed 成功，单份专家 weight payload/allocation 均为 7,474,348,032 B，跨轮 logits 逐位一致。CPU 在 1K 有明显低估，observed/predicted=1.439，而 512 为 0.970。因此添加 optional context_scales：每长度三后端整体倍率，按 token 数线性插值、端点夹取，旧模型省略时全为 1。最终层 N=1 placement 仍全 CPU；没有在线学习、权重移动或 ID 查表。

runtime-joint.json 是原 FFN fit；context.py 从全部 technical 校准样本产生 runtime-joint-context.json。新参数只用训练输入，未读取 wetland/foundry 完整模型结果。校正后的 scheduler 主机和 Android 定向测试均通过：随机 availability、完整 predict 一致性、插值/端点、错误 knot 拒绝、校正导致 shared 选择变化。编译命令同 handoff-01，未重新编译库。额外改动 tests/moe-placement-model.h；kernel 未变。

正式验证启动命令：

```sh
python3 task/handoff/handoff-2026-10-22-joint-scheduler/profile.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/fit.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/run.py calibration
python3 task/handoff/handoff-2026-10-22-joint-scheduler/context.py
python3 task/handoff/handoff-2026-10-22-joint-scheduler/run.py validation --model task/handoff/handoff-2026-10-22-joint-scheduler/runtime-joint-context.json
```

validation-frozen.json 固定 binary、库、模型、prompt 和 plan 哈希。evidence/freeze 保存校准前版本；evidence/freeze-validation 保存正式验证版本。旧原始 binary/libs/model 仍在 evidence/backup。

验证共 48 进程、288 timed requests，tokens128/256/512/1024；每文本顺序正反对称，间隔30秒。driver 日志 evidence/validation-driver.log，进度 validation-runs.json，远端 /data/local/tmp/llama-joint-scheduler-20261022。首次启动因本地相对 --model 路径解析失败，在发出任何设备测试前退出；已 resolve 修复，原日志 validation-prelaunch-path-error.log 保留，不存在丢弃设备样本。

下一步：等待完成，不更改冻结输入或 binary；运行 analyze.py --partial/最终 analyze.py；报告所有收益及失败/内存/数值边界。检查 1K 分页、shared owner、decision、arena 内存增量。2K 不运行，CPU 仍四线程。完成后归档并清理任务远端，Git 保存结果，TODO.md 不动。
