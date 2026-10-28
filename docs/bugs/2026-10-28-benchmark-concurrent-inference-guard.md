# 设备基准测试在并发冲突后继续运行

## 现象与触发条件

1K 回归的首个 NPU 进程运行时，另一套目录 /data/local/tmp/moe-hetero-lengths/ 启动了 1025-token 异构进程。五次监控均记录两进程并发，随后设备重启，当前基准没有有效计时结果。原始证据保留在 task/handoff/handoff-2026-10-27-calibration-regression/evidence/，进程路径与排除说明见同目录 interference.json。

## 根因与边界

测试驱动只在长耗时诊断快照之前检查设备空闲；检查到实际推理启动之间存在时间窗口。原监控发现并发后仅记录污染、等推理完成才拒绝结果，因此不能及时解除两套大模型共同占用内存的情况。设备重启的精确内核机制未据此证明，不能直接把该监控问题称为重启根因。

## 修改

仅修改本任务 run.py：在 before snapshot 完成后再检查空闲；运行时检测多个推理命令后，核对 PID 的当前 cmdline，只向本任务独立目录下的 fullmodel-oracle 发送 TERM。其他目录的进程不被终止。失败证据保留，不自动重试。新增 cohort 标签，用户确认继续后用 resumed 标签，避免覆盖原失败记录。

前任务已冻结 driver、计算源码/二进制、模型、计时条件不变；这是本任务的保护包装，不是所有测试程序间的分布式互斥锁。

## 验证

2026-10-28 执行宿主 Python mock 核验：并发两进程时只选择本任务 PID；PID 的当前 cmdline 已变化时不发送 TERM；snapshot 返回后才再次执行 idle check。三项断言通过，验证未发送设备命令。执行 `python3 -m py_compile task/handoff/handoff-2026-10-27-calibration-regression/*.py` 与 `git diff --check` 通过。

恢复时重新核对全部宿主/设备 binary 与库 SHA256、成本模型/输入/源码 SHA256、GGUF stat，设备空闲。正常设备路径以 resumed 对照记录验证；不人为制造双大模型并发来复现危险场景。

## 限制

第二次 idle check 和实际 launch 仍存在很短的竞争窗口，监控采样也不是连续的；该保护只能缩短发现后的并发持续时间。消除冲突仍需各测试任务遵守设备独占安排，或所有调用方共用同一互斥机制。没有修改或控制外部测试。
