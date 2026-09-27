# Decode 实验路径的权重驻留与异步生命周期

## 现象与根因

1. 首次启用 CPU GGUF 权重视图与 activation lifetime arena 时，报 `missing activation layout reservation`。arena 准备逻辑用 bank 是否拥有 `buf` 判断是否存在 routed 权重，遗漏了有效的映射视图。
2. 尝试为 CPU dense custom 路径全局开启 extra buffer 后，完整模型出现严重换页，decode 达到约412/469 ms/token。加载器对“override 到 CPU”仍会从 extra buffers 中选择优化格式，因此为 callback 不使用的原生 Expert 额外分配了约7.6 GiB CPU_REPACK；日志同时出现约7.4 GiB CPU_Mapped，进程 VmSwap 约7.6GB。该失败不是异构计算性能。
3. 新异步 shared 路径正常执行会在 finish join，但若在 launch 与 finish 之间发生异常，默认成员析构顺序可能先释放 callback 状态，再销毁仍引用该状态的 worker。静态检查发现，未在设备触发异常。

## 修改

- CPU identity Expert bank 可借用只读 GGUF 页；映射生命周期覆盖 graph/bank。arena 同时识别 owned buffer 和 mapped weights；compact 非 identity 映射显式拒绝。
- 保持全局 extra buffers 关闭，仅对 attention Q/K/V/output 与最终 output 权重明确指定 CPU_REPACK。Expert source 保持原始映射，避免无人使用的整份格式副本。v4 实际 CPU_REPACK 降为459.43MiB（dense 权重），不再是7587.43MiB。
- full_engine 在异常析构和正常退出时都先 drain/destroy shared worker，再释放 callback 状态和权重。

## 验证与边界

Android fd8657d6 使用现有 NDK 构建目录增量编译。`tests/run-moe-decode-study.py` 的 v2/v4 screening 已通过每进程四轮、每轮16步、17个相同前缀输出的 bitwise repeat 检查；sharedCPU/GPU/HTP 和反向分支均保留结果。v1 的失败日志和 v3 的换页实验全部保留，不能混入正常常驻吞吐比较。

最终清理顺序修订的构建、smoke 与原 prefill 路径回归记录见 `task/handoff/handoff-2026-09-27-decode-parallel/handoff-final.md`（完成时生成）。没有放宽同步或重复 logits 检查。不同后端/权重路径之间仍有数值差异；bitwise repeat 不等于模型质量等价。完整 shared 加速器副本约446MiB，本实验不能声称满足200MB额外内存上限。
