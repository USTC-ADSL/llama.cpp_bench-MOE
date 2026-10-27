# 激活总预算池：实现及定向验证

## 目标与当前状态

用户要求压缩 2K CPU/GPU/NPU 异构总预算，保留单 chunk、动态真实路由和三端并发。起始提交 93a136cda；默认设备 fd8657d6 在线，原有用户未跟踪 TODO.md、paper/、results/ 未改动。

已实现 `--shared-arena pooled`（环境 MOE_SHARED_ARENA=pooled）；private 保留用于基线及其他既有路径。原三端按各自 4N pairs 预留，改为一个 backing、全局 4N pairs 加一个 shared expert 的预算。每次按实际 owner/count 给三端分配不重叠、对齐的区域。当前 NPU-only shared 配置预计约 443 MiB，相比 963 MiB 节省约 520 MiB。

## 实现要点

- tests/shared-activation-buffer.h：bind_layout 支持非零 base_offset，检查剩余容量。
- tests/moe-shared-weight.h：按实际 routed pairs/shared/full_input 计算分支上界；key 加入 partition offset。已绑定 tensor 永不移动，offset 变化创建独立 cache graph，防止 OpenCL extra 缓存旧地址；HTP 提交使用图的稳定地址。图内实际 lifetime 大小必须小于分区预算，所有分区必须小于总池。执行仍 async/pool 并发并 join 后归约，层间串行复用。
- tests/fullmodel-oracle.cpp：按 4N 全局上界而非三个 4N 计算 pool，CPU/GPU/HTP 导入同一 backing；兼容 private/common shared scratch。新增物理分配去重计账，activation_arena_bytes/arena_capacity_bytes 为唯一 backing 列表，pooled 时只有一个元素。独立 early-shared decode dispatcher 尚未适配，明确拒绝与 pooled 同时启用。
- tests/run-fullmodel-oracle.py：新增 --shared-arena private/pooled。
- tests/moe-pooled-activation-probe.cpp：新设备回归，验证非零 offset、同形状不同 offset、owner/count 切换、缓存复用、并发 async/pool、private/common shared scratch。

## 已执行与结果

使用既有有效配套 host/DSP 库；只对改动涉及的 fullmodel-oracle、moe-pooled-activation-probe、test-shared-activation-layout 增量交叉编译。命令模板：

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/<name>.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/<name>
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py deploy
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py probe
python3 task/handoff/handoff-2026-10-27-pooled-activation/run.py fullmodel --tokens 2048 --prompt wetland
```

编译全部退出 0，仅既有 deprecated API warning。Python py_compile 和 git diff --check 通过。

设备原有 layout test 通过；新增 probe 360 用例全部与 private 输出逐位一致，无越界/并发区域重叠、缓存预热后无新增 job。覆盖 tokens 1/17/128/2048、全CPU/全HTP/全GPU及两种混合分配、shared owner CPU/GPU/HTP、3轮变化输入、private/common scratch。same shape different offsets 覆盖通过。

设备目录 /data/local/tmp/llama-pooled-activation-20261027；frozen.json 记录版本、hash、boot ID 和模型 stat。原 binary 备份在 evidence/baseline-fullmodel-oracle。原始日志与探针报告在 evidence/，不纳入 Git。

## 下一步

当前 wetland 2048 完整模型已启动，driver 日志 evidence/fullmodel-wetland-driver.log；run.py 会采样内存、采集 before/after dmesg/logcat 并保留失败证据，不重试。完成后核对6个计时样本、池容量、无分区重叠、logits与10-26基线逐位一致、route/owner一致、major faults及OOM/LMKD。再跑 foundry 2048；根据证据决定是否需独立1K回归/同期private对照，不机械扩展测试。

全模型尚未验收，不能宣称 OOM 已消失。完成后更新报告/任务清单，归档证据和实际命令，确认模型不变和无进程后清理本任务远端，检查 diff 并提交。不要覆盖10-26基线或重跑已有标签。新池目前为显式选项，不修改旧运行默认行为。
