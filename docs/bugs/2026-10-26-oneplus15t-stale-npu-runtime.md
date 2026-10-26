# OnePlus 15T 旧 NPU 运行库崩溃与部署更新

## 现象与触发条件

设备 `3B162U00GZY00000`（OnePlus 15T，SM8850，固件 `PLZ110_17.0.0.107(CN01)`）取得 Root 后，以 `su -c` 运行 Qwen1.5-MoE-A2.7B-Chat Q4_0 的 Hexagon 三会话 benchmark。

原先 ADB shell 的 FastRPC `0x72` 权限失败已消失，三个 HTP 会话能够建立并加载权重。但 pp128 预热触发 DSP `TLBMISS RW`，主机收到 `dspqueue_read failed: 0x2e`，进程退出码 134。尚未完成预热，因此旧运行没有有效吞吐样本。

原库搜索顺序 `.:/vendor/lib64` 还会在当前系统中触发 `liblog.so` 的 C++ 符号加载失败；使用 `.:/system/lib64:/vendor/lib64` 后启动正常。SELinux 始终为 Enforcing。

## 根因定位与证据边界

已经确认的是**旧部署版本相关的执行失败**，不是选错 Hexagon 架构：

- 不设置 `GGML_HEXAGON_ARCH` 时，驱动自动报告 v81。
- 设备原 DSP 库 ELF flags 为 `0x81`。SHA256 `c28f2cdce3b2e272e64df320d9fb7f95ee0d77018b100218c69b33e703af3d03` 与旧本地构建完全匹配。
- 精确匹配库的相对 PC `0x1eab0` 对应 `tiled_vec_dot_q4_0_32x2`，LR `0x2be74` 对应 `hvx_mm_nx_2d_repacked_q4_0`。故障地址 `0x80`，栈指针 `0x29B800`。
- 旧版改用 flat matmul 后仍崩溃，位置转为 `unary_task_f32_scale`；限制为 4 个 HVX 线程也未恢复运行。
- 同一设备、模型、Root 域和 v81 架构，换成当前源码 `aa5f2d811` 配套构建的主机库与 DSP 库后，完整吞吐、HVX 对照和数值正确性测试均通过。

**尚未证明的部分：**没有通过历史提交二分定位到单一源码缺陷。空 DMA 队列返回空指针是候选解释，未被运行时直接验证；主机/DSP 文件日期不同也不能单独证明 ABI 不一致。不能把上述猜测当作已确定的代码根因，也不能排除旧实现具有 v81 特有缺陷。

## 修改与部署

无需新增算子补丁。本次增量构建并部署当前仓库已有实现，修复设备仍在使用历史运行库的状态：

```sh
cmake --build build-android-opencl-hexagon --target htp-v81 llama-bench test-backend-ops -j 8
```

构建耗时 170.5 秒，退出码 0。DSP 库取自 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v81.so`，不可混用 `bin/` 中的历史同名副本。

先部署到固定测试目录内的诊断子目录，验证后迁移：

- 当前运行库：`/data/local/tmp/llama-moe-bench/`。
- 原运行库归档：`/data/local/tmp/archive/2026-10-26-npu-runtime-upgrade/original/`。
- 模型：`/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。
- 新运行目录内 11 个文件均与主机 SHA256 一致。没有修改模型、设备安全策略或时钟配置。

版本选项不可混用：旧 `MM_SELECT=3` 表示 HMX/tiled/flat；当前实现的 HMX/HVX 选择值为 `2`，HVX 对照使用 `1`。

## 验证

正式测试：三个 HTP 会话按 `1/1/1` 分配，`-ngl 99`，CPU 调度线程 4，batch/ubatch 512，flash attention on，mmap，lazy mode off。默认预热后，各项重复 3 次。

| 项目 | 平均吞吐（tokens/s） | 样本标准差 | 三次样本 |
| --- | ---: | ---: | --- |
| pp128 | 385.331 | 1.934 | 383.154 / 385.993 / 386.847 |
| pp512 | 830.945 | 43.456 | 819.152 / 794.602 / 879.080 |
| tg128 | 22.483 | 0.329 | 22.7239 / 22.1078 / 22.6160 |

另通过以下检查：

- 当前版本 HVX-only pp128，退出码 0。
- `test-backend-ops test -b HTP0:0 -o MUL_MAT,MUL_MAT_ID`，按记录中的 Q4_0 参数筛选，17/17 用例与 CPU 参考一致；包含普通矩阵乘法和 MoE 专家路由矩阵乘法。
- 迁移回固定运行路径后的 pp128，退出码 0。
- 最终文件 SHA256 校验、无遗留 benchmark 进程。

吞吐属于合成 pp/tg 测试，不包含模型加载、tokenization、采样等开销。系统 DVFS、散热和后台活动未控制；这些数值是当时运行条件下的结果。

## 复现及证据

主机证据在 `task/handoff/handoff-2026-10-26-oneplus15t-npu-crash/evidence/`。各目录的 `run.json` 保存了实际完整命令、退出码和耗时。`summary.json` 保存通过解析和样本数检查的结果；`deployment.json` 保存部署 SHA256。

在当前固定路径重新运行正式配置：

```sh
python3 task/handoff/handoff-2026-10-26-oneplus15t-npu-crash/run-probe.py repeat-throughput --mm 2 --prompt 128,512 --gen 128 --repetitions 3
```

每次使用新的输出标签，避免覆盖证据。旧运行库和旧 DSP 二进制仍保留供需要时进行历史缺陷定位。
