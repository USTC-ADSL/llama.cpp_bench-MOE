# OpenCL Q4_0 MoE 专用重排数值错误

## 现象与触发条件

K80 Pro（fd8657d6，Adreno 830，OpenCL 0800.71 / E031.47.18.49）执行 Q4_0 MoE 专用路径时，第 0 层真实 Qwen1.5-MoE 权重的 gate/up、down 在 n1/n128 下 RRMSE 约 0.30～0.31，超过原阈值 0.02。通用 OpenCL 路径通过。干净官方 upstream 64e9bceb2 重新构建同样复现，排除了本项目预算补丁是必要触发条件。

## 根本原因

`ggml/src/ggml-opencl/kernels/cvt.cl` 的 `kernel_convert_block_q4_0_trans4_ns` 通过 `uchar *` 写入私有 `ushort8 post_block`，再用 `as_uint4` 消费向量值。这一写法在目标设备编译器下生成错误量化字节。独立读回测试中原内核 196608 个 q 字节有 5739 个不符，scale 全部正确；保留原输入向量加载、仅用标量 uint 构造输出后，q/scale 均完全一致，真实专用 GEMV/GEMM 数值也恢复正常。这是对具体代码/编译器交互的实证定位，不将尚未分析的 OpenCL 语言规则问题一概认定为驱动缺陷。

## 修改范围

只修改上述函数的输出打包部分，删除 12 行、新增 13 行。用四个初始化为零的 uint 累积对应字节，再构造 uint4；每次移位仅为 0/8/16/24，保留原始 nibble 顺序和四个输出 plane 的布局。

输入读取、scale、矩阵/expert 索引、缓冲区、路由、内核选择、GEMV/GEMM 计算、其他量化格式均未修改。没有切换通用 fallback，没有调整精度容差。测试辅助程序仅新增可选 verify-only、穷举字节对及尺寸参数，原始诊断模式保留。

## 构建和设备回归

在有效构建目录中增量执行：

```sh
cmake --build build-android-opencl-hexagon --target ggml-opencl -j8
```

测试工具使用 NDK r27d 的 `aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++` 编译；`repack_probe.cpp` 链接 `-lOpenCL`；`expert_sanity.cpp` 使用项目 ggml/include、vendor/nlohmann 和 build-android-opencl-hexagon/bin，链接 `-lggml -lggml-base -lggml-cpu`。

| 回归 | 结果 |
|---|---|
| K80 和 OnePlus 两台设备直接运行修改后的重排 kernel | 全部字节和 scale 精确一致 |
| K2048/M64/E3、K1408/M128/E2 的 65536 种输入字节对穷举 | 每台设备分别检查 196608、180224 个 q 字节，均零差异 |
| K32/M32/E1，小矩阵及尾部 work item guard | 每台设备 512 个 q 字节，零差异 |
| K80 专用路径 off/observe/enforce，n1/n128 gate/up、down | 三种模式共 12 项全部 PASS；各模式结果一致 |
| OnePlus 3B661501LA000000，原始测试图和默认通用路径 | 四项 PASS，报告出的数值统计与修改前一致 |
| K80 全层模型 pp128/tg32 | 25/25 层卸载至 OpenCL，退出码 0；161.35 / 28.52 token/s |

K80 专用路径四项 RRMSE：0.002580873、0.002740556、0.002827038、0.002815164；归一化最大误差最大 0.040395172（阈值 0.10），输出全部有限。observe/enforce 使用 64 GiB 的非约束 extra cap，仅验证计账模式不改变修复结果，不是实际 RAM 限额验收。

测试的关键命令（在隔离设备目录内执行）：

```sh
REPACK_VERIFY_ONLY=1 REPACK_EXHAUSTIVE=1 ./repack-probe cvt.cl 2048 64 3
REPACK_VERIFY_ONLY=1 REPACK_EXHAUSTIVE=1 ./repack-probe cvt.cl 1408 128 2
REPACK_VERIFY_ONLY=1 ./repack-probe cvt.cl 32 32 1
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_ACCEL_BUDGET_MODE=off E0_DIAGNOSTIC_PATH=reorder E0_PADDED_IDS=1 ./e0-expert-sanity-regression /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_ACCEL_BUDGET_MODE=observe GGML_ACCEL_BUDGET_EXTRA_BYTES=68719476736 GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra E0_DIAGNOSTIC_PATH=reorder E0_PADDED_IDS=1 ./e0-expert-sanity-regression /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_ACCEL_BUDGET_MODE=enforce GGML_ACCEL_BUDGET_EXTRA_BYTES=68719476736 GGML_ACCEL_BUDGET_OPENCL_COVERAGE=extra E0_DIAGNOSTIC_PATH=reorder E0_PADDED_IDS=1 ./e0-expert-sanity-regression /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_ACCEL_BUDGET_MODE=off ./e0-expert-sanity-regression /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf GPUOpenCL
LD_LIBRARY_PATH=. GGML_OPENCL_KERNEL_CACHE_DIR=off GGML_ACCEL_BUDGET_MODE=off ./llama-bench -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf -dev GPUOpenCL -ngl 99 -p 128 -n 32 -b 128 -ub 128 -t 4 -r 1 -v
```

无路由环境变量的命令用于 OnePlus 原始路径；K80 专用测试使用已有诊断工具补齐独立图的路由前置条件。项目运行时路由逻辑未修改。

## 限制和保留问题

独立 MUL_MAT_ID 图的路由初始化/IDs 行跨度问题与本数值错误独立，本提交不修复它。原 E0B 运行器未经适配仍可能在 n128 报 -38；不能把旧整体验收状态直接改为全部 PASS。完整模型 benchmark 验证运行成功，不是端到端 logits/token 正确率对照；没有宣称覆盖其他 GPU 或所有形状。

证据及构建日志存放在 `task/handoff/handoff-2026-09-08-upstream-opencl/`，本次文件前缀 `fix-` 或 `build-project-fix`，原始日志有无损 `.log.gz` 归档；host/device hashes 核验部署一致性。设备隔离目录在验证后清理，原模型及 mobilebench 部署保留。仅提交本修复、对应回归辅助和文档，不包含用户已有 AGENTS.md/Exp1 修改。
