# Memory / Expert Slot Runtime

仓库内的 C++17 库和 MoE 实验。依赖方向为 **Memory Runtime → Expert Slot Runtime → Experiments**；不承诺外部分发或 ABI 稳定，也未接入 llama 的 router 或 scheduler。

| 目录 | 职责 |
| --- | --- |
| [runtime](runtime/README.md) | Buffer、ExpertPack、range reader、Loader、Repack、Arena、设备 view |
| [experiments](experiments/README.md) | demo、benchmark、模型 fixture、graph/input/reference、调度 |
| [tests](tests/README.md) | host/device 正确性测试、底层探针、专用 Profile 与分析支撑 |
| models | 本地 GGUF，供 pack generator 和完整模型 benchmark 使用 |
| results | [长期结果与复现记录](results/README.md)，不改写历史数据 |

旧 `test_mem` 目录及对应构建缓存已删除；当前入口统一为上述三个目录。

## 构建

Host core 不需要 Android/Hexagon SDK：

```sh
cmake -S MOE -B MOE/build/host-core -G Ninja
cmake --build MOE/build/host-core --parallel
ctest --test-dir MOE/build/host-core --output-on-failure
```

带 GGUF generator、CPU consumer 和 I/O Profile：

```sh
MOE/build_host.sh
ctest --test-dir MOE/build/host --output-on-failure
```

可用 `-DMOE_TEST_MODEL=/absolute/path/model.gguf` 加入真实模型 CPU I/O 集成测试。独立库也可通过 `cmake -S MOE/runtime -B MOE/build/runtime` 构建。

Android raw OpenCL/FastRPC 程序与 V79 测试 DSP：

```sh
MOE/tests/probes/build_probe.sh
```

Android ggml 程序通过仓库顶层配置 `GGML_OPENCL=ON`、`GGML_HEXAGON=ON`、`LLAMA_BUILD_MOE_SHARED_EXPERT_DEMO=ON` 构建；沿用当前 NDK/SDK 配置，例如：

```sh
cmake --build build-android-pd-final --target expert-slot-demo resident-expert-bench routed-expert-bench
cmake --build build-android-pd-final --target expert-baseline-profile expert-pipeline-bench routed-expert-profile
```

可执行文件位于构建目录的 `bin/`；raw 测试 DSP 位于 `tests/probes/libbuffer-capacity-htp-v79.so`。CPU Android 辅助入口为 `MOE/build_android_cpu.sh`。

## 程序迁移

| 原入口 | 当前入口 |
| --- | --- |
| shared-expert-dmabuf-demo | shared-buffer-demo；原 checksum/失败路径验证为 shared-buffer-device-tests |
| shared-expert-dmabuf-phase-b 普通流程 | expert-slot-demo；原完整验证为 expert-slot-device-tests |
| Phase B baseline | expert-baseline-profile |
| Phase B pipeline | expert-pipeline-bench |
| shared-expert-resident-compute-bench | resident-expert-bench |
| shared-expert-routed-bench | routed-expert-bench / routed-expert-profile |
| shared-buffer-lifecycle-bench | buffer-lifecycle-profile / expert-buffer-profile |
| expert-ufs-probe | expert-io-profile |

`expert-pack-gen`、`buffer-capacity-probe`、`htp-map-latency-probe` 保留名称。程序文件采用 `snake_case`，target 使用 `kebab-case`。Runner 和 analyzer 按用途位于 experiments、tests/profiles 或 tests/probes。

## 运行边界

GPU/HTP parity、Top-2 路由、四层模型、serial/parallel、worker queue 和运行次数由实验定义。Runtime 不知道这些策略。普通 Runtime 没有时钟采样、JSON、checksum kernel、DSP 测试服务或 `/proc` 归因。

专用 Profile 链接同源 `-profile` Runtime；一个进程只允许一种变体。pipeline benchmark 因需要内部阶段测量而链接 Profile 变体。生产 reader 与 Profile 共用实际读取实现，固定 `buffered-pread + separate`。

设备操作必须遵守根 AGENTS.md：使用正常 `adb devices -l` 检查 `fd8657d6`，操作前缀为 `adb -s fd8657d6`。不默认运行容量压力测试。

完整 MoE 模型仍使用主仓库 `llama-bench`/`llama-completion`。`-cmoe`/`-ncmoe` 改变专家放置，不改变路由或算子支持；详见 [MoE 模块说明](../docs/agent/modules/moe.md)。`pic.py` 仅绘制内嵌历史数据。
