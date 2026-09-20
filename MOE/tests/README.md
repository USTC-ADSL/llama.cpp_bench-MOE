# Tests and profiles

Host 测试覆盖状态转换、stale/跨 arena handle、写失败取消、lease 移动与析构、resident 重读、共享 completion、分散区间、CRC/range/Repack。带 `MOE_WITH_GGML=ON` 时增加 CPU consumer；设备不重复 host 的完整状态组合。

| 程序 | 边界 |
| --- | --- |
| shared-buffer-device-tests | 原 shared demo 的 checksum、DMA 可见性、异步生命周期和负例 |
| expert-slot-device-tests | 原 Phase B 完整设备计算与状态验证 |
| expert-baseline-profile | setup、buffer/view、单 Expert 基线计时 |
| buffer-lifecycle-profile | raw Shared/Private allocation、import、读写和 teardown |
| expert-buffer-profile | Expert prepare/switch/replace 测量 |
| expert-io-profile | buffered-pread + separate、GGUF source、可选 CPU compute 校验 |
| routed-expert-profile | 独立 HTP profiler 构建 |
| buffer-capacity-probe | 容量与驻留探针，默认不纳入 smoke |
| htp-map-latency-probe | 专门测量 HTP map/unmap |

Profile 链接同源 Runtime 的 `-profile` 变体。`support/expert_source_profile.cpp` 负责 `/proc` 归因、冷缓存控制和读操作统计，实际字节读取只有 Runtime Reader 一份实现。`support/device_probe_runtime` 保存 checksum kernel、测试 DSP 队列和测量支撑；其物理 allocation、raw import/map 委托公共 Memory Runtime。

```sh
cmake -S MOE -B MOE/build/host-core -G Ninja
cmake --build MOE/build/host-core --parallel
ctest --test-dir MOE/build/host-core --output-on-failure
MOE/tests/probes/build_probe.sh
```

不把 profile 的额外同步用于普通 compute 路径。专门测同步开销的 case 应明确保留测量语义。FastRPC session 改为进程级复用，allocation teardown 不再调用 deinit/dlclose；不能把新版 setup/free 数据与旧口径合并。raw Buffer 创建包含 fd export，wrapper 的 export 阶段只暴露已导出的 fd，分析时应结合该边界解释。

原始结果、失败现场和实际命令按 [结果规范](../results/README.md) 保存。容量探针的独立操作说明见 [probes/README.md](probes/README.md)。
