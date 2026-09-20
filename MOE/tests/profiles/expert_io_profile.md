# Expert load 辅助测试

本目录保留 Expert Slot 所需的权重 source 与回填验证，不承担 GPU/HTP Slot runtime。

- `../../runtime/`：公共 Reader 按 gate/up/down 独立 range 执行 buffered `pread`，直接写调用者 destination。
- `expert-pack-gen`：从 GGUF 生成 expert 索引/pack。
- `expert-io-profile`：读取真实 expert tensor range，回填预分配 CPU slot，并可验证 `MUL_MAT_ID`。
- `expert-cpu-consumer-tests`：验证 CPU backend 直接消费 canonical Expert Slot。
- `../test_cpu_pipeline.py`：pack、离散 range 读取、CRC 失败和 CPU compute 集成测试。

当前的 `buffered-pread + separate` 只是最小可靠基线，代码和 JSONL 均标记为
`read_policy_status=pending-design`。后续的异步预取、跨 Expert 合并、page-cache 控制和
direct I/O 需要基于真实 token trace 与 UFS profile 重新设计，不作为当前 Expert Slot
正确性 demo 的可选开关。

```sh
MOE/build_host.sh

cmake -S MOE -B MOE/build/host -DMOE_WITH_GGML=ON \
  -DMOE_TEST_MODEL="$PWD/MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf"
cmake --build MOE/build/host --parallel
ctest --test-dir MOE/build/host --output-on-failure
```

Android CPU 构建入口是 `MOE/build_android_cpu.sh`，默认输出到 `MOE/build/android-cpu/bin/`。
当前 schema 4 使用 CpuPrivateBuffer 与统一 Slot lease，compute buffer 单独分配，不追加 CPU 空 synchronize。
