# 阶段 1：基准工具及设备冒烟验证

## 目标与边界

Qwen2.5-3B-Instruct Q4_0 在 fd8657d6 的完整 Attention / FFN prefill microbench。用户确认：prompt 64/256，CPU 全部核心，三后端独立测量，不实现 A/F 混合执行；输入/权重就绪后预热重复同步计时，搬运单列。

- Attention：RMSNorm + Q/K/V（含 bias）+ NEOX RoPE + 当前 prompt K/V 转 F16 + 因果 GQA + O projection + residual。
- FFN：RMSNorm + gate/up + SwiGLU + down + residual。
- 块独立输入为确定性合成 F32 激活，无历史 KV；不包含 tokenizer、embedding、LM head、图构建/分配、权重加载，不是实际请求端到端性能。
- 非 FA 与 FA 均支持；将分别测量，避免实现选择造成误判。
- 无 scheduler，逐节点支持检查通过才计算，禁止把 CPU fallback 计为纯设备块。
- CPU 权重使用后端额外优化 buffer（实际 CPU_REPACK），taskset ff、8 线程，runner 验证 CPU online=0-7。

## 改动

- `tools/CMakeLists.txt`：注册新工具。
- `tools/dense-af-bench/CMakeLists.txt`、`dense-af-bench.cpp`：GGUF 元数据/权重读取、图构建、逐节点支持检查、预热同步计时、独立 upload/download、CPU 输出参考与残差外 relative RMS 校验。
- `tools/dense-af-bench/run-android.py`：顺序设备测量、精确命令/日志/样本/CPU频率与电池温度保存，拒绝覆盖已有日志。
- 代码核验确认 math/shape/GQA/causal mask 与 `src/models/qwen2.cpp`、`build_attn_mha`、`build_ffn` 一致。工具限定标准 Qwen2.5-3B、Q4_0/F32、无 RoPE scaling；拒绝混合量化权重 buffer 的不明确报告。

## 设备与模型

- Serial `fd8657d6`，设备在线；SoC SM8750，GPU Adreno 830，HTP v79（6 HVX、1 HMX），CPU 0-7 在线。
- `/data/local/tmp/models/Qwen2.5-3B-AoT/ggml/weights.gguf`：1742627744 bytes，36 层、embedding 2048、FFN 11008、Q heads 16 / KV heads 2 / head dim 128，RoPE base 1000000，eps 1e-6。
- 434 个 tensor：253 个 Q4_0、181 个 F32；直接读取原有文件，不复制模型。
- 固定部署目录 `/data/local/tmp/dense-af-bench/`，参考输出在 `reference/`，任务结束需删除这些临时参考。

## 构建和部署

已执行：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench
cmake --build build-android-opencl-hexagon -j 8 --target htp-v79
adb -s fd8657d6 shell 'mkdir -p /data/local/tmp/dense-af-bench/reference /data/local/tmp/dense-af-bench/cl-cache'
adb -s fd8657d6 push build-android-opencl-hexagon/bin/dense-af-bench build-android-opencl-hexagon/bin/libggml-base.so build-android-opencl-hexagon/bin/libggml-cpu.so build-android-opencl-hexagon/bin/libggml-opencl.so build-android-opencl-hexagon/bin/libggml-hexagon.so build-android-opencl-hexagon/bin/libggml.so /data/local/tmp/dense-af-bench/
adb -s fd8657d6 push build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so /data/local/tmp/dense-af-bench/
```

重要部署陷阱：最初误用 `bin/libggml-htp-v79.so` 旧副本，导致首轮 NPU 卡在 DSP signal wait；已终止自己的进程，并替换为上面真实 htp-v79 产物，立即恢复。不要再次使用旧 bin 副本，不需要修改后端源码或重启设备。

- 旧副本 SHA256 `582071ad44014ae88cc865f61bda1ad5eadbdb84b3d3dc136ef0d9b2b4154989`
- 正确产物 SHA256 `f0f8907e8f8a0169d65ae38bae9b3053b5e489d846dd4e5161a1edc9fc38c523`

## 已执行设备验证

```sh
python3 tools/dense-af-bench/run-android.py --output task/handoff/handoff-2026-10-01-dense-af/results --round check --layers 0,17,35 --warmup 2 --reps 3 --reference --cooldown 1
python3 tools/dense-af-bench/run-android.py --output task/handoff/handoff-2026-10-01-dense-af/results --round check --layers 0,17,35 --stage attention --flash 1 --warmup 2 --reps 3 --reference --cooldown 1
git diff --check
```

全部成功（12 个进程、各 3 层）。非 FA + FFN 最大 relative RMS：GPU 1.923%，HTP 1.073%；FA Attention 最大：GPU 1.347%，HTP 1.044%。参考为 CPU_REPACK 输出、分母排除 residual；量化后端激活量化/累加方式不同，不是逐位一致。5% 是工具筛查阈值，最终报告实际最大误差。

`results/smoke-HTP0-p64-current-dsp.log` 保留一次 profiling 证据：RMSNorm、RoPE、softmax、copy、residual 在 HTP 执行，matmul 标明 hmx-tiled。正式计时必须 PROFILE=0。

## 状态与下一步

- 本阶段将独立提交；最初用户未跟踪文件 `TODO.md` 保留，不纳入提交。
- 尚未正式测全 36 层，不能把上述冒烟数字当最终结论。
- 下一步：runner 默认 `--layers all --reps 15 --warmup 5`，先 `--round 1 --reference`，再反序设备 `--devices HTP0 GPUOpenCL CPU --round 2`。非 FA 两块、FA Attention 各两轮；结果聚合采用每层样本中位数，再跨层汇总，报告轮次变化。
- 正式数据保存在 results 的日志与 JSON；结束后生成 CSV/报告、核查 git diff、清理设备参考输出、保存第二检查点。
