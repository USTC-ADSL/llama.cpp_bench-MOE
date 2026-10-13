# 阶段 2：正式测量前的精度、线程池和 decode 校正

## 用户新增要求

新增 `prompt=1`，代表 decode；用户明确指定历史 KV 长度为 256。仅测此历史长度，不增加 64 历史或空历史实验。每次计算包含 1 个新 token、256 历史；有效 key 数 257，按 `src/llama-kv-cache.cpp:1255` 的 256 对齐规则补齐至 512，mask 屏蔽其余槽。

## 必须保留的修正

1. **CPU 必须持久线程池。** 初版工具只设置线程数，`ggml_backend_cpu_graph_compute` 在 threadpool=nullptr 时每次建线程，把线程创建开销计入块耗时，尤其严重扭曲 decode。现在通过 CPU registry 获取 new/free/set_threadpool，复用 8 线程，和 llama-bench 一致。之前结果只作排查证据，不能引用为最终性能。
2. **OpenCL 正式使用 `GGML_OPENCL_Q4_0_DENSE_DP4A=1`。** 默认路径对第 1–4 层 FFN 的合成输入，输出相对 CPU F32 反量化参考误差 7.4%–12.2%。关闭 fusion、BIN_DP4A 都无改善；启用 dense DP4A 后降至 0.6%–1.4%，略快。对应源码默认非 DP4A Adreno GEMM 使用 half 累加，DP4A 是现有量化激活+整数点积路径。没有修改后端源码。正式结果须注明此配置，不能声称 GPU 默认数值路径已经通过验证。
3. **增加 F32 参考诊断模式。** `--f32-reference --write-reference --device CPU`，将原始 Q4_0 权重反量化为 F32，权重加载/转换不计时；只用于检查，不能拿它的速度作为 CPU Q4_0 性能。
4. **decode cache 实际驻留并更新新槽。** 用 SET_ROWS 写当前 K/V，位置和 slot 都为 256；每轮覆盖相同新槽，不累积历史。禁止用 concat 整个历史模拟 cache 写入。非 FA 包含 V 布局转换，FA 不需此转换。
5. **KV 对齐。** 无 padding 的 257-key softmax 被 HTP supports_op 明确拒绝（大于32时需32对齐）。已采用引擎256对齐到512，padding全零且mask=-inf；不能把未支持的257形状回退到CPU再计为NPU。

## 证据与已有验证

- 代码第一检查点：`ae9090efa`。本阶段改动在 `tools/dense-af-bench/` 与任务文档。
- 构建：`cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench`（增量，成功）。部署仅本目录二进制，DSP 使用阶段1记录的真实产物。
- GPU 默认误差证据：`results/initial-r1-GPUOpenCL-p64-fa0-both.*`。
- F32 诊断日志：`results/diagnostic-f32-CPU-p64-current.log`、`diagnostic-vs-f32-*-p64.log`。
- 开关隔离：`diagnostic-GPU-bin0-p64.log`、`diagnostic-GPU-dense1-p64.log`、`diagnostic-GPU-fusion0-p64.log`。
- 原始未持久线程池数据已移至 `results/obsolete-no-threadpool/`，绝不能混入正式统计。
- 正式 runner 默认 DENSE_DP4A=1，写出准确命令和温度/频率快照。
- decode 初次未对齐失败日志在 `results/decode/rcheck-*`，保留作边界证据。
- 对齐、线程池修正后测试：

```sh
python3 tools/dense-af-bench/run-android.py --output task/handoff/handoff-2026-10-01-dense-af/results/decode --round alignedcheck --prompts 1 --past 256 --layers 0,1,17,35 --warmup 3 --reps 5 --reference --cooldown 1
python3 tools/dense-af-bench/run-android.py --output task/handoff/handoff-2026-10-01-dense-af/results/decode --round alignedcheck --prompts 1 --past 256 --layers 0,1,17,35 --stage attention --flash 1 --warmup 3 --reps 5 --reference --cooldown 1
```

两组全部成功，GPU最大relative RMS 2.072%，HTP 0.883%；CPU_REPACK 8线程持久池。冒烟数据只用于检查，不引用作最终全层结果。

## 下一步

1. 用当前代码按 round1 CPU/GPU/HTP，round2 HTP/GPU/CPU，15次+5预热，测全36层：prefill64/256的非FA两块、FA Attention；decode1/past256 同样配置。
2. round1 加 `--reference`，round2 不需要重复写参考。prefill放results，decode放results/decode。
3. `summarize.py results` 与 `summarize.py results/decode --prompts 1`，每层合并两轮30样本求中位数，再36层平均；保留单轮变化和每层胜出计数。
4. 最终表区分普通/FA Attention，不能将36层独立块耗时加总称为真实端到端耗时。上传/下载是host边界API成本，不是直接跨设备搬运。
5. 清理本任务设备 `reference/`、`reference-f32/` 和空 `cl-cache/`。保留benchmark及所需库以便复现，保留host日志与CSV。
