# V13：真实权重回归及 CPU/GPU 内核优化，性能仍未全部达标

## 当前状态

用户要求继续到全部完成。分支 feat/shared-weight-layout；原生备份仍是 backup/shared-weight-layout-baseline。前两个检查点 8b0596d1a（原型）、3e12ca6d0（拒绝路径及设备计时）。用户 TODO.md 未动。中间层仍是单份 HTP 32x32 tiled Q4_0，MoE 未接入。

## 本阶段修改

- GPU decode去掉显式严格fma，改为同原生的乘加表达式及-cl-mad-enable。V10只增加unroll/mad编译选项，仍约1ms；V11再替换显式fma后约44–65us。不能把前面29–36倍的差距全部归因为物理布局。
- GPU decode改为每workitem两输出行（同原生粒度），四K分区；V12单独验证通过。prefill仍保留V11的weight image读取：V12试global buffer/full-wave，N128反而退化，已恢复。
- CPU从固定循环分配32行tile改为使用原生threadpool chunk动态分配，降低异构核心负载不均。没有增加CPU线程或辅助转换线程，线程数4/mask f0不变。
- CPU曾试int16按plane乘加，V10更慢，已恢复原有SDOT实现，仅保留动态调度。
- probe新增--model（runner）读取指定GGUF的blk.0.ffn_gate_exps.weight和blk.0.ffn_down_exps.weight的expert7，三端均用同一canonical字节，激活仍为合成F32，无MoE routing。GPU/HTP使用设备计时，CPU用持久线程池wall。
- shared-weight.h及其依赖shared-dma.h加入公共安装头文件清单。

## 结果

真实权重：CPU/GPU用v13-model-*，HTP用v11-model-htp（HTP内核一直未改）。表内native best/shared微秒；GPU包含activation transpose所有kernel event之和，HTP是DSP op时间。

| K×M | N | CPU | GPU | HTP |
|---|---:|---:|---:|---:|
|2048×1408|1|21.43 / 34.53|36.86 / 61.95|50 / 50|
|2048×1408|32|198.12 / 234.90|383.87 / 469.12|86 / 85.5|
|2048×1408|128|795.10 / 816.43|613.63 / 752.00|110 / 110|
|1408×2048|1|19.48 / 29.97|33.02 / 43.01|50 / 49|
|1408×2048|32|197.34 / 235.42|263.94 / 314.37|86 / 86|
|1408×2048|128|815.00 / 868.10|448.13 / 595.97|108.5 / 108|

CPU prefill一例落入5%，另一N128约6.5%，N32约19%；decode约54–61%。GPU约19–68%。**总体目标未通过，任务保持未完成。** 实际模型数值校验及alias变异验证三端通过；v13边界回归CPU/GPU各20组，HTP沿用v9相同实现。

实际增量构建：`cmake --build build-android-opencl-hexagon --target shared-weight-probe ggml-cpu ggml-opencl -j 8`（多次按改动目标构建，未全量重建）。实际运行：

```sh
python3 tests/run-shared-weight.py --deploy --shared --profile --tag shared-v10-profile-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --shared --tag shared-v10-cpu --backend CPU --reps 48
python3 tests/run-shared-weight.py --deploy --shared --profile --tag shared-v11-profile-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --shared --model --profile --tag shared-v11-model-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --shared --model --tag shared-v11-model-cpu --backend CPU --reps 32
python3 tests/run-shared-weight.py --shared --model --profile --tag shared-v11-model-htp --backend HTP0 --reps 32
python3 tests/run-shared-weight.py --deploy --shared --profile --tag shared-v12-profile-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --deploy --shared --tag shared-v13-cpu --backend CPU --reps 48
python3 tests/run-shared-weight.py --deploy --shared --model --tag shared-v13-model-cpu --backend CPU --reps 48
python3 tests/run-shared-weight.py --shared --model --profile --tag shared-v13-model-gpu --backend GPUOpenCL --reps 48
python3 tests/run-shared-weight.py --shared --edges --tag shared-v13-edges-cpu --backend CPU --reps 4
python3 tests/run-shared-weight.py --shared --edges --tag shared-v13-edges-gpu --backend GPUOpenCL --reps 4
```

日志/JSON/部署hash全部在evidence/。只有自有远端目录保留继续实验所需二进制；没有复制或修改模型。

## 后续候选（尚未实现）

CPU原生4x8可复用原生traits，完全消除CPU运行时重排；它是block_q4_0x4=72bytes，d[4]在前、64quant按8-byte行交错并xor0x88，排列[group-of-4-rows][K/32]。不可误用144byte block_q4_0x8。

两位只读探子已核查HTP最小入口，不必再全库搜索：

- HVX普通N1是`MATVEC_2D_REPACKED_IMPL`，matmul-ops.c:346–440/715，调用tiled_vec_dot_q4_0_32x1，hvx-mm-kernels-tiled.h:539。原DMA按576→640的每Ktile展开。可新增flag后让32行chunk连续DMA（总字节仍为576*Kblocks），新dot按8个4-row组读取72byte blocks。VTCM原预留640*Kblocks比raw576*Kblocks大，或可复用预算，须实核尾部HVX overread。
- 新HVX dot可直接以128byte向量展开四行K32到int8，vrmpy得到每行8个int32部分和，按每行scale及activation scale转换累积，结束再按行reduce。现有activation tiled的8向量每个重复四个K值，可以用7个vshuff合成K32重复四行，保留现有quantizer。此为推断，尚未实现/验证。
- HMX普通`hmx_mm_2d_f32`在matmul-ops.c:2665，dequant state分派:2383；raw DMA可按group_bytes=4*weight_stride、height=n_cols/4，源chunk地址nc*weight_stride依旧正确。worker row_stride保持canonical每行大小，用 `(ct*32+row)*row_stride+kt*72` 定位4-row block。原640byte tile预算可能足够容纳更紧的raw staging而不改solver；须实核边界。
- HMX输出FP16 tile半字索引应核对 `(K/2)*64 + row*2 + K%2`，已有canonical_scatter_four_rows在hmx-mm-kernels-tiled.h:160。新4-row解码可shuffle int32 pair后scatter两向量到tile前/后1024bytes，避免完整DRAM副本。
- host新增独立CPU4x8 flag，不能冒用CANONICAL_Q4_0；关键ggml-hexagon.cpp:288、2616、5852、7866、8296、8326。普通支持/参数生成应限定2D、K%128、M%32、N1走新HVX，其余仅已支持HMX；不要误放行MoE或view。
- CPU固定layout traits可在repack.cpp模板定义后暴露internal getter，返回`tensor_traits<block_q4_0,8,4,GGML_TYPE_Q8_0>`，仅支持NEON+i8mm；共享buffer的extra traits直接返回它即可复用最佳原生调度。

不要为通过5%而选更慢native基准、藏转换、增加完整后端权重缓存。也不要从V1的失败直接断言所有共享布局不可能。
