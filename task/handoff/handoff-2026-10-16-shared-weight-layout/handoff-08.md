# 三端共享权重中间层：第一阶段验收完成

## 范围与结论

在Redmi K80 Pro `fd8657d6`上完成Q4_0普通2D MUL_MAT的CPU/OpenCL/HTP共享物理布局。MoE路由、预测及完整模型执行器是上层工作。本阶段验收覆盖指定Qwen模型的两个真实权重矩阵，K/M为2048/1408及1408/2048，N为1/32/128。三端共18组，相对于各端本轮最佳有效原生实现，算子耗时中位数退化均不超过5%，配对分块bootstrap的95%区间上界也均不超过5%。该结论不外推到其他芯片、量化类型或任意形状。

最终统计：evidence/acceptance-final.json；输入分别为shared-v41-cpu4x8-cpu-stable.json、shared-v41-cpu4x8-gpu-stable.json、shared-v41-cpu4x8-htp-final.json。

| 后端 | 每组重复数 | 最差共享/原生比值 | 最差95%区间上界 | 最大NRMSE |
|---|---:|---:|---:|---:|
| CPU | 512 | 1.03787 | 1.04845 | 0.003228 |
| OpenCL GPU | 512 | 1.02064 | 1.02374 | 0.003674 |
| HTP | 128 | 1.03509 | 1.04425 | 0.003229 |

CPU固定4线程、mask f0、持久线程池，选择CPU/CPU_REPACK中最快者为每组基准。GPU启用原生DP4A，耗时用全部kernel event之和，含activation量化。HTP用DSP op counter，含DMA/解码，不包含host/RPC以免稀释差异。八次warmup后逐轮交替执行顺序；bootstrap使用2000次配对重采样，连续8样本圆形分块，seed0。早期CPU/GPU96次数据部分区间较宽，已补测512次，不删除早期证据。

## 物理布局、内存和接口

- 采用`GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2`：每个四行/K32块72字节，8字节FP16 scales、64字节quants，8-byte interleave和XOR0x88。CPU直接复用原生4x8 trait；GPU直接访问导入DMA buffer；HTP绑定同一allocator原始存储并在既有VTCM中暂存/解码。
- 测试中的每个真实权重矩阵为1622016字节，唯一共享allocation也为1622016字节；三份backend tensor均别名同一host pointer。完整/部分canonical回读逐字节相同；绑定后测试性翻转scale符号、运行算子并恢复，证实执行不使用隐藏持久副本。
- 无额外完整持久权重或运行时CPU权重转换。HTP临时工作区仍受设备8MiB VTCM预算约束；raw chunk显式包含56-byte向量load overhang，pipeline重用两套chunk缓冲。CPU/GPU的activation工作区仍是算子临时空间。这不等于“零临时内存”。
- 独立公开接口在ggml/include/ggml-shared-weight.h。上层在加载期使用`ggml_shared_weight_create_with_layout(htp,k,m,canonical,size,GGML_SHARED_WEIGHT_Q4_0_CPU4X8_V2)`，随后对各端调用`ggml_shared_weight_bind`，把返回tensor作为MUL_MAT的权重输入并检查`ggml_backend_supports_op`。创建后可释放调用者canonical临时输入。计算完成后`ggml_shared_weight_free`同步各端并依次释放imports和owner。
- `ggml_shared_weight_create`为兼容早期实验仍选择V1；本阶段推荐V2必须显式选择。V2三端共同支持要求K为128倍数、M为32倍数；普通contiguous二维Q4_0 × F32。任意view、写入、DUP及其他不支持操作被拒绝，不能当普通canonical buffer直接修改。handle及其backends生命周期、外部bind/free序列化契约已在头文件说明；describe只是借用，不retain。

## 最终实现与回归

HTP优化：先转置4-bit压缩数据，减少shuffle；正确槽距的vlut32/vlut16直接解码；连续K分块避免重复fastdiv及八路地址乘法；HVX以qf32累加并在最终输出转回FP32；HMX按实际raw大小分配VTCM，以约三分块平衡首块解码和计算重叠，容量不足时在tile单位上二分缩小。删除未使用shared-cpu4x8.h与旧store helper，原生kernels及layout wrapper保持原行为。

V40真实模型96次、V41的128次HTP测量均通过。最终V2三端各20组边界测试通过；额外HTP六组覆盖K384/M64及K4096/M1536，N1/33/128，验证两tile分块和超过三个chunk的VTCM压力。V1三端各20组边界回归通过；所有测试同时执行原生路径并对照独立FP64累加参考，NRMSE阈值2%。

接口审查未发现确定的所有权/复制/view绕过问题。最终审查曾建议将vlut32 LUT改为连续byte，这是错误建议：QEMU指令语义和实机N1/HVX测试确认它取halfword槽的低byte，必须保留当前int16_t表。详见handoff-07中的语义出处；不能仅根据intrinsic名字推测table布局。

主要改动文件：htp/{hmx-mm-kernels-tiled.h,hvx-mm-kernels-tiled.h,matmul-ops.c,matmul-ops.h}、ggml-shared-weight.h、tests/{shared-weight-probe.cpp,run-shared-weight.py,analyze-shared-weight.py}。早期交换HMX输入及错误LUT实验已撤回；失败日志、partial JSON及patch保留为证据。

## 实际执行命令

```sh
cmake --build build-android-opencl-hexagon --target htp-v79 shared-weight-probe ggml-opencl ggml-hexagon -j 8
cmake --build build-android-opencl-hexagon --target shared-weight-probe -j 8
python3 tests/run-shared-weight.py --deploy --shared --layout cpu4x8 --model --profile --tag shared-v41-cpu4x8-htp-final --backend HTP0 --reps 128
python3 tests/run-shared-weight.py --shared --layout cpu4x8 --model --tag shared-v41-cpu4x8-cpu-stable --backend CPU --reps 512
python3 tests/run-shared-weight.py --shared --layout cpu4x8 --model --gpu-dp4a --tag shared-v41-cpu4x8-gpu-stable --backend GPUOpenCL --reps 512
python3 tests/run-shared-weight.py --deploy --shared --layout cpu4x8 --stress --profile --tag shared-v41-cpu4x8-htp-capacity --backend HTP0 --reps 8
```

V2边界tag为shared-v41-cpu4x8-{cpu,gpu,htp}-final-edges；V1回归tag为shared-v41-v1-{cpu,gpu,htp}-regression。均使用`--shared --edges --reps 8`和对应layout/backend，GPU加`--gpu-dp4a`，HTP加`--profile`。完整命令、设备/热状态及部署SHA256在每个`.run.json`中。

分析命令：`python3 tests/analyze-shared-weight.py <上述CPU.json> <GPU.json> <HTP.json> --output task/handoff/handoff-2026-10-16-shared-weight-layout/evidence/acceptance-final.json`。代码及文档的`git diff --cached --check -- . ':!task/handoff/handoff-2026-10-16-shared-weight-layout/evidence/experiment-v28-hmx-swap.patch'`通过；唯一排除的是归档补丁，其空白上下文行必须保留diff前缀空格，并非现行源码的尾随空白。增量构建成功，有已有的format-pointer/OpenCL pragma/missing-prototype警告，无本阶段构建错误。

## Git、清理及后续边界

开发分支feat/shared-weight-layout；原始内核备份backup/shared-weight-layout-baseline仍指向f6a299d8d9dbfbeadd13e63887868499415443a3。本次提交保存完整阶段，父提交88927c55c；最终提交号以该分支git log为准。未跟踪TODO.md是用户文件，未修改/提交。回退可显式选择原生路径或从备份分支恢复指定内核，不执行整体reset。

设备目录/data/local/tmp/llama-shared-weight-layout仅保留所有权标记和当前probe/共享库；失败实验V26/V27的partial JSON已拉回evidence并删除设备副本，成功运行的JSON/profile均已拉回并清理。模型未复制或修改。

当前任务清单全部完成。后续若接入MoE，应在此API之上持有expert到handle的映射、路由及预测，不把这些策略塞回物理布局层。其他量化、设备、三维/专家算子和全模型端到端加速均需独立验证，不属于本次已完成验收。
