# 共享权重中间层：首个功能原型，性能尚未验收

## 目标和状态

用户要求持续推进全部任务，不需要再次询问是否继续。约束以 tasklist.md 为准：独立于 MoE、一份长期权重、CPU/GPU/HTP 普通 Q4_0 MUL_MAT、各端最佳算子耗时退化 <=5%，允许加载时转换和有界临时空间，辅助 CPU 工作不能依赖全核。

开发分支 `feat/shared-weight-layout`；原生备份 `backup/shared-weight-layout-baseline` 指向 `f6a299d8d9dbfbeadd13e63887868499415443a3`。原有未跟踪 TODO.md 未修改。新路径显式通过 shared API 绑定，不替换原生权重路径。

**首个候选功能已在 fd8657d6 通过三端运行和数值校验，5% 总目标没有达到。不能宣称所有任务完成。** 当前布局选择是实验候选，不是已经选定的最终架构。

## 已读基础材料

- `docs/weight-manager.md`：现有 manager 有专家 cache，但不是本中间层的边界；当前任务不改 MoE。
- `task/handoff/handoff-2026-09-27-online-weights/design.md` 和 `results.md`：canonical HTP 仅部分 MUL_MAT_ID/HMX 可用，已有实测转换成本和失败优化。不要把它误认为已支持普通 MUL_MAT/decode。
- `docs/bugs/2026-09-27-canonical-weight-layout.md`：setter/copy/view/layout guards。
- `task/handoff/handoff-2026-09-12-weight-prepare/handoff-01-results.md`：现有 preparation probe 不测算子计算。
- `task/handoff/handoff-2026-09-24-shared256/results.md`：shared activation 的整体运行证据不是共享权重算子性能证据。

## 当前代码

- `ggml/include/ggml-shared-weight.h`，`ggml/src/ggml-shared-weight.cpp`：model-independent immutable handle，create/bind/describe/free；由 HTP buffer 分配 DMA，canonical Q4_0 加载时转换一次，绑定 CPU/GPU 缓冲视图。释放先同步所有绑定后端，再逆序释放 imports/owner。backend 必须比 handle 活得久，调用方不得修改返回 tensor。
- `ggml/src/ggml-shared-weight-impl.h`：描述符校验、无整矩阵临时副本的 canonical getter。
- 布局 V1 与 HTP 原生相同：`[M/32][K/32]`，每 tile 576 bytes，前 512 bytes 是 16 个相邻 K nibble-pair planes ×32 rows，后 64 bytes 是 FP16 scales。M/K 都需32对齐。
- `ggml/src/ggml-cpu/shared-weight.cpp/.h`：注册专用不可普通分配的 CPU extra buffer。prefill 每线程只临时整理32行，使用原有4x8 GEMM；NEON寄存器转置替代逐字节 gather。decode 现在用直接读取 tile 的 NEON dot-product。线程间只共享 activation quantization scratch，无完整后端权重缓存。
- `ggml/src/ggml-opencl/ggml-opencl.cpp` 与 `kernels/shared_q4_0.cl`：共享 DMA 导入后绑定 tile 标识，绕过 SoA conversion，image 对象别名原分配。prefill 同原生4行×8token FP16计算分块，只转换 activation。decode 正在评估 subgroup broadcast 版本。新增独立 buft 的 alloc_size，避免原生 SoA 对齐余量导致 tensor_alloc 边界断言。
- CMake 为上述代码和 `shared-weight-probe` 加入目标。
- `tests/shared-weight-probe.cpp`：同进程原生/共享交替测量，K/M为2048/1408、1408/2048，N为1/32/128。CPU对照含canonical和REPACK。量化合成权重、独立FP64累加参考（NRMSE<2%）。权重在计时前准备；每次调用中的重排/activation转换计入。测试保留原生对照的副本仅用于比较，不是产品内存占用。
- `tests/run-shared-weight.py`：只用fd8657d6，部署hash核验，日志/JSON/thermal记录；自有远端目录 `/data/local/tmp/llama-shared-weight-layout`，marker `.shared-weight-owned`。

## 共享与正确性验证

每个 handle 同时绑定 CPU/GPU/HTP，完整 canonical byte roundtrip 精确一致。CPU/GPU另验offset17/size37非对齐读取；HTP原生getter不支持这种任意partial，不能用它触发断言。

各端 N=1 都进行测试专用 alias 验证：所有工作完成后，直接翻转唯一DMA中每个FP16 scale的sign bit，重新计算应得到负参考，再恢复并重算。该测试能排除始终读取旧私有拷贝的实现。**不是公开可变权重API**，不得在有任务执行时这样做。指针相同本身不是零拷贝证据；驱动内部物理页统计没有独立硬件认证。

## 构建与运行

```sh
adb devices -l
cmake -S . -B build-android-opencl-hexagon -DLLAMA_BUILD_TESTS=ON
cmake --build build-android-opencl-hexagon --target shared-weight-probe ggml-cpu ggml-opencl ggml-hexagon -j 8
python3 tests/run-shared-weight.py --deploy --tag baseline-cpu --backend CPU --reps 24
python3 tests/run-shared-weight.py --tag baseline-gpu --backend GPUOpenCL --reps 24
python3 tests/run-shared-weight.py --tag baseline-htp --backend HTP0 --reps 24
python3 tests/run-shared-weight.py --deploy --shared --tag shared-v3-cpu --backend CPU --reps 24
python3 tests/run-shared-weight.py --shared --tag shared-v3-gpu --backend GPUOpenCL --reps 24
python3 tests/run-shared-weight.py --shared --tag shared-v3-htp --backend HTP0 --reps 24
python3 tests/run-shared-weight.py --deploy --shared --tag shared-v4-cpu --backend CPU --reps 32
python3 tests/run-shared-weight.py --deploy --shared --tag shared-v7-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --shared --tag shared-v7-cpu --backend CPU --reps 48
python3 tests/run-shared-weight.py --deploy --shared --tag shared-v8-gpu --backend GPUOpenCL --reps 32
```

运行结果统一 `evidence/<tag>.json/.log/.run.json`，tag不可覆盖；manifest记录实际二进制hash，不依赖内嵌commit版本。v1/v2/v5是失败实验（分别HTP partial getter、OpenCL SoA额外alloc_size、Adreno拒绝half8动态subscript），不能当性能证据。

最初 configure 与 build 不慎并发触发 package configure 失败，configure 完成后顺序重跑已成功。后续构建必须等待前一configure/build结束，不要重复全量重建。

## 主要测量结果（耗时比 = shared / 最快原生，越低越好）

- v3：三端功能通过。CPU逐字节整理慢1.66–6.92倍；GPU简单F32直接读prefill慢24–63倍。已替换这些路径。
- v4 CPU NEON整理：旧非持久线程池协议下仍慢约15–57%。不能作为严格算子基线。
- v7 CPU：启用持久threadpool，固定CPU mask f0（cores4/5/6/7），4线程。K80 Pro实机policy0为0..5 @3532800，policy6为6/7 @4320000，所谓“小核”应理解为较低频cluster，设备并无传统A5xx小核。无CPU辅助全核转换。
  - 2048×1408 N1：native29.4us/shared91.8us；N32：198.1/287.0；N128：792.1/1070.8。
  - 1408×2048 N1：19.4/41.2；N32：197.4/292.1；N128：815.9/1075.3。
  - 全部失败5%门槛。之前线程启动主导的约300us不能拿来掩盖实际差距。
- v7 GPU image/FP16 prefill：N32/N128比值1.098/1.116和1.014/1.141；decode仍约3.2–3.3倍。v8改decode为64-lane subgroup broadcast，仍未通过：2048×1408 N1 native560.0us/shared1832.8us、N32 973.0/1059.4、N128 1214.4/1348.7；1408×2048 N1 527.0/1288.2、N32 766.5/718.1、N128 874.9/960.0。数值和别名校验均通过。
- v3 HTP六组比值0.924、0.995、1.004、1.006、0.990、0.945，功能通过。与原生相同布局/内核，差异多为噪声；只能称当前同步单算子图计时通过，不能当独立纯DSP指令计时已完成。

**计时边界仍为单算子graph_compute+同步**，GPU含driver submission、HTP含RPC；CPU已排除线程创建。用户最终要求算子计算，不是端到端。最终还需GPU event/DSP profile核对，不能靠固定调度开销稀释差异来宣布5%通过。

## 尚未完成与下一步

1. 收取最新v8 GPU结果，检查数值与decode是否改善。
2. 保存功能原型检查点，明确性能未验收；检查独立API的拒绝路径/尺寸上限/生命周期边界，补必要设备回归。
3. 当前HTP tile无法证明适合CPU。另一个有意义候选是CPU原生4x8布局（**4行×8-byte交错，block_q4_0x4=72bytes，不是block_q4_0x8=144bytes**；探子曾混淆，勿照抄）。若尝试，需为HTP HVX decode和HMX解码增加布局分派，不能把它只改stride当canonical。
4. HTP关键位置：host repack `ggml-hexagon.cpp:899`；HVX tiled_vec_dot dispatch `htp/matmul-ops.c:543,715,1580,1651`；普通HMX `hmx_mm_2d_f32`约2665，DMA shape约2696，scratch约2734；canonical ID模板约3531；`hmx-mm-kernels-tiled.h:136` canonical四行scatter。普通HMX最终消费FP16 tile，必要改动主要是输入DMA/解码，不必改HMX矩阵乘本体。
5. 未实现多模型通用布局选择、任意views/shape/量化类型，也没有接入MoE上层。公开初版仅描述受支持2D Q4_0 MUL_MAT，显式选择，不应自动替换所有权重。
6. 不要提高阈值、将CPU_REPACK基准换成canonical或把layout转换挪出计时以“通过”。未达到目标要保留未完成状态。
7. 任务结束清理自有远端目录；目前仍保留必要二进制供继续实验，没有复制模型。
