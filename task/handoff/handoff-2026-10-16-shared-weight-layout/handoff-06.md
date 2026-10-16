# CPU4x8 HTP 功能检查点，性能未通过

V2 已实现同一 allocator raw DMA storage 上的不可变 HTP tensor metadata 绑定，独立 host/DSP flag、canonical getter 和 graph/copy/view/fusion 拒绝检查；无额外长期权重。HVX/HMX 使用已有 VTCM raw staging，DMA 连续读取 CPU4x8；所有准备计入 DSP 算子时间。原生内核和 V1 保留。

V21 三端绑定、别名变异、真实模型六组和 HTP 边界 20 组均通过。V22–25 优化 HVX 与 HMX 数据整理；当前 V25 真实模型六组通过，HTP native/shared 微秒依次为 K2048 M1408 N1:53/59.5、N32:89/115、N128:115/141；K1408 M2048 N1:50/55、N32:87/115、N128:113/138。**尚未达到 5%，不得标成完成。** 最新边界仍需复测。

主要文件：ggml-shared-weight.cpp、ggml-shared-weight.h、ggml-hexagon.cpp、htp-ops.h、matmul-ops.c、hvx-mm-kernels-tiled.h、hmx-mm-kernels-tiled.h、新 shared-cpu4x8.h、shared-weight-probe.cpp。

实际增量构建为 `cmake --build build-android-opencl-hexagon --target htp-v79 -j 8`（DSP 必须显式编译），host 使用 `--target shared-weight-probe ggml-hexagon`。V25 运行命令：`python3 tests/run-shared-weight.py --deploy --shared --layout cpu4x8 --model --profile --tag shared-v25-cpu4x8-htp-scale --backend HTP0 --reps 48`。证据和二进制 hash 见 evidence/shared-v20* 至 shared-v25*，V20 是过时 probe guard 导致的失败，不能作为通过证据。设备 fd8657d6 已确认在线。

HVX V25 用向量 gather 已载入 raw block 的 scale 前缀，再转置 8 组四行到原生 dot lanes。V24 scalar memcpy 读取 VTCM scale 慢至 182–191us，禁止恢复。V22 八个 float accumulator 有严重 spill，约78us；V23 两个 accumulator 约90us。当前 V25 55–60us 最好。

HMX 当前逐四行解码并做寄存器转置、整向量 store，无 scatter；约115–141us。下一实验交换 HMX 左右输入，权重解码为左输入，将转置移至较小 activation 和 output。必须亲自核对 A/output 的半字交错格式，不能照抄探子声称 row-major 的结论：hvx_vec_f32_to_f16_shuff 实际交错两行。交换后输出也必须转置回 [token][feature]。所有准备均计入算子时间。

下一步完成 HTP 优化、三端最终稳定统计、完整接口/内存回归并更新任务清单。用户 TODO.md 不动。自有设备测试目录保留后续二进制，无模型副本；最终再清理。当前任务仍未完成。
