# V19 GPU：六组实际权重初测通过5%，HTP适配在途

CPU4x8 V2现在GPU直接读取连续packed buffer，prefill使用与原生相同的64行×32token DP4A数据流，只有activation转Q8临时存储，没有权重副本。小batch保留FP16，decode用连续vector读取及8个K分区。V1未改算术路径。

发现原生GGML_OPENCL_Q4_0_DENSE_DP4A默认关闭但明显更快，runner增加--gpu-dp4a以纳入最佳有效基准；--shared仍会创建原生和shared两类job，不能误认native对照也走shared kernel。

实际增量构建：cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8。
测试均为python3 tests/run-shared-weight.py --shared --layout cpu4x8 --model --gpu-dp4a --backend GPUOpenCL，首次构建后--deploy。
V15对照(shared-v15-cpu4x8-native-dp4a,32reps)、V16连续buffer prefill(shared-v16-cpu4x8-vector-gpu,32reps)、V17 DP4A(shared-v17-cpu4x8-dp4a-gpu,48reps)、V18 decode buffer(shared-v18-cpu4x8-mv-buffer,48reps)、V19 decode8分区(shared-v19-cpu4x8-mv-eight,48reps)。V19边界--edges --reps4，tag shared-v19-cpu4x8-edges-gpu，20组通过。

V19 native/shared kernel event微秒（包含激活准备）：

| K,M | N | native | shared |
|---|---:|---:|---:|
|2048,1408|1|36.096|29.952|
|2048,1408|32|250.880|256.000|
|2048,1408|128|423.936|391.168|
|1408,2048|1|33.024|34.048|
|1408,2048|32|189.952|171.008|
|1408,2048|128|438.016|402.048|

六组初测均<=1.05，prefill与原生DP4A数值误差一致；CPU之前V14也通过初测。尚需稳定统计/完整内存及接口回归，HTP CPU4x8还没有实测结果，不能声称总任务完成。

当前工作区另外有未提交HTP改动：独立host/DSP CPU4x8 flag；allocator raw_storage直接绑定metadata、immutable getter；host能力与graph guard；HVX直接四行dot、HMX VTCM scatter decoder、连续DMA。正在构建htp-v79（需要显式目标，ggml-hexagon目标不会自动编DSP）。先收构建结果，再更新host增量构建并部署进行HTP数值验证。现有DSP重建命令cmake --build build-android-opencl-hexagon --target htp-v79 -j 8。默认设备fd8657d6在线。

待检查的HTP风险：HVX unpack与shuffle次序、HMX tile scatter地址、额外56byte unaligned load仅限有padding的VTCM、全图融合及copy不得绕过CPU4x8约束。已禁用CPU4x8的MMNX/ADD融合及raw异步copy。接下来扩展probe到HTP拒绝路径/partial getter。用户TODO.md未动。
