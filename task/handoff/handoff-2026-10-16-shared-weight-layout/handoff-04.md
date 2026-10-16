# CPU4x8 V2 候选检查点：CPU通过初测，GPU未达标，HTP未实现

目标与约束仍按tasklist.md，全部任务未完成。原V1及原生路径保留，默认create仍为V1。用户TODO.md不动。

新增create_with_layout与CPU4x8枚举。raw_storage由HTP分配DMA，但不创建HTP tensor；HTP绑定明确返回空，防止错误解释布局。加载时转换为72字节的四行8-byte交错格式，CPU直接复用原生4x8 traits。GPU仅适配地址读取；两端getter恢复canonical，均没有完整私有权重。probe/runner加入--layout cpu4x8。

实际构建：cmake --build build-android-opencl-hexagon --target shared-weight-probe ggml-cpu ggml-opencl -j 8。

实际测试：tests/run-shared-weight.py --shared --layout cpu4x8，分别运行--model --backend CPU --reps 64 (shared-v14-cpu4x8-model-cpu)、--model --backend GPUOpenCL --reps 48 (shared-v15-cpu4x8-model-gpu)，两端--edges --reps 4 (shared-v15-cpu4x8-edges-cpu/gpu)。首次部署加--deploy。所有数据及部署hash在evidence/同名文件，设备fd8657d6。

CPU六组shared/native比率1.008/1.001/1.000/1.000/1.001/1.001，GPU kernel event比率1.468/1.632/2.854/1.326/1.430/2.542（顺序K2048 M1408 N1/32/128，再K1408 M2048 N1/32/128）。数值与alias均通过，边界各20组通过。CPU性能仅初测，完整稳定统计未完成。

下一步：优化V2 GPU权重读取/算术数据流；审计并测量原生dense DP4A对照。其环境开关GGML_OPENCL_Q4_0_DENSE_DP4A默认关闭，只有N>8/K%32/M%64且设备支持integer dot才用。probe已有native和shared两个job，不能将shared分支错误地认为覆盖native对照。HTP V2仍按handoff03的独立flag/直接HVX与HMX解码路线待实现。

两位只读探子完成：gpu_native_dp4a_audit、gpu_cpu4x8_dataflow。后者声称local={1,128}与get_global_id不一致是错误推论，不能照改为{1,1}；OpenCL合法多线程workgroup完全可以使用global ID，需核查enqueue的round-up。其指出重复image读取有优化价值。探子不可复用。

本检查点只保存候选实现，不代表三端目标完成。设备自有目录仍保留下一轮实验二进制，无模型副本。
