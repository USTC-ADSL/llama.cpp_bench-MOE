# 原生CPU加载模型时共享权重能力探测空描述符崩溃

## 现象与触发条件

Redmi K80 Pro（fd8657d6）上使用fullmodel-oracle的cpu模式加载Qwen1.5-MoE-A2.7B-Chat Q4_0，在blk.0.ffn_gate_inp.weight的buffer类型选择阶段SIGSEGV（exit 139，fault address 0x24），尚未进入prefill或decode。CPU额外buffer枚举启用了CPU_SHARED_Q4_0；原生CPU正常加载需要遍历这些类型。

## 根本原因

llama-model-loader.cpp的weight_buft_supported()创建`ggml_backend_buft_alloc_buffer(buft, 0)`作为临时能力探测buffer。该buffer携带类型，但context为空。ggml-cpu/shared-weight.cpp的supported()识别到共享类型后，直接把context解释为ggml_shared_weight_desc并读取d->k，导致空指针访问。

崩溃栈：supported(shared-weight.cpp:139) → ggml_backend_cpu_shared_weight_op_allowed → ggml_backend_cpu_device_supports_op → weight_buft_supported。它是import-only buffer对模型加载器探测契约处理不完整，不是模型损坏、内存不足或decode计算错误。

## 修改

supported()在读取描述符前检查context，缺少真实import descriptor时返回false，使canonical权重继续走普通CPU/CPU_REPACK类型选择。没有关闭CPU_REPACK、禁用共享权重或改动计算kernel。真实import descriptor路径保持原有检查和执行行为。

新增tests/test-cpu-shared-weight-support.cpp，重现加载器的零大小探测：F32和Q4_0矩阵均应拒绝import-only类型，并允许普通CPU类型。

## 构建和设备回归

实际执行：

```sh
cmake --build build-android-opencl-hexagon --target ggml-cpu -j4
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include tests/test-cpu-shared-weight-support.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -o build-android-opencl-hexagon/bin/test-cpu-shared-weight-support
adb -s fd8657d6 shell 'cd /data/local/tmp/llama-decode-scheduler-20261023 && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./test-cpu-shared-weight-support'
```

- Android定向测试通过：F32/Q4_0共享类型探测返回false，普通CPU探测返回true。
- 原生CPU完整模型成功加载，日志确认CPU_REPACK模型buffer约7587.43 MiB；128前缀、32步连续decode、两次warmup和两次timed sequence通过，各轮logits逐位一致。
- 相同新库下的shared-weight/common路径也完成上述decode回归；33个位置logits与修复前有效shared路径逐位一致，warmup后无新增job分配。
- 增量构建只重编共享权重支持检查和版本信息涉及的ggml.c，重新链接CPU/base库。原库备份保留。

证据目录：task/handoff/handoff-2026-10-23-decode-scheduler/evidence/。关键文件：pilot-wetland-128-cpu.log（原始失败）、cpu-pilot-crash.log、cpu-pilot-symbolized.log、cpu-fix-build.log、cpu-support-test.log、verify-wetland-128-cpu.*、verify-wetland-128-common.*。fixed-validation.json汇总校验；原源码备份和库位置见cpu-fix-backup.json、backup.json。

## 限制

这是对有效的空context能力探测的处理，并不验证任意损坏的非空descriptor。支持查询不代表实际导入成功；真实共享权重仍需通过既有import与布局校验。此回归基于当前Android/ARM64设备，未扩展为其他架构的运行验证。
