# Canonical Expert 权重布局的边界校验

本轮新引入 canonical Q4_0 HTP 路径；尚未作为默认路径发布。

问题与触发：二维 setter 会重新设置 REPACK 标记；既有异步复制会覆盖目标 tensor 的布局标记；对已写入的 packed tensor 再标记 canonical 会重新解释已有字节。上述路径均可能令数据与 HTP kernel 预期布局不一致。

修复：普通/二维写入统一保留 canonical 标记；记录已写入状态，拒绝事后改变 packed 布局；涉及 canonical 的异步复制回退到语义正确的 get/set；拒绝 view、普通 matmul 和 HVX fallback，DSP 侧另设拒绝条件。

验证：fd8657d6 上 canonical-guards-fixed 通过二维 strided set/get、双向布局复制、fresh/已写入/unsupported shape/view/MUL_MAT/HVX 检查；实际 gate/up/down、8/32/128/256tokens 的 canonical/packed 输出逐位一致，另有 FP64 解量化参考。NDK 对 fullmodel-oracle.cpp、dynamic-moe.cpp、expert-routing-probe.cpp 的兼容性语法编译通过。构建使用 ggml-hexagon、htp-v79、ggml-opencl 增量目标，命令及日志在 online-weights handoff 目录。

另一个实验核验发现：Adreno GPU-to-HTP 转换的 byte shift 表达式会在实机生成错误高 nibble，关闭程序缓存后仍复现；改为读取 ushort pair 并显式提取四个 nibble 后，gpu-htp-pack-v5 的稀疏 Expert 映射、原始字节往返和 NPU 计算逐位对照均通过。前一个 struct-copy 表达式还触发了驱动编译器 address-space 错误。失败日志保留；此诊断不推广到其他驱动版本。

限制：当前支持连续 base Q4_0、K%128/N%32 的 HMX MUL_MAT_ID；不支持直接在 canonical HTP 权重上执行 decode/HVX，decode 使用 CPU 同源视图。算子校验不能替代整模型质量评估或端到端性能验证。
