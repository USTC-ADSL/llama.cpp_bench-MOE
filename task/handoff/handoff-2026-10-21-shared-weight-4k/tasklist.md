# 三端共享权重：4K MoE 吞吐、分配优化与 router 验证

- [x] 确认默认设备 fd8657d6 在线，保留用户已有修改
- [x] 恢复既有实验设置、输入和构建产物，确定 4K 对照协议
- [x] 在原专家分配下完成 1024 token 正式 ABBA；4096 token 完成运行诊断但不作稳定加速结论（用户允许 1K 优先）
- [ ] 分析三端负载并实测有依据的分配优化候选
- [ ] 对照 router 的 GPU / HTP 算子时间与端到端影响
- [x] 完成真实 layer 0 / 12 router projection 的 GPU event / HTP DSP op 计时与数值检查
- [x] 验证 GPU router 整模型可执行，24 层 probs/top-k buffer trace 为 OpenCL
- [x] 在 1K 以进程间 30 秒空闲重建完整稳定对照
- [ ] 汇总结果、正确性和内存限制，保存代码与实验检查点
- [ ] 归档证据并清理本任务设备临时产物

目标：使用当前 Redmi K80 Pro 与既有 Qwen1.5 MoE 模型，区分布局接入收益、分配收益和 router 放置收益。延续先前真实路由 prefill 测试口径，将 length 扩展至 4096；不将 prefill 吞吐描述为 decode 吞吐。

用户更新：4K 若异常，先验证 1K。4K 整批首次触发 kernel_panic；分成 4×1024 可完成，但共享路径正式样本出现约 4–5 万次 major faults/请求与温度/耗时波动。停止扩展 4K，改以 1024 token 完成三个问题。已开始的 wetland native-b 请求允许结束并保留诊断证据。
