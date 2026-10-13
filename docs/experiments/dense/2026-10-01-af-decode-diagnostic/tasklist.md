# 真实decode轻量劣势诊断

用户要求稍微检查mixed相对全NPU同host的约14.19ms/token劣势。仅做少量短诊断，不重扫工作点。沿用PKR110无线设备、Qwen2.5-3B、亮屏、NOM/DCVS0及双大核1.4016GHz。

- [x] 检查现有Hexagon timing接口和profile扰动；选择BASIC+aggregate，保留原fusion
- [x] runner加入计时窗口前后计数器快照与decode/sample拆分，构建设备部署
- [x] 每配置少量真实decode诊断，对比DSP batch/op与host queue时间、输出一致性
- [x] 给出有证据的瓶颈范围及未分离因素，恢复设备，整理报告并提交

不将DSP执行时间与host等待简单相加，它们可能重叠。原正式100-token结果仍以a05af0039为准，诊断模式另存。本轮起点a05af0039；保留TODO.md及前轮未跟踪power-trace文件。
