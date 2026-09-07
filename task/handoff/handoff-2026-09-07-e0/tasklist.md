# E0 测量与执行路径核验

结果目录：`docs/Motivation_Exp/E0`。仅核验工具和真实路径，不进入缓存、调度或配额实验。

- [x] 确认设备、模型、构建、代码版本与原始 dirty diff
- [x] 审计已有脚本、fault 统计、参数语义与后端算子能力
- [x] 实现必要的轻量时间戳、边界采样与 sanity tests（CPU扰动验收仍BLOCKED）
- [x] 展示 dry-run 并交叉编译、部署最小测试工具
- [x] CPU / OpenCL / HTP 单 PD 诊断与真实 Q4_0 shape 数值测试（HTP n128数值FAIL已记录）
- [x] 采样开关三次配对及必要补测（CPU/OpenCL最终6对，HTP3对）
- [x] 汇总 PASS/BLOCKED、原始记录、handoff.json，清理设备临时产物与 Git 封板

约束：保留用户修改；使用已配置默认设备和模型；不改系统限制/缓存/swap/LMKD/SELinux/频率；加载 300 秒、请求 180 秒安全截断；MemAvailable < 1 GiB 持续 3 秒或 SEVERE thermal 停止。

最终状态：E0工作完成并停止，整体验收BLOCKED。OpenCL本次范围内PASS；CPU异步PSS后六对p95 ITL仍显著变化；HTP n128 HMX RRMSE=1。两个阻塞均保留原始证据，不改容差/路由/精度。所有181个设备文件已与本地证据/构建产物核对SHA256；仅清理本次设备目录，保留模型、既有缓存和原用户文件。详细交接见 handoff-02.md 与 E0/handoff.json。勾选代表核验工作已处理，不代表所有路径通过验收。
