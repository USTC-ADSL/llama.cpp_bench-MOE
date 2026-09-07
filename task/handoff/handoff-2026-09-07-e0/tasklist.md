# E0 测量与执行路径核验

结果目录：`docs/Motivation_Exp/E0`。仅核验工具和真实路径，不进入缓存、调度或配额实验。

- [x] 确认设备、模型、构建、代码版本与原始 dirty diff
- [x] 审计已有脚本、fault 统计、参数语义与后端算子能力
- [ ] 实现必要的轻量时间戳、边界采样与 sanity tests
- [x] 展示 dry-run 并交叉编译、部署最小测试工具
- [ ] CPU / OpenCL / HTP 单 PD 诊断与真实 Q4_0 shape 数值测试
- [ ] 采样开关三次配对及必要补测
- [ ] 汇总 PASS/BLOCKED、原始记录、handoff.json，清理设备临时产物与 Git 封板

约束：保留用户修改；使用已配置默认设备和模型；不改系统限制/缓存/swap/LMKD/SELinux/频率；加载 300 秒、请求 180 秒安全截断；MemAvailable < 1 GiB 持续 3 秒或 SEVERE thermal 停止。

当前发现：CPU 同步边界 PSS 测量开销约 25%，异步改进正在构建。OpenCL 真实权重 n=1/128 数值通过；HTP n=1 HVX 通过，n=128 HMX 失败（RMSE=1），独立只读核验未发现构图/repack/sync 缺失。保留原始失败记录，不改容差/路由/精度。Android ADB 后台 shell 会等待子进程，已改为前台 ADB Popen+PID gate，第一条 watchdog 观测后才放行。
