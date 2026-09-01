# Exp1 任务清单

- [x] 完成 Qwen1.5-MoE mmap/OpenCL/HTP 基础可运行性验证
- [x] 阅读并确认 `docs/Motivation_Exp/Exp1/Exp_require.md` 的实验要求
- [x] 确定统一的 TTFT、TPOT、PSS、MemAvailable、major faults 与 LMKD/OOM 测量方法
- [x] 准备设备端实验运行环境、固定 prompt 和监控脚本
- [ ] 完成 CPU + mmap，重复 3 次
- [ ] 完成 CPU + no-mmap 对照
- [ ] 完成 OpenCL + mmap，重复 3 次
- [ ] 完成 HTP 1 PD，重复 3 次并记录 HTP-REPACK buffer
- [ ] 完成 HTP 2 PD，重复 3 次并记录 HTP-REPACK buffer
- [ ] 完成 HTP 4 PD，重复 3 次并记录 HTP-REPACK buffer
- [ ] 汇总原始结果，核验异常值与系统稳定性
- [ ] 在 `docs/Motivation_Exp/Exp1/` 输出最终实验报告
- [ ] 检查最终 Git diff、清理设备端临时产物并保存阶段性提交
