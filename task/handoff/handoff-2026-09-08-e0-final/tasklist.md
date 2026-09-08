# E0 / E0B 最终补验

- [x] 确认 fd8657d6 在线，读取历史验收与现有运行器
- [x] 验证修复版 OpenCL 固定 128/32 请求 off/observe/enforce 输出一致，全层卸载
- [x] 验证 4 GiB 工程预算（允许 5%），记录 R 估计、拒绝行为和可观测边界
- [x] 两份简短最终报告放在 docs/Motivation_Exp/E0_E0B_final/，检查证据与 Git 封板

不改推理逻辑，不覆盖原始实验数据。用户已有 AGENTS.md 和 Exp1/Exp_require.md 不纳入提交。

预算补验：OpenCL/rpcmem/交替三组均实际触碰 3 GiB extra +512 MiB host，第49块64MiB拒绝且最终账本归零；模型 GPU/HTP 加载均 budget_denied 退出1，cgroup failcnt=0，无安全截断。证据 evidence/final/budget.json。首轮三模式32 token一致，但默认verbosity未打印层卸载，自动路径断言未通过；-v补跑后三种模式均PASS，首轮证据保留。任务完成不代表全物理RAM±5%已证实，见最终E0B报告。
