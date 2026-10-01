# 真实自回归 A/F decode

用户要求验证真实decode吞吐。沿用无线PKR110、Qwen2.5-3B、双大核，mixed CPU A 1.4016GHz + NPU F NOM/DCVS0，对照全NPU NOM host2.6496GHz。亮屏，不重复扫描工作点。

- [x] 确认现有llama-bench采用随机token，不能替代真实采样；定位Qwen2图及权重/缓存放置入口
- [x] 实现完整模型真实采样runner与显式A/F图放置，增量构建
- [x] 设备验证实际放置与生成，测256-token prompt后100次真实decode的吞吐
- [x] 整理原始token/文本/耗时及限制，恢复设备、检查diff并Git封板

设计：完整模型embedding、36层、final norm/head，真实tokenize及greedy logits采样，KV随生成增长。首个prompt输出token单独报告；随后100次单token decode+采样计时，排除加载/prefill及文本输出I/O。显式禁止EOG以固定工作量并记录该条件。模型输入/输出头沿默认放置，两方案一致；不把完整模型与旧主体microbench混为一谈。

起点9e6e636ed。保留TODO.md与前轮三个未跟踪power-trace产物。新部署固定目录/data/local/tmp/dense-af-real-decode，不覆盖旧实验。
