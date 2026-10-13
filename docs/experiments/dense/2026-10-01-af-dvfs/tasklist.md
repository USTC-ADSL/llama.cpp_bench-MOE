# A/F 独立配置的能效机会：FastRPC + DVFS

用户授权：以FastRPC代替QNN，调整NPU DCVS及GPU/CPU频率，CPU只用双大核；比较A/F独立放置相对全CPU/GPU/NPU的最佳点。沿用无线设备192.168.50.180:44159（执行前重新确认），Qwen2.5-3B。先采用零额外边界成本的块级组合，不据naive切换实现下结论。

- [x] 读取原离线profile，确认工作点、context范围和现有延迟约束
- [x] 核对设备CPU/GPU频率接口、恢复方法及FastRPC DCVS修改路径
- [x] 实现必要的DCVS配置与小规模扫描工具，构建设备验证
- [x] 采集A/F工作点能量—延迟数据（CPU双大核），完整恢复设备状态
- [x] 比较最佳统一配置与独立配置，输出能效收益与Pareto结果
- [x] 保存最终Git检查点（包含本清单的提交；见git log）

起点Git仅用户未跟踪TODO.md，不能改动。已有功率采样工具和两电芯校正在dense-af-bench与上阶段handoff中。原profile为 `/home/miog/pzw/download/pzw/HeteroCompute/llama.cpp-acom/Paper_Writing/offline-OnePlusAce5/offline-OnePlus-3B/Decode/system_benefit_offline_profile.csv`，字段包含throughput_tps,power_mw,energy_mj_per_token，不能直接作为新设备新后端的实测值。

最终用户缩小范围：无固定延迟上限，给最低能耗和Pareto；先仅KV256的decode1，对同一A/F图连续预热5s+测量5s，不追求恰好100次，不测低档。CPU6–7两核（mask c0）有效档2246400/2438400/2649600kHz；GPU607/832/1050MHz；NPU nom/nom_plus/turbo（固定core/bus/HMX corner，DCVS动态浮动关闭）。最终18窗口全部亮屏，独立块组合乐观忽略边界成本。原来考虑的1792上下文暂不执行。

默认NPU、NOM DCVS1及非法配置检查完成。host/DSP增量构建成功并部署到独立目录/data/local/tmp/dense-af-dvfs。请求3.2832/3.8016GHz未能维持，已剔除；NOM+不稳定窗口延长预热复测并保留原证据。CPU A + NPU NOM F的估计能耗6.2462mJ，相对最佳统一NPU节能8.44%。最终见report.md和screen-on-analysis。频率、超时、wake lock已恢复。

## 追加：独立块 Pareto 曲线

用户确认复用原有每后端3点、各新增3点，总计每后端Attention/FFN各6点。新增CPU1.4016/1.6896/1.9584GHz，GPU443/734/967MHz，NPU svs/svs_plus/max。继续亮屏、同一层/KV/参考/5s预热+5s测量；输出screen-on-extension。

- [x] 核对现有独立块数据、在线设备与可用频率
- [x] 确认点数并选择不重复的新工作点
- [x] 保持亮屏，采集新增工作点的A/F独立窗口并恢复设备
- [x] 扩展汇总工具，输出每块的后端内及跨后端Pareto曲线
- [x] 校验结果、保存报告和Git检查点（包含本清单的追加提交）
