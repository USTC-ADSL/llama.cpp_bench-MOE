# 36层真实激活交接

用户授权：在前轮36层resident实验中真正传入中间激活，测CPU Attention1.4016GHz→NPU NOM FFN→下一层CPU Attention的复制/同步速度和总吞吐功率。沿用无线PKR110、亮屏、decode1/KV256、两大核。

- [x] 核对tensor-copy API和Hexagon共享内存/同步语义
- [x] 增加chain/frozen输入模式、正确性检查和分阶段耗时统计，构建设备部署
- [x] 生成36层CPU链参考并验证71条传递边，测混合chain与同输入frozen对照、全NPU chain
- [x] 分析复制/同步、总时间/能量与数值误差，恢复设备并清理临时参考
- [x] 报告与Git检查点

chain模式每轮从固定A0输入开始，依次传递71个8192-byte张量。历史KV固定不增长，无embedding/head；即使有真实层间依赖仍是transformer主体microbench。frozen模式先串联生成同样的各块输入，再在测量时冻结，隔离数值变化。CPU↔HTP复制在生产者graph完成后调用同步tensor_copy；HTP↔HTP使用async copy并由后继图同步完成。graph_compute_async+一次synchronize分开记录提交/等待，避免原来的重复同步。

起点c8580ed4c，保留用户未跟踪TODO.md。使用新设备目录/data/local/tmp/dense-af-handoff，旧数据/部署保留。只做少量窗口，不扫描工作点。
