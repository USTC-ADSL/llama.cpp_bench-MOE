# 完整真实decode能耗比较

用户询问真实mixed相对单后端J/token变化。之前完整decode仅测吞吐，不能套用microbench功率。本次沿用PKR110无线设备、亮屏、Qwen2.5-3B、双大核，mixed CPU1.4016GHz/NPU NOM，与全NPU NOM host1.4016及2.6496GHz比较。

- [x] runner加入可选电池采样与token时间边界，增量构建部署
- [x] 每配置一次200步真实decode，采集电池电流/电压及频率
- [x] 独立积分整段与后100步稳定窗口，比较J/token和吞吐、检查真实输出
- [x] 恢复设备，记录边界和报告，Git封板

prompt256，实际KV随200次decode增长到456；后100步统计对应历史KV356开始。采用后100步作为稳定窗口，避免prefill后约1s更新电池遥测影响。不重扫档位，不重复打乱顺序。功率口径整机亮屏、双串电池电压近似×2，非单加速器轨。

起点342dc47e1。保留用户TODO.md和旧3个power-trace产物，隔离部署/data/local/tmp/dense-af-real-energy。
