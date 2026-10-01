# 36层固定输入A/F放置实验

用户授权：Qwen2.5-3B全部36层，CPU计算Attention、NPU计算FFN，沿用已测最低能耗工作点，固定输入输出、不做跨块激活交换，检查多层是否获益。沿用无线192.168.50.180:44159，亮屏，decode1/KV256，双大核。

- [x] 核对已有多层参数与图生命周期：现有--layers all仅逐层独立窗口，不是全层常驻
- [x] 增加全层常驻、逐层A/F交替执行模式，固定每块输入，构建并部署
- [x] 校验所有层输出与原CPU参考，验证72块覆盖和旧逐块入口
- [x] 测混合CPU1.4016GHz/NPU NOM及全NPU NOM对照，5s预热+5s功率窗口
- [x] 检查稳定性、设备恢复与清理，输出报告并保存Git检查点（包含本清单的提交）

混合模式CPU维持1.4016GHz（含NPU host调用），不在每层做sysfs调频。统一NPU先沿用host2.6496GHz，另以host1.4016GHz作匹配对照。只忽略激活搬运，实际调度/同步/多后端常驻功耗进入测量。无真实跨层数据依赖，不是整模型TBT，不计embedding/head。

起点提交35881a896；用户TODO.md未跟踪，保留。源码预计修改tools/dense-af-bench/dense-af-bench.cpp并新增独立resident模式头文件、Android runner。设备新目录/data/local/tmp/dense-af-multilayer，旧部署不覆盖。不得仅把单层结果乘36当成测量。
