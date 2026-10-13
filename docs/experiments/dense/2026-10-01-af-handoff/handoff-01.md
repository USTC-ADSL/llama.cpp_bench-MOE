# 激活交接实验交接

目标已完成：基于c8580ed4c的36层resident microbench，真正传入CPU Attention / NPU FFN的中间激活，量出复制时间并记录含同步的全轮耗时能耗。结果和全部限制见report.md。

## 代码

- `dense-af-bench.cpp`：新增resident专用`--handoff none|chain|frozen`。
- `resident-bench.h`：真实链式输出传递、同输入冻结对照、71边byte-exact检查、独立阶段计时；采用async graph compute + 一次sync，去掉旧路径的重复同步。CPU↔HTP blocking copy；同backend native async copy。chain参考名加`-chain`，误差使用output-relative RMS。
- `run-resident-android.py`：`--handoff-experiment`执行CPU参考、mixed frozen、mixed chain、all-NPU chain；带频率/显示检查与finally恢复。
- `summarize-handoff.py`：独立积分及结构/时钟/误差/恢复验证、CSV/JSON/PNG/PDF。

## 验证与状态

Android增量构建通过（build.log）；无线PKR110运行成功，每个功率窗口≥5秒。精确命令和结果在report.md及results中。Python编译通过。旧none模式CPU/HTP单层17设备冒烟通过；七个实际部署文件哈希匹配；72个临时CPU参考已从设备清理。无待运行测量。设备频率与超时恢复、wakelock释放，屏幕仍亮。

只读审查`handoff_final_audit`核验残差输出、链式依赖、frozen及同步语义，未发现功能错误；指出全NPU异步copy计时只是入队，此限制已写入报告。早先`chain_audit`被中断，没有有效审查结论。

混合chain108.764 ms、9.194轮/s、2.944 W、320.194 mJ；复制0.105575 ms/轮；相对全NPU对照节能17.91%、吞吐2.069倍。frozen109.609 ms，微小差异不能作显著加速结论。禁止把93.14 ms NPU sync当作纯同步开销；它包含FFN计算。没有embedding/head/采样和真实增长KV，不要把轮/s标为实测文本生成tokens/s。

## 后续

本任务无需追加扫描或重复功率测试。若继续论文实验，应由用户决定是否做完整模型/真实KV端到端验证；当前证据只支持此微基准与已选工作点。权重和KV驻留在负责该块的后端，不迁移；全NPU对照是72张分块图而非融合模型。

Git封板包含上述4个源码/脚本和本目录实验产物；用户未跟踪TODO.md保留不动。起点c8580ed4c，阶段提交号可用`git log -1`查询。本目录无需再次生成同义任务清单。
