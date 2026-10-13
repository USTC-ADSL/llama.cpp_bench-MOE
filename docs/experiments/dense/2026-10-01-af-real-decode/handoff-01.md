# 真实decode阶段完成

用户要求由固定KV主体microbench推进到真实生成。完整实现、Android构建部署、两种放置审计、三组100步真实生成均已完成。结果与解释见report.md，原始数据均在本目录。

## 重要结果

mixed（CPU A1.4016GHz/NPU F NOM）13.446 token/s；全NPU NOM host2.6496GHz 16.897；全NPU同host1.4016GHz 16.617。mixed吞吐低20.42%/19.08%。三组相同prompt256 token和输出101 token（首个来自prefill，100个来自被测decode）；KV增长256→356。旧microbench优势不能外推。未测本轮能耗，禁止套用旧2.94W或17.9%节能。

## 修改

- `src/models/qwen2.cpp`：实验环境变量`LLAMA_QWEN2_CPU_ATTENTION=1`时明确pin完整A到CPU、F到层默认设备；新增分段graph expand与pin；offload_kqv必须关闭。开关关闭时原执行图路径保持，已在全NPU正式生成中验证。
- `tools/dense-af-bench/dense-af-decode.cpp`及CMake：全模型runner。固定256个真实文本token，完整预热，清KV后重做prefill，greedy采样实际序列并计时100次decode+sample；EOG抑制，有限性检查计入耗时；输出文本在计时后转换保存。
- `run-real-decode-android.py`：隔离部署、CPU/屏幕控制与恢复、audit/measure及同host选项，产物拉回清理。
- `summarize-real-decode.py`：placement/时钟/生成tokens/KV长度/时间/恢复验证及CSV/图表。

## 执行和验证

具体命令见report.md。增量target dense-af-decode构建通过，8个文件部署哈希一致。mixed和NPU各一次placement audit、4步decode；验证36层A/F、36FA、72KV updates及输出头后端。正式mixed及NPU各100步，再补一次NPU同host100步。每组真实输出完全一致，内部perf均100eval及99次graph reuse；独立汇总脚本通过。

设备无线192.168.50.180:44159，PKR110；每次命令频率结束读回正确；恢复CPU1017600–4320000kHz、timeout10800000、释放wake lock，仍亮屏。新设备目录只保留8个二进制/库，prompt/text/token/audit文件均已拉回并清理。不要覆盖旧handoff/multilayer目录。

实现期间遇到两项已解决的小问题：ggml_is_view_op是内部辅助函数，改为枚举判断四类view op；root mkdir后的设备目录归属导致adb push失败，部署时chown shell:shell后成功。没有设备测量失败或待处理后台工作。

只读审查核对了decode计数与logits同步；其“模型应为24层MoE”发现不适用于本任务，用户明确指定Qwen2.5-3B且设备日志确认36层，未按项目默认MoE改变任务模型。

## 边界和后续

每配置只测一次、一个短上下文区间；无置信区间。输入embedding在CPU，head在NPU，所有方案相同。全NPU图2splits，mixed72splits，是重要可观测差异但未隔离每个因素，不能把总差值全部叫同步开销。没有真实生成能耗数据，也没有QNN对照。

任务无需继续补测，除非用户要求。后续若问能量应在此真实runner上单独采集，不移植microbench功率。Git起点9e6e636ed，阶段提交由git log查询。保留用户TODO.md和前轮未跟踪3个power-trace产物，勿混入提交。
