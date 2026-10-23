# 正式CPU/current对照已启动

用户明确最新范围：只比较纯CPU与当前调度，取消纯NPU性能基线。NPU此前四组诊断仅生成共同teacher tokens，不能当正式性能结果。设备fd8657d6在线，默认设置；TODO.md与paper/未动，2K暂缓。

实现检查点aacd2c0f9允许shared-weight连续decode，128 common/private三十三个logits位置逐位一致。随后原生CPU加载在CPU_SHARED_Q4_0的能力探测中SIGSEGV，fault address 0x24。根因是模型加载器给零大小、context为空的dummy buffer，supported()读取空descriptor。4de2280d5修复该检查，保留CPU_REPACK并加入tests/test-cpu-shared-weight-support.cpp及docs/bugs/2026-10-23-cpu-shared-weight-probe-null-descriptor.md。Android定向回归、纯CPU 128+32 decode及shared路径回归通过，修复前后有效shared logits逐位一致。

增量构建命令：cmake --build build-android-opencl-hexagon --target ggml-cpu -j4。重编shared-weight.cpp及版本信息涉及的ggml.c，链接CPU/base库；旧库备份在evidence/backup。新库已部署到/data/local/tmp/llama-decode-scheduler-20261023。cpu-support-test.log、fixed-validation.json、cpu-fix-build.log以及verify-wetland-128-{cpu,common}.*记录验证。失败pilot-wetland-128-cpu.*及cpu-pilot-crash.log必须保留，不应覆盖。

当前正在运行：python3 task/handoff/handoff-2026-10-23-decode-scheduler/formal.py，stdout在evidence/formal-driver.log。此入口才是CPU/current正式实验；不要使用run.py formal（它是用户澄清前未执行的NPU入口）。16进程，128/1024×wetland/foundry×CPU/current×两进程，每进程两次sequence warmup+四次timed，32 steps，ABBA/BAAB，20s冷却；总2048个计时decode token，prefill重建在计时外。protocol.json和formal-frozen.json冻结了范围、四组teacher、binary/库、模型/plan、输入与源码。

不要修改formal-frozen.json里列出的源码/脚本/库/模型，不要重跑入口覆盖tag。进度在runs.json（含pilot、teacher、一次预期保留的失败CPU pilot和verify），正式记录tag以formal-开头。启动时已有两个正式进程完成，current首轮约51.23ms/token，CPU预检约34.86ms/token，尚不能当最终均衡对照。

analyze.py已改为CPU baseline，支持--partial。它核验重复及跨进程logits逐位一致、相同teacher/输入、24层N=1、4 routed pairs、最后层CPU约束、common容量/owner、权重总量、hashes及预热后不再增加job。输出summary.json，CPU/current吞吐收益，以及分支预测/实测、决策和归约统计、数值相对CPU误差。report.py生成results.md、comparison.csv、branch-summary.json。两者不在冻结列表，可以继续完善分析。

下一步：等待formal 16进程完成，运行analyze.py与report.py；如失败保留日志先定位，不筛选慢样本。补全结果、确认负收益的边界（N=1超出128–1024校准域；GPU是否实际使用；CPU计算4线程），归档所有证据并逐文件验hash后再清理本任务设备目录。最终Git封板，保留原kernel和binary备份。本任务要求测当前策略，不为了得到正收益修改代价模型。
