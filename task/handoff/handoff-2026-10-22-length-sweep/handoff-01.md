# 当前代价模型长度外推：实验协议与恢复入口

用户要求测试128–2048输入相对纯NPU静态后端的收益。选择128/256/512/1024/2048五档，wetland/foundry两输入各取前nt个token；每组合NPU与hetero各两进程，wetland ABBA / foundry BAAB，每进程2预热+6计时，间隔30s。共40进程240计时请求。所有样本保留，主统计先取每进程中位数再平均两个进程；另报总体样本中位数。

代码基线5b32b8743，设备fd8657d6已确认在线。无需重建，使用上一轮已经构建/验证的binary和host/DSP libraries，运行前全部冻结并校验。模型/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf。

异构：固定runtime-fast-cpu150，CPU4x8单份权重，indexed DMA gather，NEON reduce，async分支。当前真实route驱动每层owner选择，计入调度时间；不调参、不重拟合、不读取预存route作为运行输入。两个输入的resident eligibility完全相同：前23层60专家三端可用，shared固定NPU；最后层单token全部CPU。计划中的初始selected在每次scheduler choose时被替换。

纯NPU：原生`--mode npu`，HTP0全层offload、原生布局/MMID，保留框架不支持算子的CPU fallback。已亲读fullmodel-oracle.cpp:723–740核实CPU expert override数组**仅在custom条件中赋给mp**，不影响mode=npu，不能把数组声明误判为无条件CPU分配。两路径同计时循环：n_ctx=n_batch=n_ubatch=nt，F16 KV，最后token logits，加载/权重布局准备/KV清空不计时，llama_decode+synchronize计时。router默认放置不变。CPU outer4线程，hetero expert4线程。

固定远端/data/local/tmp/llama-length-sweep-20261022。脚本run.py拒绝覆盖frozen.json；失败时保留证据并停止，不自动删日志重试或降级chunk。2048如发生设备问题，保留已完成长度，明确记录原因。任何后续续跑需明确新tag/记录，不能重写失败记录。

执行：`python3 task/handoff/handoff-2026-10-22-length-sweep/run.py`。进度runs.json，冻结配置frozen.json，原始结果/日志/hash/thermal在evidence/。首次部署后不改运行版本。任务清单随长度完成即时更新。

后续：统计各长度的完整prefill中位数和tokens/s、NPU/hetero比；核验同长度token一致以及不同长度前缀关系、进程重复输出bitwise一致、跨NPU/hetero logits NRMSE/cosine/top1/top10（数值不必逐位一致）；记录major faults、arena、专家分配、调度耗时。冻结配置不能用长度结果反过来调参。归档冻结输入/代价模型/库和全样本，再清理仅本任务远端并提交。用户TODO.md不动。
