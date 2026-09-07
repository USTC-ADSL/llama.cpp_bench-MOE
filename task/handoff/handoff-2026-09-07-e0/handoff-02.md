# E0 最终交接：完成核验，验收 BLOCKED

目标仅为手机MoE工具和真实路径核验；本次已结束，禁止据此自动继续后续研究实验。入口：`docs/Motivation_Exp/E0/REPORT.md`、`verdict.json`、`handoff.json`。

代码基线 `d46e707010c74d2c1f8d3654fdd64abe7cfaacf2`；instrumentation检查点 `3b17c9d2b`；最终结果版本为包含本文件的Git提交。保留用户AGENTS.md修改和Exp1/Exp_require.md未跟踪文件，原SHA256核验通过。没有push。

设备 `3B661501LA000000`，OnePlus PLU110 / SM8735 / Adreno825 / Hexagon v73 / Android16。模型原路径 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`；完整SHA256与实际tensor shape见E0/environment.json及raw/model.txt。

主要修改：common/e0-measure.h与common/common.cpp、tools/server/server-http.cpp、tools/server/server-context.cpp增加opt-in内部边界/token-ready记录；E0/scripts实现设备运行、watchdog、真实Q4_0算子比较、fault sanity、分析和日志路径审计。完整核心代码差异见E0/raw/instrumentation.patch。所有部署binary和最终脚本hash见artifacts.json、manifest.json。

实际构建/运行命令全部保存在E0/commands.jsonl及raw构建日志。复用已有build-android-opencl-hexagon：

```sh
cmake --build build-android-opencl-hexagon --target llama-server htp-v73 -j8
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run cpu --pairs 6 --label cpu-validated
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run opencl --pairs 6 --label opencl-confirm
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --run htp --label htp-validated
python3 docs/Motivation_Exp/E0/scripts/analyze.py
python3 docs/Motivation_Exp/E0/scripts/path_audit.py
```

详细profile和最终numeric分别在runs/{cpu,opencl,htp}-profile、runs/numeric-{opencl,htp}-verified。主要33次请求均128输入/32输出/cache_n0，token IDs与RUSAGE计数检查通过。CPU六对p95 ITL采样开启变化−11.522%，不能证明±3%；OpenCL六对与HTP三对采样检验通过。OpenCL原Q4_0四项数值PASS；HTP n1 HVX两项PASS，n128 HMX两项RRMSE=1，FAIL。没有改精度、路由或容差来通过测试。

重要语义：--no-warmup不关闭server序列删除能力的[0,0]两token探测；探测前后清KV。path-audit请求范围已排除启动探测，HTP NX实际请求完成768次（生命周期792）。PSS只报采样峰值，异步边界延迟保留。cgroup历史peak、PSS、KGSL计数和FastRPC映射容量分别报告，不相加；ION/rpcmem cgroup覆盖未知。

失败试验与不要重复的路径：旧同步PSS显著扰动；Android后台ADB shell导致watchdog启动过晚；awk三元表达式缺括号导致过密采样；停止后未等清理完成导致cpu-final低内存安全截断。已修launcher/PID gate/互斥/周期并完成设备验证；原记录保留在handoff.json.invalid_trials，不当作合格测量。不把安全截断、数值失败或unsupported记OOM。不在E0修HMX或继续CPU性能优化。

清理：181个设备文件SHA256均与本地raw或build artifacts一致；确认无测试进程后只删除 `/data/local/tmp/llama-e0-20260907`。模型与 `/data/local/tmp/llama.cpp/cl-cache`保留。见cleanup.json。原始.log另存无损.log.gz纳入Git，索引raw-log-archive.json。

未通过项已经明确记录为最终BLOCKED，不再有进行中的测试。下一会话应先读取E0/handoff.json；只有收到新的明确任务才开始后续工作。不要重新执行已完成测试来填补聊天上下文。
