# 并发失败保留，设备重新连接后恢复

原首个 1K NPU 测试 eval-n1024-01-wetland-npu 与另一套 /data/local/tmp/moe-hetero-lengths/ 的 1025-token custom 测试并发（PID 18692 / 18686），五次监控均见两个进程，随后设备重启。runner exit255，外层 exit1，未生成有效计时，已在 runs.json 中保留并排除；详见 interference.json、evidence/*memory.jsonl、recovery-*。用户先要求等待另一个测试结束，后明确继续；10月28日首次连接检查无设备，用户重新连接后确认 fd 在线且无推理进程。

全部宿主/设备 binary 与库 SHA256、源码/输入/成本模型 SHA256、GGUF stat 均核对原冻结记录不变。当前 boot 1ba761f6-5b18-403a-87d1-fcf146ec8fa0 记录在 execution-epoch.json，原 frozen.json 不改。没有重新编译、没有改模型。恢复使用 resumed 标签，禁止覆盖原失败 tag。

任务 run.py 增加 --cohort，并在诊断快照后、推理启动前再次检查空闲；监控识别多个推理进程后，校验 /proc/PID/cmdline，只向本任务目录下的 fullmodel-oracle 发 TERM，不干预外部测试。通过 mock 验证只终止自身、PID identity 变化时不终止、snapshot 后复查顺序；未为验证启动设备负载。这解决上一轮“启动前检查与实际启动有时间窗口、发现并发只记录却继续运行”的监控问题，计算实现和测量参数不变。

当前运行：`python3 task/handoff/handoff-2026-10-27-calibration-regression/run.py compare --tokens 1024 --cohort resumed > task/handoff/handoff-2026-10-27-calibration-regression/evidence/compare-1024-resumed.log 2>&1`，tool session 63924。以 runs.json/日志为最终状态，不重复启动。结束后分析1K，然后 `--tokens 1536 --cohort resumed`；如1K无回退，补2K new/NPU同条件对照与阶段增长分析。

已有2K离线分析保存在 prior-2k-paging.json。修正手算：wetland 零fault子集加速比是 1.14137×（不是前交接的1.138×），foundry为1.14207×。这是描述性子集，不是消除换页的因果实验。当前没有有效的新1K/1.5K结果，不声称无回退。下一步继续完成原任务，最后归档和清理仅本任务设备目录。用户 TODO.md/paper/results 不动。
