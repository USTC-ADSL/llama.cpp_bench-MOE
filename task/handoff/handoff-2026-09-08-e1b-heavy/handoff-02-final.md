# E1-B 较重任务补验收尾

任务完成。用户后续明确要求减少实验量，因此只完成原定image/PDF各三对，没有追加实验、重跑或额外交叉验证。最终报告docs/Motivation_Exp/E1-B/report-heavy.md，原report.md只加补验链接/限定结论，原初筛和轻量证据不变。

- 12/12条件最终PASS；6/6 LLM请求成功，512输入/128输出/cache0，token逐项匹配原CPU，trace无溢出，无stop_reason，thermal0。
- 六对均变慢，但HWUI帧延迟增量204～410ms，未达500ms AND 50%门槛；12次状态/继续操作保留、返回PID不变。H1-R仍not supported，限两个任务。
- PDF后台约257MiB、image后台110MiB；image独立导出15样本峰值1.045GiB，不是持续高内存场景。没有可用原轻量App PSS比较。
- 图片初始continuation_ok=false为异步导出早查。正式轮次完成后只读同次PID/路径完成日志与文件，三个LLM均成功，证据export-followup；原始false保留，analyzer分别输出original_continuation_ok和最终continuation_ok。最早noLLM日志后续滚出，采用其原始成功记录。没有重新导出。
- 导出2.25～2.285s→2.803～2.909s，与推理实际重叠1.592～1.737s。不能声称完整导出或峰值均与推理重叠。
- Python分析器重新计算有效HWUI帧、核对返回和保存时刻/原始token/安全记录；无效帧边界的纯函数检查通过，12个正式时延不变。计时不是完整屏幕呈现，recovery_upper含UI工具开销。

实际命令：`python3 tests/run-e1b-heavy.py --stage pairs --task image --execute`和`--task pdf`，各只跑一次；`python3 tests/analyze-e1b-heavy.py --collect-exports`（只读设备，不重跑任务）；`python3 tests/analyze-e1b-heavy.py`exit0。正式harness检查点e29d1ee9b，未修改后端/runtime或重新构建；继承原固定OpenCL库。

清理：`python3 task/handoff/handoff-2026-09-08-e1b-heavy/cleanup.py --execute`验证143个设备文件、9张实验导出与主机文件SHA256后，精确删除对应MediaStore记录和图片、两个素材、专用目录/data/local/tmp/llama-e1b-heavy与/sdcard/Download/E1B-heavy，以及多看external_books/E1B-raster-160pages.pdf。未删除DCIM/Creative目录或私人文件，未清App数据/缓存/策略。模型SHA256仍8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473；运行前无遗留llama-server/helper进程。第一次cleanup在遍历空device_path时TypeError，发生在删除前；改为空字符串后成功，不曾误删。

设备验证在evidence/heavy/device-artifact-verification.json，清理记录cleanup.json；缺少本地副本的三个测试scratch文件补存device-archive。server.log无损gzip，验证清单记录的是解压后哈希；源素材/完整导出JPG本地主机保留但Git-ignore，可恢复设备测试产物。源素材由固定seed生成器重建。

最终提交包含实验分析器、补验报告、协议的事后取证说明及新证据。用户AGENTS.md修改和docs/Motivation_Exp/Exp1/Exp_require.md不纳入、不回滚。到此结束，无待跑实验。
