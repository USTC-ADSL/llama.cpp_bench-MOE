# E1-B 完成收尾

目标已完成：Static_App改名E1-B，补齐旧OpenCL覆盖，最终报告`docs/Motivation_Exp/E1-B/report.md`。旧308个文件经git blob逐一核对内容完全不变；新证据`evidence/completion/`，CSV27正式行。

- 21/21初筛、3/3放置诊断、3/3共存LLM请求全部成功；27份输出均128token逐项匹配CPU，512输入/cache0；无trace溢出/安全停止。
- 初筛B1全层OpenCL最佳，median(TTFT+decode)=8601.329ms；B2=10884.553ms。load B1=45.649s比B2=14.879s慢，不能混淆排序口径。
- 主线程程序核验真实512-token请求段routed/shared/QKV；摘要与关键行`placement-check.txt`。独立探子未及时交卷被中断，没有把未完成子代理作为验证结论。
- 首次预演准备阶段没有armed；同坐标晚些成功，增加固定准备稳定等待/事件轮询后preflight2及六正式条件成功。未改任务/测量判据，失败证据保留；Bug简报`docs/bugs/2026-09-08-e1b-app-readiness.md`。
- 三对浏览器noLLM/LLM：95.104/348.742、97.996/393.645、104.035/345.882ms；计算器35/171、35/158、35/158ms。六条件pass，H1-R not supported，虽3/3变慢但未达+500ms AND +50%。三次LLM所有切回均在请求中，实际browser约5.70–5.98s。
- 主线程已目视三次LLM的browser-continued与calculator-continued截图：draft preserved、827,115。六条件状态/公式结果/PID一致/保护窗口kill0由CSV交叉检查。LLM最小MemAvailable2.386–2.555GiB、thermal0。
- 设备文件290个SHA256匹配，仅fixture.pid/ui.xml为废弃scratch；模型SHA256未变。已清理`/data/local/tmp/llama-e1b`约193MiB，确认目录不存在且无相关进程，保留模型/原部署/App数据。所有server.log无损gzip。

实际运行命令：prepare、screen-all、diagnostics；pairs首次预演失败后`python3 tests/run-e1b-completion.py --stage pairs --preflight-rep 2 --execute`exit0（PTY71133完成）。`python3 tests/analyze-static-app.py`在压缩前后均通过；五个相关Python脚本py_compile通过；最终报告链接检查通过。具体ADB命令在completion/commands.jsonl，清理在cleanup.txt，hash完整记录device-artifact-verification.json。

代码改动只涉及实验脚本：新目录/七配置/交错顺序/选择与App配对编排/准备等待/分析去除过时BLOCKED。没有修改预算或推理，也未做额外缓存清理和配额实验。沿用有效构建，不重建后端。

Git静态阶段检查点`aae66028f`；本文件与最终报告、配对证据由随后的收尾提交保存。原始help输出中的历史空白保留，代码/新文档diff检查通过。用户`AGENTS.md`、`docs/Motivation_Exp/Exp1/Exp_require.md`仍不纳入。

不需要继续实验。若未来扩展更重App/低内存，必须单独设计协议；不要把H1-R阴性当作应反复加压直至阳性。复跑不可覆盖现有run或selection，应使用独立证据目录/标签。先前handoff-01为过程记录，以此文件为准。
