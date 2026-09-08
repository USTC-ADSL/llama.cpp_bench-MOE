# E1-B 浏览器草稿准备时序

2026-09-08；仅测试自动化准备逻辑，不涉及推理或预算代码。

- 现象/触发：fd8657d6浏览器首次打开本地fixture，preflight-B1-noLLM-1有load/pageshow_painted，但在0.5s后点击、0.3s后检查时没有armed，setup抛出browser draft setup failed；未开始LLM和正式配对。
- 定位：任务标题与截图确认正确页面；同坐标稍后手动点击立即得到armed/draft_ok，排除坐标永久失效。确定的脚本缺陷是把页面rAF当成浏览器整体输入就绪、且只做单次短等待检查；入口动画吞掉早期点击是吻合证据的解释，未对浏览器内部事件派发做跟踪，不把这一内部机制写成已证实。
- 修改：tests/run-app-pairs.py仅在模型加载/测量前固定等待3s、点击一次、轮询armed最多3s；没有在推理中重建草稿或掩盖状态丢失。run-e1b-completion.py新增预演编号，只有尚无正式paired目录且selection完全相同才能恢复；预演状态失败会停止。
- 验证：`python3 -m py_compile tests/run-app-pairs.py tests/run-e1b-completion.py`通过；`python3 tests/run-e1b-completion.py --stage pairs --preflight-rep 2 --execute`设备端exit0，一次新预演及六个正式条件全部成功准备并保留任务。`python3 tests/analyze-static-app.py`验证21初筛/6配对和实际推理重叠通过。
- 证据：`docs/Motivation_Exp/E1-B/evidence/completion/preflight-B1-noLLM-1/{setup-failure,manual-tap-check}.png`、`app-events-all.jsonl`、`preflight-B1-noLLM-2/`及`paired-B1-*/`。失败预演保留，不计正式重复。
- 限制：固定3s不是跨设备/浏览器版本通用就绪保证；坐标及控件改变时仍应由准备断言拦截。未放宽任何共存判据，未修改App或系统策略。
