# E1 补验协议（正式运行前固定）

原Static_App改名E1，范围仍为无配额静态配置与真实App共存。旧报告/CSV分别保留report-before-opencl.md、summary-before-opencl.csv，旧evidence不覆盖。OpenCL旧阻塞由252aa4396修复及E0数值回归解除；E0B已有工程收尾，不再引用其旧BLOCKED阻止本实验。

原B0/B2/B4/B6不变，新增对称OpenCL定义：B1=GPUOpenCL -ngl99；B3=B1+--cpu-moe；B5=B1+--n-cpu-moe12。完整B0..B6七配置同一当前构建各三次独立启动；OpenCL磁盘kernel cache显式off，不切通用路径。顺序：0/1/2/3/4/5/6，6/5/4/3/2/1/0，2/5/0/6/1/4/3。无全局cache清理、无文件级cache回收、无预算/人为压力；独立启动不称冷文件。

固定原512真实token prompt、128输出、context2048、F16 KV、t/tb4、b512/ub128、mmap、temperature0、seed1234、cache_prompt=false、cache-ram0、fit off/no-warmup。复用E0 perf和原安全watchdog，不把PSS采样混入性能。数值门槛继承当前已验证固定库和原0.02/0.10阈值；本轮逐token对比CPU并记录，不把正常浮点差异导致的文本分歧自动等同数值失败；输出smoke人工核验，不做多任务质量宣称。

补做B1/B3/B5放置诊断，不进入排名；既有HTP/CPU放置证据保留。按请求成功、数值门槛、输出smoke筛选，用三次中位数(TTFT+server decode耗时)选一个最佳，不按最容易导致内存压力选配置。记录TTFT/p95 ITL，保留每次结果。

App仍为已安装浏览器本地文章/未提交草稿和计算器12345×67，不使用私人历史/账户。仅对这两个测试App每条件force-stop后重建任务，不清数据。3对AB/BA/AB，无LLM对照等待时长取新最佳配置load中位数向上取整。切回计算器目标请求起点后1秒；浏览器目标取min(6秒, 最佳初筛最短请求总耗时/2)，下限3秒，实际重叠必须由device monotonic与server事件核验。该规则在正式App对照前落实为固定秒数，LLM/noLLM两侧相同；若UI流程无法确保重叠，记录无效而不冒充共存阴性。

保留原判据：新增reload或切回同时增加≥500ms且≥50%、任务仍可继续=degraded；受保护状态丢失、任务中断或10秒无法恢复=fail；其它=pass。同类新增影响至少2/3次才支持H1-R。阴性结果限两个轻量App、单prompt、ADB进程优先级环境；H1-R可为not supported，实验完成与假设支持是两回事。

全部新证据在evidence/completion；正式App阶段开始前记录选择、固定等待/切回时间和App版本。无需实现缓存/预取/Bundle，也不自动扩展为低内存矩阵。

正式配对前准备修订：preflight-B1-noLLM-1在草稿准备检查被拦下（只有load/pageshow，无armed；未启动LLM）。同页同坐标稍后手动点击产生armed，坐标有效，怀疑浏览器入口动画输入就绪晚于页面rAF。准备等待从0.5s改为固定3s，并轮询armed最多3s；任务和测量窗口不变。保留失败截图/事件，新预演编号2；selection-before-pairs.json不改（B1，等待46s，浏览器目标4.2846895s）。
