# E1 补验进行中

用户要求补全Static_App旧OpenCL阻塞、直接做实验，并重命名为E1。已将docs/Motivation_Exp/Static_App整体git mv至E1；旧report/summary另存report-before-opencl.md和summary-before-opencl.csv，原始evidence保持内容不变。

主线程已完整读取旧report、protocol-before-pairs、run-static-coexist/run-app-pairs/analyze-static-app/static-app-http/static-app-web及E0 guard相关代码。探索确认没有额外独立E1协议；不采用用户未要求的Exp1低内存矩阵。新增protocol-completion.md在正式前定义B1/B3/B5为GPU对应B2/B4/B6。七配置各三次当前构建交错初筛，按median(TTFT+decode)选一个最佳再三对App AB/BA/AB。H1-R支持与否是研究结果，不强求阳性。

现有源改动：三个活跃Static_App路径更新；静态runner增加B1/B3/B5、七配置顺序及失败停止；新证据默认evidence/completion。App runner仅参数化noLLM等待与browser切回时刻，默认仍15/6；analyzer增加三GPU配置、不把正常跨后端token不同/真实App状态失败当脚本异常、移除硬编码OpenCL BLOCKED行。新增run-e1b-completion.py负责3个GPU放置诊断和winner App配对。无推理/预算代码修改。

实际已执行：

```sh
python3 tests/run-static-coexist.py --prepare --execute
python3 tests/run-static-coexist.py --screen-all --execute
python3 -m py_compile tests/run-static-coexist.py tests/run-app-pairs.py tests/analyze-static-app.py tests/run-e1b-completion.py tests/static-app-web.py
```

screen-all已exit0，21/21成功；主线程逐一验证512/128/cache0、128个token-ready等于响应、无overflow/stop_reason，全部输出与CPU逐token一致。B1 median(TTFT+decode)=8601.329ms胜出，B2=10884.553ms；B1 load median45.649s慢于B2 14.879s，必须分开陈述。三个GPU诊断也成功；主线程程序检查实际512请求到首decode区间每个MUL_MAT_ID位置，B1=72 OpenCL、B3=72CPU、B5=36CPU+36OpenCL，shared/QKV各72均OpenCL。关键行见completion/placement-check.txt，server.log已无损gzip。固定remote=/data/local/tmp/llama-e1b，fd8657d6；无quota/文件cache回收/全局drop_caches。OpenCL磁盘kernel cache off，产物push逐一hash校验。

App进行中：首次`--stage pairs --execute`在preflight-B1-noLLM-1准备草稿被拦下，未开始正式组。页面load/pageshow正常，无armed；同坐标稍后手动点击成功，怀疑入口动画就绪差。准备等待固定3s+armed轮询最多3s，协议追加记录；失败截图保留（setup-failure/manual-tap-check），手动诊断fixture PID2592已核对路径并TERM。新脚本支持--preflight-rep 2，仅当无paired目录且选择完全相同才可恢复预演，不覆盖selection。当前`python3 tests/run-e1b-completion.py --stage pairs --preflight-rep 2 --execute`正在运行，PTY 71133，不要重复启动。B1等待46s，browser目标4.2846895s；核对实际overlap。fixture为现成helper/PAGE，任务/判据不变。之后analyzer、截图核验、报告、cleanup、提交。

之后主线程做analyze-static-app.py及必要证据检查（该analyzer最后要求21screen+6pairs），生成简短E1/report.md；旧E0B BLOCKED改引用当前工程PASS，但E1不新增预算实验。完整保留没有支持共存强motivation的可能。最后停止fixture/LLM/watchdog、拉取证据核验、压缩新server日志、删除仅本任务remote目录，保留模型/App数据，Git分阶段封板。当前起点1bc58e1ec，用户AGENTS.md和Exp1/Exp_require.md禁止纳入提交。所有rename已在index，新增修改尚未完整stage。
