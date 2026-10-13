# decode轻量诊断完成

需求：稍微检查真实mixed decode相对全NPU同host劣势。只测各8步，不继续扫描或优化。结论及数据解释见report.md。

增量修改`dense-af-decode.cpp`：通过后端registry解析现成HTP timing接口；prefill完成后到被测decode完成后获取计数器delta，启用时额外分离llama_decode调用和sample/logits等待。`run-real-decode-android.py`新增`--htp-diagnostic`，同时设PROFILE1/TIMING1。无scheduler/kernel/融合修改。

构建通过；独立部署`/data/local/tmp/dense-af-decode-diagnostic`（8文件hash匹配），mixed及NPU-host1401各8步通过，真实输出9token与彼此及前轮正式前9token完全一致。检查时钟、KV增长、时间加和与恢复通过。临时prompt/token/text已拉回并清理；恢复原CPU min/max、屏幕超时和wake lock，原亮屏保留。

实测每token：mixed墙钟76.013ms、DSP batch46.578ms、host read55.747ms、host非read20.266ms、36batches；全NPU60.558/56.303/58.509/2.049ms、1batch。DSP少9.725ms，但host非read多18.217ms；CPU Attention与主机调度交接路径是主要需要关注的范围。不能将DSP时间与host等待相加；不能把残差叫纯同步或copy成本；未精确分离CPU Attention、scheduler、响应间隔和缓存因素。mixed第8步90ms，不能用短诊断更新前轮100步正式吞吐结果。

本轮没有能量测量、没有修复/优化实现、没有进一步待跑实验。若用户后续要求更细归因，可再独立测CPU split计算时间；本轮到这里即可。详细复现命令见report.md。Git起点a05af0039，阶段提交由git log查询。用户TODO.md和旧三个power-trace产物保持不动。
