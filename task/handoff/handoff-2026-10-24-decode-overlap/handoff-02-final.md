# N=1 overlap 实测交接

## 完成范围

实现提交 66f56015a（feat/shared-weight-layout）。新增独立 tests/moe-decode-overlap-probe.cpp，生产代码/内核/代价模型未更改。使用既有 CPU4x8 single-layout、common scratch、async/pool 执行路径；新增诊断不会修改默认行为。

Redmi K80 Pro fd8657d6 上 pilot 88 样本及 formal 7040 样本均完成。formal driver session 25442 已退出 0，无待运行实验。四进程，每进程 4 层，每层 40 block ×11 候选；输入来源 technical capture 行 127/511，进程顺序127/511/511/127。数值检查通过、所有样本输出逐位稳定、5个缓存图预热后无新分配、全部0 major faults。minor faults、慢样本全部保留。

## 结果

16个上下文的 FFN 总时间中位数：全CPU687.40 µs；CPU+NPU async713.15、pool716.24；CPU+GPU async883.96、pool861.42。以各上下文中位数之和比较吞吐，四种split分别-3.82/-4.22/-22.36/-20.34%，全部0/16上下文超过全CPU。详细每进程结果、paired比值、区间和限定见results.md及summary.json。

独立branch work：CPU routed约318 µs，CPU shared340 µs，NPU shared618 µs，GPU shared813 µs。NPU并行时CPU routed约560 µs，为同dispatch孤立分支约1.76倍；NPU自身约660 µs。GPU并行CPU routed约334–344 µs，GPU约822–830 µs。区间重叠真实存在，非串行执行；这些是主机compute+synchronize区间，不是硬件kernel trace。

isolated CPU派发额外wall由async47.68 µs降为pool22.71 µs；GPU split配对总时间约降2.47%，NPU无稳定改善。不能只归因派发，也没有证据唯一归因DDR竞争。真实decode全模型缓存工作集不同，本轮没有端到端复测，不覆盖10/23结果。

## 修改与命令

- tests/moe-decode-overlap-probe.cpp：一个FFN驻留实例、11候选交错测量及数值/分支输出契约。
- 当前任务目录run.py：独立远端目录部署、哈希冻结、pilot与formal管理。
- analyze.py：严格校验报告哈希/形状/完整性，按block配对、按上下文汇总。
- finish.py：验证本地原始证据、受限清理、归档逐文件SHA256核验。
- build-command.json：实际Android clang++增量编译命令；build日志在evidence/build.log。

实际运行：run.py deploy；run.py pilot；run.py formal；analyze.py；python3 -m py_compile run.py analyze.py finish.py；git diff --check。归档与清理以finish.py输出、archive-record.json、cleanup.json为准。原始报告evidence/formal-0-127.json、formal-1-511.json、formal-2-511.json、formal-3-127.json及对应.log/.run.json，均不可覆盖。

## 状态与边界

当前测试目录 /data/local/tmp/llama-decode-overlap-20261024；仅此目录由finish.py清理，模型不动。最终归档 /mnt/sda1/pzw/moe/experiment-artifacts/decode-overlap-20261024.tar.gz，包含测试binary、实际动态库、输入、源码关键路径和原始报告。归档时任务清单可能仍处收尾状态，最终完成状态以Git中tasklist.md及archive-record.json为准。

TODO.md与paper/仍为用户未跟踪内容，未纳入提交。没有修改生产dispatch默认值、N=1代价表、内存布局或计算内核。CPU仍4线程，1–2小核目标与2K均未验证。下一步若继续：优先拆N=1 shared后端graph提交/同步/kernel时间，再调查CPU与NPU并发慢化；不要直接根据单层结果声称端到端加速。
