# 同驻留 N=1 overlap 探针

目标为诊断 CPU routed + NPU/GPU shared 为何没有稳定收益，不更改生产内核、默认调度或前一轮标定表。默认设备 fd8657d6 已在线。

新增 tests/moe-decode-overlap-probe.cpp，通过既有 FULLMODEL_ORACLE_NO_MAIN 复用生产 full_engine / partitioned_ffn / moe_shared_execution。每层只建一份权重、三个 routed arena、一个 common shared scratch，缓存 5 个图（CPU 合并图、CPU routed、CPU/NPU/GPU shared）。每个候选预热 5 次。11 个候选在每个 block 中随机排列；完整 FFN 直接调用生产 run_into，isolated 用同一缓存图并复现复制/compute/synchronize。CPU isolated 包含 inline/async/pool 三种调用位置。完整模式的 pool 使用已有三线程池，不引入新调度实现。

数值验证：完整 FFN 与历史 capture 的逐 token 4% 检查；isolated outputs 与对应完整图的 branch outputs 逐位比较；每个计时样本结束后与本路径预热输出逐位比较。检查 ff.allocations 不再增加。所有样本保存，包括慢样本、minor/major faults。isolated total 与完整 total 的报告序列化成本不同，仅分支 compute/gather 与 wall 用于跨两者归因。

已执行：build-command.json 中 Android clang++ 增量构建；run.py deploy；run.py pilot；git diff --check；python3 -m py_compile run.py。pilot-127 四层共 88 样本通过，0 major faults，各层 5 个缓存图。型号/输入/二进制哈希见 frozen.json。CPU 仍四线程、不锁频不绑核，未验证 1–2 小核。

正在执行 run.py formal，进程 session 25442。顺序为 position 127,511,511,127，每进程 4 层，每层 40 blocks ×11 候选，合计 7040 正式样本，进程间 20 秒冷却。每个 run 有唯一 tag，禁止覆盖失败或完成证据。日志位于 evidence/formal-*.log，JSON 和 run records 由脚本拉回。远端目录 /data/local/tmp/llama-decode-overlap-20261024，测试结束后仅清理该目录。不要重跑原 10/23 实验。

限制：输入仍是 technical prefill capture 抽出的 N=1 FFN 行；每层工作集单独驻留，不模拟全模型 24 层权重/缓存竞争，也不是连续 decode。性能结论只能描述该诊断条件。CPU core begin/end 仅 isolated 分支记录，不能反推完整分支绑核情况；未测带宽计数器，不能断言带宽争用。

下一步：分析按 block 配对的三种分配、async/pool 比较；计算分支区间 overlap、墙钟超过最长分支的差额，以及同 dispatch isolated/parallel 分支时长比。完成结果、归档原始证据/构建版本、设备清理和独立结果提交。用户 TODO.md 与 paper/保持未跟踪，禁止纳入提交。
