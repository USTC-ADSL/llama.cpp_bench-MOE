# V1：分配优化已确认，25% 目标未达

分支 `feat/shared-weight-layout`。实现检查点 `5cab4e301`；独立 OpenCL DMA 导入资源修复 `2898132f6`。用户 `TODO.md` 未修改。设备 fd8657d6，模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。本阶段仍用 4 个 expert CPU 线程及默认 CPU 亲和性，不满足仅 1–2 小核的长期限制。

## 已完成

- 独立 technical-1024 capture 的 0/12/23 层，最终 27 种分配，每种 2 warmups +5 timed。81 个候选全部通过逐 token 4% FFN 误差检查。0、12 层为 N=1024，23 层仅 N=1。
- 新增 `tests/moe-shared-cost-plan.py`，非负成本模型拟合分支启动、专家个数、routed pairs、32 行 tile、shared expert 成本。`model-complete.json` 为最终模型。训练不是正式测试输入；用独立 technical 完整模型修正分支上下文成本。
- GPU profile 补入轻专家、多专家组合后，明确需要 per-expert 固定成本。shared expert 在 HTP 比 GPU 更快，因此维持 HTP；末层 N=1 全 CPU。
- `moe-route-scheduler.h` 新增 opt-in `shared_additive`。当前真实路由计数驱动、整专家分配、min-max 三分支完成时间、贪心初始解及有限 move/swap。每次最多 2048 个便宜的增量比较，无全局最优保证。generic 路径默认保留。
- 新增调度元数据与 opt-in NEON 有序 FMA 合并；scalar 默认与备份保留。完整模型 scalar/NEON logits 逐位一致。设备 `test-moe-route-scheduler` 通过，包含 200 个受限 residency/不同 N 的确定性及预算测试；未将 generic 路径的小规模穷举最优误称为 fast 路径最优保证。
- 动态全模型绑定暴露 DMA fd 重复导入问题，已修复并独立提交，见 `docs/bugs/2026-10-21-shared-weight-import-fd-exhaustion.md`。640 个 slice 绑定只增加 1 个 fd，三后端算子回归通过。无权重复制，无 kernel 算术改动。

## 正式 V1 结果

`frozen.json` 在运行前冻结 22 个文件。`confirm.py` 执行两输入各 NPU/shared/shared/NPU，2 warmups +6 timed/process，30 秒空闲，共 48 次正式请求。`analyze.py` 与 `summary.json` 保留完整统计。完整 1024-token prefill，包括最后 token logits、真实 router、在线调度；不含模型/权重加载。

| 输入 | NPU / ms | 新分配 / ms | 吞吐提升 |
|---|---:|---:|---:|
| wetland | 1294.742 | 1055.629 | 22.651% |
| foundry | 1292.547 | 1065.375 | 21.323% |

统计量是两个进程中位数的平均，不是取最快请求。两输入的 shared-a 分别约 1010/1018 ms，shared-b 分别约 1102/1113 ms，不能丢弃慢进程。相同条件跨进程 logits 逐位一致；NPU/shared top1 相同且 top10 集合一致，NRMSE 为 2.726%/1.331%。单份权重别名及固定请求预热后图缓存数量检查通过。

三端顺序始终 CPU/HTP/GPU。wetland 分支计算均值 415/462/451 ms，backend gather 13/65/39 ms；expert wall 542 ms，host combine 41 ms，外部非 FFN 456 ms，调度 1.82 ms。调度包含于 gather；backend gather 已包含于 expert wall；不能把重叠分支时间相加。慢进程的 NPU/GPU compute 基本不变，主机准备/合并/外部路径明显变慢。目前只是怀疑协调线程迁移/资源竞争，未确证具体根因。

## 探索与失败保留

`exploration-summary.json` 包含全部已完成探索：最初静态模型 1157 ms、改进静态 1109 ms、host1 1124 ms、GPU router 1338 ms、通用动态搜索 1219 ms（搜索约127 ms）、轻量 scalar 1079 ms、轻量 NEON 1030 ms、CPU 成本 penalty1.5 1006 ms。探索最好值不能替代正式 V1 的 21–23%。未完成加载的 `wetland-dynamic-scalar-explore` 是 fd 耗尽证据，不是设备重启。

## 命令及后续

最终 profile：`python3 task/handoff/handoff-2026-10-21-cost-placement/profile.py --tag technical-profile-complete --extended --gpu-shapes`。

模型/计划使用 `tests/moe-shared-cost-plan.py --profile .../evidence/technical-profile-complete.json --context-report .../evidence/technical-context-calibration.json --cpu-penalty 1.5 --runtime-output .../runtime-fast-cpu150.json`，calibration-plan 为此前 wetland/foundry-c8-g15.json.gz。完整命令可由 frozen/model source 字段及 run manifest 恢复；执行器命令逐个在 `evidence/*.run.json`。

编译：NDK r27d 的 `aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle`。scheduler test 同 compiler、`-std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-route-scheduler.cpp`，设备 LD_LIBRARY_PATH 指向本任务目录。Python py_compile 和 git diff --check 已通过。现有 deprecated API 编译 warning，无新错误。

下一阶段正在诊断 opt-in `--main-cpu 7`，仅在预热后绑定协调线程，新生成的 branch worker 显式恢复原 CPU mask；NPU 必须同样设置。专家线程数保持4，不能误称小核方案。V1 二进制已另存 `evidence/fullmodel-oracle-frozen-v1`；原始 kernel/库备份也在 evidence。V1 freeze 对应旧源码版本 5cab4e301；后续修改后分析不能要求当前工作区仍等于 V1，应该按原始 run manifest/归档哈希核验。

设备 `/data/local/tmp/llama-cost-placement-20261021` 仍需保留到后续验证及归档结束。不要同时启动第二个设备负载。最终仍需：确认诊断是否有效、若有效冻结新候选并独立对照、如实报告未达标数据、归档并清理设备目录、提交实验结果。
