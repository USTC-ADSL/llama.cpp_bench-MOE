# 1K 共享权重 MoE：成本驱动分配的确认结果

已将原先约 5% 的吞吐收益提高到 **21.32%–22.65%**。两输入的独立完整对照未达到用户要求的 25%；不能用探索阶段约 28% 的结果代替。后续校准在权重加载阶段遇到设备 kernel panic，未产生可用计时数据。本报告结论仅来自重启前已经完成的正式对照。

## 对照条件与结果

设备 Redmi K80 Pro / fd8657d6；Qwen1.5-MoE-A2.7B-Chat Q4_0；实际 1024 个输入 token，单个 1024-token prefill chunk，真实 router，包括最后 token logits。权重加载、准备和 context reset 不计入推理时间，在线调度、host gather、合并和同步计入。每输入 NPU/shared/shared/NPU，2 warmups +6 timed/process，进程间空闲30秒，共8个进程、48个正式请求。统计量为每方案两个进程中位数的平均。

| 输入 | 标准 NPU / ms | 新共享方案 / ms | NPU / tokens/s | 新共享 / tokens/s | 吞吐提高 | 延迟下降 |
|---|---:|---:|---:|---:|---:|---:|
| wetland | 1294.742 | 1055.629 | 790.89 | 970.04 | **22.651%** | 18.468% |
| foundry | 1292.547 | 1065.375 | 792.23 | 961.16 | **21.323%** | 17.576% |

标准 NPU 是原生 `--mode npu`，保留不支持算子的正常 CPU 回退，未削弱基线。outer host 和 expert CPU 均沿用4线程、默认亲和性；不是仅用1–2个小核的方案。router保持默认HTP路径。两种方案运行同一二进制/库、同样的输入token。冻结记录 `frozen.json`，完整原始命令/环境在 `evidence/*.run.json`，统计在 `summary.json`。

两次 shared 进程存在约90–96ms的差异：wetland 为1009.746/1101.512ms，foundry为1017.515/1113.234ms。全部保留，没有选择最快进程；本轮只有每输入每条件两个进程，不能据此声称跨工作负载稳定保证21%或25%。此前 c8-g15 的4.95%/5.35%来自上一轮独立对照，不与本轮混合统计。

## 原来为什么只有约5%

“GPU算力是NPU的1/3”在理想完全并行条件下对应吞吐提高33.3%、延迟降低25%。它不是完整模型必然至少提高25%的下界。例如只有比例f的工作能这样并行，忽略开销时加速比为 `1 / (1-f+f/(1+1/3)) = 1/(1-f/4)`；仅靠这个GPU增量，要达到1.25倍吞吐需f至少为0.8。CPU能增加并行能力，但不能消除串行路径和同步成本。

本任务还存在以下具体差异：

- c8-g15只是按校准路由选8个轻专家给CPU、其余约15% routed量给GPU，不是测量后的三端等时分配；NPU分支此前约540ms，而GPU约160ms，明显失衡。
- 小专家的GPU成本不只是矩阵FLOPs。独立profile补入轻专家/较多专家的组合后，每个active expert的固定成本不可忽略；小batch、32行填充和launch开销使单一1:3比例不适用所有专家。
- shared expert本身是另一项工作。实测1024形状下HTP约7–9ms/层，GPU约35ms，CPU约80–100ms，因此保留HTP；最后层只有1个token，采用CPU。不能只平衡routed数量而忽略shared。
- NPU/GPU/CPU host gather分别不同；将算子计算时间配平还不够。非FFN工作和合并也留在完整请求的关键路径上。

## 实现

1. 在独立technical-1024 capture的0/12/23层，测试27组CPU/HTP/GPU及shared组合，每组2次预热+5次计时。81组都通过逐token FFN误差检查。成本拟合仅用1024形状，N=1独立处理。
2. 使用非负加性模型：分支启动成本、active expert数量、routed pairs、32行tile数量、shared expert成本。目标观测为 `backend gather + compute`。再用独立technical完整模型校准全局后端成本系数。最终CPU成本加1.5倍保守系数，系数及来源hash保存在model/runtime JSON。
3. 每层router产生真实IDs后，统计当前各expert的token数，以最小化三端预计完成时间的最大值为目标。贪心初始化，使用有预算的单专家移动和两专家交换；每次最多2048个增量比较。更新只操作三条分支的标量负载，不再重复构造完整feature向量或候选图。它是确定性的有界启发式，不保证全局最优。
4. 三端只预绑定同一物理权重的alias。改变expert owner不搬权重、不重新pack；执行按专家构建2D图。相同请求预热后图缓存固定。不同新请求可能创建新的shape图；本轮不证明任意请求流的图元数据缓存有界。
5. 保留generic调度和scalar合并默认路径。正式候选选择`shared_additive`、有序NEON FMA合并。NEON与scalar的完整模型logits逐位相同；未用两次探索总时间差独立宣称NEON加速幅度。

核心文件为 `tests/moe-shared-cost-plan.py`、`tests/moe-route-scheduler.h`、`tests/moe-shared-weight.h`、`tests/fullmodel-oracle.cpp`。实现提交 `5cab4e301`。

## 分阶段时间与内存

以下是正式请求均值，单位ms。后端计算并行重叠，不能相加后与总耗时比较。

| 项目 | wetland | foundry |
|---|---:|---:|
| CPU分支计算 | 415.423 | 412.216 |
| HTP分支计算 | 462.423 | 461.045 |
| GPU分支计算 | 450.841 | 451.892 |
| CPU / HTP / GPU gather | 12.620 / 64.699 / 38.747 | 14.299 / 68.359 / 39.462 |
| 并行expert阶段墙钟，包含backend gather | 542.376 | 548.509 |
| 分支执行前准备，包含调度 | 5.006 | 5.978 |
| 其中在线调度 | 1.825 | 2.204 |
| host合并 | 40.983 | 43.718 |
| FFN以外的请求时间 | 456.229 | 468.519 |

原通用搜索约127ms/请求，轻量搜索约2ms。routed pair分配约CPU9.3%、HTP55.5%、GPU35%；这不是expert个数比例，也未把shared工作算入分母。仍可看到HTP gather使它成为主要尾部。

全部routed/shared专家的唯一payload和实际weight allocation均为 **7,474,348,032 B（约6.961 GiB）**，未生成三份backend权重。三端activation arena分别为137,363,456 / 231,768,064 / 137,363,456 B，总计 **483.031 MiB**，与此前共享方案相同。每请求缓存图累计70个，正式重复中不增长。三端全绑定会增加descriptor/映射/图元数据；这不是“总内存零开销”的证明。VmRSS不完整反映DMA分配，不能拿它直接当模型总物理内存。

全绑定还暴露了OpenCL重复import整个DMA backing导致fd耗尽的问题，已通过按backing身份共享import修复。640个slice绑定只增加1个fd，释放后恢复；CPU/HTP/GPU算子回归通过。提交 `2898132f6`，简报 `docs/bugs/2026-10-21-shared-weight-import-fd-exhaustion.md`。

## 正确性及未采用尝试

- 每个正式shared进程内部logits逐位一致；相同方案跨进程也逐位一致。NPU/shared的top1及top10集合一致，最终logits NRMSE为2.726%/1.331%，cosine为0.999634/0.999911。该检查不是困惑度或完整生成质量评测。
- scheduler设备测试通过200组availability、不同N、预算及确定性检查；通用路径既有小规模穷举测试也通过。所有22个冻结文件hash在回退后重新核验通过。
- GPU router探索1338ms、host1探索1124ms，均未采用。全程pin7探索1554ms，callback内部pin7探索1215ms，也均未采用；未把慢NPU设置引入对照。
- 删除GPU shared transpose中未使用input view对象的实验通过9项算子回归，但完整模型新校准在加载第14层附近断连，exit255、无正式样本。补丁保存在`experiments/view-elision.patch`，没有合入当前代码，原库已恢复；不能声称这个补丁提高了端到端速度。

设备自动恢复后boot ID由`f8c81ca4-027e-4197-bfc3-12433b0d31fc`变为`f3665dc6-f63f-4aec-8777-8fd2ef58f997`，系统boot reason包含`kernel_panic`。ramoops含大量乱码/控制字节，无法可靠确定具体根因；加载日志停在计算图执行之前，也不足以将重启归因于新计算路径。原始ramoops、恢复状态、加载日志已保留。正式V1对照早于这次重启，完整且可复核。

## 后续边界

25%目标仍未完成。进一步工作应先定位加载阶段kernel panic，再在默认亲和性下用独立输入修正分支成本、减少HTP gather或GPU逐expert启动开销，随后重新冻结候选并做独立对照。GPU目前共享CPU4x8路径只支持2D MUL_MAT；真正将多expert合并成少量dispatch需要专用batched/MUL_MAT_ID共享kernel及数值回归，不能通过换一个比例免费获得。仅1–2小核、4K长度和跨请求长期图缓存内存也未在本轮完成验证。
