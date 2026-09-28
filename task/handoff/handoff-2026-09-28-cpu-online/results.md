# CPU布局与分组执行

设备fd8657d6，Qwen1.5-MoE Q4_0，单请求128-token完整Transformer首次prefill；已加载模型、空Expert转换缓存、真实在线router，计时包含决策/准备/同步/聚合。继续使用K8 NPU预取、mixed graph、batch pack和source views。所有性能运行都是新进程，不使用目标prompt预跑或分配重放；加载、输出权重恢复、源映射建立另报。实现提交2b4149d55。

## 设计与取舍

- 原CPU路径直接别名canonical DMA，使用MUL_MAT_ID，并不使用CPU_REPACK；output层的CPU_REPACK不代表Expert布局。
- `repack-id`：8-Expert工作区，整组gate/up/down重排后用ID算子；三矩阵槽序一致，末组合法padding无路由，目标+单矩阵staging为49.5MiB。
- `repack-mm`：单Expert三矩阵工作区4.640625MiB，将该Expert全部token聚合后用普通MUL_MAT。当前CPU_REPACK的ID内核逐row GEMV，普通MM能利用多row内核；纯计算更快不等于首次执行更快。
- `canonical-mm`：按Expert分组直接借用源权重的连续2D切片，不复制、不重排权重。gate/up/down采用同组token，中间结果保持该Expert局部顺序，最后写回原token/top-k槽位并按原顺序聚合。当前实现每Expert构图，并没有声称实现跨Expert fused grouped-GEMM或完全消除调度开销。
- 可选`GGML_CPU_REPACK_DIRECT`：NEON读取四行，直接交错写入并融合quant XOR；scale和packed格式不变。此轮设备验证4×8格式逐字节等价，4×4分支仅静态核验，非NEON回退原实现。

常驻源保持一份，所有CPU workspace跨层复用，初始没有转换好的Expert。最终采用canonical-MM候选，CPU新增权重数据容量为0，复用原CPU激活池。终层单token FFN继续canonical-ID。本轮不改全局默认，保留原路径和所有消融开关。

## 为什么没有直接启用CPU_REPACK

真实layer0的两个Expert各处理指定token，独立FP64矩阵累加/float SwiGLU参考通过。`cpu-matrix-direct.json`是同一进程的完整FFN准备+计算微测中位，单位ms；有独立warmup，不能当成整模型首次prefill，也不能直接拿其交叉点替代并发下的决策。

| 每Expert token | canonical分组MM | 直接NEON repack＋分组MM |
|---:|---:|---:|
|4|0.430|0.832|
|8|0.817|0.997|
|16|1.707|1.236|
|32|3.550|1.858|
|64|7.123|3.152|

常规repack的两个Expert准备约1.6–2.1ms，直接NEON约0.5–0.7ms。但开发输入分给CPU的Expert大多只有1–9 token；完整并发中标量repack-ID / repack-MM首次prefill为1593.48/1196.75ms，直接NEON repack-MM仍为761.39ms，全部慢于当轮canonical-ID控制。完整层累计准备分别1035.47/895.65/399.65ms，说明转换显著优化后仍会吞掉这类小工作量的计算收益。不能将“CPU未使用repack”直接等同于“改成repack就更快”。负结果全部保留。

canonical-MM开发两轮CPU分支累计348.32/344.59ms，对照canonical-ID382.78/387.30ms；对应原分配模型完整prefill556.85/550.04ms，对照566.46/598.89ms。非MoE部分有明显跨进程波动，因此独立确认预先规定四轮顺序平衡，不选择最好的一次。

## 独立确认

在读取两份新输入结果前冻结`confirmation-plan.json`。CPU成本校正对两条路径采用相同方法：开发b/c共46层的实际CPU分支总时间/旧预测值的中位比率；canonical-ID因子0.9102124，canonical-MM因子0.7901807，仅更新CPU scale/overhead，NPU/GPU系数和候选集合不变。实际在线分配仍按当前router计数选择低token CPU tail，shared交GPU、其余routed交NPU。

| 新输入 | 原生NPU | 原在线异构 | 仅校准的ID异构 | 分组MM＋对应校准 | 分组较原在线吞吐 | 分组较校准ID吞吐 | 分组较原生吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
|reservoir|549.23|536.89|533.57|528.60|+1.57%|+0.94%|+3.90%|
|clockwork|552.65|539.75|540.53|513.98|+5.01%|+5.17%|+7.52%|

单位ms，均为四轮中位；各路径各占一次执行顺序位置，不删除样本。分组reservoir范围521.50–575.24ms，clockwork511.12–522.43ms。reservoir新增收益较小，且存在慢样本；这是两份输入上的中位收益证据，不是逐请求保证或普遍统计显著性证明。相对原生的全部收益属于整套异构系统，不能全归于本轮CPU改动。

MoE callback累计中位：原ID→校准ID→分组分别为reservoir433.51→428.58→417.95ms，clockwork433.52→436.93→411.74ms。逐层max累计也下降，但不能用CPU/NPU跨层累计之和的max替代它。首次分组运行分配：reservoir21层tail32、2层tail36；clockwork15层tail32、8层tail36；终层4个真实路由配对全部CPU。具体60-Expert owner与真实counts见每个结果的layers，不使用这些确认输入继续调参。

## 正确性、内存与边界

矩阵微测覆盖2个真实Expert、1/4/8/16/32/64 token完整FFN；canonical-ID/repack-ID/repack-MM/canonical-MM均通过独立反量化参考。direct布局对三矩阵及未路由padding逐字节验证为4×8。开发完整模型对23个非终层、相同输入和真实路由重复原CPU参考，最大relativeL2为7.32322e-5；该验证运行明确不计吞吐。

确认输入的23层同输入/路由参考亦全部通过：reservoir最大relativeL2为7.75108e-5，clockwork为1.85411e-4（0.01854%）；追加参考计算未改变其原确认运行的完整logits，逐字节一致。见`correctness-summary.json`。这验证的是本轮CPU执行路径，不能替代跨后端的通用质量评估。

CPU内核改变会积累浮点差异并可能改变后续路由，因此不宣称完整logits逐位相同。首次分组确认相对原生logits relativeL2为reservoir4.133%、clockwork3.777%，top1均相同；原ID对原生本来已有3.924%/3.192%差异。没有做困惑度或任务级质量评估。

canonical-MM不增加权重副本，局部额外容量账本仍146.515625MiB，含74.25MiB NPU双槽。它不是完整物理RAM上界；驱动映射、原生dense/KV、代码/图元数据等边界同前阶段。CPU_REPACK控制的额外工作区单独报告，未保留多层/完整CPU packed模型。

## 复现入口与后续

`tests/run-online-weights.py --cpu-experts canonical-mm`，配本目录`cost-canonical-mm.json`及与plan相同的在线/NPU预取选项。准确设备命令、源文件/二进制/输入/输出SHA256在各`*.run.json`；汇总命令`python3 tests/analyze-cpu-online.py task/handoff/handoff-2026-09-28-cpu-online`。

本轮先完成128-token CPU优化，尚未验证其他输入长度与decode。更高CPU负载可进一步研究按实际token数选择canonical/packed并估计并发转换代价，但当前未实现自动阈值，也不依据微测直接设定16-token阈值。其次可复用Expert分组描述/图，减少逐Expert构图与调度开销；收益需新开发/确认输入验证。
