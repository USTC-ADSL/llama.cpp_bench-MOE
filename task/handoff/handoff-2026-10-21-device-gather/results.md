# CPU4x8 单份权重：输入 gather 融合结果

设备 Redmi K80 Pro / fd8657d6，Qwen1.5-MoE-A2.7B Q4_0，1K完整prefill。实现提交 **62876447f**。本轮比较基线是当前三端共享权重、CPU/GPU/NPU异构分配的 **host gather**，不是纯NPU，也不是旧的三份权重布局。

## 完整模型结果

使用原 runtime-fast-cpu150 分配模型和真实动态路由、NEON单线程合并、async分支、CPU expert 4线程/outer 4线程。候选与基线使用完全相同的executable和库，只切换输入准备方式。每输入2个host进程+2个fused进程，每进程2warmup+6timed；wetland ABBA、foundry BAAB，进程间隔30s，固定1024-token单chunk，包含完整模型与最后token logits，权重已驻留。设备boot ID全程不变，无调频或affinity调整。

主统计量先取每进程6次请求的中位数，再平均同一方案的两个进程。吞吐提升=host延迟/fused延迟−1。

| 输入 | Host gather | DMA融合gather | 吞吐提升 | 全部12样本总体中位数：host→fused | 按总体中位数算吞吐提升 |
|---|---:|---:|---:|---:|---:|
| wetland | 1070.664 ms | 1023.445 ms | **+4.614%** | 1062.934→1026.186 ms | +3.581% |
| foundry | 1083.614 ms | 1030.681 ms | **+5.136%** | 1089.793→1028.562 ms | +5.953% |

进程中位数：wetland host=[1092.757,1048.571]、fused=[1015.059,1031.831]；foundry host=[1077.434,1089.793]、fused=[1001.573,1059.789] ms。原始48样本全部保留。

全部样本均值：wetland1104.245→1032.595ms，foundry1238.943→1037.045ms。这一均值受host慢样本影响显著，不能拿它作为更大的稳定收益宣传。foundry-host-1包含2464.113ms和1645.723ms请求，前者CPU专家分支累计1514ms，同时FFN外时间约779ms，其他后端也变慢；没有通过硬件counter证明其具体成因，不能全部归因于gather。现有样本支持本设备、这两个输入上约数个百分点的改善，不构成跨设备或长时间稳定收益保证。

本轮48次计时中，同一输入的各方案内外 logits bitwise 相同，跨本轮与修改内核前的host基线hash也一致。token IDs、实际路由counts、owner/shared分配均一致，缓存图70个且重复请求不增长。

## 改了什么

原来CPU按每个HTP专家的token列表复制输入，在DDR形成多个连续F32矩阵；gate/up再把这些连续矩阵搬到VTCM并转FP16 tile。新路径上传一次完整输入及小型I32 token-row map，HTP将按索引读取的行直接DMA到既有VTCM双buffer，再使用原FP16 tile转换与HMX乘法。小M按列、较大M按32行tile并行。

公开计算图仍是GET_ROWS→普通MUL_MAT。Hexagon内部识别安全子图后跳过GET_ROWS执行，给MUL_MAT添加内部MM_ROWS标志和src[9]索引描述符，logical M仍按原gathered输出计算。权重继续使用同一份CPU4x8物理布局，未重排或增加GPU/NPU权重副本；CPU、GPU、NPU共享权重接口不变。

源输入和indices必须retained且连续、非view；gather的所有consumer必须支持普通CPU4x8 HMX matmul；可见gather、其他consumer、插入计算、单行HVX路径等拒绝融合。MoE层在nt1、HTP只有shared专家、或上传全输入不比host gather少的稀疏分配时回退host。DSP在读取前检查索引范围。

## 内存和复制量

| 项目 | Host | Fused |
|---|---:|---:|
| 唯一权重payload/allocation | 7,474,348,032 B | 相同 |
| CPU activation arena | 137,363,456 B | 相同 |
| HTP activation arena | 231,768,064 B | 231,784,448 B（+16 KiB） |
| GPU activation arena | 137,363,456 B | 相同 |
| wetland HTP主机输入复制/请求 | 592.242 MiB | 184.000 MiB |
| foundry HTP主机输入复制/请求 | 595.359 MiB | 184.000 MiB |
| HTP索引写入/请求 | 0 | 209,020 / 210,616 B |

HTP输入复制分别减少408.242和411.359 MiB/请求。以上是代码逻辑复制字节，**不是硬件实测DRAM流量**。VTCM复用原scratch，不新增大块持久buffer；公开图中的GET_ROWS临时tensor仍占arena预约，尚未消除其容量预留。另有图描述符/索引元数据，未测得精确增量，不声称总RSS零增长。

## 单层与失败尝试

真实捕获层0/12各12组AB/BA，同进程、同权重、同输入：

| 层 | Host总耗时 | DMA融合总耗时 | 耗时降低 | HTP计算host→fused | HTP主机gather host→fused |
|---|---:|---:|---:|---:|---:|
| 0（1024 tokens） | 22.568 ms | 21.452 ms | 4.946% | 18.491→19.064 ms | 2.302→0.651 ms |
| 12（1024 tokens） | 22.677 ms | 21.812 ms | 3.817% | 18.539→19.140 ms | 2.348→0.687 ms |

收益来自输入准备/物化开销减少；HTP计算计时包含indexed准备，本身略增约3.1%/3.2%。不宣称HMX乘法算力提高，也不将这两个整层结果当作所有算子都满足5%预算的独立验证。层23仅1token，保持host回退，不声明收益。输出合并沿用已有指针表和fused weighted reduction，没有大块拼接拷贝可移除。

保留的失败/中间方案：

- 显式HTP GET_ROWS：减少相同数量host复制，但增加独立设备gather。完整1K吞吐wetland−4.151%、foundry+0.426%，未证明稳定收益；见summary.json与原8进程证据。
- 融合直接HVX读DDR：正确，但层0/12总耗时增加约4–5%，HTPcompute多约2.6ms；未采用。源代码和库保存在evidence/fused-direct-freeze。
- 最终采用DMA→既有VTCM scratch→原tile转换，保留两个开关以独立回退。

## 使用、验证和回退

完整runner在原参数上增加：

```text
--shared-weights cpu4x8 --shared-gather htp --shared-fused-gather
```

对应环境变量MOE_SHARED_GATHER=htp、GGML_HEXAGON_FUSE_CPU4X8_GATHER=1。默认host不变；移除这两个gather参数即回退原路径。仅关闭fusion而保留htp gather则执行显式GET_ROWS。开关在进程/图创建前设置；不要在同一已缓存图上动态切换融合开关。

原host/DSP库及源文件成对备份在evidence/kernel-backup，manifest.json含hash。若需要恢复二进制，应同时恢复libggml-hexagon.so及libggml-htp-v79.so，不混用启用新flag的host和旧DSP库。归档位置见archive-record.json。

设备验证：CPU4x8 175个rewrite guard/cache case；native MMID HMX/NX-off/HVX各20 case；最终MoE probe 28 cases、172 timed samples均逐位一致，含1/2/3/17/32/33边界、全60专家、稀疏临界、shared-only和缓存更新；真实参考FFN每token NRMSE<4%。日志确认新INDEXED_ROWS确实执行，且不安全guard case均未融合。

复现入口：regress.py、probe.py、confirm_fused.py。各脚本拒绝覆盖已有证据，重跑请使用独立cohort目录/tag。统计复算：`python3 task/handoff/handoff-2026-10-21-device-gather/analyze.py --prefix dma- --candidate fused`。构建命令和历史记录见handoff-01.md；fused-frozen.json和各run.json记录精确binary/library/source hashes及命令。

本轮只验证1K prefill，未验证4K和多HTP分区。CPU仍采用原4+4线程设置，未完成长期仅用1–2小核的目标。
