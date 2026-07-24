# MoE 后端内存与 Expert I/O 分析（组会汇报）

## 1. 本周问题

在 `fd8657d6`（16 GB RAM、V79 HTP）上回答两个问题：

1. CPU、GPU/OpenCL、HTP/FastRPC 各自及混合使用时，实际可稳定驻留多少 buffer？
2. CPU Expert 缓存 miss 时，`O_DIRECT`、冷 mmap、热 mmap 哪种加载方式更合适？

## 2. 核心结论

- 不触发明显 ZRAM 时，三后端可用总量约 **10.0–10.6 GiB**，而不是宣传口径中的
  “16 GB + 16 GB”。
- GPU 有明确硬限制：OpenCL global memory 约 **7.38 GiB**，单个 `cl_mem` 最大
  **1 GiB**。
- HTP 的瓶颈不是 host 侧 rpcmem 总申请量，而是 DSP 同时可见的 VA。固定/pinned
  映射约 **3.7–4.0 GiB**；delayed/windowed 映射可让 host 侧维护约 **9.5 GiB**，
  但 DSP 每次仍只映射当前窗口。
- CPU 借助 ZRAM 可以申请到 **22.875 GiB**，但一次完整扫描约 **8 分 32 秒**，
  不具备实际推理价值；量化权重可压缩性也有限。
- Expert 已在 Page Cache 时，mmap 回填最快：单 expert **0.334 ms**，双 expert
  **0.712 ms**。冷数据没有固定赢家：单 expert direct 更快，双 expert mmap-cold
  借助顺序读取/预读后更快。

## 3. 后端 buffer 容量

### 3.1 设备内存口径

测试前采样：

```text
MemTotal:      15.48 GiB
MemAvailable:  10.40 GiB
SwapTotal:     16.78 GiB
SwapFree:      15.30 GiB
```

`MemAvailable` 是内核估算的当前可用物理内存；ZRAM 是占用物理 RAM 的压缩 swap，
不能与 RAM 直接相加。测试以 `SwapFree` 下降不超过 64 MiB 作为“未明显触发 ZRAM”
的边界。GPU/HTP buffer 不能假定像普通匿名页一样可换出；其分配导致 SwapFree
下降时，通常是系统把其他进程的匿名页压入 ZRAM 来腾出物理页。

### 3.2 测试方法

- CPU：匿名映射后每 4 KiB 写入并校验。
- GPU：创建 OpenCL buffer，由 kernel 每 4 KiB 写入，`clFinish` 后校验。
- HTP：申请 rpcmem、取得 FD、host 触页，再由 DSP 分窗口读写并校验。
- 多后端测试会保持三类 buffer 同时存活并轮流访问，但**不是三后端同时并发计算**。

### 3.3 单后端稳定上限

| 测试路径 | 稳定值 | 首个不稳定值/限制 |
|---|---:|---:|
| CPU 匿名驻留（不明显使用 ZRAM） | 7904 MiB | 7920 MiB |
| OpenCL 单 buffer | 1024 MiB | driver 单 buffer 上限 |
| OpenCL 聚合 | 7552 MiB | global memory 上限 |
| rpcmem 单次申请（仅 host） | 9680 MiB | 9696 MiB |
| rpcmem 多次 512 MiB 申请（仅 host） | 10144 MiB | 10160 MiB |
| HTP 单 FD 固定映射 | 4048 MiB | 4064 MiB |
| HTP 多 FD 同时 pinned | 3824 MiB | 3840 MiB |
| HTP delayed/windowed | 9760 MiB | 9776 MiB |
| 三后端普通内存最大总量 | 10608 MiB | 10624 MiB |

`rpcmem` 行只证明 host 能申请、触达共享内存；HTP 行才包含 DSP 真实访问。
`FASTRPC_MAP_FD_DELAYED` 能把“host 登记 FD”和“DSP 当前活跃映射”分开，从而绕过
DSP VA 总量限制，但会引入 map/unmap 和页表更新开销。

### 3.4 典型混合分配

| CPU:GPU:HTP | CPU MiB | GPU MiB | HTP MiB | 稳定总量 |
|---|---:|---:|---:|---:|
| 1:1:1 | 3472 | 3472 | 3488 | 10432 MiB |
| 2:1:1 | 5264 | 2624 | 2640 | 10528 MiB |
| 1:2:1 | 2624 | 5264 | 2640 | 10528 MiB |
| 1:1:2 | 2464 | 2464 | 4976 | 9904 MiB |
| CPU 优先 | 10096 | 256 | 256 | 10608 MiB |
| GPU 优先 | 256 | 7552 | 256 | 8064 MiB（受 OpenCL 上限约束） |
| HTP 优先 | 256 | 256 | 9456 | 9968 MiB（delayed/windowed） |

结论是：普通内存总预算主要由约 10.4 GiB 的 `MemAvailable` 决定；GPU global
上限和 HTP DSP VA 决定各后端能占多少。HTP delayed mapping 扩大的是可管理权重
总量，不是 DSP 同时可见的映射量。

完整数据见：

- [memory_data/README.md](test/results/memory_data/README.md)
- [capacities.csv](test/results/memory_data/capacities.csv)
- [CPU ZRAM 边界报告](test/results/near-oom-20260720/cpu-swap-summary.md)

## 4. CPU Expert I/O

### 4.1 测试口径

- 模型：`Phi-mini-MoE-instruct-Q4_0.gguf`。
- 范围：layer 0，共 16 个 expert；每次选择 1 或 2 个 expert。
- 单 expert：gate Q4_0 + up Q4_0 + down Q4_1，共 **6.5625 MiB**。
- CPU backend，4 threads，20 次重复，表中为 Median。
- `miss_fill` 包含读取和写入预分配 CPU expert slot；完成后执行一次计算校验。

三种路径：

| 路径 | 含义 |
|---|---|
| direct | `O_DIRECT + pread` 到 staging buffer，再 memcpy 到 expert slot |
| mmap-cold | 丢页 hint 后访问 GGUF mapping，触发 page fault，再回填 slot |
| mmap-warm | 权重已在 Page Cache，直接从 mapping 回填 slot |

### 4.2 关键结果

| I/O 路径 | 单 expert fill | 双 expert fill 总计 | 双 expert 平均 | fill 带宽（单/双） |
|---|---:|---:|---:|---:|
| direct | 2.740 ms | 5.444 ms | 2.722 ms/expert | 2395 / 2411 MiB/s |
| mmap-cold | 4.430 ms | 4.902 ms | 2.451 ms/expert | 1482 / 2678 MiB/s |
| mmap-warm | 0.334 ms | 0.712 ms | 0.356 ms/expert | 19648 / 18448 MiB/s |

### 4.3 结果解读

- **缓存命中收益最大**：mmap-warm 比冷读取快约一个数量级，Expert cache 的首要
  目标应是提高命中率。
- **单 expert 冷 miss**：direct 为 2.740 ms，优于 mmap-cold 的 4.430 ms；单个
  小范围 page fault/预读成本较明显。
- **双 expert 冷 miss**：mmap-cold 平均 2.451 ms/expert，优于 direct 的
  2.722 ms/expert，说明连续 expert 的合并读取和文件系统预读可以摊薄开销。
- direct 每个 expert 因 4 KiB 对齐多读 12 KiB，I/O 放大仅约 1.002 倍，不是主要
  瓶颈。
- mmap-cold 的物理读取量波动较大：单 expert 样本为 6708–21832 KiB，受
  `POSIX_FADV_SEQUENTIAL` 和文件系统预读影响，不能把 payload/time 直接当成块
  设备真实吞吐。
- `mmap-warm` 的物理读取为 0，证明正式计时期间命中了 Page Cache。

完整结果见：

- [CPU I/O 汇总](results/expert-miss-cpu-fd8657d6-20260722-no-crc/cpu-io-summary.md)
- [原始 JSONL](results/expert-miss-cpu-fd8657d6-20260722-no-crc/cpu-io-comparison.jsonl)

## 5. 当前判断

建议把 MoE 权重管理分成两层：

1. GGUF mmap/Page Cache 保存完整模型，避免为 CPU 再复制一份常驻权重。
2. 小型活跃 Expert cache 保存近期 expert；cache hit 走内存，miss 时合并相邻 expert
   的冷读取。

后端适配方向：

- CPU：优先验证 tensor 直接 alias GGUF mmap，减少一次 slot memcpy。
- GPU/OpenCL：验证 `CL_MEM_USE_HOST_PTR` 的实际 zero-copy、对齐和一致性；不能只因
  API 创建成功就认定没有 driver copy。
- HTP/QNN：当前仍需 mmap GGUF → rpcmem → DSP 映射；通过 delayed/windowed mapping
  控制活跃 VA，并复用高频 expert 的映射。

## 6. 下一步

1. 在真实 llama.cpp MoE 推理中测 Expert cache 命中率、端到端收益和 P95/P99 miss
   延迟，而不只看 probe。
2. 测量 HTP map/unmap 单次成本，确定映射缓存大小和淘汰策略。
3. 对比单 expert 随机读取与相邻多 expert 合并读取，确认 mmap 预读收益边界。
4. 验证 huge page 是否在目标 Android 内核可用，以及它对 TLB miss 的真实收益；
   暂不假设 `MAP_HUGETLB` 一定可用或物理连续。
5. 所有性能结论优先使用“不明显触发 ZRAM”的配置，避免把压缩/换页时间混入推理。
