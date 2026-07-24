# 可参数化 Expert Miss：UFS→后端 Buffer 填充计时实验

## 1. 实验目标

本实验验证：

> 当发生 Expert Cache Miss 时，从 UFS 读取指定数量、指定大小的专家，并将其完整注册到 CPU、OpenCL GPU 或 HTP 后端的可计算 buffer 中，需要多长时间，能够达到多大的有效带宽？

实验不模拟完整 MoE 推理链路，不采集真实 Router 输出，不执行真实 prompt，也不要求使用模型中的真实专家权重。

以下变量均通过命令行参数控制：

```text
专家 Miss 数量
单专家数据大小
专家内部 tensor 组成
I/O 模式
文件布局
读取策略
目标计算后端
重复次数
```

Phi-mini-MoE Q4 仅作为默认 profile，用于提供一组接近真实模型的参数。
用Phi-mini-MoE Q4的实际情况做测试
---

## 2. 可配置实验变量

### 2.1 Expert Miss 数量

一次 attempt 加载的专家数量由参数控制：

```text
--miss-count N
```

或者批量扫描：

```text
--miss-counts 1,2,4,8
```

推荐范围：

```text
1 <= N <= 8
```

也允许扩展到更大数值，用于模拟：

* 单 token Top-K Miss；
* 多 token 合并 Miss；
* 跨层预取；
* Cache 批量回填；
* 高压力测试。

对于 Phi-mini-MoE：

```text
miss-count = 1:
  Top-2 中一个专家缺失

miss-count = 2:
  单 token 两个激活专家全部缺失

miss-count > 2:
  多 token、批量预取或压力测试
```

---

### 2.2 单专家大小

单专家大小不写死，由参数控制：

```text
--expert-size-mib SIZE
```

例如：

```text
--expert-size-mib 2.5
--expert-size-mib 4
--expert-size-mib 6.328125
--expert-size-mib 8
--expert-size-mib 16
```

也可以批量扫描：

```text
--expert-sizes-mib 1,2,4,6.328125,8,16
```

单专家大小表示：

```text
gate payload
+ up payload
+ down payload
```

的总和。

程序根据指定的专家总大小，按照配置比例分配给 gate、up 和 down。

默认比例可设置为：

```text
gate : up : down = 1 : 1 : 1
```

接口：

```text
--tensor-size-ratio 1,1,1
```

如果需要模拟非均匀 tensor 大小，也可以设置：

```text
--tensor-size-ratio 1,1,2
```

程序根据比例计算三段大小：

```text
gate_bytes =
    expert_bytes × gate_ratio / ratio_sum

up_bytes =
    expert_bytes × up_ratio / ratio_sum

down_bytes =
    expert_bytes × down_ratio / ratio_sum
```

所有 tensor payload 的最终大小必须满足目标 ggml tensor type 的 block-size 对齐要求。

---

#### `phi-mini-moe-q4`

使用 Phi-mini-MoE 的默认参考配置，例如：

```text
hidden size       = 4096
intermediate size = 960
experts per layer = 16
top-k             = 2
```

专家大小由实际采用的 Q4 类型决定。

即使选择该 profile，仍允许使用命令行参数覆盖默认值：

```text
--profile phi-mini-moe-q4
--expert-size-mib 8
--miss-count 4
```

#### `custom`

由用户显式指定全部结构参数。

参数优先级为：

```text
命令行显式参数
>
profile 默认值
>
程序内置默认值
```

---

## 3. 三种 I/O 模式参数化

I/O 模式由以下参数控制：

```text
--io-mode direct
--io-mode buffered-cold
--io-mode buffered-warm
```

也可以一次运行全部模式：

```text
--io-mode all
```

或者指定列表：

```text
--io-modes direct,buffered-warm
```

### 3.1 Direct I/O

```text
--io-mode direct
```

使用：

```text
O_DIRECT
```

特点：

* 绕过 Linux page cache；
* 数据主要经过块设备读取路径；
* 需要满足 buffer、offset 和 length 对齐；
* 用于测量 UFS→应用→后端的主要实验结果。

程序记录：

```text
payload_bytes
physical_io_bytes
I/O amplification
/proc/self/io read_bytes delta
```

如果设备或文件系统不支持 Direct I/O，必须返回：

```text
direct-io-unsupported
```

不能静默回退到 buffered I/O。

### 3.2 Buffered Cold-Hint

```text
--io-mode buffered-cold
```

使用普通 buffered `pread()`，正式读取前执行：

```text
posix_fadvise(..., POSIX_FADV_DONTNEED)
```

特点：

* 经过 Linux page cache；
* 尝试使目标文件页在正式读取前不处于 page cache；
* 更接近普通应用首次访问文件的路径；
* 不能保证清除 UFS 控制器内部缓存；
* 也不能保证 Linux 一定立即回收目标页。

因此结果应标记为：

```text
buffered-cold-hint
```

而不是绝对冷盘。

### 3.3 Buffered Warm

```text
--io-mode buffered-warm
```

正式计时前先完整读取目标专家，使数据进入 Linux page cache。

特点：

* 正式读取主要来自 DRAM page cache；
* 通常不会产生明显的块设备 `read_bytes`；
* 用于剥离 UFS 读取成本；
* 测量内存读取、host prepare、后端上传、repack 和同步开销。

---

## 4. 合成专家数据

### 4.1 数据合法性

专家内容不要求来自真实模型，但必须满足后续可计算性验证要求。

每个专家包含：

```text
gate
up
down
```

可以提供两种数据模式：

```text
--payload-mode valid-quantized
--payload-mode raw-bytes
```

#### `valid-quantized`

推荐作为正式实验模式。

流程：
参考llama.cpp框架中使用MOE/models/Phi-mini-MoE-instruct-Q4_0.gguf拿到数据后进行MUL_MAT_ID计算前的工作
1. 写入专家 pack；
2. 后续可以直接用于 `MUL_MAT_ID` 验证。

#### `raw-bytes`

只用于纯 I/O 和 buffer 注册测试。

特点：

* 数据按固定模式填充；
* 不保证能够执行有效的量化矩阵计算；
* 不能用于“可直接计算”的正式验收结果。

如果要求 Expert Miss 完成后能够直接计算，则必须使用：

```text
--payload-mode valid-quantized
```

---

### 4.2 Tensor 类型

通过参数指定：

```text
--gate-type q4_0
--up-type q4_0
--down-type q4_0
```

也可写成：

```text
--tensor-types q4_0,q4_0,q4_1
```

支持的具体类型以当前仓库 ggml 后端能力为准。

如果指定的类型不被某个后端支持，则该后端 attempt 返回：

```text
unsupported-tensor-type
```

不得回退到 CPU 并混入结果。

---

## 5. Expert Pack 生成接口

Host 工具：

```text
expert-pack-gen
```

接口：

```text
expert-pack-gen
  --profile generic|phi-mini-moe-q4|custom
  --expert-count N
  --expert-size-mib SIZE
  --tensor-size-ratio G,U,D
  --tensor-types GATE,UP,DOWN
  --payload-mode valid-quantized|raw-bytes
  --layout contiguous|random-gap
  --alignment 4096
  --seed N
  --gap-min-kib N
  --gap-max-kib N
  --output PATH
```

示例：

```bash
./expert-pack-gen \
  --profile phi-mini-moe-q4 \
  --expert-count 16 \
  --expert-size-mib 6.328125 \
  --tensor-size-ratio 1,1,1 \
  --tensor-types q4_0,q4_0,q4_0 \
  --payload-mode valid-quantized \
  --layout random-gap \
  --alignment 4096 \
  --seed 1 \
  --output phi_mini_moe_experts.pack
```

通用大小扫描示例：

```bash
for size in 1 2 4 8 16; do
    ./expert-pack-gen \
      --profile generic \
      --expert-count 16 \
      --expert-size-mib "${size}" \
      --tensor-size-ratio 1,1,1 \
      --tensor-types q4_0,q4_0,q4_0 \
      --payload-mode valid-quantized \
      --layout random-gap \
      --alignment 4096 \
      --seed 1 \
      --output "experts_${size}mib.pack"
done
```

---

## 6. Android Probe 接口

```text
expert-ufs-probe
  --dataset PATH
  --backend cpu|opencl|htp|all
  --io-mode direct|buffered-cold|buffered-warm|all
  --miss-count N
  --miss-counts LIST
  --expert-selection fixed|random|round-robin
  --read-policy separate|merged
  --repeat N
  --reserve-mib N
  --validate-compute true|false
  --output-jsonl PATH
```

单组测试示例：

```bash
./expert-ufs-probe \
  --dataset /data/local/tmp/moe/phi_mini_moe_experts.pack \
  --backend htp \
  --io-mode direct \
  --miss-count 2 \
  --expert-selection round-robin \
  --read-policy separate \
  --repeat 20 \
  --validate-compute true \
  --output-jsonl results.jsonl
```

一次扫描 1–8 个专家和三种 I/O 模式：

```bash
./expert-ufs-probe \
  --dataset /data/local/tmp/moe/phi_mini_moe_experts.pack \
  --backend all \
  --io-mode all \
  --miss-counts 1,2,3,4,5,6,7,8 \
  --expert-selection round-robin \
  --read-policy separate \
  --repeat 20 \
  --validate-compute true \
  --output-jsonl results.jsonl
```

---

## 7. 后端 Slot

程序按照最大 miss 数量预分配 slot：

```text
--max-miss-count 8
```

计时外完成：

* 后端初始化；
* 最大数量的 slot 分配；
* tensor descriptor 创建；
* host aligned buffer 分配；
* graph 创建；
  -后端 warm-up；
* 验证 activation 和 output 分配。

一次 attempt 只启用前 `miss_count` 个 slot。

例如：

```text
max-miss-count = 8
miss-count     = 3
```

则本轮只加载并注册 3 个专家，其余 5 个 slot 保持 `EMPTY`。

---

## 8. 主计时区间

每次 attempt 的计时流程为：

```text
选择 miss_count 个专家
        ↓
记录 miss_fill_start
        ↓
从 UFS 读取指定专家
        ↓
处理 Direct I/O 对齐前后缀
        ↓
checksum 和 payload 切分
        ↓
写入或上传到目标后端 slot
        ↓
执行 repack、映射或设备上传
        ↓
backend synchronize
        ↓
所有目标 slot 标记为 READY
        ↓
记录 miss_fill_end
```

主指标：

```text
miss_fill_us =
    miss_fill_end - miss_fill_start
```

分项指标：

```text
file_read_us
host_prepare_us
backend_register_us
backend_sync_us
miss_fill_us
```

由于部分操作可能异步交叠：

```text
miss_fill_us
```

必须使用端到端 wall-clock 测量，不能简单依赖分项相加。

---

## 9. 带宽指标

### 9.1 UFS Payload 带宽

```text
payload_read_bandwidth =
    payload_bytes_total / file_read_us
```

反映读取有效专家数据的速度。

### 9.2 Physical I/O 带宽

```text
physical_read_bandwidth =
    physical_io_bytes_total / file_read_us
```

主要用于 Direct I/O。


反映 DRAM 数据进入 CPU、OpenCL 或 HTP 可计算 buffer 的速度。

### 9.4 端到端填充带宽

```text
miss_fill_bandwidth =
    payload_bytes_total / miss_fill_us
```

这是最重要的总体吞吐指标。

它回答：

> 整个 UFS→后端可计算 buffer 链路，每秒能填充多少 MiB 的 Expert 权重？

### 9.5 单专家平均时间

```text
average_per_expert_us =
    miss_fill_us / miss_count
```

但该指标只作为辅助，因为多个专家的读取和上传可能被合并或并行，不能假设总时间严格线性。

---
