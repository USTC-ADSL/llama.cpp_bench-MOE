# Exp1：4 GiB memory cgroup 限制下的 Qwen1.5-MoE 实验报告

- 日期：2026-09-03
- 设备：OnePlus Turbo 6，ADB Serial `3B661501LA000000`
- 模型：`/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`
- 模型识别：Qwen2MoE A2.7B，Q4_0，14.32B parameters
- 测试代码：`6b386ae8c`，llama.cpp build `10742`
- 对照基线：`Exp_report.md` 中的无限制实验

## 1. 结论摘要

本轮施加的 4 GiB 限制对 **cgroup v1 memory controller 能够计费的 RAM+swap** 生效，但它不是包含 OpenCL、ION、DMA-BUF、FastRPC/rpcmem 和 HTP buffer 在内的整机物理内存总量上限。

结果如下：

1. CPU mmap 和 CPU no-mmap 均为 0/3 成功。两组都在加载期间达到 4 GiB memcg 上限，并持续停留在模型 tensor/repack 加载阶段；正式样本等待 180 秒后由 runner 发送 `SIGTERM`，不是 LMKD 或 cgroup OOM kill。
2. OpenCL mmap 和 HTP 1 PD 均为 3/3 成功，完成精确的 512-token prompt 和 128-token decode；但成功不代表它们满足“总物理内存不超过 4 GiB”。
3. OpenCL 仍创建约 `7599.83 MiB` model buffer，HTP 仍创建约 `7356.29 MiB` model buffer。大部分后端分配没有等量反映在该进程的 memcg usage 中，系统最低 `MemAvailable` 反而分别降到 `1.202 GiB` 和 `1.846 GiB`。
4. 12 次正式运行的 `memory.oom_control.oom_kill` 和 `memory.failcnt` 都为 0，也没有针对 `llama-server` 的 LMKD kill 证据。
5. 因此，当前 CPU Q4_0 repack 路径无法在 4 GiB 可计费内存预算内完成加载；OpenCL/HTP 的成功则利用了 memcg 计费边界之外的后端内存，不能视为满足用户期望的 4 GiB 整体内存预算。

## 2. 4 GiB 限制方法与边界

设备存在 cgroup v1 memory controller。本轮为每次运行创建独立 cgroup：

```text
/dev/memcg/apps/llama-exp1-4g-<config>-<repeat>
```

同时设置：

```text
memory.limit_in_bytes       = 4294967296
memory.memsw.limit_in_bytes = 4294967296
memory.swappiness           = 0
```

启动时由原 shell 进程通过一次短暂的 `su -c` 将自身 PID 写入 `cgroup.procs`，再 `exec llama-server`。每次都通过 `/proc/<pid>/cgroup` 验证 server 已加入目标 memory cgroup。这样不会把整个 server 改为 root 上下文。

设备端约每 200 ms 采样以下指标：

```text
/proc/<pid>/smaps_rollup
/proc/<pid>/stat
/proc/meminfo
memory.usage_in_bytes
memory.max_usage_in_bytes
memory.failcnt
```

进程退出后另读取 `memory.memsw.*`、`memory.oom_control` 和 `memory.stat`，再删除本次 cgroup。

### 2.1 本限制能够证明什么

它能够证明 `llama-server` 中由该 cgroup v1 controller 计费的匿名页、文件页和 swap 合计没有突破 4 GiB。12 次运行都验证了 cgroup membership，且 cgroup 历史峰值均贴近或等于 4 GiB。

### 2.2 本限制不能证明什么

它不能证明下列内存的整机物理占用合计小于 4 GiB：

- GPU/OpenCL driver buffer；
- ION 或 DMA-BUF；
- FastRPC/rpcmem 与 HTP 共享 buffer；
- 驱动代进程持有、但映射到应用地址空间的设备内存；
- 未归属到该进程 memcg 的内核和厂商运行时分配。

这也是 OpenCL/HTP 能够在 memcg 达到 4 GiB 时仍然成功的原因。报告中的 “4 GiB 限制” 此后均指 **memcg 可计费 RAM+swap 限制**，不指整机总物理内存限制。

## 3. 固定实验条件

| 项目 | 固定值 |
|---|---|
| Prompt | 精确 512 tokens |
| Decode | 精确 128 tokens，`ignore_eos=true` |
| Context | 2048 |
| 重复次数 | 每组 3 次 |
| CPU threads | 4 |
| Batch / ubatch | 512 / 128 |
| Flash Attention | off |
| Prompt cache | off |
| Server warmup | off |
| CPU 加载超时 | 180 s |
| memory cgroup | v1，4 GiB RAM+swap，swappiness 0 |

Prompt token 数组与无限制基线相同。每次重复都重新启动 server，并调用 `POSIX_FADV_DONTNEED` 尽量丢弃模型文件的干净 page-cache 页。

配置保持为：

```text
CPU mmap:      -lm mmap -lzm off -dev none      -ngl 0
CPU no-mmap:   -lm none -lzm off -dev none      -ngl 0
OpenCL mmap:   -lm mmap -lzm off -dev GPUOpenCL -ngl all
HTP 1 PD:      -lm mmap -lzm off -dev HTP0:0    -ngl all -sm layer -ts 1
```

目标设备 Hexagon 为 v73，当前后端对低于 v75 的 SoC 固定只注册一个 PD。因此本轮没有重复 HTP 2/4 PD 的已知参数失败；该限制与 4 GiB memory cgroup 无关。

## 4. 总体结果

除成功数外，数值均为三次均值；括号中为三次范围。`最低 MemAvailable` 的括号下界也是该组观测到的整机绝对最低值。

| 配置 | 成功 | 加载等待 | 峰值 PSS | 最低 MemAvailable | memcg 峰值 | Major faults（进程 / 请求） | Server / E2E TTFT | TPOT | 目标进程 OOM/LMKD kill |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CPU mmap | 0/3 | 180.571 s（180.443–180.727） | 4.121 GiB（4.107–4.133） | 4.429 GiB（4.380–4.458） | 4.000 GiB | 5,281 / 不适用 | 不适用 | 不适用 | 0/3 |
| CPU no-mmap | 0/3 | 180.114 s（180.004–180.299） | 4.009 GiB（4.009–4.009） | 4.424 GiB（4.407–4.438） | 4.000 GiB | 1.3 / 不适用 | 不适用 | 不适用 | 0/3 |
| OpenCL mmap | 3/3 | 17.384 s（15.924–19.210） | 4.081 GiB（4.065–4.089） | 1.532 GiB（1.202–1.709） | 4.000 GiB | 5,783 / 121 | 7134.1 / 7494.6 ms | 58.19 ms | 0/3 |
| HTP 1 PD | 3/3 | 15.911 s（15.195–16.306） | 4.057 GiB（4.032–4.072） | 1.923 GiB（1.846–1.965） | 4.000 GiB | 68,225 / 1,016 | 4784.3 / 5142.0 ms | 140.28 ms | 0/3 |

`峰值 PSS` 略高于 4 GiB 不表示 cgroup 限制失效。PSS 是地址空间映射页的比例统计，memcg usage 是 controller 的页计费；设备映射、共享页归属和采样时刻不同，两者不应要求逐字节相等。限制是否生效以 `memory.max_usage_in_bytes <= 4294967296` 为准。

## 5. CPU 加载结果

### 5.1 CPU mmap

三次日志都到达：

```text
CPU_Mapped model buffer size = 7424.21 MiB
CPU_REPACK model buffer size = 7587.43 MiB
```

随后 server 一直返回 `Loading model`，未完成 health ready。cgroup 历史峰值三次都精确达到 `4294967296` bytes；在 180 秒超时前没有 cgroup OOM kill，最后的退出码 `143` 是 runner 主动发送 `SIGTERM`。

这说明当前 mmap 并不是“只在计算某个 expert 时才临时读入其权重”的朴素 offload。GGUF 文件映射本身可以按缺页按需读入，但本分支的 CPU Q4_0 路径还建立了约 `7587.43 MiB` 的 `CPU_REPACK` 表示。4 GiB 限制下无法形成完成加载所需的 working set。

加载期进程 major faults 平均约 5,281 次。由于模型没有进入请求阶段，不能得到 TTFT、TPOT 或 decode faults，也不能在本轮证明 CPU mmap 的长时间推理稳定性。

### 5.2 CPU no-mmap

三次同样无法在 180 秒内完成加载，日志报告：

```text
CPU model buffer size = 179.21 MiB
CPU_REPACK model buffer size = 7587.43 MiB
```

no-mmap 不保留 7.42 GiB 的 `CPU_Mapped` 区间，但仍需建立完整 CPU repack buffer，因此也无法满足 4 GiB memcg 预算。进程 major faults 只有 1–2 次并不表示没有文件 I/O；普通 `read`/`pread` 导致的存储读取不会计入进程 major page faults。

## 6. 异构后端结果

### 6.1 OpenCL mmap

三次均完成 512/128 推理：

```text
CPU_Mapped model buffer size = 166.92 MiB
OpenCL model buffer size     = 7599.83 MiB
```

请求 major faults 平均 121 次，其中 prompt 50、decode 71。三次 TPOT 范围为 `57.531–59.095 ms`，重复间没有明显抖动。

OpenCL 的进程 PSS 从无限制基线的 6.909 GiB 降到 4.081 GiB，但最低 `MemAvailable` 从 1.942 GiB 进一步降到 1.532 GiB，且绝对最低达到 1.202 GiB。这组相反方向的变化直接说明：进程 PSS 或 memcg usage 下降不能解释为整机总物理内存也下降。OpenCL run-01 期间系统回收了多个后台缓存应用，但没有杀死 `llama-server`。

### 6.2 HTP 1 PD

三次均完成 512/128 推理：

```text
CPU_Mapped model buffer size = 166.92 MiB
CPU_REPACK model buffer size = 243.43 MiB
HTP0:0 model buffer size     = 7356.29 MiB
```

这里的 `7356.29 MiB` 是 backend 报告的 model buffer 地址范围，不应理解为 7.36 GiB 全部驻留在 HTP 片上存储，也不能与所谓 HTP “3.5 GiB 可用空间”直接比较。当前数据没有分解其中的 host shared ION/rpcmem、设备映射和真实驻留量。

HTP 的生命周期进程 major faults 平均为 68,225，显著高于 OpenCL。本轮请求 faults 分别为 `2780`、`135`、`134`；run-01 的 2,709 次 decode faults 是明显离群值，但三次 TPOT 仍稳定在 `139.297–140.802 ms`。HTP run-02 出现厂商 `ion_limit` 监控告警，但请求成功且 server 正常退出。

当前分支没有输出独立的 `HTP-REPACK buffer total`。报告继续将 `HTP0:0 = 7356.29 MiB` 作为重排后 backend model buffer 的代理值，不能将其解释为可独立剥离的纯 repack 额外开销。

## 7. 与无限制基线比较

| 配置 | 无限制结果 | 4 GiB memcg 结果 | 解释 |
|---|---|---|---|
| CPU mmap | 3/3 推理成功；峰值 PSS 9.735 GiB | 0/3，加载超时 | mmap 加上 CPU repack 的完整 working set 超过 4 GiB |
| CPU no-mmap | 3/3 加载成功；峰值 PSS 8.023 GiB | 0/3，加载超时 | 完整 CPU repack 无法在限制内建立 |
| OpenCL mmap | 3/3；PSS 6.909 GiB；E2E TTFT 7588 ms；TPOT 60.60 ms | 3/3；PSS 4.081 GiB；E2E TTFT 7495 ms；TPOT 58.19 ms | 性能相近，但更低 PSS 不代表更低的总物理占用 |
| HTP 1 PD | 3/3；PSS 7.235 GiB；E2E TTFT 5304 ms；TPOT 145.43 ms | 3/3；PSS 4.057 GiB；E2E TTFT 5142 ms；TPOT 140.28 ms | 性能相近，后端 buffer 绕开了大部分 memcg 计费 |

三次重复不足以把本轮几百分点的 TTFT/TPOT 差异归因于 4 GiB 限制；page cache、后台回收和设备运行状态都可能造成该量级波动。

## 8. OOM、LMKD 与稳定性复核

12 次正式运行均满足：

```text
memory.oom_control.oom_kill = 0
memory.failcnt              = 0
```

CPU 六次退出码 `143` 都发生在 runner 的 180 秒 health timeout 之后，是主动清理，不是 OOM kill。日志中的 `lowmemorykiller ... disable killing` 明确表示“内存足够，禁用 killing”，不能作为实际 kill 证据。

OpenCL run-01 有 37 条 `am_kill`，目标为其他后台缓存应用而非 `llama-server`。CPU 运行出现厂商 `total_limit_native` 和个别 ENOMEM 监控日志，HTP run-02 出现 `ion_limit` 告警；这些都说明系统或厂商内存策略受到了压力，但不是目标进程被杀的证据。

达到 `memory.max_usage_in_bytes == limit` 而 `failcnt == 0` 并不矛盾：controller 可以通过回收已有页满足新的 charge，未必产生一次失败 charge 或 OOM kill。CPU 的表现更接近持续回收/缺页压力下无法完成加载，而不是立即 OOM。

## 9. 最终判断

若“运行内存上限 4 GiB”定义为 **目标进程在 cgroup v1 中可计费的 RAM+swap**，本轮限制已经正确施加并验证：CPU 路径无法加载，OpenCL/HTP 能完成推理。

若其定义为 **模型推理导致的整机总物理内存，包括 GPU/ION/rpcmem/HTP buffer**，则本轮 OpenCL/HTP 不能被判定为满足 4 GiB。它们仍创建 7 GiB 以上的后端 model buffer，并把整机 `MemAvailable` 压到约 1.2–2.0 GiB。

因此当前四种配置中，没有证据表明 Qwen1.5-MoE Q4_0 能在严格的 4 GiB 整体物理内存预算内运行。若要实现后一个目标，需要增加驱动/allocator 级别的 ION、DMA-BUF、GPU 和 rpcmem 统计与配额，或者降低模型权重和 repack 规模；仅使用进程 memory cgroup 不足以约束异构后端总内存。

## 10. 数据与复现

结构化单次结果和聚合结果：

```text
task/handoff/handoff-2026-09-03-exp1-4g/results/
task/handoff/handoff-2026-09-03-exp1-4g/results/aggregate.json
```

运行器、设备监控和本轮聚合工具：

```text
task/handoff/handoff-2026-09-01-exp1/exp1_runner.py
task/handoff/handoff-2026-09-01-exp1/device_monitor.sh
task/handoff/handoff-2026-09-03-exp1-4g/aggregate_results.py
```
