# E0 手机 MoE 环境与测量核验

本次核验已结束，整体结论为 **BLOCKED**。OpenCL 在本次固定输入、真实 Q4_0 expert shape 和计时采样范围内通过；CPU 采样扰动与 HTP n=128 HMX 数值异常阻止整套环境通过验收。这里的 PASS 仅表示已测工具与可用路径可信，不表示 ION/rpcmem 已受 cgroup 控制，也不证明研究 motivation 或任何缓存、调度设计有效。

## 环境与固定条件

- 代码基线：`d46e707010c74d2c1f8d3654fdd64abe7cfaacf2`，已包含官方 upstream `5202104b5`；E0 instrumentation 检查点 `3b17c9d2b`。完整版本、dirty patch、文件 SHA256 见 `environment.json`、`artifacts.json`、`manifest.json`。原有用户文件保留。
- 按 AGENTS.md 配置使用 `3B661501LA000000`：OnePlus PLU110，SM8735，Adreno 825，Hexagon **v73**；Android 16/API36；内核 `6.6.89-android15-8-g7e1f3c083cc6-abogki467167594-4k`。MemTotal 15,524,920 KiB，约 16 GB / 256 GB 设备。另一在线设备未使用。
- 原模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8,149,865,792 bytes，SHA256 `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`。未重新量化。完整哈希在主要测试之后只读计算。
- GGUF 实际为 qwen2moe：24 层、d_model=2048、60 routed experts、top4、routed d_ff=1408、shared d_ff=5632。第0层 gate/up shape `[2048,1408,60,1]`，down `[1408,2048,60,1]`，均原始 Q4_0。
- 每次请求使用 `prompt.json` 同一个实际 **128-token 数组**，目标32输出，context512，单请求/单slot，F16 K/V，t=4、tb=4、batch128、ubatch128。三个主要验收组的33次请求都实际 prefill128、生成32，`cache_n=0`，server token-ready IDs 与响应一致。另三次详细 profile 各为128/32。
- 先展示 dry-run，再在 Android 上执行；复用 `build-npu-opencl.sh` 对应的已有交叉编译目录增量构建。NDK r27d、Hexagon Tools19.0.04。因上游 UI 构建下载阻塞，在本机构建缓存关闭 UI 下载/构建，本次只使用 server API。没有升级依赖。

## 验收结果

| 路径 | 固定请求与计数 | 真实权重算子数值 | 采样开关配对 | 路径结论 |
|---|---|---|---|---|
| CPU-only + mmap | 13次128/32，PASS | Android CPU Q4_0 reference | 优化后6对仍超出±3% | BLOCKED：采样扰动 |
| OpenCL offload | 13次128/32，PASS | n=1/128，gate+up/down 四项PASS | 初始3对不确定，补6对PASS | PASS（本E0范围） |
| HTP 单PD | 7次128/32，PASS | n=1 HVX两项PASS；n=128 HMX两项FAIL | 3对PASS | BLOCKED：数值异常 |

详细 profiling 单独运行，不进入采样开销统计。HTP 多PD按本地v73注册策略记 unsupported，未运行多PD测试，也未把参数不支持记为OOM。

### 数值证据

使用第0层原始 Q4_0 expert 权重、相同固定激活与 expert IDs；Android CPU `ggml_backend_cpu_set_use_ref(true)` 为参考。执行前固定容差：relative RMSE ≤0.02、max absolute error / RMS(reference) ≤0.10，且输出全部有限。没有为了通过测试更改路由、权重、精度或容差。

| 后端/输入n | gate+up RRMSE | down RRMSE | 结果 |
|---|---:|---:|---|
| OpenCL / 1 | 0.002581 | 0.002741 | PASS |
| OpenCL / 128 | 0.002786 | 0.002774 | PASS |
| HTP HVX / 1 | 0.003806 | 0.003873 | PASS |
| HTP HMX / 128 | **1.000000** | **1.000000** | FAIL |

HTP n=128 两项 max-error/RMS 分别为5.3586和22.5671。构图、backend supports_op、repack/tensor_set、compute/sync/get 经过源码和独立只读复核，未发现遗漏必要步骤；这不是对后端根因的完整定位。保留失败证据，不在E0修复HMX。最终引用 `runs/numeric-{opencl,htp}-verified/{numeric.jsonl,backend.log.gz,execution.json}`；两组watchdog错误日志为空、无安全截断，HTP退出码1来自数值失败。

### 真实执行与参数语义

请求阶段 scheduler 日志将 routed experts、shared experts、attention 分别指向 CPU、OpenCL、HTP0，符合三种配置。scheduler 打印节点数是图分配证据，不是算子调用次数。结合完成的请求、同步边界和 HTP 完成算子日志核验，不能仅凭选择参数断言执行路径。

HTP 实际请求段完成日志含 `hmx-tiled=140`、`hmx-pipe=24`、`hvx-tiled=6004`；routed/shared/attention 各组均有 HMX/HVX 记录。`MUL_MAT_ID_NX` 在实际128/32请求内完成 **768次**，全生命周期792次；多出的24次属于启动探测。未观察到所审计三类节点转给CPU。HMX/HVX命中并不抵消数值失败。详见 `path-audit.json` 的分组、行号与关键原文，以及三组 `*-profile/server.log.gz`。

- `--cpu-moe` 是加载时 routed-expert tensor 的CPU buffer override，不包括 shared experts 或 attention；`--n-cpu-moe N` 指前N层，不是N个expert。
- tensor override 是 tensor-name regex 到 backend buffer type 的加载映射，不提供缓存、逐请求迁移或配额管理。最终执行仍取决于scheduler与backend支持。
- `--load-mode mmap` 映射GGUF；设备上传、repack和其他buffer仍可能额外分配内存。参数与枚举以 `raw/help.txt` 和本地源码为准。
- `--no-warmup` 关闭 `common/common.cpp:1516` 的显式BOS/EOS warmup，但 server 在 `tools/server/server-context.cpp:1239` 调用 `common_context_can_seq_rm`。`common/common.cpp:1588–1628` 仍执行 `[0,0]` 两个token的删除能力探测，前后清KV并同步。HTP profile实际请求从第38050行开始；启动2token不计入128token prefill或请求算子统计。
- 同一server内每次 `cache_prompt=false`、`--cache-ram 0`，使请求从 `n_past=0`、清除该序列KV后重新prefill128，权重保留。没有清全局page cache或驱逐权重。启动探测已经可能触碰权重，首次请求不能自动称“冷文件”。
- OpenCL支持本模型Q4_0 `MUL_MAT_ID`；Hexagon风格NX未在OpenCL实现，不将其它epilogue fusion误称NX。HTP默认NX启用，MM/FA选择策略可能选HMX、HVX或fallback，实际以完成日志为据。

## 计时与采样扰动

TTFT定义为完整 `/completion` POST进入server callback到第一个完成backend同步、sampling/accept后的token-ready时间；ITL是相邻token-ready时间戳之差，p95由31个间隔作线性插值经验分位数。记录CLOCK_MONOTONIC，并保留CLOCK_BOOTTIME用于与设备uptime对齐。没有使用SSE包间隔或平均TPOT替代。load/context/prefill/decode边界和逐token ID保存在各组 `events.jsonl`。

配对交替OFF/ON与ON/OFF，在同一server保留权重，每次清请求KV。watchdog和必要时间戳在两臂都启用；这里检验的是内存采样的增量扰动，未证明整套instrumentation相对完全无测量server的开销。判据是配对相对变化平均值的4000次固定seed bootstrap 95%区间全部落在±3%内；小样本PASS仅限本次范围。

| 最终组/配对数 | TTFT变化均值；95%区间 | p95 ITL变化均值；95%区间 | 判定 |
|---|---|---|---|
| CPU / 6 | −1.112%；[−3.148%, +1.464%] | **−11.522%；[−15.670%, −7.349%]** | BLOCKED |
| OpenCL / 6 | −0.037%；[−0.195%, +0.131%] | +0.189%；[−1.517%, +1.948%] | PASS |
| HTP / 3 | −0.289%；[−1.804%, +0.888%] | −0.214%；[−0.574%, +0.200%] | PASS（仅采样） |

CPU开采样反而更快，也属于明显观测扰动/未解释混杂，不能解释成“负开销所以合格”。同步PSS初版使TTFT增加约24%–27%，已改异步队列并补测；异步后CPU仍未通过。

PSS目标周期1秒，并在阶段边界触发异步读取；保存触发/采样开始/结束时间。主要组最大边界队列延迟CPU232.516ms、OpenCL17.821ms、HTP17.718ms；最大周期约1.38s/1.05s/1.05s。异步边界值不是边界瞬时值，以下均标为采样峰值。

首个诊断请求（非冷文件、非性能排名）的TTFT/p95 ITL分别为：CPU5420.878/53.568ms，OpenCL1877.857/69.531ms，HTP777.964/102.991ms。CPU计时受上述扰动限制，HTP尚未通过数值验收，不能用这些数值比较可信加速收益。

## Fault 与内存口径

旧表1,232,047是三个进程major-fault请求增量 `[1315459,1228895,1151788]` 的均值1,232,047.333。旧脚本去掉最后 `) ` 之前的pid/comm后，shell `${10}` 对应原stat字段12 majflt，未误读为字段10 minflt。但旧请求边界靠host/SSE附近采样点，且没有历史RUSAGE_SELF，无法追认成精确server边界。见 `old-fault-audit.json`。

新工具正确处理含空格/括号的comm，读取进程字段10/12，并用前后 `getrusage(RUSAGE_SELF)` 夹逼。设备sanity以 `E0 ) comm (x)` 为comm，匿名16MiB首次写仅增加minor；主要组所有请求计数夹逼检查通过。没有叠加线程、子进程或全系统fault。CPU首请求majflt增量1,323,254，与RUSAGE一致；另列read_bytes增量5,561,036,800 bytes。读取是边界附近有起止时间的有限窗口，不是原子快照。fault×页大小不是实际UFS读取量，`read_bytes`也不是物理UFS总线实测。

| 主要组 | PSS采样峰值GiB | 最小MemAvailable GiB | 继承cgroup历史peak bytes |
|---|---:|---:|---:|
| cpu-validated | 10.112 | 2.672 | 10070388736 |
| opencl-confirm | 7.145 | 2.340 | 10070388736 |
| htp-validated | 6.375 | 2.914 | 10070388736 |

cgroup读数来自既有继承组的历史峰值，可能覆盖其它进程/旧负载，不是E0私有峰值。未创建或限制cgroup。详细profile有独立内存记录，不混入上表。

OpenCL报告容量7580MiB、max allocation1024MiB是驱动能力信息，不是实际进程RAM。OpenCL profile的KGSL快照：gpumem_mapped=8,661,237,760、gpumem_unmapped=196,608、kernel=8,661,434,368、kernel_max=8,661,532,672 bytes；是驱动定义的分配/映射计数，保存在 `kgsl-counters.txt`，不推断独占物理RAM。HTP生命周期同时FastRPC映射容量峰值7,871,344,640 bytes，由unique session/base/fd日志重建；它不是独占物理内存。HTP的小KGSL上下文注册也不证明GPU执行。

`/sys/kernel/debug/adsprpc`、`fastrpc`、`dma_buf`接口不可用；部分DMA-BUF sysfs和KGSL接口可读。ION/rpcmem是否由本机cgroup计账/限制未被证明，覆盖未知项为NA/null。PSS、cgroup、KGSL、PD映射不相加制造“总RAM”。

## 安全、证据与结束状态

全程未改全机swap、LMKD、SELinux、频率或cgroup限制，未清全局page cache。watchdog阈值：MemAvailable<1GiB持续3秒或当前HAL达到SEVERE；加载300秒、请求180秒超时均记安全截断。早期 `cpu-final` 在上一轮退出清理未收敛时触发 `safety_low_memavailable`，不是OOM。最终运行器增加互斥、PID gate、首条watchdog观测后放行及退出收敛等待。

早期失效试验原样保留并在 `handoff.json.invalid_trials` 标注：后台ADB启动导致watchdog过晚；同步PSS扰动；Android awk三元表达式错误造成过密采样；手动停止和低内存安全截断。最终验收只引用明确列出的validated/confirm/profile/verified组，不能从其它目录的单项PASS推断有效验收。修复简报见 `docs/bugs/2026-09-07-e0-measurement-instrumentation.md`。

`environment.json` / `capabilities.json` / `manifest.json` / `commands.jsonl` 保存环境、参数、dry-run及实际命令；`measurement-summary.json`、`counter-sanity.json`、`path-audit.json`、`verdict.json`保存机器可读结果。原始token、请求/响应、内存、maps/fdinfo与算子日志均在 `runs/`；原始 `.log` 另存无损 `.log.gz` 纳入Git，SHA256清单见 `raw-log-archive.json`。代码patch、构建记录与模型哈希在 `raw/`。

设备证据拉回并核验后，仅清理本次 `/data/local/tmp/llama-e0-20260907`；保留模型、既有 `/data/local/tmp/llama.cpp/cl-cache` 和原用户文件。实际清理记录见 `cleanup.json`。最终交接见 `handoff.json`。E0任务已处理完毕并停止；两个验收阻塞保留，不自动进入修复HMX、继续CPU优化或后续研究实验。
