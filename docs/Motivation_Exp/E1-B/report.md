# E1-B：共同内存配额下的官方静态配置单次扫描

2026-09-09。E1-B 仅用于本任务命名。按用户最后追加指示，正式阶段每配置/分区只做一次独立启动、同 server 的 C1/W；取消三次重复、额外归属复验、关键点五输入复验及串行20请求。R 沿用用户明确接受的 **R_est=256MiB**，全程固定，不再要求独立校准。结果是限定设备与 wrapper 口径下的单次观测，不宣称全局最优、稳定退化或严格 total-RAM cap。

**结论：无约束下B0–B6全部完成C1/W；5GiB已测17个配置/分区全部在加载阶段失败，其中13个为budget_denied、4个为300秒加载安全超时。CPU-only补测同样触发加载安全超时，未进入推理。没有观测到这些配置在预算压力下自动换出/轮换权重并继续推理。** 用户澄清目的只是了解llama.cpp有无自动swap支撑，并要求失败即可报告；原扫描在B4/T5E2完成并清理后停止。随后按用户明确追加要求，仅补跑一次B0/T5E0，不重跑其他点。没有把未执行点记为失败。

范围：5GiB的B1/E0.5、B2/E1未执行；所有observe补充候选及4/3GiB均取消。原计划的公平完整分区扫描未完成，因此受限数据不能用于给七类baseline做全面排名。CPU-only仅证明本次构建、参数、组策略及300秒加载窗口内未完成，不外推所有CPU配置或启用swap后的可行性。已完成：已有验收复用、dry-run/小测试、单次无约束对照、17个受限加载尝试、结果整理和设备清理；其余扫描按用户要求终止。

正式阶段共 24 次启动、31 行结果，其中完整 C1/W 成功的启动为 7 次。正式逐行状态：`{'success': 14, 'budget_denied': 13, 'safety_stopped': 4}`。包括此前校准在内，results.csv 共 41 行：`{'success': 22, 'measurement_invalid': 2, 'budget_denied': 13, 'safety_stopped': 4}`。加载失败行代表拟执行 C1 的尝试，实际请求未发送、tokens 为0；没有添加虚构的 W 行。

## 预算与执行设置

设备 `fd8657d6`（实际为 Redmi K80 Pro / 24122RKC7C，Adreno830、HTP v79；不冒充普通 K80）。MemTotal=15476712KiB，8个在线CPU核。模型为 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8149865792 bytes，沿用 E0 的 Qwen1.5-MoE-A2.7B-Chat Q4_0；实际架构 qwen2moe，24层、60 routed experts、top4。

`B_total = B_charged + B_extra + R_est`，1GiB=1073741824 bytes；R_est=268435456 bytes。B_charged 约束实验组已记账页；B_extra 是 OpenCL/HTP 共用的未记账 backing 账本 cap，两后端不各获一份。used/reserved、pool-idle、in-flight、pending-free 都占额，沿用已有最终释放语义。同 run 不借配额。

| 名义预算 GiB | extra=0 的 charged | extra=0.5 的 charged | extra=1 的 charged | extra=2 的 charged |
|---|---:|---:|---:|---:|
| 5 | 4.75 | 4.25 | 3.75 | 2.75 |
| 4 | 3.75 | 3.25 | 2.75 | 1.75 |
| 3 | 2.75 | 2.25 | 1.75 | 0.75 |

原计划：B0无加速backing，有限预算只取extra=0；B1–B6初始化/模型路径需要正extra，粗候选为0.5/1/2GiB，各配置分配相同候选机会。粗候选均失败时，按该配置无约束observe生命周期extra峰值向上取整到1/16GiB补一个候选；与粗候选重合则再加1/16GiB，且须charged>0。本次提前停止，尚未执行补充候选。上述4/3GiB分区仅为固定定义，没有实际运行。observe峰值不是任意fallback路径的物理下界，不构成全局不可部署证明。

实际控制器为 **cgroup v1**，专用目录 `/dev/memcg/llama-e1-budget/<run_id>`。有限分区在 exec/backend初始化/模型加载前设置 `memory.limit_in_bytes=B_charged`、`memory.memsw.limit_in_bytes=B_charged`、`memory.swappiness=0`，然后只将本次子进程移入组。真实口径是该组受记账的 memory 与 memory+swap 限额；swappiness=0 不等价 v2 的 memory.swap.max=0，不宣称严格无 swap。未修改全机 swap/LMKD/SELinux。启动 shell 继承 oom_score_adj=-1000；专用组 oom_kill_disable=0。保留该启动政策，不能把配额下等待/回收后的 watchdog 退出改写成 kernel OOM kill。

无约束组 memory/memsw 不设有限 cap，wrapper 使用 observe、extra非约束参考值64GiB；这个数不是实际RAM。有限组使用 enforce、extra cap 在第一次初始化前冻结。当前未覆盖完整 DSP/runtime 私有容量，独立 image enforce 仍有限制；复用 [E0B 最终工程验收](../E0_E0B_final/E0B.md) 的范围及用户接受的 R_est，不把 coverage_gaps=0 解释为完整物理覆盖。

前置资料已读取AGENTS.md、E0/REPORT.md与E0_followup/summary.csv、E0B/report.md与results.csv、E1/report.md与summary.csv，以及E0_E0B_final/E0.md和E0B.md；实际文件命名差异未判作缺项。复用最终工程PASS中的覆盖、跨DSO共享账本、池/in-flight/pending-free释放和开销验证，不要求另交JSON/日志清单。共享BudgetManager位于ggml/src/ggml-accel-budget.cpp，仅归属ggml-base；OpenCL入口为ggml/src/ggml-opencl/ggml-opencl-budget.h，HTP为ggml/src/ggml-hexagon/htp-drv.cpp。执行复用tests/run-cache-closure.py的文件准备/先入组，以及tests/run-static-coexist.py和E0/scripts/watchdog.sh的计时/安全逻辑。

USB供电、原有DVFS/融合/异步提交策略；正式 run 启动门限统一为 thermal=0 且电池≤38°C。早期校准门限35°C，在正式数据开始前调整，校准不并入正式比较。保留运行自身温升。原 watchdog 每秒观察：MemAvailable<1GiB持续3秒或HAL thermal≥SEVERE停止，加载300秒/每请求180秒超时；这些是安全截断，不是交互SLO或OOM。C1/W之间等待watchdog实际观察到idle，使W获得独立超时窗口；不清权重或文件缓存。

原计划预算顺序无约束→5→4→3GiB，实际在5GiB阶段结束；原粗候选顺序用Python Random(9092026 + 预算GiB×100)打乱，无约束取0。后续B0/T5E0是用户指定的单点补测，固定排在最后，未声称该点顺序随机。每点只启动一次，实际顺序、时间、起止温度保留在CSV。起跑另要求MemAvailable≥2GiB及USB供电，等待冷却不计入加载时间。

每个独立启动前，复用已验证 `model-cache-prepare --evict <GGUF>`：检查无模型持有者，仅对该文件DONTNEED并确认残留≤16MiB、元数据不变。无全局drop_caches。模型先入组后加载。按用户“直接测试、不用验证”追加指示，正式阶段不再运行 model-page-charge 归属探针；相关列留空，不能解释为本次重新证明组外页为0。历史缓存归属闭环依据 [E0B最终报告](../E0_E0B_final/E0B.md)。

## 静态配置、输入与实际命令

| ID | 放置参数 | routed CPU层数 | 既有已核验范围 |
|---|---|---:|---|
| B0 | `--device none -ngl 0` | 24 | CPU routed/shared/attention，mmap+官方repack |
| B1 | `--device GPUOpenCL -ngl 99` | 0 | OpenCL全层，保留CPU支持操作 |
| B2 | `--device HTP0 -ngl 99` | 0 | HTP单PD，保留CPU embedding/output/router softmax等 |
| B3 | `B1 + --cpu-moe` | 24 | routed CPU；shared/attention OpenCL |
| B4 | `B2 + --cpu-moe` | 24 | routed CPU；shared/attention HTP |
| B5 | `B1 + --n-cpu-moe 12` | 12 | 前12层routed CPU，后12层及shared/attention OpenCL |
| B6 | `B2 + --n-cpu-moe 12` | 12 | 前12层routed CPU，后12层及shared/attention HTP |

参数由本地 common/arg.cpp:2787–2801 及设备 --help 确认；qwen2moe.cpp 的 routed/shared 是分离分支。0/12/24是专家CPU层数，不是专家数量；--cpu-moe不把shared搬到CPU。放置依据复用 [E1](../E1/report.md)，本次记录日志的实际offloaded层数与最终成功/拒绝，不重跑独立放置诊断。未启用新缓存/预取/流式调度、改量化或强制绕过官方fallback。

mmap并不保证所有CPU专家最终存储都直接映射GGUF。当前官方构建启用CPU_REPACK；CPU override在 src/llama-model-loader.cpp:1227 进入CPU候选buffer选择，src/llama-model.cpp:1030 优先考虑extra buffer。直接用mmap创建backend buffer还要求默认buffer类型（src/llama-model.cpp:1734）；CPU_REPACK属于非默认类型，会另行分配并repack（ggml/src/ggml-cpu/repack.cpp:4726）。因此CPU-MoE也可能承受额外已记账分配压力。本次未关闭repack；源码机制不能替代逐run的RSS/分配证据，也不能据超时直接断言具体驻留构成。

对“自动swap”的回答应拆开：① 本地src/llama-mmap.cpp:471–481使用MAP_SHARED、PROT_READ映射模型，内核可以回收干净文件页，之后从GGUF重新读入；这属于文件按需分页，不需要把这些干净页写进swap。② 匿名内存的swap/zram由内核与系统/组策略决定，不是llama.cpp自行管理的模型交换。本轮memory与memsw同限、swappiness=0限制了该路径，未评估打开系统swap后的效果。③ 本次所测静态offload分配在13次extra拒绝后直接加载失败，未出现通过自动减小offload或逐层交换继续成功的行为；不能把这一有限测试外推为所有llama.cpp版本/选项都无任何相关机制。对本轮的CPU_REPACK及加速器另行分配，单靠mmap选项也不能保证文件按需分页能覆盖全部容量。

各类只使用共同初始组合 `(t,tb,ub)=(4,4,128)`，没有从四种允许组合中分别挑选有利结果，也未追加6/18层或部分offload点。固定 logical batch512、context2048、slot1、F16 KV、mmap、no-warmup、cache-ram0、fit off；OpenCL磁盘kernel cache延续E1设为off。每次请求temperature0/seed1234、ignore_eos=true、cache_prompt=false，目标128输出；C1为首请求，W为同server第二请求，成功行均实际prefill512、输出128、cache_n=0。C1不称为完全冷权重，官方启动探测保持原样。

本地确有 `--fit`：common/arg.cpp:2895–2896将其定义为调整未设置的参数以适应设备内存。本轮为固定静态配置显式使用 `--fit off`，因此没有测试该启动参数拟合功能；它与推理期间自动换出权重不是同一机制。本轮失败不能解释为已启用并否定了所有自动内存适配能力。

```sh
# 专用组已经创建；所有变量取自该行固定分区
echo "$B_charged" > "$group/memory.limit_in_bytes"
echo "$B_charged" > "$group/memory.memsw.limit_in_bytes"
echo 0 > "$group/memory.swappiness"
echo $$ > "$group/cgroup.procs"
# 随后 exec：完整 env/命令逐行保存在 results.csv 的 command 列
./llama-server -m /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf \
  -c 2048 -np 1 -t 4 -tb 4 -b 512 -ub 128 -ctk f16 -ctv f16 \
  --load-mode mmap --no-warmup --cache-ram 0 --fit off \
  --host 127.0.0.1 --port 18091 <上述放置参数>
```

环境统一为本次私有目录的 LD_LIBRARY_PATH/ADSP_LIBRARY_PATH、GGML_HEXAGON_DEVICES=1、GGML_OPENCL_KERNEL_CACHE_DIR=off、GGML_ACCEL_BUDGET_REPORT=1、两后端COVERAGE=extra、E0_MODE=perf。GGML_ACCEL_BUDGET_MODE及EXTRA_BYTES按上述分区设置。实际执行入口：`python3 /tmp/e1b_scan.py plan`、`prepare`、`small`、部分`calibration`/定向`run`，随后`scan`。等待期间的host runner重启按run_id跳过已完成点，没有增加正式重复。原构建有效、未重新编译；部署逐文件hash核对。OpenCL库SHA256为 `70bac8ab6b0a692507e180e04ece6c9dc5e3e1d4bca4901416b832cbb666c20f`，HTP v79库为 `48b3aeb4fa140d8ec0469c60f46ac381811d1a25d8bf6ec4619bba9fcf19900f`，与E1已验证产物一致。

正式单次扫描固定输入ID `cal1`（叙述/规划类）。其文本为以下主题、一换行、以下补充句组用单个空格连接重复20次；经本模型 `/tokenize`、add_special=false 后截取前512个token ID，HTTP请求直接传该数组。两条校准输入和五条保留输入已经冻结；按追加指示，五条保留输入未运行，代码类 cal2 只存在于先前无约束校准，未用于受限预算结论。

> Plan a volunteer repair workshop at a public library. There are four volunteers, six tables, thirty chairs, and a quiet reading session from ten until eleven. Electrical repairs are prohibited. Explain safety, priorities, staffing and closing checks.

> Use these facts as a planning exercise. State assumptions clearly. Distinguish requirements from suggestions. Explain the sequence of actions, the reason for each action, and ways to check the outcome. Include a concise example. Avoid unsupported claims.

## 结果与边界

时间口径：加载是启动器go放行到health成功的墙钟，含轮询误差；加载未完成时load_ms为空。TTFT是server接收完整POST到首token-ready；p95 ITL为127个相邻token-ready间隔的线性插值95分位数。完整请求时间为同一请求的接收至末token-ready，不含加载，不拼不同配置的最好分量。以下按完整C1请求时间选择该配置/预算的最佳已扫描成功分区；每行C1/W来自同一次启动。

| 预算 | 配置 | 分区/结果 | load s | C1 TTFT ms | C1 p95 ITL ms | C1完整 s | W完整 s | C1相对同配置无约束 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 无约束 | B0 | U | 19.51 | 10650.15 | 47.00 | 16.36 | 14.86 | 1.00× |
| 无约束 | B1 | U | 42.15 | 3188.82 | 43.57 | 8.56 | 8.53 | 1.00× |
| 无约束 | B2 | U | 14.67 | 2071.36 | 77.35 | 10.70 | 10.65 | 1.00× |
| 无约束 | B3 | U | 30.63 | 6759.42 | 56.47 | 13.81 | 13.26 | 1.00× |
| 无约束 | B4 | U | 22.09 | 7527.34 | 58.94 | 14.77 | 13.38 | 1.00× |
| 无约束 | B5 | U | 43.56 | 5896.60 | 50.96 | 12.17 | 11.10 | 1.00× |
| 无约束 | B6 | U | 18.29 | 5634.53 | 65.90 | 13.30 | 12.82 | 1.00× |
| 5GiB | B0 | T5E0；safety_stopped×1 | NA | NA | NA | NA | NA | NA |
| 5GiB | B1 | budget_denied×2 | NA | NA | NA | NA | NA | NA |
| 5GiB | B2 | budget_denied×2 | NA | NA | NA | NA | NA | NA |
| 5GiB | B3 | safety_stopped×1, budget_denied×2 | NA | NA | NA | NA | NA | NA |
| 5GiB | B4 | budget_denied×1, safety_stopped×2 | NA | NA | NA | NA | NA | NA |
| 5GiB | B5 | budget_denied×3 | NA | NA | NA | NA | NA | NA |
| 5GiB | B6 | budget_denied×3 | NA | NA | NA | NA | NA | NA |

- 无约束最短完整C1的已扫描成功点：**B1 / U**，C1 8.56s，TTFT 3188.82ms，p95 ITL 43.57ms，加载 42.15s。
- 5GiB：没有完整C1/W成功点；这只表示当前配置、候选与安全窗口内未完成，不外推全局不可部署。

charged_peak_bytes 是该独立cgroup的生命周期 max_usage；extra_peak_bytes 是共享账本退出SUMMARY中的生命周期 used+reserved峰值。它们包含加载及同server全部请求，在C1/W行重复附带，peak_scope已明确；不是每请求独占峰值，二者不相加。若强制终止来不及输出SUMMARY，extra峰值留空，不能以cap或0冒充实测峰值。普通CPU正式点无加速backing时extra为0。swap_peak_sample_bytes未独立测量，留空。

## 失败、限制与收尾

预算拒绝、驱动失败、unsupported、明确kernel OOM、未知SIGKILL及安全截断分别记状态；fallback最终成功仍记success。Cgroup failcnt不单独证明OOM。所有启动/请求失败和原始数值保留在同一结果表，没有按失败或运行后温升删行。

13个预算拒绝的临时日志均为GGML_BUDGET_DENIAL后模型加载失败；原始拒绝记录已逐条写入CSV reason，字段顺序为operation、backend枚举、请求bytes、当时已占用bytes、cap bytes、count。HTP日志中的rpcmem_alloc failed在这些点是预算拒绝的后果，未重复记为driver_failed。原3个安全超时分别为B3/T5E2、B4/T5E1、B4/T5E2；其charged峰值分别为2952790016、4026531840、2952790016 bytes，达到各自上限，extra退出峰值缺失。CPU-only补测增加第4个加载安全超时，详见下段。没有得到明确kernel OOM kill证据。

CPU-only补测：`scan-B0-T5E0-cal1-r1`，2026-09-09T13:20:38Z启动，命令复用上文公共参数并加 `--device none -ngl 0`；wrapper为enforce/extra=0，cgroup memory与memsw均为5100273664 bytes（4.75GiB），R=268435456 bytes。一次启动后加载触发 `safety_load_timeout`，exit=143，未发送C1/W；load_ms、TTFT及ITL留空。charged峰值5100273664 bytes，extra=0，extra拒绝数0，cgroup failcnt=0。电池温度36.1→40.7°C，最低MemAvailable=6172924KiB，最高thermal状态0。该行保留为安全截断，不记成预算拒绝或OOM。

该CPU-only加载过程的一次只读快照为：memory.usage_in_bytes=5100187648、rss=5004357632、cache=50372608、swap=0、workingset_refault_file=73410654、pgmajfault=4469，oom_control的under_oom=0、oom_kill=0。它显示组内大量匿名RSS及文件页反复读入，不能把“CPU-only+mmap”理解为全部权重只占可回收文件页；这与上文官方CPU_REPACK另行分配机制相容，但该快照没有逐buffer归因，也不是swap峰值或完整OOM历史。未尝试关闭repack、改变swap政策或延长安全窗口。

B4/T5E2加载期间一次只读快照：memory.usage_in_bytes=2952716288，限额2952790016；memory.stat的rss=2884288512、cache=25563136、swap=0、workingset_refault_file=29379441、pgmajfault=1813；memory.oom_control为under_oom=0、oom_kill=0。该时刻显示已记账容量接近上限并有大量文件refault，支持存在回收压力的解释；这是单时刻观测，不是swap峰值、完整OOM历史或CPU_REPACK独占RSS测量。

前置跨后端mixed分配小测试8项、lifecycle19项均通过。早期校准B3/cal1/r1的C1/W完成推理，但运行后的归属探针检查失败，保留2行measurement_invalid及耗时；组外页观察值7471104 bytes小于16MiB，原执行未保存完整探针细项，不能追认确切失败子条件。已在途的r2检查通过，同样保留。用户随后明确取消进一步验证，正式阶段不再运行该探针，未将这些测量无效行伪装成OOM或预算拒绝。

本轮每点仅一次，正式受限扫描只用一种输入；**两类输入、相邻预算下约1.5×稳定退化的建议推进门槛未评估**。有限预算没有成功请求，无法计算同配置的有限/无约束延迟退化，失败和300秒安全阈值不替代延迟比值。不能用C1/W当作两次独立重复，不能做稳定因果/显著性宣称。本次不能判定CPU-MoE在目标预算下已足够，也不能凭失败就建议必须开发新调度；用户的自动swap摸底到此结束。

原扫描已执行 `python3 /tmp/e1b_scan.py cleanup`，返回CLEANUP PASS。CPU-only补测使用临时入口 `python3 /tmp/e1b_cpu_once.py`，重新部署相同构建产物并核对传输hash，仅运行上述一个点，沿用文件级模型缓存准备与原watchdog，未追加校准或推理重复；结束后同样返回CLEANUP PASS。两次清理均在server退出、专用组为空后删除本次实验组及部署目录；GGUF模型和其他实验目录保留。仅交付本报告与results.csv，历史E1目录迁移及用户其他修改保留。计算用临时时序/日志与临时runner不归档，不新增图、manifest、handoff或额外汇总。
