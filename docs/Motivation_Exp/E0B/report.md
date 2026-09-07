# E0B：加速器 backing 计账与配额检查

**整体验收：BLOCKED，不能作为“全部 LLM RAM 已有严格上限”的已完成实现。** 已实现并在 K80 Pro 验证两个 host 后端共用的 BudgetManager、普通 OpenCL buffer／buffer-with-properties 与 host rpcmem 的预留、登记、回滚和最终生命周期退费。尚未接入 DSP 端显式 DDR 申请，运行时内部申请的覆盖也不完整；独立 OpenCL image 没有可靠的预申请容量依据，enforce 明确不支持该路径。R 未独立校准。CPU／HTP 数值和固定请求通过，K80 上的 Qwen OpenCL 数值失败在官方 upstream 同样复现；可归因的额外开销结论仍 BLOCKED。

代码封板 commit：`897034f03`。测试以原工作树 `71da3296f` 加本次修改构建；官方对照为已有 `73ab7599b`，未更新 upstream。本次读取了 AGENTS.md、`build-npu-opencl.sh`，以及 E0 的等价结果 `E0_followup/report.md`、已有测量入口和能力／产物记录；`E0/report.md` 不存在不构成阻塞。未覆盖此前结果和用户的 AGENTS.md、Exp1 修改。没有新增 manifest、coverage JSON、patch 导出或 handoff。

- [x] 审计当前两个后端的 host 显式分配与共享生命周期，接入同一动态库账本
- [x] 私有 cgroup 中的 64 MiB 分块、跨后端、并发预留、回滚、别名、fd 复用与异步释放检查
- [x] CPU／HTP upstream、observe、宽松 enforce 的数值／128-32 单请求对照
- [x] 保存边界、开销和失败分类；清理本次设备产物
- [ ] DSP DDR／运行时全部显式 backing 覆盖，独立 image enforce 覆盖
- [ ] K80 Qwen OpenCL 数值和整模型验收
- [ ] 独立 R 校准与可归因的开销验收

设备为 `fd8657d6`，K80 Pro／SM8750／Adreno 830／Hexagon v79，设备状态在运行前确认为 `device`。内核为已有 E0 确认的 4 KiB 页配置。模型仍是 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8149865792 bytes；未复制或改写模型。复用 E0 的 128 个 token ID、32 输出、context 512、slot 1、t/tb 4、b/ub 128、F16 KV、默认 mmap、`cache_prompt=false`、`--cache-ram 0`、原放置／融合／HMX 选择。没有 expert cache、逐出、预取、流式加载或新的优化 fallback。

**预算语义与实现边界。** `B_total = B_charged + B_extra + R` 是分区约束，不是把互有重叠或不同时间的测量值相加。设备 memory 控制器实际为 cgroup v1，使用实验子组的 `memory.limit_in_bytes`／`memory.usage_in_bytes`，不是写一个不存在的 v2 `memory.max`。每个测试子进程在加载后端前进入实验组；extra cap 和 coverage 在第一次初始化时读取、之后固定。没有按 memory.current／usage 轮询借配额。watchdog 的读取仅用于安全退出。

小测试固定 `B_charged=512 MiB`、`B_extra=128 MiB`，每条物理申请序列最多三块 64 MiB。64 MiB 仅是测试块大小。整模型对照固定 charged 12 GiB、extra 64 GiB，后者是包含原有 VMEM 探测的**非约束性能控制值**，不是宣称设备有 76 GiB RAM 或该组满足某个低内存 total cap。未运行 3/4/5 GiB 完整模型扫描。模型测试保留 MemAvailable <1 GiB 连续三次及 thermal ≥SEVERE 的退出条件；没有写全机 cgroup、swap、SELinux、LMKD 或频率设置。

`GGML_ACCEL_BUDGET_MODE=off|observe|enforce`；默认 off。`GGML_ACCEL_BUDGET_EXTRA_BYTES` 为十进制字节数。`GGML_ACCEL_BUDGET_OPENCL_COVERAGE` 和 `GGML_ACCEL_BUDGET_RPCMEM_COVERAGE` 可为 `extra|charged|unknown`，默认 unknown 并保守扣 extra；这不是根据 GPU／CPU 名称自动判断 charge。性能对照的 extra 分类仅依据本机普通 CL buffer、system-heap/default-flags rpcmem 的小实验，不外推其他设备、heap 或任意扩展标志。真正已记账共享 owner 的 alias 不重复扣 extra；对应 charged-owner 检查是账本结构测试，不能冒充真实跨后端 import 的硬件证据。`CL_MEM_USE_HOST_PTR` 始终标 unknown，不自动认作零复制。

账本位于 `ggml/src/ggml-accel-budget.cpp`，仅编进 `libggml-base.so`；两后端引用其导出 API。设备测试以两个 DSO 查得的同一 snapshot 函数地址与主程序引用相等，验证没有各自的静态副本。mutex 覆盖累计 used+reserved 检查和状态转换；成功后登记 native backing ID 与单调、不复用的 generation，alias 保留 root backing generation。fd 整数不作为物理存储去重键。内存对象回调使用准确的 OpenCL 回调签名，函数驻留 ggml-base；模块引用及账本状态保留到进程结束，避免 backend 卸载／静态析构造成回调悬空。

OpenCL 不拦截首次 `clReleaseMemObject` 退费，而在对象真正销毁的回调退费；实测 child、额外 retain 和未完成命令仍保留父 backing 的扣款。尚未发布给调用者的对象若回调登记失败，成功释放其独占引用后才回滚；释放失败保守保留账目并标覆盖缺口。rpcmem 在真实 free 调用返回后按原 generation 退费，clone／PD unmap 不退费。原有 static scratch、extra pool、KV image pool、in-flight 引用继续占额；没有为了计账加入任何 clFinish 或逐 expert 同步。驱动对象销毁之后的隐式缓存是否归还 OS 不由这些回调证明。

**审计范围与尚未覆盖项：**

| 路径 | 接入／证据位置 | 结论 |
|---|---|---|
| 普通 CL buffer、COPY_HOST_PTR、独立重排／staging buffer、计算临时 buffer | `ggml-opencl.cpp:35` 统一引入 `ggml-opencl-budget.h`；原 `ggml_cl_buffer`、backend alloc、set_tensor、FA temp 的 create 调用均经过封装 | host 显式调用已接入；普通 buffer 设备边界 PASS。没有只统计 model buffer |
| CL buffer-with-properties／原 QCOM large-buffer 备用 API | 同一封装保留属性、flags、size；原 backend alloc 控制流未改 | 64 MiB QCOM 属性调用成功，CL_MEM_SIZE 和账本均 67108864；完整“普通申请先失败再 fallback 成功”链条未人为强制触发 |
| sub-buffer、image1d-buffer、tensor view、KV image pool、保留 scratch | `ggml_cl_img_pool_get_or_create`、tensor extra 与各算子原有创建点；对明确 `desc.buffer` 的 image1d-buffer 登记 alias | 实际 parent/sub-buffer/image 去重与最终释放 PASS；池持有期不退费 |
| 独立 image2D／xmem scratch image | `ggml_cl_make_image2d_half4` 等；封装查询 CL_MEM_SIZE | observe 记录驱动返回容量并标 unknown；enforce 在申请前返回 CL_INVALID_OPERATION。完整覆盖 BLOCKED |
| host rpcmem_alloc／alloc2／free | `ggml-hexagon/htp-drv.cpp:111` 起；上游 `ggml_hexagon_rpcmem_block`、shared_buffer、opqueue、VMEM 探测都走这些封装 | 申请／失败／真实 free 接入。按本机 4 KiB 取整，包含上游 fence／guard 加量；PD mapping 不当物理申请 |
| Hexagon clone 与映射池 | `ggml_hexagon_shared_buffer` 的共享 mem、`clone_buffer`、DSP `prep_op_bufs` | shared_ptr 最后释放才触发 rpcmem_free；VMEM、单块 MBUF、16 个映射 slot 均非累计物理 cap |
| DSP 显式 DDR 与 handle | `htp/main.c:83` calloc handle、`:413` unified memalign、`:553` 512 KiB DDR scratch；stop／close 释放 | **未接入 host BudgetManager，BLOCKED**。统一块包含主栈、work/DMA/HMX queues，不能把这些漏项说成已校准的 R |
| FastRPC／dspqueue 内部申请、PD、VTCM、驱动缓存 | SDK 内部没有由当前封装可观察的完整 backing/释放证据；VTCM 为独立 DSP scratch 资源 | 逐项未知，未给物理总量。没有把 PD map 大小加到 rpcmem 上 |
| SVM 与 external fd import | 本次目录审计只有 SVM capability 查询，没有 SVM allocator 或 external import/export 调用 | 当前路径 NOT_PRESENT；没有伪造真实跨后端 import 去重已测状态 |

独立 image 的反例是本次阻断 enforce 的实际证据：RGBA/F16、512×512，名义 2097152 bytes，驱动报告 `CL_MEM_SIZE=2129920`、row pitch 4096。原先只按像素量计账会漏掉 32768 bytes；最终 observe 账本改为 2129920，enforce 拒绝没有预申请容量依据的路径。该拒绝是 coverage 不支持，**不是 budget_denied**。不能通过申请后发现越界再退费来声称严格预留。更大的 driver 物理驻留量是否等于 CL_MEM_SIZE 仍未知。

**64 MiB 小测试。** 最终普通分块组为 CSV `run_id=20260907T155645Z`，最终生命周期／QCOM 属性组为 `20260907T160528Z`。下表是 observe 三块相对同进程 initialized 边界的差量；cgroup 小幅负增量来自其他已记账页变化，不能解释成“负物理内存”。

| 申请类型，合计 201326592 bytes | cgroup usage 差量 bytes | extra 账本 bytes | 可读驱动／fd 容量差量 bytes |
|---|---:|---:|---:|
| 新匿名页并触碰 | 201261056 | 0 | NA |
| 新文件页 MAP_SHARED 并触碰 | 201879552 | 0 | NA |
| OpenCL fill 并等对应 event | -569344 | 201326592 | KGSL mapped +201326592 |
| rpcmem 并 memset | -90112 | 201326592 | 各 allocation 的 fd st_size 合计 201326592 |
| OpenCL／rpcmem／OpenCL 交替 | -974848 | 201326592 | KGSL +134217728；rpcmem fd capacity 67108864 |

OpenCL 初始化时可读 `gpumem_mapped=16039936` bytes，初始化 cgroup usage 为 16547840 bytes；rpcmem driver 初始化时 cgroup usage 为 11010048 bytes。它们是当前组的观察值，不是完整初始化物理占用或 R 校准。KGSL mapped 是映射容量，fd st_size 是 backing 容量；没有将它们与 cgroup、PSS 相加计算总 RAM。FastRPC 独占物理驻留量、PD 物理量及驱动缓存上界仍 NA。

文件页对照使用新创建文件以观察本组首次 charge，不能证明已有缓存的 Qwen mmap 页全部记在新实验组。未清全局 page cache、改写模型 I/O 或迁移已有 page charge；真实模型缓存页的跨组 charge 归属没有逐页核验，同样是 B_charged 覆盖未知项。不能按“CPU／文件页”名称自动把这些页视为已被本组上限完整约束。

observe 允许第三块成功；128 MiB enforce 在单后端和交替后端的第三块拒绝。16 个并发 reservation 竞争同一 cap，enforce 仅两项获得预留，其余拒绝；该并发用例是纯账本测试，没有同时实际申请 1 GiB。额外检查包括真实 CL 无效 flags 失败回滚、rpcmem 函数指针故障注入回滚、旧 rpcmem_alloc int-size 窄化拒绝、结构化 charged alias、真实 fd 重用、各 32 次创建／触碰／销毁、额外 retain 的池复用、user-event 阻塞下的释放和完成后的退费。上述最终断言全部 PASS。缺省 off 账本为禁用状态，CSV 中的零不表示加速器实际内存为零。

budget_denied 只在 results.csv 按测试、allocation stage、请求字节、已用＋预留、上限和次数聚合记录。没有逐事件文件；较早诊断版本的拒绝行标有 `diagnostic_prior_revision`。代表性配额错误是原分配接口的 `CL_MEM_OBJECT_ALLOCATION_FAILURE (-4)`／rpcmem 返回 nullptr。上游已有 fallback 控制流保持原样；成功的最终申请记 success，不按中间失败记 OOM。

**数值和 128/32 对照。** 最终 CPU／HTP 组 `20260907T155808Z`，所有实际请求均为128输入、32输出、cache_n=0。CPU 和 HTP 的三个模式均输出相同32个 token；CSV 保存 token 序列摘要。HTP 四项小算子的三种模式结果逐项相同，沿用 RRMSE≤0.02、max_abs_error/RMS(reference)≤0.10 和全有限标准：

| HTP 用例 | RRMSE | 归一化最大误差 | 数值 |
|---|---:|---:|---|
| n1 gate/up | 0.003550824 | 0.013148587 | PASS |
| n1 down | 0.003568041 | 0.014693214 | PASS |
| n128 gate/up | 0.002798966 | 0.019480680 | PASS |
| n128 down | 0.002787277 | 0.045816010 | PASS |

HTP observe/enforce 显式账本峰值均为 7871344640 bytes（约7.33 GiB），退出汇总时仍持有43593728 bytes；汇总发生在一些静态 owner 的销毁之前，不应提前把它改成零。未触发配额拒绝。此峰值不能与健康检查时的 cgroup usage 相加构造峰值，更不包含未接入的 DSP DDR。该组 `coverage_gaps=0` 仅表示已进入封装的调用未新增缺口，不代表所有 DSP／驱动申请已覆盖。

K80 的 Qwen OpenCL 在官方、observe、enforce 三种模式均出现相同问题：n1 gate/up 的 RRMSE=0.300084266、归一化最大误差=1.295472473；n1 down 为0.311562470／1.225582186，均 **FAIL**。随后 n128 在官方 `ggml-opencl.cpp:23904`、本分支对应23905报 `status error -38`（CL_INVALID_MEM_OBJECT），由原 GGML_ASSERT 终止，exit134。相应整模型请求没有执行，记 BLOCKED。没有修改融合、放置、量化、精度容差或 fallback 来绕过；原设备上的 E0 OpenCL PASS 不能外推到本机。单独的 F32 仿射小 kernel 精确数值通过，不替代 Qwen 检查。

| 单请求，毫秒 | upstream prefill / decode / sum | observe prefill / decode / sum | enforce prefill / decode / sum |
|---|---|---|---|
| CPU | 2403.748 / 1018.927 / 3422.675 | 2694.656 / 1142.103 / 3836.759 | 2672.078 / 990.765 / 3662.843 |
| HTP | 484.852 / 1761.725 / 2246.577 | 502.575 / 1891.916 / 2394.491 | 515.571 / 1999.104 / 2514.675 |

sum 是 server 的 prompt_ms+predicted_ms，非完整 HTTP 延迟或 TTFT。单请求 sum 增量分别为 CPU +12.098%／+7.017%、HTP +6.584%／+11.934%。这不是可归因的稳定计账开销估计；CPU 控制组也有显著变化，没有给“无扰动 PASS”或性能排名。

为补足短 kernel 时序证据，另运行三轮独立启动，每次1000次预热后1000次定时，顺序 off/observe/enforce、逆序、正序。模式整个进程固定，未动态换 quota。结果为：

| 轮次 | off ms | observe ms | enforce ms |
|---|---:|---:|---:|
| 0 | 281.377 | 596.014 | 659.007 |
| 1 | 486.295 | 654.869 | 668.926 |
| 2 | 659.513 | 650.387 | 445.145 |

全部数值精确，但 off 控制本身从281到660 ms漂移。配对百分比均值 observe +48.367%、enforce +46.420%，范围分别[-1.384%,111.821%]、[-32.504%,134.208%]；不能删掉这些不利数值，亦不能把它们全部归因于账本。这里 off 是同一测试的 wrapper bypass，不冒充官方 ggml 的微基准。**可归因的额外开销验收 BLOCKED**；CSV 保留逐轮时间和 INCONCLUSIVE 汇总，未进一步调频、改变系统政策或展开性能优化。

本次有两类已解决的准备错误：最初 root 创建的部署目录不允许 adb shell 写入，改为测试目录归 shell 所有；最初 vendor-first 的 LD_LIBRARY_PATH 造成 Binder 符号加载失败，恢复 E0 的仅指定私有目录方式。另一次 HTP observe 队列错误 `dspqueue_read failed: 0x0000002e` 来自部署了旧 bin/ 下的 v79 副本。已改为使用并核验 E0 真正验证过的构建目标输出 `build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so`，随后 HTP 三模式复验通过。这些错误都没有记为 Android OOM／LMKD。

**关键运行入口与实际构建命令**（脚本默认只打印 dry-run；所有实验写操作在打印计划后才执行）：

```sh
adb devices -l
cmake --build build-android-opencl-hexagon --target ggml-base ggml-opencl ggml-hexagon llama-server -j8
cmake --build build-android-opencl-hexagon --target ggml-base ggml-opencl ggml-hexagon -j8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ -Iggml/include tests/test-accel-budget.cpp -Lbuild-android-opencl-hexagon/bin -lggml-hexagon -lggml-base -lOpenCL -ldl -o build-android-opencl-hexagon/bin/test-accel-budget
python3 tests/run-accel-budget.py
python3 tests/run-accel-budget.py --apply --modes observe enforce off
python3 tests/run-accel-budget.py --apply --tests lifecycle
python3 tests/run-accel-budget.py --apply --tests operator --modes off observe enforce --repeat 3 --operator-iterations 1000 --operator-warmup 1000
python3 tests/run-accel-budget-model.py
python3 tests/run-accel-budget-model.py --apply
python3 tests/run-accel-budget-model.py --apply --backends htp
python3 tests/run-accel-budget-model.py --apply --backends cpu htp
```

最后两个定向命令分别用于纠正 DSP 部署后的验证与最终代码复验；OpenCL 原始失败没有反复重跑。全量 build 脚本仅阅读，复用有效 Android 构建目录和既有 upstream 构建；未改 DSP 代码或 SDK。构建成功，保留原 OpenCL unused-parameter／cast-qual／TODO 警告，没有把它们写成测试失败。

收尾只留下本报告和 `results.csv` 两个实验结果文件。CSV 合并最终逐项测试、边界和时序数据，并保留有标记的早期拒绝聚合、准备错误简报及覆盖缺项；没有保留逐事件 raw、日志、JSON 或图包。设备上的本次测试进程、私有部署目录和空 cgroup 均已清理，模型保留，原 apps memory limit 仍为9223372036854771712。结果中的空字段表示未测，-1表示计数器不可读，NOT_PRESENT 不等于硬件验收 PASS。后续若要启用所有 baseline 的完整 enforce，必须先补齐明确列出的 DSP／runtime／image 覆盖和 R 证据；本次没有宣称这些工作已经完成，测试到此停止。
