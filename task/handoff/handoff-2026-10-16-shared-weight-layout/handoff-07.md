# HTP 解码与流水优化，验收仍未完成

本文是V38阶段的历史记录；最终V41已通过本阶段验收，当前状态以handoff-08.md为准。

## 目标及状态

继续保持 CPU/OpenCL/HTP 一份持久 CPU4x8 V2 权重，MoE 在接口之上。各端算子含调用时准备不得比最佳有效原生路径慢超过5%。当前分支 feat/shared-weight-layout；原始内核备份 backup/shared-weight-layout-baseline。上一提交88927c55c。用户未跟踪 TODO.md 不动。

CPU V32与GPU V33各96次真实模型交替重复均满足5%，数值和别名检查通过。GPU基准开启原生DP4A。HTP V38六组数值通过，但性能未达标，不得标记完成。

## 已做实验

- V26–28交换HMX左右操作数：最终修正了HMX左输入/输出半字交错，以及activation scratch与raw weight staging别名导致的覆盖。正确但更慢，已撤回；失败日志和experiment-v28-hmx-swap.patch保留。不要重新实施该朴素gather/scatter方案。
- V29–33先转置压缩4-bit数据，再解码；HVX直接vshuff/vdeal，HMX FP16查表。无完整权重副本。
- V31增加VTCM组间128字节padding无稳定收益，已撤回。四HVX线程慢于默认六线程。
- V34按CPU4x8实际字节数分配raw staging，额外56字节保护末尾128-byte load，按最少两个均衡chunk启用双缓冲；N32过门槛，N128仍慢12%–14%。原生layout wrapper不变。
- V35/36 HVX vlut32表槽距错误导致NRMSE失败，未采用。指令语义实际为从halfword槽取低byte，nomatch索引低5位；V37改为int16_t表，重复两遍16个有符号nibble值。HMX vlut16按word槽取low half，低nibble用nomatch消除mask。V37的20组边界和六组真实模型正确性均通过。
- 指令语义参考QEMU target/hexagon/imported/mmvec/ext.idef，vlutvvb_nm: `idx=(idx&0x1F)|(matchval<<5); fGETBYTE(oddhalf,VvV.h[idx % fVELEM(16)])`；SDK只有声明，无此细节。网址 https://raw.githubusercontent.com/qemu/qemu/master/target/hexagon/imported/mmvec/ext.idef 。
- V38 HMX连续K块内不重复fastdiv及八组行stride乘法；分组起点只在跨32输出行时重算，内部指针每次增加72。反汇编确认移出循环，代码无vector spill。真实模型通过。

## 实测摘要

HTP DSP op median微秒，顺序K2048/M1408的N1、32、128，再K1408/M2048的N1、32、128：

| 版本 | 原生 | 共享 |
|---|---|---|
| V34 | 51,86,115,51,89,110 | 54,85,129,51.5,87,125 |
| V37 | 54,88,112,51,86,109 | 54,87,127,51,84,123 |
| V38 | 51,87,112,51,86.5,109 | 54,81,121,51,81,117 |

V32 CPU共享/最佳原生比值：1.0033,1.0008,1.0001,1.0027,1.0004,0.9995。
V33 GPU共享/原生DP4A比值：0.8582,1.0196,0.9237,1.0310,0.9208,0.9221。

默认设备fd8657d6处于device状态。日志/JSON/hash均在本目录evidence中对应tag。HTP诊断trace必须TIMING=0；runner --htp-trace已做到。接受性能只用DSP profile1，不用host/RPC时间。V30 trace显示额外时间集中在权重解码/转置，未发现重复解码；HMX乘法本身类似。HMX输入/输出不是普通row-major，切勿再次误判。

## 当前修改及命令

主要未提交文件：htp/hvx-mm-kernels-tiled.h、hmx-mm-kernels-tiled.h、matmul-ops.c/.h、shared-cpu4x8.h，tests/shared-weight-probe.cpp和run-shared-weight.py（诊断选项、错误输出）。

增量DSP：`cmake --build build-android-opencl-hexagon --target htp-v79 -j 8`。
设备测试：`python3 tests/run-shared-weight.py --deploy --shared --layout cpu4x8 --model --profile --tag <unique-tag> --backend HTP0 --reps 48`；边界改用`--edges --reps 8`。
CPU/GPU最终复测仍需边界；GPU加`--gpu-dp4a`。CPU持久线程池4线程/mask f0；辅助CPU无运行时权重转换。

## 待完成

- 收取V39三chunk实验；V40实验将HVX累加保留qf32至最终输出，尚待正确性/性能测试。每次部署前等对应构建完成。
- HTP <=5%最终稳定验收仍是硬性未完成项；decode K2048也有51/54us波动，需要持续优化和重复测量。
- 当前HMX chunk从3开始是实验选择，最终需根据实测选择，处理n=64小矩阵，并采用有界选择过程。
- shared-cpu4x8.h现只有未使用旧helper，cpu4x8_store_hmx_half也已不再使用；删除前确认引用，保留原生内核，将VTCM overread说明移到实际读取处。
- 接口只支持普通2D Q4_0 MUL_MAT，生命周期由handle负责，adapter借用storage。审查未发现确定bug；补齐公开说明、相邻V1/native回归、最终边界、稳定统计和内存报告。
- 整理diff并提交独立检查点；保留失败证据，清理自有设备目录失败JSON等。设备目录/data/local/tmp/llama-shared-weight-layout有所有权标记，不触碰其他目录或模型。
