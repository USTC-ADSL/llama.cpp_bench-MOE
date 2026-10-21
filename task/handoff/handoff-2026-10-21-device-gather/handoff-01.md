# 显式 gather 完成，进入 CPU4x8 tile preparation 融合

目标是减少单份 CPU4x8 MoE 的输入物化开销；不设固定收益，默认 async、host gather 不变。分支 feat/shared-weight-layout，阶段基线 1d5951bdc。用户 TODO.md 未动。

## 已完成与结果

1. tests/moe-shared-weight.h 增加可选完整输入 tensor + 各专家 I32 token-row map，MOE_SHARED_GATHER=htp 时 HTP 分支上传一次输入、GET_ROWS 在设备物化。只有 nt>1 且 routed_pairs+shared_rows>nt 才使用；稀疏、单 token、shared-only 保持 host。缓存 key 含 full_tokens，防止同 expert counts 不同输入长度混用。
2. tests/fullmodel-oracle.cpp 对 HTP arena 预留 4*nt*sizeof(I32)，shared 不在 HTP 时另预留完整输入。真实1K arena 仅 +16384 B，权重仍 7474348032 B。
3. 原库、原 executable 与显式候选的 source/binary hashes 已冻结在 frozen.json、evidence/explicit-freeze/。原 HTP host/DSP 内核和源码在 evidence/kernel-backup/manifest.json，备份先于内核修改。
4. probe.py --tag explicit-htp：19 cases / 136 timed samples，host/HTP 输出 bitwise 一致，真实捕获 layer0/12/23 均通过每 token NRMSE<4%。layer0 total median 22.273→21.735ms，layer12 22.369→22.018ms；HTP compute 因 GET_ROWS 增加约1.2ms。
5. confirm.py 完成8进程、48样本、1K完整模型；wetland ABBA，foundry BAAB，各2warmup+6timed，间隔30s，设备boot ID未变。相同输入的 logits、token ids、routing及owner/shared均相同。analyze.py 可重放全部统计。两进程中位数均值：wetland host1074.661→HTP1121.204ms，吞吐-4.151%；foundry1032.729→1028.346ms，+0.426%。波动明显，不删异常慢样本，不声称稳定加速。HTP主机输入复制分别减少428072960、431341568 B/请求。

## 新融合实现（尚在验证）

ggml-hexagon.cpp 增加独立 GGML_HEXAGON_FUSE_CPU4X8_GATHER 开关，匹配所有 consumer 都是普通CPU4x8 HMX MUL_MAT的GET_ROWS。要求输入及indices retained、连续且无view，不能有可见gather输出/其他consumer/插入计算。仅内部重写，公开图可回退。逻辑M参数按原gathered shape计算。

htp-ops.h 新 MM_ROWS=1<<2，不复用已有MMID_ROWS。matmul-ops.c 校验rows范围和形状，logical M取dst rows，按token_rows读取完整输入。正常/HVX/nativeMMID路径保持原参数；F16 batched传NULL。

直接HVX DDR读取版已通过小尺寸和28-case捕获测试（trace确认INDEXED_ROWS），但层0/12 total约+4–5%，HTPcompute约+2.6ms，不采用。源码/库/probe在evidence/fused-direct-freeze；fused-small、fused-captured日志均保留。

当前源码已改为 indexed DMA→现有VTCM双buffer→原FP16 tile转换。每线程最多4行在途；小M按列并行，大M按32行tile并行；不新增持久DDR输入副本。尚未取得此版性能结论。tests/htp-cpu4x8-gather-probe.cpp 新增1/2/31/32/33/71/1025行、重复和非单调indices、缓存更新、unretained/visible/interposed/extra-user回退回归。regress.py同时运行原native MMID HMX/NX-off/HVX回归。

## 命令和位置

- cmake --build build-android-opencl-hexagon --target ggml-hexagon htp-v79 -j 8
- Probe编译：NDK `/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++`，`-std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/<probe>.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/<probe>`。
- python3 task/handoff/handoff-2026-10-21-device-gather/regress.py
- python3 task/handoff/handoff-2026-10-21-device-gather/probe.py --tag <unique> --fused [--trace]
- 本任务远端 `/data/local/tmp/llama-device-gather-20261021`。默认设备fd8657d6，运行前确认在线，不改频率/affinity。

## 下一步与限制

先收齐DMA版regress，再测fused-captured DMA新tag。若单层有收益再做完整1K。完整 runner 新 --shared-fused-gather 需与 --shared-gather htp / --shared-weights cpu4x8 一起使用。fullmodel的effective字段已改为实际测量层汇总，显式旧冻结数据仍使用原字段但23层确实启用。

当前均未提交；完成验证后分阶段commit。需要最终归档原始结果/源码/库以及设备自有目录清理。CPU仍4 expert线程+4 outer线程，未满足长期1–2小核目标。当前关注prefill1K，不声明4K结果。融合跳过GET_ROWS执行，但公开图仍保留该tensor的arena预约，不声称已消除其预留内存。

## DMA 版验证补充（代码封板前）

regress.py 全部成功：CPU4x8 175 case，trace中120条indexed op均仅属于fused-projection；native HMX / NX-off / HVX各20 case，indexed执行分别10/20/0条。每case测试连续同图输入/indices更新以及交叉图缓存复访。

probe.py --tag fused-dma-captured --fused 完成28 cases、172 timed samples，均bitwise相同。真实layer0 host total22567.682→21451.537us，layer12 22676.901→21811.537us；HTPcompute分别18491.198→19064.063、18538.932→19140.469us；host gather2301.901→651.406、2347.735→687.188us。融合没有让纯矩阵执行更快，但减少了主机重复输入物化，净层延迟下降。layer23 nt1保持host回退，无优化收益声明。

已启动 confirm_fused.py 的同binary/同library host-vs-fused 1K ABBA/BAAB。fused-frozen.json及evidence/fused-final-freeze冻结全部候选库/源文件。测试过程中不修改部署版本；后续读取fused-confirmation-runs.json与dma-*.json。全部测试结束后执行 `python3 task/handoff/handoff-2026-10-21-device-gather/analyze.py --prefix dma- --candidate fused`，不要仅比较第一进程或选择稳定样本。
