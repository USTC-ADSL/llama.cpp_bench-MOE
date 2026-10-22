# 单份 shared scratch：实现与定向验证

基线274e555f4，分支feat/shared-weight-layout；默认设备fd8657d6在线。用户要求继续减少激活内存并询问shared为何不能使用同一内存。底层DMA本来支持，旧helper只绑定单一后端且shared/routed混排；本轮将共享对象从权重扩展至独占执行的shared激活。

改动：shared-activation-buffer.h增加root子集布局、多backend view和按layout绑定；moe-shared-weight.h将shared新建root与必要的NPU full_input划入common scratch，其余留在private routed arena。fullmodel-oracle.cpp支持MOE_SHARED_SCRATCH=common，三端private routed +一份common shared backing；原private路径默认保留。runner增加--shared-scratch private/common。不会将不同backend并发写入的routed区域混在一起；common每层仅有一个owner，device.compute同步、所有branch join、host归约后下一层才复用。

NPU full_input特殊处理：如果NPU是shared owner，full_input属于common并保留至graph结束供indexed matmul读取；如果shared在别端，NPU full_input留在private，避免并发覆盖。common模式因此比原shared仅NPU配置仍多一个NPU full-input预算及1MiB padding，1K合计约9MiB；相比上轮三份shared预算减少171.0625MiB。

已经执行：以NDK27d aarch64-linux-android34-clang++、-std=c++17 -O2、-Iggml/include -Iinclude -Ivendor/nlohmann，以及-Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base，分别编译tests/fullmodel-oracle.cpp、tests/moe-shared-scratch-probe.cpp、tests/test-shared-activation-layout.cpp至同名build/bin可执行文件。仅既存deprecated warning；未改/重编kernel库。Python py_compile及git diff --check通过。

Android运行moe-shared-scratch-probe MODEL OUTPUT：324用例通过，输入长度1/2/17/32/33/128/256/512/1024 × routed全CPU/全NPU/全GPU/混合 × shared轮换CPU/GPU/NPU × 3轮变化输入。与private路径逐位一致，所有owner预热后无新job分配，一份physical shared backing导入三端，base指针相同。1K容量695320576→515948544 B。记录在evidence/scratch-probe.json和.log。既存test-shared-activation-layout在Android通过，日志evidence/layout-test.log。

原binary、库、源码、模型与plan备份在evidence/backup，哈希backup.json；新部署哈希deployed.json。远端仅使用/data/local/tmp/llama-shared-scratch-20261022。

完整模型对照已经启动：python3 task/handoff/handoff-2026-10-22-shared-scratch/run.py。32进程，128/256/512/1024 × wetland/foundry × private/common × 两进程；每进程2 warmup +6 timed，ABBA/BAAB，间隔30秒。上轮joint cost model与plan冻结，CPU仍四线程，2K不跑。frozen.json固定所有输入/源码/binary；runs.json与evidence/driver.log保存进度。不要重跑覆盖。当前尚未宣布完整模型性能通过。

下一步等待完成、逐位检查新旧logits和route/owner完全相同，汇总总物理arena时必须加shared_scratch_capacity_bytes（不可只sum三个arena_capacity_bytes）。检查shared唯一owner、warmup后分配、缺页、时延。完成后归档、清理本任务设备目录、报告并Git封板。TODO.md不动。
