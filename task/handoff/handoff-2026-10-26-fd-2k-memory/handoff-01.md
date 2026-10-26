# fd 设备 2K 内存复测

## 目标与协议

用户要求在 fd8657d6 上检查 2K 是否仍然 OOM。本轮复用 10-22 长度扫描的 2048-token 单 chunk prefill、wetland/foundry 输入和 runtime-fast-cpu150 异构配置，使用当前源码 a7dffbce1 增量构建。原生 NPU 对照及 CPU4x8 单份共享权重异构各两个输入，每进程 2 预热 + 6 计时。没有 decode，没有缩小 chunk，没有调频、绑核、清空缓存或主动停止后台应用。

推理为 shell UID 2000；仅内存采样和 dmesg 读取使用已有 Root。观察 /proc/meminfo、/proc/vmstat、memory PSI、进程 status/stat，约每 5–7 秒采样；before/after 保存 dmesg、logcat、boot ID。原始日志在 evidence/，不纳入 Git；汇总、命令与哈希纳入 Git。

## 构建和启动

首次 `cmake --build build-android-opencl-hexagon --target htp-v79 fullmodel-oracle -j 8` 未执行编译即失败：fullmodel-oracle 不是 CMake target。

实际成功命令：

```sh
cmake --build build-android-opencl-hexagon --target htp-v79 llama -j 8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 task/handoff/handoff-2026-10-26-fd-2k-memory/run.py
python3 task/handoff/handoff-2026-10-26-fd-2k-memory/run.py --resume-setup
python3 task/handoff/handoff-2026-10-26-fd-2k-memory/analyze.py
```

首次 run.py 在启动任何模型进程之前因系统日志含非 UTF-8 字节失败。仅为日志解码增加 backslashreplace，再由受限 --resume-setup 恢复（必须不存在 runs.json 和模型 run manifest，冻结 hashes/boot/model stat 必须完全一致）。旧采集和异常日志保留；没有重跑任何失败推理样本。

部署固定目录 /data/local/tmp/llama-fd-2k-memory-20261026，9 个二进制/库均校验 SHA256，DSP 来自 build/ggml/src/ggml-hexagon/libggml-htp-v79.so。沿用原 runner MM_SELECT=3；当前源码的 >=2 条件启用 HMX，该值不是禁用 HMX。

## 已确认的中间结果

- wetland NPU：完成 8 次请求，退出 0，中位 2356.135 ms，计时 major faults 0–1。
- wetland 异构：完成 8 次请求，退出 0，中位 16954.781 ms，计时 major faults 30789–44000。
- 异构期间新增两条 `kgsl: out of memory`，以及大量 LMKD 后台进程终止日志。不能把推理退出 0 描述为无 OOM。KGSL 两条日志未标识申请者，不能据此认定是 fullmodel-oracle 自身 GPU buffer 申请失败。
- 异构系统可用内存采样最低 520228 KiB，进程换出采样最高 940180 KiB；RSS 不覆盖全部 DMA-BUF/设备共享分配，不用 RSS 代表总物理工作集。

## 最终状态与后续入口

四组均已完成并退出 0。foundry 异构中位 16905.998 ms、NPU 2988.295 ms。最终共捕获 5 条 KGSL OOM 和至少 319 条 LMKD Kill；详见 results.md、summary.json、memory-events.json。32 次完整请求、24 次计时完成；tokens/chunk=2048、配套库 hash、有限 logits、boot ID 均已核验。不要重跑 run.py 或覆盖 frozen.json；后续复测需新目录。

已保留 72 个原始 evidence 文件并生成 evidence-manifest.json；确认无测试进程后仅删除本任务远端目录，模型 stat 未变，见 cleanup.json。Git 检查点包含报告、脚本、汇总和证据哈希，原始 evidence/ 留在主机本地且忽略 Git。用户 TODO.md、paper/、results/ 均未改动。没有修改推理实现，不把内存问题记录为已修复。

若后续要求优化，应以本轮系统 OOM/LMKD 和分页证据为起点，针对 2K 异构总工作集做分配归因；本轮没有确定单一 buffer 的净内存贡献，不应直接把 KGSL 失败归因到推理进程。
