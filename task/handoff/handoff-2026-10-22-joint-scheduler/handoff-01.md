# 实现检查点

当前任务是 128–1K 长度感知 shared/routed 联合调度；用户暂缓 2K。基线 26b835889，分支 feat/shared-weight-layout，用户 TODO.md 不动。

快速调度支持既有模型的 shared length/tile 特征 13/14，并校验增量评分与完整 predict 一致。维持原 greedy + bounded moves/swaps，三端 shared 候选共同参与。旧 JSON 兼容，默认模型未替换。runner 允许 profile 使用 CPU4x8 单份权重。

已备份原 fullmodel-oracle、host/DSP 库、旧模型及改动前源码于 evidence/backup；backup.json 记录原文件哈希。未修改 kernel 或重建库。

已执行：

```sh
g++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-route-scheduler.cpp -o /tmp/test-moe-route-scheduler
/tmp/test-moe-route-scheduler
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-route-scheduler.cpp -o build-android-opencl-hexagon/bin/test-moe-route-scheduler
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
adb -s fd8657d6 shell 'LD_LIBRARY_PATH=/data/local/tmp/llama-joint-scheduler-20261022 /data/local/tmp/llama-joint-scheduler-20261022/test-moe-route-scheduler'
python3 -m py_compile task/handoff/handoff-2026-10-22-joint-scheduler/profile.py task/handoff/handoff-2026-10-22-joint-scheduler/fit.py tests/run-fullmodel-oracle.py
git diff --check
```

主机及 Android 调度测试均通过，exhaustive optimum=22；随机约束与长度边界测试通过。fullmodel 编译只有既存 deprecated warning。完整模型功能、收益与内存尚未验证。

profile.py 正在设备运行，driver 日志 evidence/profile-driver.log，远端仅使用 /data/local/tmp/llama-joint-scheduler-20261022。128 profile 已完成，随后 256/512/1024。训练协议见 protocol.md；fit.py 待采集结束运行，禁止重复 profile.py 覆盖已有结果。下一步检查数值和验证层排序，再决定训练上下文校准是否必要，冻结模型之后才开始完整模型留出验证。
