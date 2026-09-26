# MoE 延迟反馈误学习首次图分配

- **现象与触发条件**：完整模型首次调用分配 Expert 图，随后进入稳定推理。128-token 初次实验中，第一次 FFN 约 877 ms，下一次约 258 ms；无条件 EWMA 将冷启动成本学入稳态预测，计时区间预测误差从冻结模型的 2.70% 上升到 11.69%。
- **根因**：反馈更新没有区分首次执行/新增计算图和已有图复用，冷启动不是要预测的稳态工作负载。
- **修复**：首个 invocation 不更新；后续只在累计 Expert 图分配数与上次相同时接受样本。记录 `update_accepted`、分配数、已接受观测数及更新前后状态，审计脚本逐步重放。第二次预热若已稳定，可以因果地更新第一个正式样本的预测；不是排除全部预热样本。
- **验证**：Android 设备 fd8657d6，编译 `fullmodel-oracle.cpp` 后执行 `python3 tests/run-moe-context-study.py prepare-test` 和 `confirm`。修正后 128-token A/B 首次 FFN 865/849 ms 均被拒绝更新；冻结→反馈 MAPE 分别为 4.15%→1.14%、7.42%→5.76%，更新核心开销约 76/82 µs。全部长度的逐样本重放由 `python3 tests/summarize-moe-context-study.py task/handoff/handoff-2026-09-26-context-model` 验证。
- **证据与限制**：失败原始记录保留在该任务的 `evidence/pre-feedback-fix/`，最终记录在 `evidence/`。此过滤只识别首次调用和新增 Expert 图，不保证排除缺页、频率变化等其他扰动；B 组末轮突发变慢仍保留在误差统计中。反馈仅改预测，不改变当前权重归属，因此不将其误差改善表述为在线调度吞吐收益。

Android 编译命令：

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
```
