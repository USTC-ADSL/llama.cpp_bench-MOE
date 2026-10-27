# 多长度重新校准：工具和协议阶段

用户要求重新校准1K以上，并报告校准后收益。目标fd8657d6在线，模型固定Qwen1.5-MoE Q4_0。前任务结论：2K旧模型预测NPU31.6ms而实测44ms，shared固定7.03ms；当前新任务重新采样，不拿wetland/foundry拟合。

只修改采样工具：tests/fullmodel-oracle.cpp添加可选MOE_CAPTURE_LAYERS，仍记录24层形状，仅保存选择的层张量；tests/run-fullmodel-oracle.py添加--capture-layers并只拉取对应bin。默认未传时仍捕获全层。此举避免4K捕获常驻全部24层激活。已增量交叉编译fullmodel-oracle，无库/内核改动。1024、2048选择捕获成功，1024的54个候选层profile逐token4%校验通过。

编译命令：

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
python3 task/handoff/handoff-2026-10-27-long-context-recalibration/run.py deploy
python3 task/handoff/handoff-2026-10-27-long-context-recalibration/run.py calibration
```

日志在evidence/build.log、deploy.log、calibration-driver.log；frozen.json固定binary/库/源码和prompt来源。设备目录/data/local/tmp/llama-long-calibration-20261027，仅本任务使用。run.py calibration目前运行中，不可重启覆盖已有tag；阶段状态以runs.json和driver日志为准。

technical-long.txt由既有独立technical.txt重复6节生成，确保4K前缀充足，明确不是wetland/foundry。采样长度顺序1024/2048/4096/1536/3072，层0/12/22；每层18种CPU/GPU routed分配与三种shared owner，2预热5计时。profile使用全三端resident alias、pooled、CPU4x8、HTP fused gather、NEON reduction。捕获bin拉取哈希核验后本地gzip并验证解压，再删除设备上对应bin，保留其余证据。

拟合fit.py：6特征(active branch, active experts, routed pairs/128, tiles32, shared bool, shared tokens/1024)，使用原NNLS算法。仅长度1024/2048/4096的层0/12训练，长度1536/3072及层22留出；所有样本验证映射到C++26项系数的数学一致性。导出runtime-isolated.json；之后需在独立technical完整模型采集1024/2048/4096的分支实测，fit.py context生成带context_scales的runtime-calibrated.json。CPU不继承人为1.5倍，使用实测校正。应检查内存/温度/误差再冻结，不能把明显换页污染当纯计算成本。

收益验证confirm.py：冻结模型后，2048和4096各12进程；wetland NPU/old/new/new/old/NPU，foundry new/NPU/old/old/NPU/new；每进程2预热6计时，间隔30秒，单chunk。同一个binary/库，old/new同resident计划和pooled，仅切换route-model。模型不得按held-out结果再修改。若4K失败，保留证据并明确未通过，不自动重试或静默减chunk。

下一步：等待校准采样完成，核对所有270个候选层数值及内存事件，执行fit.py isolated；再做technical context校准、冻结模型和收益对照。不要动用户TODO.md、paper/、results/。最终需分析/归档/清理设备目录并提交结果。
