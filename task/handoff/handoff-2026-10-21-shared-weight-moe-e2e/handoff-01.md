# 接入与设备验证检查点

目标与对照协议见 protocol.md。当前已完成 shared-weight retained row slice API、HTP 对齐有界偏移支持、OpenCL plain F16 DMA import，以及 fullmodel-oracle 的 opt-in MoE grouped 2D 执行器。旧 bank/MMID 路径保留，不设置 MOE_SHARED_WEIGHTS 即可恢复。所有新投影权重由每层六个 bank owner 持有，分片与各端 tensor 验证指针别名；小 F16 scalar gate 每层一个 owner。

修改文件：ggml/include/{ggml-shared-weight.h,ggml-opencl.h}，ggml/src/ggml-shared-weight.cpp，ggml/src/{ggml-hexagon/ggml-hexagon.cpp,ggml-opencl/ggml-opencl.cpp}，tests/{moe-shared-weight.h,fullmodel-oracle.cpp,shared-weight-probe.cpp,moe-shared-weight-probe.cpp,run-fullmodel-oracle.py,run-moe-shared-weight-e2e.py}。

已执行：

```sh
cmake --build build-android-opencl-hexagon --target llama ggml-cpu ggml-opencl ggml-hexagon htp-v79 -j 8
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
# 相同编译参数构建 moe-shared-weight-probe；shared-weight-probe 只链接 ggml / ggml-base。
python3 -m py_compile tests/run-fullmodel-oracle.py tests/run-moe-shared-weight-e2e.py
python3 tests/run-moe-shared-weight-e2e.py smoke
python3 tests/run-moe-shared-weight-e2e.py bench
```

设备 fd8657d6；自有目录 /data/local/tmp/llama-shared-weight-moe-e2e-20261021。初次部署哈希 deployment.json；之后修正 OpenCL F16 import 并更新了该库，正式各进程 run.json 中 hashes 为实际最终版本。首次 standalone 链接早于库构建结束导致缺符号，等库成功后重编成功，不是源代码错误。构建保留旧 deprecated 告警。

设备分片验证：三个后端各 20 个 edge shape/batch cases 通过；测试使用两侧不同内容的三专家 bank 中间 slice，释放 parent 后运行，验证 canonical 回读、非法参数拒绝，以及修改共享 scale 后计算同步改变，排除隐藏完整复制。文件 evidence/slice-{CPU,HTP0,GPUOpenCL}.{json,log}。

设备单层 FFN：实际 layer0 权重，专家 0/7/31/59，合成输入，N=1/17/32，三端与各自 native oracle_job 对比。CPU/HTP NRMSE=0，GPU N=17/32 NRMSE 约 0.0142，所有 token 的 NRMSE 小于原有 0.04 门限。首次发现 GPU 共享 import 不接受 F16 gate，补齐 plain F16 后通过。证据 evidence/moe-layer-probe.json 和 moe-layer-probe-f16.log；原失败 log 保留用于溯源。

完整模型 smoke：wetland/c8-g15/768，2 warmups +1 timed，913699 us；有限 logits、重复 bitwise 相同、计时阶段没有新 graph allocation。唯一 expert payload 与 owner allocation 均 7474348032 bytes；三个 activation arena 合计 380657664 bytes。VmRSS 不包含全部 DMA 实体，不可把 VmRSS 当整个模型总内存。

当前 bench ABBA 已启动，输出 bench-run.log 与 evidence/{wetland,foundry}-{native,shared}-{a,b}.*。尚需分析 E2E/分段/logits/route/weight bytes、检查正式进程 hash一致、保存原始证据、报告限制并清理自有设备目录。不要把 smoke 时延当 A/B 结论；不要重用旧 archived wall time 当现场对照。不要改动 TODO.md。
