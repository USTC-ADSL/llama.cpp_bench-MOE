# 同后端拆分实验交接

起点 fa7a20484。已完成用户要求的同后端显式层间同步/数据交互实验，实测而非独立层时间相加。三模式复用同一批 36 层权重、张量和节点序：single_graph / split_shared / split_copy。主结果 GPU split_copy 耗时+62.29%、能量+11.76%；NPU +15.69%/+8.87%；CPU 耗时约−2.5至−3.0%，能量首轮−3.16%、补测+1.53%，不能报CPU稳定负优化。详见 report.md。

新文件 layer-chain-bench.h 实现层图拼接及模式对照，dense-af-bench.cpp 加 --layer-chain；run-layer-energy-android.py 加 --chain 和 --devices；新增 summarize-layer-chain.py。

实现关注点：预分配所有层，在 block graph 的两个 input consumers（RMSNorm/残差）上切换到上一层 output 或独立 input；保留所有 buffers，模式切换前同步。不重新分配 tensor，不依赖 graph_cpy 深复制。组合图用 ggml_graph_add_node 保持原图节点顺序；直接 backend 执行，不使用 scheduler 或重新分配。CPU/GPU/NPU 两层与 36 层都已验证。GPU 36层相对误差约 3e-5，CPU/NPU bit-exact；所有 35 条 copy 边 bit-exact。最终验证由主代理执行。

原无线地址 192.168.50.178:43879 失效，用户提供新地址 192.168.50.178:37949。只在这个 PKR110 上测试，没有使用有线设备。

成功命令：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench > task/handoff/handoff-2026-10-03-same-backend-split/build.log 2>&1
python3 tools/dense-af-bench/run-layer-energy-android.py --serial 192.168.50.178:37949 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-03-same-backend-split/pilot --layers 0,1 --seconds 0 --settle 0 --chain --deploy > task/handoff/handoff-2026-10-03-same-backend-split/pilot-run.log 2>&1
python3 tools/dense-af-bench/run-layer-energy-android.py --serial 192.168.50.178:37949 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-03-same-backend-split/results --layers all --seconds 5 --settle 2 --chain > task/handoff/handoff-2026-10-03-same-backend-split/run.log 2>&1
python3 tools/dense-af-bench/run-layer-energy-android.py --serial 192.168.50.178:37949 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-03-same-backend-split/cpu-repeat --layers all --seconds 5 --settle 2 --chain --devices CPU > task/handoff/handoff-2026-10-03-same-backend-split/cpu-repeat-run.log 2>&1
python3 tools/dense-af-bench/summarize-layer-chain.py task/handoff/handoff-2026-10-03-same-backend-split/results --voltage-multiplier 2 > task/handoff/handoff-2026-10-03-same-backend-split/analysis-run.log
python3 tools/dense-af-bench/summarize-layer-chain.py task/handoff/handoff-2026-10-03-same-backend-split/cpu-repeat --voltage-multiplier 2 --devices CPU > task/handoff/handoff-2026-10-03-same-backend-split/cpu-repeat-analysis.log
python3 -m py_compile tools/dense-af-bench/run-layer-energy-android.py tools/dense-af-bench/summarize-layer-chain.py
git diff --check
```

已核验部署 hash、raw 积分、数值输出、边界数据、设备恢复；图目视检查。设备目录 /data/local/tmp/dense-layer-energy 仅 7 个二进制/库；CPU/GPU 控制未改，屏幕原超时和 wake lock 已恢复。无待测项目。单图只是复用 microbench 内存布局的受控基线，不是完整 llama 模型，不含输出头，不能将 pass 能量称 J/token。此前未跟踪 TODO.md 和旧三份 power-trace 文件继续保留。
