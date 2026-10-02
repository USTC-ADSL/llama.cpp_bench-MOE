# Attention 核心 CPU 实验交接

起点 a398ffc66。用户要求把 Attention 的线性投影迁到 NPU，仅核心留 CPU，测试真实吞吐及功耗。任务已得到设备结果：新 core-cpu 为 13.256 token/s、3.733 W、0.28160 J/token；本次同频全 NPU 为 16.634 token/s、4.026 W、0.24204 J/token，稳定窗口能耗高 16.35%。详见 report.md，不要混用不同日期的小幅功率变化解释为新切分收益。

改动：

- src/models/qwen2.cpp：新增可选 `LLAMA_QWEN2_CPU_ATTENTION=core`。先展开 embedding，在 QKV 和 RoPE 之间记录图边界；调用不带 wo 的 build_attn，pin 核心 CPU，再独立添加数学等价的 O 投影并 pin NPU。原 mixed 的整段 CPU pin 和全 NPU 默认路线保留。
- dense-af-decode.cpp：接受 core-cpu，保留 NPU 权重与 CPU KV；审计 TSV 增加 src0。
- run-real-decode-android.py：增加 core-cpu case，沿用 1.4016 GHz。
- summarize-real-energy.py：支持 --cases 和 --candidate，原默认用法保留。

已执行命令：

```sh
cmake --build build-android-opencl-hexagon -j 8 --target dense-af-decode > task/handoff/handoff-2026-10-02-attn-core/build.log 2>&1
python3 tools/dense-af-bench/run-real-decode-android.py --serial 192.168.50.180:44159 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --directory /data/local/tmp/dense-af-real-energy --output task/handoff/handoff-2026-10-02-attn-core/audit-verified --phase audit --steps 200 --cases core-cpu --deploy > task/handoff/handoff-2026-10-02-attn-core/audit-verified-run.log 2>&1
python3 tools/dense-af-bench/run-real-decode-android.py --serial 192.168.50.180:44159 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --directory /data/local/tmp/dense-af-real-energy --output task/handoff/handoff-2026-10-02-attn-core/results --phase measure --steps 200 --cases core-cpu npu-host1401 --power > task/handoff/handoff-2026-10-02-attn-core/run.log 2>&1
python3 tools/dense-af-bench/summarize-real-energy.py task/handoff/handoff-2026-10-02-attn-core --cases core-cpu npu-host1401 --candidate core-cpu
python3 -m py_compile tools/dense-af-bench/run-real-decode-android.py tools/dense-af-bench/summarize-real-energy.py
git diff --check
```

有效审计在 audit-verified/：部署哈希逐项对比最终完成构建，全部一致；逐层验证 144 个 NPU Attention 投影、108 个 NPU FFN 矩阵乘、72 个 CPU RoPE、36 个 CPU FA、72 个 CPU KV 写入。对应 verification.json。正式测量 profiling=0，功率积分、CPU 时钟、无休眠、KV/token 数和设备恢复均通过断言。图已目视检查。额外列 buffer 的 TSV 与原 DictReader 分析兼容。

失败尝试：首次 audit/ 启动于增量构建尚未结束时，部署了旧 dense-af-decode/libllama，程序报 invalid placement，未执行模型测量。该目录仅作失败记录；之后在构建 exit=0 后重新部署到 audit-verified，哈希验证通过，不要使用 audit/ 当作有效放置数据。没有崩溃产生的设备大型临时文件。开发时一次主机审计脚本有括号笔误，修正后验证通过，无设备影响。

设备清理和恢复已验证，只保留 /data/local/tmp/dense-af-real-energy 下 8 个二进制/库。当前新二进制替换了旧部署；旧结果均已归档，可由原提交重建旧版本。无待测项目，不需要再重复运行。用户 TODO.md 及旧 af-handoff 三个 mixed-chain-power-trace 产物保持未跟踪，不纳入本提交。
