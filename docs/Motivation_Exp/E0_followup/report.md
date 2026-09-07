# E0 定向补验（2026-09-07）

**结论：CPU 正式 perf 验收仍 BLOCKED；OpenCL 必要回归 PASS；默认 HTP 数值仍 BLOCKED。HMX 根因尚未确定，本次没有交付 HMX 修复。** 全部推理和数值测试均在指定 Android 手机执行。没有采用 HVX-only、CPU fallback、降低 ubatch、关闭默认融合或放宽容差作为验收方案。

- [x] 复用 E0 环境、真实输入和原失败用例，完成输出统计与局部定位
- [x] 实现 perf/memory 分离，完成三次启动六对计时开销检查及独立 memory 运行
- [x] 完成必要 OpenCL 回归、恢复默认 DSP 并复验原数值用例
- [ ] CPU decode 开销达到 ±3% 可判定要求
- [ ] HMX 根因修复、相关算子回归及修复后的 128/32 整模型验收

仓库 `/mnt/sda1/pzw/moe/llama.cpp_bench-MOE`；起始 HEAD `bbfd2b22e`，与 E0 完成版本无代码漂移。仅原有 `AGENTS.md` 和 `docs/Motivation_Exp/Exp1/Exp_require.md` 用户修改，最终 SHA256 与 E0 环境记录一致。未更新 upstream、依赖或模型。设备 `3B661501LA000000`，PLU110 / SM8735 / v73（运行时报告 4 HVX、1 HMX、8 MB VTCM），Android 16、内核 6.6.89；详细身份沿用 `../E0/environment.json`。模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`，8149865792 bytes，原完整 SHA256 `8b704c763230935a096c9c810765c8a5483f1724c83e34e253704138ca040473`。本次未改写模型，shape/元信息见 `raw/model.txt`。

固定请求复用 `../E0/prompt.json` 的 128 个 token ID，目标32、context512、slot1、F16 KV、t/tb4、b/ub128、mmap。`cache_prompt=false`、`--cache-ram 0`；所有 CPU/OpenCL 请求均验证 `prompt_n=128`、`cache_n=0`、生成32，ON侧32个 token-ready ID 与返回 token 完全一致。未驱逐权重；首次请求不称为冷文件。保留 E0 互斥、退出检查和 watchdog：每秒读取安全状态，MemAvailable <1 GiB 持续3秒或当前 HAL thermal ≥SEVERE 停止；加载300秒、请求180秒超时为安全截断。本次正式组无安全截断；未改 swap/LMKD/SELinux/频率/cgroup，也未清全局 page cache。

`common/e0-measure.h` 增加 `E0_MODE=perf`：固定1024条内存缓冲，请求结束后写时间戳；不构造旧 recorder/pss_worker、不读 PSS/smaps/stat/RUSAGE/io。`watchdog.sh` 的 perf 分支只保留安全观察；runner 禁止 perf 与详细 profiling 同跑。memory 单独启动、单独标记，延迟排除在性能表外；legacy 审计保留。OFF/ON 使用同一构建、同一 watchdog、同一门控入口；OFF仍有公共门控/互斥及请求入口配置读取（含一次丢弃的起点时钟读取），无 token 时间戳或事件写盘，比较的是打开记录的增量开销。

CPU 在3次独立启动中，每次先 OFF/ON 预热，再 AB/BA 两对，共6对正式请求。证据 `runs/cpu-perf-s1..s3/`：无 PSS sidecar，watchdog 的 PSS/cgroup 字段均为 -1，预热后和各请求后的线程数始终17，源码分支无采样队列/后台采样任务。TTFT 从完整 POST 回调入口到首 token 就绪；p95 ITL 来自31个相邻 token 就绪间隔，不使用 SSE。OFF侧 TTFT/ITL 为 NA。共同耗时取 server 响应 `prompt_ms`、`predicted_ms`；两者之和命名 `slot_request_ms`，是 slot 计算起点至最后 token 的耗时，**不等同完整 HTTP 接收至响应结束**。

下表为每启动两对的 ON/OFF−1 均值（%）。请求和独立启动是重复单位，token 不作为独立重复；完整每请求、每对及启动结果见 `summary.csv`。

| 启动 | prefill | decode | slot_request |
|---|---:|---:|---:|
| s1 | +0.118 | -5.789 | -2.313 |
| s2 | +1.327 | +1.580 | +1.379 |
| s3 | -0.425 | +0.790 | +0.060 |
| 均值 | +0.340 | -1.139 | -0.291 |
| 启动分组 bootstrap 95% 区间 | [-0.425, +1.327] | [-5.789, +1.580] | [-2.313, +1.379] |

固定 seed1234、4000次按启动重采样。prefill 和 slot 区间满足 ±3%，decode 不满足，不能判定最小埋点总体验收通过。decode 单对变化 -13.687% 到 +5.246%；只有3个独立启动，区间本身也受样本数限制。不改阈值、不继续无限补测，不能把均值变小解释为无扰动。独立 `runs/cpu-memory/` 的 PSS **采样峰值**为10508165 KiB（约10.02 GiB，含加载与请求采样），不是 perf 同时内存，更非实际物理峰值；没有将 cgroup、驱动容量和 PD 映射相加。

HTP 复用第0层真实 Q4_0：gate/up `[2048,1408,60,1]`、down `[1408,2048,60,1]`、top4，原固定激活与 IDs 不变；参考仍为 Android CPU Q4_0 reference。阈值始终 RRMSE≤0.02、max_abs_error/RMS(reference)≤0.10、全有限。初始 `runs/hmx-repro/` 与去掉临时诊断后的 `runs/hmx-default-final/` 结果完全一致：

| n / 用例 | 实际路径 | RRMSE | 归一化最大误差 | 结果 |
|---|---|---:|---:|---|
| 1 / gate+up | HVX，默认选择 | 0.00380649 | 0.01438793 | PASS |
| 1 / down | HVX，默认选择 | 0.00387328 | 0.01536291 | PASS |
| 128 / gate+up | HMX tiled，MUL_MAT_ID_NX | 1.0 | 5.35863352 | FAIL |
| 128 / down | HMX tiled，MUL_MAT_ID | 1.0 | 22.56713947 | FAIL |

n128 两项 target 的 min/max/RMS/非零比例均0、非有限数0，reference RMS分别0.26356082/0.22096489。`backend.log` 同时给出 execute/profile-op、NX融合、hmx-tiled、flags0；局部日志确认 HMX worker 实际执行。不能把 n1 的通过解释为 n128 的替代验收。

定位证据与排除范围：

- 初始 host/DSP 库与 E0 指纹一致；最终恢复的 DSP SHA256 `59348f4df78c815293324f7beb2f456a06df2b46eded0dbe41853959a06b2291` 与初始完全相同。运行时打开本次目录的 v73 DSP 库；最终设备指纹见 `raw/final-device-sha256.txt`，perf host 指纹见 `raw/perf-build-sha256.json`。原始 `artifacts.json` 是本次初始部署指纹，不冒充 perf 更新后的指纹。
- 部署只更新了perf相关host库，`libggml-base.so` 沿用E0（SHA256 `4c66d1163827034851c0be9ac4498e12bdece8005d4ea0355e512560483c5de2`）。它的版本字符串为 `d46e70701`，本地重建版为 `bbfd2b22e-dirty`；对应 `ggml.c`/头文件/CMake 没有代码差异。**实际测试库**保存在 `build-android-opencl-hexagon/e0-followup-tested/libggml-base.so`，不得把当前bin目录同名库冒充已测版本。全部配对使用同一套设备库，完整指纹已记录。
- `expert_sanity.cpp` 遵循本地 backend 测试核心调用顺序：分配/写张量、`ggml_backend_graph_compute`、同步、`ggml_backend_tensor_get`。本次仅增加输出统计；测试权重、IDs、精度、容差没有变动。
- `raw/hmx-worker-verified.log`：首 expert 的有效 activation 与反量化 weight 非零；HMX worker 把首 tile 的 poison 改写为0；普通逐tile HMX诊断也输出0。因此不只是 host 未回读或 deep-only 异常。该诊断会改写首tile，**不是验收配置**。
- `raw/hmx-lock-diag.log`：资源句柄非0、`HAP_compute_res_hmx_lock` 返回0。源码 `htp/main.c` 中 `vtcm_alloc` 在 queue_init 前调用，排除“queue拿到尚未初始化句柄”的假设。
- `raw/hmx-bias-diag.log`：bias回读 `3c00,0,3c00,0`，后半部0，与输入一致，activation/weight地址满足tile对齐。证明bias已加载，**未证明整个 repack/activation 布局及 HMX 算术契约正确**。bias诊断借用输出区，不能用其数值结果验收。
- 分拆 activation/weight 指令包的临时构建被汇编器拒绝：`packet has HMX activation without a weight`，见 `raw/hmx-packet-build.log`。未部署、未作为修复。默认同包指令已恢复。

所有临时 DSP 代码已移除，诊断补丁仅保留在 `raw/hmx-local-diagnostics.patch`。`hmx-local-diag` 未启用 FARF；`hmx-worker-diag` 曾在编译完成前误推旧库，不能作为该版worker证据；随后 `hmx-worker-verified` 等有效局部日志单独保存。summary 对诊断组明确标记 `numeric_diagnostic_not_acceptance`。

**未解决项**：尚未证明 Q4_0 repack/activation 的完整值与布局正确，也未确定为何 HMX 运算写出全零。gathered activation 的线程分片与每worker补齐32行存在待核查的边界疑点，但没有证明它是全零根因，未据此贸然修改。需要进一步在 HMX 运算/输出变换边界核验实际矩阵内容与本机契约。未交付任何候选HMX修复，因此“修复后的相关算子回归”和“修复后的128/32默认HTP整模型smoke”均未执行；旧E0整模型能生成文本不能替代失败的数值验收。失败路径的性能不进入排名。

OpenCL 仅跑 `runs/opencl-perf-regression/`：同构建 OFF/ON 预热后1对，128/32、cache_n0、token-ready与输出一致、无PSS采样，通过公共计时代码功能回归。原 E0 OpenCL 数值结论保留；本次单对不声称完成新的性能开销统计验收。CPU可用于计数/独立内存诊断，尚不能进入正式性能比较；OpenCL可在既有已验证范围继续；默认HTP不能进入下一阶段。

实际命令（工作目录为仓库根；ADB始终指定上述serial）：

```sh
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup
cmake --build build-android-opencl-hexagon --target llama-server -j8
cmake --build build-android-opencl-hexagon --target htp-v73 -j8
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ -Iggml/include -Ivendor/nlohmann docs/Motivation_Exp/E0/scripts/expert_sanity.cpp -Lbuild-android-opencl-hexagon/bin -lggml -lggml-base -lggml-cpu -o build-android-opencl-hexagon/bin/e0-expert-sanity
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --prepare
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --numeric htp --label hmx-repro
# CPU实际通过import复用runner，依次调用下列等价配置；每次均独立启动：
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --run cpu --mode perf --pairs 2 --label cpu-perf-s1
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --run cpu --mode perf --pairs 2 --label cpu-perf-s2
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --run cpu --mode perf --pairs 2 --label cpu-perf-s3
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --run cpu --mode memory --label cpu-memory
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --run opencl --mode perf --pairs 1 --label opencl-perf-regression
python3 docs/Motivation_Exp/E0/scripts/run_e0.py --followup --numeric htp --label hmx-default-final
python3 docs/Motivation_Exp/E0/scripts/analyze.py --followup
```

构建复用 `build-npu-opencl.sh` 对应缓存，NDK r27d、Hexagon Tools19.0.04，无重新配置或依赖升级。完整手机启动/请求/拷贝命令见 `commands.jsonl`；增量构建日志在 `raw/*build*.log*`，原始响应/token/内存在 `runs/`。各步骤单独增量构建并在成功后部署；诊断命令的具体label及环境在各 `execution.json`。`--prepare` 为首次执行记录，不应直接在现有结果目录重复运行。收尾检查与设备清理记录见 `raw/final-validation.txt`。

收尾：分析器检查全部perf记录、计数/线程/无PSS约束；原Android counter sanity通过，minflt增量4102、majflt增量0，与RUSAGE_SELF bracket一致。125个设备文件全部与保留证据或构建产物逐一hash匹配后，仅删除 `/data/local/tmp/llama-e0-followup`；无遗留推理/watchdog进程，原模型及已有缓存保留。Git检查点为包含本报告的补验提交；任务至此停止。
