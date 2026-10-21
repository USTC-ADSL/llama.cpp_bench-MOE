# 共享权重长度扩展与放置验证协议

设备：`fd8657d6`，Redmi K80 Pro；模型：`/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。

用户最初要求扩展到 4096 token；遇到运行异常后明确允许先完成 1024 token。用户确认没有额外性能调节或并行压力测试。本任务不改变设备频率或性能模式，CPU 仍为既有 4 线程设置。

## 输入与对照

- 延用 wetland / foundry 两个文本。为保证可截取到 4096 token，文本按原顺序重复 8 次，以 tokenizer 输出前 N 个 token 为实际输入。该扩展是重复文本长度压力测试，不等价于自然长文泛化测试。
- 原分配采用先前冻结的 c8-g15 plans，`--transfer-placement` 仅继承专家归属，运行时计算真实路由，不重放 768-token IDs。
- native 使用后端原生布局/MMID；shared 使用单份 CPU4x8 V2 backing 和 grouped 2D 执行器。因此收益是接入方案的联合收益，不是纯布局因果消融。
- 固定分配对照顺序 native-a / shared-a / shared-b / native-b，每个进程 2 次预热、6 次计时；后续正式 1K 对照在进程之间空闲 30 秒。
- 计时包括完整 prefill、真实路由、expert gather/compute/reduce、attention/KV 和最后 token logits；排除模型加载、权重重排和请求前 KV 清空。
- 每条件使用两个进程中位数的平均值；吞吐为输入 token 数 / 完整请求耗时。不得相加三端重叠的执行窗口作为端到端时间。

## 4K 诊断与 1K 优先

首次整批 4096 在主图预留附近失联，恢复后启动原因为 kernel_panic；pstore 证据异常且无法给出可靠调用栈，根因未确定。

增加显式 `--prefill-chunk 1024`：上下文仍为 4096；一个请求内依次 llama_decode 四个 1024-token batch，保留 KV 与连续位置，只在完整请求之前清空。输出 JSON 的 `chunks` 包含所有分块的阶段记录，旧 `layers` 字段只代表最后一块。分块记录的主机开销也在请求计时内。

4K 分块可完成，但正式 wetland 对照出现大量 major faults、明显进程顺序漂移；不得据此宣称布局在 4K 的稳定加速或退化。随后 1K 的旧布局在权重准备阶段也失联，恢复后的 boot reason 为 reboot；此事件不同于首次 kernel_panic，不能未经证据推定为同一原因。

第一轮未完成的 1K 对照移动到 `evidence/interrupted-first-1k/` 保留，后续重新建立完整对照。

## 专家分配优化

从既有 768-token 校准 counts 生成 c8-g30、c8-g45、c4-g30。算法沿用原研究：最轻的 c 个活跃专家放 CPU，剩余按 counts 从重到轻取最接近 g% 累计负载的前缀放 GPU，其余放 HTP。共享专家保持 HTP，末层共享专家保持 CPU。不使用被测 1K 的未来真实路由生成候选。

先在 wetland 探索候选，选定候选后在两种输入上做完整请求复测。候选选择结果不能被表述为在线预测器或全局最优分配。

## Router

独立探针读取真实 layer 0 / 12 的 F32 router 权重 `[2048,60]`，使用对应层旧 capture 的 128-token normalized input，较长长度循环平铺输入。测试 N=1/128/768/1024/2048/4096，8 次预热、64 次交替计时。

分别测 projection 和 projection+softmax+top4；显式检查后端 supports_op，不将 CPU fallback 当作纯 HTP 完整图。GPU 记录 kernel event 总时间，HTP 记录 DSP op 时间（包括 op 内数据准备，不含 host RPC）；另保留同步 host wall 时间。reference 使用相同模型权重和输入的 double 累加点积，记录误差、概率归一化与 top4 差异。

整模型另设置 GPU router 权重 override，所有常规层权重仍指定 HTP，并通过 diagnostic placement trace 验证实际图路径；正式计时关闭 trace。该路径可能改变路由数值和后续专家分配量，需同时报告 logits 差异。

## 可复现入口

```bash
python3 tests/run-moe-shared-weight-4k.py plans
python3 tests/run-moe-shared-weight-4k.py baseline --tokens 1024 --chunk 1024 --deploy
python3 tests/analyze-moe-shared-weight-4k.py --tokens 1024
```

增量编译 fullmodel-oracle：

```bash
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/fullmodel-oracle
```

router probe 使用同一编译器和 include 路径，源文件改为 tests/moe-router-probe.cpp，链接 `-lggml -lggml-base`，输出 `bin/moe-router-probe`。

设备自有目录 `/data/local/tmp/llama-shared-weight-4k-20261021`；每进程 run.json 保留具体设备命令、哈希、热状态、启动时间。未完成测试不得从缺失 JSON 推断结果。
