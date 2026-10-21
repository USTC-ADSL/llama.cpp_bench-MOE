# 输入 gather / 输出合并调查与单层检查点

用户目标已经改为检查实现开销，不再要求强行达到 20% 或 25% 收益。基线为 feat/shared-weight-layout 的 0ecd962f3；未跟踪 TODO.md 保持不动。

## 已确认的机制

`tests/moe-shared-weight.h` 先按 expert 汇集 top-4 token 行，分别写入三个 backend 的 DMA activation arena。shared expert 另拷贝一份完整输入。输出没有大块拼接：仅生成指针表，四个有序 FMA 加 shared output 直接写 callback 的 dst。任意 callback host pointer 不满足 HTP DMA 导入条件，不能用改 data 指针消除复制。当前每层 std::async 会新建最多两个线程。

探索了三个正交改动：连续 token 行合成 memcpy、NEON 合并展开到 16 元素、跨层共用两条持久分支线程。前两个没有稳定阶段收益，已从当前源码删除；八组合补丁与 probe 源码保存在 experiments/，构建产物在 evidence/eight-variants-probe。只保留第三个 opt-in `MOE_SHARED_DISPATCH=pool` / runner `--shared-dispatch pool`，默认仍为 async。

## 当前代码

- `tests/moe-shared-weight.h`：持久 dispatch 指针、GPU inline / HTP worker / CPU worker，单 backend 直接 inline；报告 branch_dispatch 为配置的策略。
- `tests/fullmodel-oracle.cpp`：full_engine 持有一份 moe_token_workers(3)，所有顺序 layer 共用；不新增权重或 activation arena。
- `tests/moe-token-workers.h`：只更新适用契约注释，无线程池实现修改。
- `tests/test-moe-token-workers.cpp`：加入 3-way 和 inline 异常后 drain/reuse 验证。
- `tests/moe-shared-overhead-probe.cpp`：使用已捕获真实输入，逐层加载，AB/BA 交替执行同一图与权重，比较逐位输出。
- `tests/run-fullmodel-oracle.py`：显式可回退的 dispatch 参数。

所有 backend kernel 未修改。旧 V1 可执行文件另存 evidence/fullmodel-oracle-v1。CPU expert 仍为 4 线程，未满足更早的 1–2 小核目标。

## 已执行验证

交叉编译器：`/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++`。

完整模型/probe 构建模板：`<compiler> -std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann tests/<name>.cpp -Lbuild-android-opencl-hexagon/bin -lllama -lggml -lggml-base -o build-android-opencl-hexagon/bin/<name>`，实际 name 为 fullmodel-oracle 和 moe-shared-overhead-probe。唯一 warning 是已有 deprecated ggml_mul_mat_set_prec。线程测试：`<compiler> -std=c++17 -O2 tests/test-moe-token-workers.cpp -o build-android-opencl-hexagon/bin/test-moe-token-workers`。

设备 fd8657d6 已确认在线。`python3 <本目录>/probe.py`（八组合版本）及 `--tag pool-final`（最终版本）均通过：layer 0/12 为 1024 tokens，layer23 为 capture 的 1 token。八组合共 288 次、最终两组合共 72 次测量，全部与本进程 async 输出逐位相同。与旧 captured FFN reference 的每 token NRMSE <4%。原始数据、环境、hash 在 evidence/{probe,pool-final}.{json,run.json,log}。

设备运行 `LD_LIBRARY_PATH=/data/local/tmp/llama-gather-merge-20261021 /data/local/tmp/llama-gather-merge-20261021/test-moe-token-workers` 通过，见 evidence/token-workers.log。独立只读复核未发现新路径数据竞争/异常 drain/arena 生命周期问题；不是实机测试的替代。

初次八组合测试：layer0 async→pool 整层 21681.6→21585.0us；layer12 22101.1→21951.3us。分支启动约 80–134→11–22us。连续行聚合只减少 8–16% 调用，未稳定减少 gather；NEON16 未稳定改善 merge。不能外推大幅 E2E 收益。

## 正在进行 / 下一步

`python3 <本目录>/confirm.py` 进行 wetland ABBA / foundry BAAB，A=旧 V1 binary、B=最终 pool binary，两组输入各 4 进程，每进程 2 warmup + 6 timed，30s idle。remote 固定 `/data/local/tmp/llama-gather-merge-20261021`，模型沿用用户原路径；不改时钟/系统参数。自动验证 boot ID、不重用失败结果。frozen.json 与 confirmation-runs.json 记录状态。

完成后运行 `python3 <本目录>/analyze.py`；它核验 logits、token IDs、route counts/placement、binary hash、共享 payload/arena、图缓存稳定性，再写 summary.json。所有结果都必须保留，负收益也要报告。尚未确认端到端收益。

收尾需检查 diff、记录完整结果/归档、清理本任务 remote，保留模型、公共 cache、用户文件；随后最终 Git 检查点。当前性能改动不是 kernel bug，不需要伪造 bug 简报。
