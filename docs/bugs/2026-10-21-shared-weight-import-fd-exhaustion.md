# 共享权重专家分片重复导入耗尽文件描述符

## 现象与触发条件

K80 Pro（fd8657d6）上为 Qwen MoE 全部专家建立 CPU/HTP/OpenCL 绑定时，模型加载到第 15 层附近报 `backend weight does not alias shared storage`，未进入推理。设备未重启。原固定分配只绑定少量 GPU 专家，因此未暴露该规模问题。

## 根因

每个专家分片都调用 OpenCL 的 shared DMA import；该函数为同一物理 backing 反复 `dup()` DMA-BUF fd 并创建 cl_mem。每个绑定保留一个独立 fd，直到绑定释放。同一 backing 的导入资源没有复用。

用 1,622,016 字节的真实权重 backing、重复 32 行切片建立小规模复现，并将测试进程 RLIMIT_NOFILE 软限制设为 512（仅影响该测试进程，不改系统设置）：初始 24 个 fd，第 488 次绑定失败，errno=24/EMFILE，已打开 fd 恰为 512。释放切片后恢复 23 个 fd，排除正常释放后的持久泄漏；问题是并存绑定的资源放大。

## 修改

`ggml-opencl.cpp` 为同一 OpenCL context、DMA-BUF inode、host 地址和 extent 复用 cl_mem 导入。缓存持有弱引用；各 buffer context 保持独立的 tensor 形状/offset/布局元数据，并引用同一个导入所有者。缓存清理过期项，最后一个 buffer/导入释放时通过原析构回调关闭保留的 fd。

身份使用 fstat 的 device/inode，而非可被复用的 fd 数字。导入查找和创建受 mutex 保护。未复制权重、未改 GPU/CPU/HTP 计算 kernel，也未提高系统 fd 上限来绕过错误。

## 验证

```bash
cmake --build build-android-opencl-hexagon --target ggml-opencl -j 4
```

使用 NDK aarch64-linux-android34-clang++ 编译 `tests/shared-weight-import-stress.cpp`，链接既有 llama/ggml/ggml-base。部署新 OpenCL 和构建时更新的 base 库，在 fd8657d6 执行：

```bash
env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 ./shared-weight-import-stress /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf import-stress-after.json
env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. GGML_HEXAGON_DEVICES=1 ./moe-shared-weight-probe /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf import-regression.json
```

- 同样 512 fd 限制下，640 次绑定全部通过，fd 从 24 仅增加到 25；释放 parent 后保持 25（切片仍有效持有 backing），释放切片后恢复 23。
- MoE FFN 回归覆盖 CPU/HTP/GPU、N=1/17/32、专家 0/7/31/59，共 9 组。CPU/HTP NRMSE=0；GPU 最大整体 NRMSE 约 1.43%，全部通过既有 4% 每 token 门限。
- 原始日志/JSON：`task/handoff/handoff-2026-10-21-cost-placement/evidence/import-stress-{before,after}.*` 和 `import-regression.*`。
- 旧 OpenCL/base 库已备份到同一 evidence 目录的 `libggml-{opencl,base}-before-import-cache.so`，最终随实验归档。

## 限制

该修复解决导入 fd/映射随分片重复增长的问题，不说明之前 4K/1K 整机重启已被修复。每个 tensor 绑定的元数据与 kernel 对象仍各自存在。全模型动态分配与吞吐验证属于同日 cost-placement 任务，最终状态以其报告为准。
