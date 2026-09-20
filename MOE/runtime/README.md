# Runtime contracts

`moe-memory-runtime` 管理物理存储；`moe-expert-slot-runtime` 管理 Expert metadata、加载、布局与 Slot 生命周期。Host core 仅使用 C++17/POSIX 和仓库内 ggml 类型声明，不链接设备 SDK。

## Memory

`moe::Buffer` 统一容量、backend 能力、host mapping 和显式 read/write。`CpuPrivateBuffer`、`GpuPrivateBuffer`、`HtpPrivateBuffer` 各自拥有独立 allocation；`SharedDmaBuffer` 的各设备 import 引用同一 allocation。GPU private 没有伪造的 host pointer。不支持的 backend/view 返回错误。

FastRPC session 是进程级共享资源，第一次使用初始化，进程退出时 deinit/dlclose；每个 Buffer 仍独立释放 allocation。Profile 中 session 初始化可能仅在首个样本出现，per-buffer teardown 的 deinit/dlclose 为未执行的零值。不能把该口径与旧的 per-allocation session setup 结果直接比较。

`moe-memory-opencl` 适配指定 raw context；`moe-raw-adapter` 提供 IO-coherent DMA import 和 HTP mapping。`moe-ggml-adapter` 通过对应 ggml backend 导入。raw OpenCL 私有 allocation 不跨 context 导入 ggml。HTP range 可见性协议保留在现有 ggml/FastRPC 适配路径；完成等待与数据可见性不是同一个操作。

## Slot ownership

`ExpertSlotArena` 可按 slot count/stride 配置连续布局，也可用 `SlotPlacement` 把逻辑字节区间映射到不重叠的物理区间。配置时检查容量、重叠及对齐；后续不再扫描 metadata。分散布局不提供虚假的单一 base pointer。

```
EMPTY -> LOADING -> READY -> COMPUTING -> READY
             |       |
          cancel     +-> LOADING (explicit replacement)
             v
           EMPTY
```

`begin_load` 立即使旧 handle 失效。`WriteLease` 不可复制，可移动；未 publish、加载失败或析构时取消为 EMPTY。不会恢复已可能损坏的旧内容。Private destination 切换是显式操作；取消后该 Slot 仍保留新 allocation，但没有有效 Expert。

`ExpertHandle` 的 arena identity、slot 和 generation 不可由调用者构造。`ReadLease` 持有存储及 backend view，首版每个 Slot 只允许一个活动读 lease。READY 可重复读取；一个 arena 内同一 ExpertKey 只允许一个有效 Slot。

异步提交前绑定 `moe::Completion`；同一批次共享 token，wait 至多执行一次。成功后 lease 释放不再同步；失败状态是 sticky，显式 `finish()` 报错，析构将 allocation/view/completion 隔离保留并保持 COMPUTING。无法确认设备完成时，有意保留资源到进程结束，禁止提前回收。

## Pack, load and repack

`expert_pack` 保留 EXPKP001 磁盘格式，仅描述 tensor 类型、shape、payload/source range 与 layout metadata。`ExpertLoader` 打开 pack 后缓存 native plan，也可以选择经范围校验的 GGUF source。

```cpp
auto write = arena.begin_load(slot);
auto layout = loader.prepare(write, expert, BackendId::gpu);
auto handle = write.publish({loader.source_layer(), expert}, BackendId::gpu, layout);
auto read = arena.acquire(handle);
```

组合接口 `loader.load(arena, slot, expert, backend)` 调用相同 prepare/publish 路径。需要分别观测 read/repack 的 Profile 可以在同一 WriteLease 下调用 reader 与公共 Repack API，再 publish。CPU canonical 可直接进入目标区间；不能直接写入的设备/分散布局复用一个 native tensor workspace。CRC 为显式验证选项，性能样本默认不执行 CRC。

Repack 独立于 Arena，支持现有 Q4_0/Q4_1 padding、预构建 plan 和原地 GPU↔HTP 转换。未增加模型、量化类型或 padded compute graph 支持。固定 Phi graph fixture 留在 experiments/support。

## Build variants

`-profile` 目标来自相同 Runtime 源文件，以 `MOE_RUNTIME_PROFILE` 编译观测点。普通库不编译计时。CMake 的 compatible interface 检查禁止同一消费者混合 normal/profile。统计、`/proc`、page-cache profile 和输出实现位于 tests/support。
