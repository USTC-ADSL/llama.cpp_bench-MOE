# 2K 异构专家激活重复上界预留

## 现象与触发条件

fd8657d6 / Redmi K80 Pro，Qwen1.5-MoE-A2.7B Q4_0，CPU4x8单份权重、动态三端放置，2048-token单chunk prefill。原三端激活分配963.094 MiB；10-26测试出现大量major faults、KGSL OOM及LMKD杀后台应用。10-27同版本private对照再次出现KGSL OOM，wetland中位16.391秒。

## 已定位的原因

`full_engine::prepare_arenas()` 原来对CPU/NPU/GPU分别预留全部 `4N` routed pairs 所需的激活上界。实际路由将同一批 `4N` pairs 分给三个并发分支，三个后端不会同时各执行全部pairs。图内已有生命周期复用，但外层DMA backing仍按三份独立最坏情况容量分配。当前NPU-only shared配置因此可消除约两份routed上界，即约520 MiB。

这定位了明确的过量分配，不表示它解释了所有主机换出、驱动内存或加载峰值；原始GGUF映射也不能按虚拟长度当作物理副本计费。

## 修改

实现提交 a625c391e：新增 `--shared-arena pooled`。一份DMA backing导入三端；每层按当前真实counts划分不重叠区域，三端保持并发。图缓存key加入分区offset，已绑定tensor不移动，防止OpenCL缓存旧地址。检查每图lifetime大小、分区和总池容量；分支全部完成并归约后才跨层复用。容量报表按物理backing去重。原private模式保留作对照。

相关文件：tests/fullmodel-oracle.cpp、tests/moe-shared-weight.h、tests/shared-activation-buffer.h、tests/run-fullmodel-oracle.py；新增 tests/moe-pooled-activation-probe.cpp。

## 验证

实际交叉编译三个目标 fullmodel-oracle、moe-pooled-activation-probe、test-shared-activation-layout，命令和日志见 `task/handoff/handoff-2026-10-27-pooled-activation/build-commands.json`、`evidence/build-*.log`。仅既有deprecated warning；Python语法和git diff检查通过。

设备原layout测试通过；360个新探针用例全部与private逐位一致，包含同形状不同offset、缓存重访、shared owner换端、三端并发、private/common scratch和1–2048 token形状。

完整模型两个输入各2预热+6计时，均与旧基线logits逐位一致、实际路由/owner一致。容量963.09375→443.094116 MiB，减少545259136字节（53.99%）。同版本wetland private→pooled中位16.391→2.420秒，foundry pooled为2.403秒。三个进程全部正常退出，设备无重启。

完整协议、原始数据和系统日志见 [测试报告](../../task/handoff/handoff-2026-10-27-pooled-activation/results.md)。设备测试目录已清理，模型不变。

## 剩余限制

两个pooled窗口未出现KGSL OOM，但仍有47/2条LMKD Kill；wetland仍有812–28481次major faults/计时请求，foundry为0。系统内存压力未全部解决。性能仅代表本轮状态，不是多进程ABBA或长期稳定性保证。

普通单请求按层复用已验证；独立early-shared decode dispatcher明确拒绝启用pooled，多请求同池并发未验证。长期不同route/offset产生的图元数据缓存增长仍需单独评估。
