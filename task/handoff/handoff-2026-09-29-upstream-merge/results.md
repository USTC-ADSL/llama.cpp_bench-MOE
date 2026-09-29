# Upstream 合并与当前进度影响评估

## 结论

将 upstream/master `00af63567a3ac073919be039db8492f229de5ffc` 的完整更新合入 `feat/dynamic-moe-placement`，本地起点为 `7f224605a`。共纳入 420 个上游提交，其中 24 个涉及 Hexagon。整专家调度、prepared resident 布局、共享激活、HMX retained-input 索引融合，以及 canonical/packed Q4_0 权重转换路径均保留。当前 K80 Pro/Qwen1.5 MoE Q4_0 回归通过。

本次更新没有推翻当前工作方向，也没有解决剩余的权重切换和长输入收益问题。旧标定参数与旧实验数字应保留为历史版本结果；后续优化应先在新版本重新标定各后端和输入准备开销，再调整负载分配。

## 主要上游变化及影响

| 上游变化 | 对当前工作的实际影响 |
| --- | --- |
| `0c3626ec0` 64-bit mapping/DMA 重构 | 主机与 DSP 协议、IDL、DMA 队列等一起升级，必须成套部署。扩展映射只在 arch >79 默认开启，K80 Pro v79 不支持，不能把它当作当前设备权重容量收益。 |
| `eafe15a5e` 多设备 row-split | 新的分区参数已与本地 mixed-weight HMX 调用接合；当前单设备执行通过，多设备未验证。 |
| `07fc586e3` 动态量化改进及 HVX 路径调整 | 后端计算/转换时间可能变化。`MM_SELECT=2` 现在选择 HMX，`1` 才是 HVX；已修正索引融合回归驱动，避免假测回退路径。旧 `3` 仍落入 HMX 分支。 |
| `1ec818809`、`4de092659` K-quant 支持 | 增加 Q4_K/Q6_K/Q5_K 选项；当前 Q4_0 模型不直接受益，不能外推正确性或性能。 |
| `9710a3217` MUL_MAT_ID 输入精度检查 | 显式要求 F32 输入精度的算子可能不再被 Hexagon 接收；应保留上游的精度约束，不能为恢复 offload 删除检查。当前自定义图回归通过。 |
| `7ac59a6e3`、`e613ef2c8` GET_ROWS 扩展 | 支持 tiled Q4_0/Q8_0 及 I32；本地 F32 retained-input 索引融合仍是独立可选优化，未被这几个提交替代。 |
| `2b129ccfa` 后端采样，`58367713a` HMX GATED_DELTA_NET | 可帮助相应 decode/模型路径；不能直接解释当前 Qwen1.5 MoE prefill 收益。 |
| GGML、模型精度策略、OpenCL 和 server 更新 | 保留 weight provider、Qwen2MoE 回调、GPU Q4 准备及 E0 计时；模型加载策略和 server API 冲突已接合。 |

## 合并处理

10 个文本冲突文件逐处解决，未整文件覆盖本地扩展：

- Hexagon `ggml-hexagon.cpp`、`htp/htp-ops.h`、`htp/matmul-ops.c`：采用上游 ABI/序号队列和 DMA API，同时保留 canonical 权重标志、`src[9]` 行映射、`act_row`、mixed expert map 和聚合计时。行号/slot map 在 DSP 直接解引用前拒绝 extended 地址。Mixed NX 的 VA 预算跟随上游排除 extended 权重缓冲。
- OpenCL：并存本地 expert slice/dense Q4 准备逻辑与上游 upload/download buffer 分配回退。
- `src/CMakeLists.txt`、`llama-model.cpp/.h`：保留 weight manager、回调和 `no_mmap_prefetch`，加入上游 precision policy，并遵循新的模型 unity build 划分。
- `llama-bench.cpp`：`use_extra_bufts = repack && !no_repack`，同时支持上游与本地参数。
- `server-context.cpp`、`server-http.cpp`：将原有事件记录接到新的 batch view/llama_process 和多监听器回调。

## 验证结果

目标设备 `fd8657d6`（Redmi K80 Pro，HTP v79），模型 `/data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf`。

- 增量交叉编译通过：llama、llama-bench、llama-server、ggml-hexagon、ggml-opencl、htp-v79；重新生成主机 stub 和 DSP skel。五个研究探针也使用新头文件/库重编译。
- 索引融合：NX 开、NX 关、HVX 回退各 20 项检查通过，覆盖非单调/重复行号、图缓存再访、可观测 gather、未保留输入和插入中间算子的回退；输出逐位相等，日志验证融合只出现在允许的图。
- Resident probe：12 个配置通过每 token 4% 数值阈值，反向再访逐位相等。该阈值不等价于所有后端相对原生计算逐位相等。
- Canonical/mixed-weight：gate/up/down × 8/32/128/256 token，共 12 项 canonical 与 packed 输出逐位相等，混合 miss/partial/all-hit、类型/布局保护通过；跨布局复制、二维 set/get、GPU→HTP 转换和三个 shared expert GPU 准备路径通过原有断言。
- 整模型：128/768 token，各 native/base/fused 一个进程、2 次 warmup + 4 次采样，共 24 次计时 prefill。base/fused 的 logits、各层 route pair/owner/shared 分配和 allocation 计数相等，custom 的重复输出逐位稳定。

本次整模型耗时仅作回归观察（中位数，毫秒）：

| 输入 tokens | Native NPU | 原 custom 调度执行 | Custom + 索引融合 |
| --- | ---: | ---: | ---: |
| 128 | 531.11 | 359.33 | 362.91 |
| 768 | 1113.57 | 973.77 | 965.62 |

这不是旧版/新版同条件 A/B，也没有反转进程顺序和完整热状态控制，因此不能归因于 upstream 加速或退化，更不能据此重写 9 月 28 日的正式结论。融合在 128 下本次略慢、768 下略快，也不足以宣称稳定提升。

Native 与 custom logits 相对 L2 分别约 3.10% 和 1.39%，top-1 相同；base 与 fused 则逐位一致。本次仍不宣称模型质量等价。1536/2048 的旧确认实验未重跑，尚不能宣称新版本长输入性能保持不变。

## 复现、证据与后续工作

实际命令见 `handoff-01.md`、`probe-build-commands.json`；新增完成命令：

```sh
python3 task/handoff/handoff-2026-09-29-upstream-merge/validate.py full
python3 task/handoff/handoff-2026-09-29-upstream-merge/validate.py summarize
python3 -m py_compile tests/run-moe-route-execution-study.py task/handoff/handoff-2026-09-29-upstream-merge/validate.py
python3 task/handoff/handoff-2026-09-29-upstream-merge/archive-cleanup.py
```

`validation-summary.json` 保存可检查的验证结论；`evidence/` 保存构建日志、原始报告、logits、命令/退出码和设备状态。归档路径与成员 SHA256 记录在 `artifact-manifest.json`，按哈希核验后删除的设备文件记录在 `device-cleanup.json`。不复制或删除原模型。旧实验及用户未跟踪 TODO.md 均保持不变。

优先建议：在这次合并版本固定 native/base/fused 基线，重测 CPU/NPU/GPU 的实际分支完成时间及 NPU 输入准备成本，再重拟合路由代价模型；继续独立评估权重转换/缓存切换，不把 prepared resident 的结果当作在线权重切换已解决。

未覆盖：其它 Android 型号、v73/v75/v81 DSP、extended DMA64、多设备、新量化格式、后端采样、server 的完整 HTTP 测试，以及旧 --weight-config 流式权重管理器全模型场景。它们没有因当前 Q4_0 回归而获得验证结论。上游 Q6_K HVX NX dispatch 的静态疑点留作单独核查，不混入本次集成修复。
