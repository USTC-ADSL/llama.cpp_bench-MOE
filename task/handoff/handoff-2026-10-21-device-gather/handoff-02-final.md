# DMA融合gather：最终交接

用户目标为继续减少输入gather和输出合并等实现开销，不追求指定速度。实现代码已提交62876447f；最终结果详见results.md、dma-summary.json、probe-summary.json。默认host gather保持，候选需要显式--shared-gather htp --shared-fused-gather。

## 完成状态

已完成显式HTP gather及其1K负向对照、直接DDR读取融合负向尝试、最终DMA/VTCM融合、Android算子与MoE回归、同binary/同library完整1K ABBA/BAAB。全部结果均保留。最终wetland吞吐+4.614%，foundry+5.136%；总体中位数口径+3.581%/+5.953%，存在慢样本与进程波动，不能用全部样本均值夸大收益。

原内核在evidence/kernel-backup，直接读取失败版在evidence/fused-direct-freeze，显式与最终冻结快照在evidence/explicit-freeze及fused-final-freeze。新内核保持CPU4x8权重不变，以索引DMA进既有VTCM并转原FP16 tile；CPU/GPU及输出reduction算法未修改。70缓存图稳定，权重7474348032B，真实1K HTP arena+16384B，其余arena不变。

## 验证与命令

- 增量构建：cmake --build build-android-opencl-hexagon --target ggml-hexagon htp-v79 -j 8。日志fused-build.log、fused-dma-build.log、fused-dma-final-build.log。
- NDK手工编译fullmodel-oracle、moe-shared-gather-probe、htp-cpu4x8-gather-probe，完整编译形式见handoff-01.md。只有既有deprecated warning；首次开发构建缺少NULL参数/误解dst指针两处编译问题已修正，未部署失败构建。
- python3 task/handoff/handoff-2026-10-21-device-gather/regress.py：175 CPU4x8 cases+3×20 native cases，全通过，trace核验guard。
- python3 task/handoff/handoff-2026-10-21-device-gather/probe.py --tag fused-dma-captured --fused：28 cases/172 samples，全bitwise一致。
- python3 task/handoff/handoff-2026-10-21-device-gather/confirm_fused.py：8进程48timed请求，全成功，boot ID不变，无设备异常。fused-confirmation-runs.json为完整终态。
- python3 task/handoff/handoff-2026-10-21-device-gather/analyze.py --prefix dma- --candidate fused：核验权重、arena、cached graphs、复制量、token/logits/分配，生成最终统计。
- 已检查git diff、git diff --check、Python语法。实现评审未发现确切生命周期/DMA越界/默认路径回归问题。

## 归档与清理

归档已完成：`/mnt/sda1/pzw/moe/experiment-artifacts/device-gather-1k-20261021.tar.gz`，82,438,452 B、162成员逐项hash验证，SHA256 `4979e7d51cd9b9057b618765b8067fbe4b29634e6c740b9ef12e2538fd405baa`。archive-record.json和artifact-manifest.json记录详情。归档先于设备清理，归档内交接/清单保留清理前状态，最终状态以Git中的本文和cleanup-record.json为准。已确认无本任务进程，并删除本任务远端 `/data/local/tmp/llama-device-gather-20261021`，模型stat前后不变，其他目录未动。

## 限制和下一步

本轮任务完成后无必须继续运行的测试。当前候选为opt-in，不应自动替换默认；没有声明相对纯NPU的新收益。未验证4K、其他机型、multi-HTP和长期热稳态；CPU仍4 expert/4 outer线程。GET_ROWS执行被跳过，但其公开图arena预约仍在，元数据额外开销未独立测量。主要可探索方向是gate/up共享一次已准备的activation tile；这需要CPU4x8投影融合支持，尚未实现。不要重复直接HVX读取DDR版或上一轮线程pool实验来追求固定收益。

用户未跟踪TODO.md保留原样。最终结果与归档清理记录单独提交，不覆盖已有提交。本任务已完成，没有运行中的设备测试。
