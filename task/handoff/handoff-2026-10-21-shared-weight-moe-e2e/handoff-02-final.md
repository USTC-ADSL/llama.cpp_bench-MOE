# 端到端对照完成

本轮请求已完成：在 fd8657d6 用原 threeway-768 c8-g15 frozen wetland/foundry 设置，对同版本 native expert bank 和单一 CPU4x8 中间层完成 ABBA。代码实现检查点 b3e9e0f34，结果详见 results.md / comparison.json。

主结果：wetland 1012.694 → 936.305 ms（时延 -7.543%）；foundry 1045.978 → 921.589 ms（-11.892%）。每条件 2 个进程，各 2 warmup +6 timed，共 48 个正式请求。所有进程 binaries hash、prompt IDs、plan hash、执行 flags 已由 tests/analyze-moe-shared-weight-e2e.py 核验，全部通过。

内存：单份 expert payload 7474348032 B（6.961 GiB），没有常驻 native expert bank 副本；旧路径原本也按专家唯一放置，故权重 bytes 几乎相同。新 activation arena 比旧增加 160382976 B（152.953 MiB）。这是保守容量的已知代价，未宣称达到“无额外内存”的最终目标，也未把进程 RSS 当 DMA 总物理内存。

数值：三端 retained row-slice edge suite 共 60 cases 和实际 layer0 三端 FFN suite 9 cases 通过。完整模型 logits 同条件跨进程 bitwise 相同；两个输入 top1 与旧路径相同，cross-layout NRMSE 1.80%/1.42%。真实后续路由有所变化。不能从有限 prompt 推断整个模型生成质量，也没有测 decode。CPU 算子线程保留旧设置 4，没有对 1–2 小核约束另做验收。

归档 `/mnt/sda1/pzw/moe/experiment-artifacts/shared-weight-moe-e2e-20261021.tar.gz`，65 个文件，25,169,907 B；包括实际部署库/二进制、原始 samples/logits/logs、layer/slice 验证、固定 plans。artifact-manifest.json 记录 archive/member SHA256；逐成员解压后重新核验全部通过。注意实际 HTP 库来自 build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so，不能用 bin/ 内的旧同名文件替代。初始 deployment.json 在 OpenCL F16 修正之前，最终 hash 以 comparison.json 和每进程 run.json 为准。

设备自有目录 `/data/local/tmp/llama-shared-weight-moe-e2e-20261021` 已在确认无测试进程后清理。原始模型和公共 OpenCL cache 未改动。清理记录 device-cleanup.json。宿主 evidence/ 保留并由本目录 .gitignore 排除，inputs/、summary、manifest 和复现脚本提交 Git。用户 TODO.md 未动。

后续如果继续优化：优先缩小 grouped-graph activation arena 的安全上界并独立复测，不要把本轮旧数据与新容量/执行策略混作一组。其次才是扩展直接 MMID 消费公共布局或验证 1–2 小核辅助线程约束。当前所有原始后端 MMID 实现均保留，取消 MOE_SHARED_WEIGHTS=cpu4x8 即恢复旧执行器；改动前完整版本为 425ee44fd。
