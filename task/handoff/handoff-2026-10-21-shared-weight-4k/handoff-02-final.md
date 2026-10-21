# 最终交接：有效测量完成，扩展确认受设备重启阻断

本文件替代 handoff-01.md 的“当前工具/进程”和待办状态。目标是验证默认 fd8657d6 上共享布局的 MoE 吞吐、探索专家分配，并对比 GPU/HTP router。用户允许 4K 异常时先测 1K，确认设备默认设置、无额外压力任务。

## 已完成与边界

- 1024 token、原 c8-g15 分配、两个输入完整 ABBA 共 48 次正式请求：wetland 1252.27 → 1182.67 ms，吞吐 +5.89%；foundry 1261.60 → 1165.90 ms，吞吐 +8.21%。这是完整 prefill，不含加载，也不是 decode。
- 权重 payload 两路径均 6.961 GiB；共享路径专家 arena 多 202.938 MiB。CPU 仍为原 4 线程，未验证 1–2 小核限制。跨布局 top-1/top-10 集合一致，logits 有约 1.1–1.6% NRMSE。
- 三候选探索完成，冻结 c8-g45（wetland 1121.29 ms），未完成独立确认，不能作为已确认的额外收益。
- Router 最终 probe：真实层 0/12、六种 N、每条件 64 次交替测量。1K projection GPU 算子速度为 HTP 的 2.06×/2.31×；全部支持的完整 router top-4 与参考一致。只引用 router-summary.json 与 router-*-final.json，初稿 view stride 读取有误，已修正并重跑。
- GPU router 整模型 smoke 与 24 层 buffer 放置核验通过，11.25 MiB 权重迁移，无额外 router 副本。端到端 ABBA 未执行，不能引用诊断请求耗时推导加速。
- 4K 整批在图预留附近 kernel_panic；4×1024 连续 KV 能运行但缺页/顺序漂移严重。另两次 1K 在权重加载时整机中断，共同根因未确定。第三次后停止重负载，不应自动重跑。

## 代码、命令与证据

分支 feat/shared-weight-layout；既有检查点 469104b85（测试工具）、d52687a19（1K 基线），最终结果另存 Git 提交。用户 TODO.md 保持未跟踪且未修改。生产计算 kernel 未改。

测试代码涉及 tests/fullmodel-oracle.cpp、tests/run-fullmodel-oracle.py、tests/moe-router-probe.cpp、tests/run-moe-router-probe.py、tests/run-moe-shared-weight-4k.py、tests/analyze-moe-shared-weight-4k.py。本阶段最后两个脚本增加确认组/router ABBA 和仅完整组才输出对比的分析逻辑；准备好了但扩展整组尚未通过设备验证。

实际执行命令：

```bash
python3 tests/run-moe-shared-weight-4k.py baseline --tokens 1024 --chunk 1024
python3 tests/run-moe-router-probe.py
python3 tests/analyze-moe-shared-weight-4k.py --tokens 1024
```

NDK 增量编译命令见 protocol.md；只有既有 ggml_mul_mat_set_prec 废弃警告。已做 Python 语法检查和 git diff --check。完整设备命令/环境在 evidence/*.run.json，二进制 hash 在 binaries.json。HTP 实际库位于 build-android-opencl-hexagon/ggml/src/ggml-hexagon/libggml-htp-v79.so，不要误用 bin 下旧副本。

正式汇总见 results.md、summary-1024.json、router-summary.json，候选冻结见 placement-selection.json；失败证据含 evidence/interrupted-first-1k/ 与第三次 reboot/pstore。原始 evidence 不加入 Git，已归档：

`/mnt/sda1/pzw/moe/experiment-artifacts/shared-weight-moe-1k-router-20261021.tar.gz`

157 个成员，37,015,633 字节；SHA256 `6ae29aace0cde1f7eaa5bad8870bf71e02b3c8eed9c3f4b1b4be0ec3baa14265`。全部成员逐项核验，清理前再次核验整个归档 hash。清理/本最终交接及报告的恢复命令小修订发生在归档之后，以 Git 为最新记录。

## 设备最终状态与未完成事项

设备在线，无 fullmodel-oracle/moe-router-probe 进程。已删除且确认不存在 `/data/local/tmp/llama-shared-weight-4k-20261021`，释放约 104.8 MiB 临时文件；模型与公共缓存保留。记录见 cleanup-record.json。无在运行的工具 session。

仍需在设备稳定性恢复后完成分配独立确认及 GPU router 端到端 ABBA，4K 还需解决缺页/稳定性后重测。不要将未完成项打勾，也不要从诊断结果抽取有利对照。后续恢复时先归档/移走 evidence 中失败的 wetland-confirm-control-a 同名文件，保留证据，再运行：

```bash
adb devices -l
python3 tests/run-moe-shared-weight-4k.py confirm --tokens 1024 --chunk 1024 --candidate c8-g45 --deploy
python3 tests/run-moe-shared-weight-4k.py router --tokens 1024 --chunk 1024 --deploy
python3 tests/analyze-moe-shared-weight-4k.py --tokens 1024
```

候选已经冻结，不要利用确认组重新选候选。不得把三次整机中断描述为已定位/已修复的共享布局问题。
