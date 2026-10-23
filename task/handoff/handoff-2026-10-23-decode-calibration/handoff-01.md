# N=1 标定准备完成，等待设备验证

## 目标与状态

用户同意在纯 CPU/current decode 对照之后继续做 N=1 专用标定。基线 757936add，分支 feat/shared-weight-layout。当前 adb devices -l 多次返回空列表，默认 fd8657d6 不在线。按照 AGENTS.md 第 120 行，未执行 Android 构建、部署或测试。以下仅为源码与数据准备，不能声称已标定、改善分配或获得性能收益。TODO.md、paper/ 未动，2K 暂缓。

前轮结果位于 ../handoff-2026-10-23-decode-scheduler/results.md：128 CPU 优于当前约 26–31% 吞吐，1K 接近持平；当前代价模型在 N=1 低估 NPU 分支，GPU 未分配。此次不根据 wetland/foundry 的结果直接修改系数。

## 修改内容

- tests/moe-route-scheduler.h：构造器可选 single_token_model，generic/additive 的分支成本、搜索和最终预测均按实际 nt==1 选择；缺省继续用旧模型。additive 对两份模型都验证契约，所有驻留限制保留。
- tests/fullmodel-oracle.cpp：加载 MOE_DECODE_ROUTE_MODEL，要求同时指定原 MOE_ROUTE_MODEL、calibration_tokens=1 和相同 algorithm；记录是否启用，以及各层 single_token_model。按实际 FFN N 区分，因此单行 prefill 最后层也会使用 N=1 模型，不是仅凭外层“decode阶段”标签切换。
- tests/run-fullmodel-oracle.py：增加 --decode-route-model 参数，早期检查依赖，并复用 SHA-256 校验、部署和记录流程。
- tests/test-moe-route-scheduler.cpp：generic/additive 均验证 N=1 改为 CPU、N=2/128/1024 的 placement/prediction/evaluations 不变、不可用后端不能被选择、非法 additive decode 模型被拒绝。
- 本目录 profile.py prepare/run：独立 technical-1024 capture 的第 127/511 行（0-based），层 0/12 训练、6/18 留出；每层 15 种三端 routed 数量分配 × 3 个 shared 位置，合计 360 个候选。输入是完整 FFN 激活/概率/路由/参考输出的一行，并非完整 decode KV 轨迹。各专家同尺寸，以数量泛化的假设需要独立层与全模型验证。
- 本目录 fit.py：只拟合启动、每活跃 expert 和 shared 三项非负系数；N=1 的 pairs/active/tiles 共线，不同时拟合。使用三个变量的穷举 active-set 最小二乘，无正则参数调优。只在层 0/12 拟合，层 6/18 输出候选排序和 regret。输入要求 45 候选、5 次重复、数值检查通过、common scratch/NEON/共享权重契约、无预热后新增分配；缺页和慢样本不丢弃。输出 runtime-decode-ffn.json 必须再验证 full-model technical 并冻结，不能直接宣称适用于端到端。

当前没有生成任何设备标定模型。原 kernel、二进制和 prefill 模型未修改；只有测试执行器/调度器源码和脚本改变。原生 CPU 整图不能通过改 plan 保留当前实例 KV 并切换，当前可选的是共享布局内全 CPU FFN。保留原生 CPU 作为独立性能基线，不能混称“已实现原生 CPU 回退”。

## 已执行检查

```sh
adb devices -l
g++ -std=c++17 -O2 -Ivendor/nlohmann tests/test-moe-route-scheduler.cpp -o /tmp/moe-test-route-n1
/tmp/moe-test-route-n1
g++ -std=c++17 -fsyntax-only -Iggml/include -Iinclude -Ivendor/nlohmann tests/fullmodel-oracle.cpp
python3 task/handoff/handoff-2026-10-23-decode-calibration/profile.py prepare
python3 task/handoff/handoff-2026-10-23-decode-calibration/fit.py --self-test
python3 -m py_compile tests/run-fullmodel-oracle.py task/handoff/handoff-2026-10-23-decode-calibration/profile.py task/handoff/handoff-2026-10-23-decode-calibration/fit.py
git diff --check
```

宿主 scheduler tests 通过（exhaustive optimum=22，evaluations=144）；fullmodel C++ 语法通过，仅既有 deprecated/ignored-attributes 警告；Python 语法和三项合成系数恢复检查通过。额外核验 prepared.json 全部文件哈希、360 个候选的 45 种 shape 覆盖及四 expert 唯一性通过。prepared.json 保存数据来源及产物哈希，输入在 evidence/。这些不是 Android 性能/数值回归。

独立 scout 审查确认模型选择隔离与契约；其报告的 fit.py comprehension 变量遮蔽问题经最小执行核验不成立：Python 3 comprehension 的迭代变量自有作用域，全部 360 个候选 pairs 总和仍为 4。未因该线索引入无必要改动。

## 恢复设备后的具体步骤

1. adb devices -l 确认 fd8657d6 为 device；检查没有遗留 fullmodel-oracle。不换其他设备。备份当前 Android fullmodel-oracle，使用此前 NDK27d 编译命令增量编译测试执行器；库无需因本轮源码改动重编。编译 scheduler contract test 并在设备运行。
2. 执行本目录 profile.py run。它只使用 /data/local/tmp/llama-decode-calibration-20261023，验证部署与输入哈希，运行两组 N=1 captured FFN profile，每候选两次 warmup、五次计时。profile-frozen.json 记录 boot/model stat/binary hashes，逐组保存 .run.json/.log/.json。失败时保留证据，不覆盖已有 tag 或冻结记录。
3. 执行 fit.py（无 --self-test）得到 runtime-decode-ffn.json，先检查留出层 backend MAPE 和 candidate regret；若简单模型无法解释，先定位并发/启动或 shared/routed 交互，不能直接以旧 NPU 2.3–2.7 倍系数代替标定。
4. 使用独立 technical 文本完成 128/1024+32 连续 decode 的旧/新模型及共享布局全 CPU 诊断，验证预测、驻留、common owner、缓存和 logits；不要用 wetland/foundry 调参。现有 joint-plan.json.gz 保留 resident 三端和最后层 CPU-only。N=1 参数开关为 --route-model <旧prefill模型> --decode-route-model <新模型>。
5. 冻结后在 wetland/foundry、128/1024 前缀与原生 CPU 对照，复用前轮固定 teacher tokens、ABBA/BAAB、两进程、两轮 warmup+四轮32步计时协议。记录 full llama_decode+synchronize、分支预测/实测、分配、major faults、数值和内存。profile 的 N=1 小 arena 与完整模型保留的 prefill arena 不同，不能省略全模型验证。
6. 证据归档后再清理本任务设备目录，保留 GGUF、旧模型/旧binary。全部设备结果完成前保持 tasklist 未完成项。

本阶段以明确标注“设备验证待完成”的独立 Git 检查点保存，后续无需重新检索入口。
