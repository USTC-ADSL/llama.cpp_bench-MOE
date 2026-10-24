# 提前 shared：实现与设备校验中

目标：用户要求尝试 CPU router 计算期间提前派发 NPU shared。基础 3df51e080，当前 feat/shared-weight-layout。默认设备 fd8657d6 在线，boot ID 与 10/23 相同。不要触碰 TODO.md、paper/。本任务与前一轮 overlap 已完成任务不同。

## 实现

- include/llama.h、src/llama-model.h/.cpp：新增可选 moe_input_callback，在 replacement FFN router 前返回同形状输入依赖；新增 moe_router_cpu 策略 setter。默认均禁用。
- src/models/qwen2moe.cpp：callback 分支中 router 依赖 input callback 返回tensor；给 projection/argsort/scale增加命名。
- src/llama-context.cpp：显式 CPU router 模式将 projection、softmax、argsort/topk、scale固定 CPU，不能只靠权重的 CPU buffer。
- tests/moe-early-shared.h：N=1 input hook + finish callback，常驻 moe_branch_worker；early 在router前复制8KB输入并submit，late 在router后、CPU routed之前submit；CPU routed复用 CPU4x8 grouped缓存图；join后NEON有序FMA归并。cpu控制走原单份布局全CPU专家；regular控制走原async split。
- fullmodel-oracle.cpp：--decode-shared-launch 的四种策略early/late/cpu/regular，decode_active 在quality及正式decode中打开、prefix重建时关闭。prefill继续原调度；末层只额外绑定NPU shared view，prefill resident仍维持原CPU-only限制。共享worker先于layer对象析构排空。
- run-fullmodel-oracle.py：--router-backend cpu与--decode-shared-launch；router trace按name/nrows区分prefill/decode并在结束时保存。

独立只读审查未发现生命周期/输入差异问题。new API和算图调度有改动，但计算kernel未改。所有旧源码/动态库/执行器在 evidence/backup，哈希backup.json。当前和失败版本完整冻结副本分别 evidence/frozen 与 evidence/frozen-v1-failed；禁止覆盖。

## 已完成与失败

实际命令记录build-commands.json。cmake --build build-android-opencl-hexagon --target llama -j8成功；修改llama-model.h导致必要增量重编。第一次执行器link抢在库完成前，undefined symbol，记录build-executor.log；库完成后重新编译成功，记录build-executor-after-library.log。

v1 pilot-early进程134退出，HTP dspqueue_read 0x2e。设备未重启。初始日志只有graph reserve后出错，没有足够标记精确判断prefill还是decode，不能写成已证实prefill错误。diagnostic-off-router-cpu（关闭input hook/早启动）成功完成完整decode，trace显示projection/softmax CPU，但topk VIEW/source仍HTP。HTP device->session为registry共享；早启动与HTP路由链并发提交是明确冲突风险。随后补充完整CPU router调度，不加全局序列化锁、不吞错误。失败日志及logcat保留（logcat含混合编码，读取需显式windows-1252或原始字节）。

v2库构建build-llama-router-cpu.log成功，执行器build-executor-router-cpu.log成功。冻结v2后正在run.py pilot，session6780。计划pilot-v2-early/late/regular/cpu/off五进程；early开router-trace，其余无trace；每进程technical128、32decode步、1计时序列+2warmup。每进程间20秒冷却。旧tag禁止重跑。

## 后续具体步骤

1. 轮询session6780，若失败先读对应log。当前设备远端 /data/local/tmp/llama-early-shared-20261024，二进制已部署v2。
2. pilot全成功后运行 `python3 task/handoff/handoff-2026-10-24-early-shared/analyze.py pilot`，验证4类router节点24层均CPU、早/晚/regular decode logits逐位相同、cpu/off相同、各policy prefill logits相同。trace性能不当作正式收益。
3. 正确性通过后保存实现独立提交，并更新tasklist；创建docs/bugs简报记录CPU router仅放权重不能确保argsort CPU及其并发风险。
4. `run.py formal`：wetland/foundry，128/1024，early/late/cpu每种每输入2进程，4计时序列×32步，合计24进程3072计时token，顺序反向均衡，20秒冷却。prefill同模型同route model、同teacher tokens。
5. `analyze.py formal`会校验所有报告、实际分配/arena/输出，并汇总early-vs-late、early-vs-全CPU专家。全CPU专家仍是custom dense路径，不能称原生CPU整图。
6. 汇总提前量、shared实际启动是否早于router结束、join等待及e2e时间。原始证据归档、设备清理、文档/结果独立提交。尚无正式性能结论。

注：analyze.py尚未实际运行，需根据真实报告验证其契约。run.py冻结源码/库，修改冻结文件必须保存旧冻结版本并另设tag，不可覆盖原有证据。不要重跑10/23或已完成10/24 overlap实验。
