# 连续 decode 接入与128定向验证

目标：原生NPU vs 当前joint调度+单份CPU4x8权重+common scratch，128/1024前缀、32步teacher-forced连续decode。2K不跑，模型不重拟合，保留最后层CPU-only resident；CPU expert仍4线程。设备fd8657d6在线。用户TODO.md和paper/不动。

变更仅tests/fullmodel-oracle.cpp的single_layout配置guard：去掉!quality_steps，其余shared/direct/fused/overlap/arena/lifetime、native decode graph互斥规则保留。已有run_into允许nt=1；shared job按完整counts+full_input长度缓存，scratch布局按prefill预算足以覆盖decode，KV context已为continuation预留容量。不改kernel、调度器或模型参数。

旧binary、库、源码备份在evidence/backup，backup.json记录hash。使用NDK27d的aarch64-linux-android34-clang++，-std=c++17 -O2 -Iggml/include -Iinclude -Ivendor/nlohmann，编译tests/fullmodel-oracle.cpp，链接build-android-opencl-hexagon/bin中的lllama/lggml/lggml-base。仅既有deprecated warning。部署固定目录/data/local/tmp/llama-decode-scheduler-20261023。

运行python3 task/handoff/handoff-2026-10-23-decode-scheduler/run.py pilot：先NPU生成wetland128的32个teacher tokens，再执行common/private两个进程，各两次sequence warmup、两次timed序列，每序列32步；每轮计时外重建KV前缀。analyze.py --pilot通过：33个位置logits逐位一致，route/placement一致，重复序列logits逐位一致，warmup后job allocations稳定为1448，所有decode步骤为24层、每层N=1、4 routed pairs。记录pilot-validation.json。common约50.55ms/token仅为诊断，不与无正式decode计时的teacher NPU混作收益。

初步分配：128 pilot前23层CPU/NPU各2 routed pairs，shared全部CPU，最后层4 routed pairs与shared全CPU；GPU未分配。每token调度决策约0.065ms，MoE墙钟约25.35ms。这是N=1外推的实测行为，不是重新拟合结果。

新增run.py：pilot/teachers/formal三阶段，均拒绝覆盖tag；正式16进程，两个前缀×两个文本×两个方案×两个进程，每进程2次warmup+4次timed序列（32步），20秒进程间冷却，ABBA/BAAB。analyze.py检查hash、同token/KV前缀、各轮/跨进程数值稳定、private/common容量与单owner、真实route和分配稳定，并汇总NPU/当前策略性能及logits差异。teacher只固定输入token，不回放router IDs。模型context scales仅在128–1024校准，decode N=1使用128端点scale及公式外推。

接下去：run.py teachers生成其余三组teacher；完成后run.py formal冻结teacher与binary/source/模型。不要并发跑其他设备任务，不要修改已冻结run.py/源码/库/模型。formal结束运行analyze.py，补报告和归档；确认归档完成后清理本任务设备目录，保留GGUF。runs.json及evidence/*-driver.log是进度。若中断先检查进程与已有tag，不能重新启动覆盖运行。
