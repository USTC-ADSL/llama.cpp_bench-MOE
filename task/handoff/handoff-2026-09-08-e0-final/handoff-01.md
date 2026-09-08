# E0/E0B 最终补验交接

用户要求补验 OpenCL 三模式整模型与实际预算容忍度，并把两份短报告放一起。已完成受测路径验证并交付 `docs/Motivation_Exp/E0_E0B_final/E0.md`、`E0B.md`。结论：E0 在原约定范围 PASS；E0B 核心/受测路径 PASS，但全部 LLM 物理 RAM 的约5%误差尚无证据，不标整体验收 PASS。R允许估计，不要求精确校准；256MiB是预留估计，不能冒充已验证的隐藏内存峰值。

本轮不改 runtime。新增 `tests/run-e0-final.py`，复用原 guarded/watchdog、固定128/32请求、独占设备目录、私有cgroup；每个产物push后验证SHA256，逐次日志gzip保存。新增 `tests/test-e0-envelope.cpp`，复用原真实分配辅助，仅新增3GiB边界用例。旧运行器/旧报告均未改。现用 OpenCL SHA256 `70bac8ab6b0a692507e180e04ece6c9dc5e3e1d4bca4901416b832cbb666c20f` 与修复版数值回归一致。

实际构建：

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ -Iggml/include tests/test-e0-envelope.cpp -Lbuild-android-opencl-hexagon/bin -lggml-base -lggml-hexagon -lOpenCL -o build-android-opencl-hexagon/bin/test-e0-envelope
python3 tests/run-e0-final.py --tests controls --apply
python3 tests/run-e0-final.py --tests budget --apply
python3 tests/run-e0-final.py --tests controls --apply
python3 -m py_compile tests/run-e0-final.py
```

第一次controls是在新增tag/-v之前执行的，产物在 `evidence/`：三模式请求与token均正常，但默认日志缺少offloaded层数，故自动valid=false、总退出1，属于路径证据不足，不是数值失败。随后增加-v及默认final子目录，补跑三模式全部valid=true；预算五项也全部valid=true。不要删第一轮或误读其valid=false；新运行需使用新字母数字tag，脚本拒绝覆盖已有结果。

正式数据在 `evidence/final/controls.json`（3项）、`budget.json`（5项）及对应日志。控制模式：25/25层GPU、128/32、cache_n0、32token一致、无截断，非约束extra64GiB。边界：charged768MiB+extra3072MiB+R256MiB，三组helper接受48块、拒绝49块，used/reserved归零；真实GPU/HTP加载在账本超额处退出1，无cgroup限额失败。拒绝时GPU峰值2348MiB，HTP2075.81MiB。所有guarded测试无安全截断。

最终主线程检查：jq断言3个controls有效且token完全一致、5个budget有效；检查日志真实25/25及Adreno专用路径；语法检查、git diff --check；fd上 `test ! -e /data/local/tmp/llama-e0-final` 与私有 `/dev/memcg/llama-e0-final` 通过，无llama-server/test-e0-envelope进程，模型仍在。临时设备目录/cgroup已清理，原部署未动。

Git起点 `252aa4396`，本轮结果独立提交（可通过 `git log -1 --oneline` 获取）。用户已有 `AGENTS.md`、`docs/Motivation_Exp/Exp1/Exp_require.md` 不纳入。不push。

本轮没有剩余报告交付任务；若未来要使E0B全RAM验收PASS，需缩小/核验真实模型页charge与DSP/驱动隐藏驻留的不确定范围，而非重复跑已通过的token或64MiB边界测试。不可把cgroup、KGSL mapped、账本、PSS的独立峰值直接相加当物理总RAM；未支持image路径仍需标注。
