# 共享权重拒绝路径、尾部回归和设备计时

目标仍为三端同一物理权重、每端最佳算子性能退化 <=5%。MoE 是上层，未接入。任务没有完成。

## 检查点及修改

- 首个原型已保存为 `8b0596d1a`，明确标记 performance target unmet；原生备份分支不变。
- CPU 的 direct graph / graph plan 入口预检共享权重，不支持的操作返回 FAILED 或不创建 plan，避免走普通 Q4_0 kernel。能力查询追溯 view_src，输出形状和 descriptor 维度检查同步加强。
- OpenCL 执行入口和能力查询同样预检，拒绝 view、非原始设备、错误输出形状及 index 溢出。导入明确限制到已测试 Adreno A8x，检查 image 最大容量。context 判别用实际 buffer free callback（与仓库已有类型判别习惯一致），不对任意 buffer 强转。
- create 在写入前验证 exporter 描述符；CPU/GPU getters 用减法检查 range，避免加法溢出。
- 新 OpenCL proc `ggml_backend_opencl_get_compute_timing`：聚合 kernel event END-START，含 activation transpose；使用现有 GGML_OPENCL_PROFILING 开关。profiling event 持有 kernel 引用直至 flush，防止共享 handle 先释放 kernel 后查询失效。
- probe 读取 OpenCL 与现有 Hexagon DSP counters，在每次同步完成后取 delta；host wall 时间仍保留，不能和 DSP/GPU 时间相加。
- runner 新增 --edges / --profile。edges 两组128×32/256×96，N=1/2/3/4/5/7/8/9/31/33。新增 null、非法维度/descriptor、重复bind、首尾partial get、unsupported DUP直接执行拒绝、view拒绝。

## 实际命令和证据

```sh
cmake -S . -B build-android-opencl-hexagon -DLLAMA_BUILD_TESTS=ON -DGGML_OPENCL_PROFILING=ON
cmake --build build-android-opencl-hexagon --target shared-weight-probe ggml-cpu ggml-opencl ggml-hexagon -j 8
python3 tests/run-shared-weight.py --deploy --shared --edges --tag shared-v9-edges-cpu --backend CPU --reps 4
python3 tests/run-shared-weight.py --shared --edges --tag shared-v9-edges-gpu --backend GPUOpenCL --reps 4
python3 tests/run-shared-weight.py --shared --edges --profile --tag shared-v9-edges-htp --backend HTP0 --reps 4
python3 tests/run-shared-weight.py --shared --profile --tag shared-v9-profile-gpu --backend GPUOpenCL --reps 32
python3 tests/run-shared-weight.py --shared --profile --tag shared-v9-profile-htp --backend HTP0 --reps 32
python3 tests/run-shared-weight.py --shared --tag shared-v9-cpu --backend CPU --reps 48
```

所有证据在 evidence/ 同名 JSON / .run.json / .log；GPU profiling CSV/trace 同目录。每次运行 manifest 包含已部署二进制 hash。CPU/GPU/HTP 各20个edges全部通过，原生及共享都测。CPU/GPU最大NRMSE分别0.00164/0.00211。生命周期释放在每组矩阵后发生，下一组重新create/import。

## 真实设备计算结果

以下均为 native/shared 微秒；GPU 是每次 MUL_MAT 所有 kernel event 之和（含 transpose），HTP是 DSP op usecs。

| K×M | N | GPU | HTP |
|---|---:|---:|---:|
|2048×1408|1|38.91 / 1408.51|52 / 51|
|2048×1408|32|380.93 / 468.99|88 / 88|
|2048×1408|128|609.79 / 744.96|115 / 115|
|1408×2048|1|33.02 / 972.03|50 / 51|
|1408×2048|32|263.42 / 309.38|89 / 89|
|1408×2048|128|445.95 / 594.18|111 / 111|

**GPU decode实际慢29–36倍，而之前graph同步计时仅2–3倍。这正是不能用host固定开销稀释差距的证据。** HTP维持原布局、原kernel，DSP差异0–2%。CPU仍明显未通过，详见v9-cpu JSON。

## 下一步与限制

1. GPU decode的循环sub_group_broadcast索引目前是变量j，原生为常量展开；需要核查编译展开/内核计算方式。没有证据证明该差距全由物理布局造成。
2. CPU现布局需要较多寄存器转置或分块重排。可评估直接逐plane乘加，或更换CPU4x8共同布局并为HTP写直接VTCM解码。不能把临时整矩阵CPU副本当作单份权重实现。
3. 当前是确定性合成数据，尚需指定模型实际权重测试、稳定统计和完整阶段验收。
4. API显式选择，非通用自动scheduler重写；不支持任意views/types/批维。HTP owner仍依赖公开的immutable契约。
5. GPU profiling仍为编译开关，开发构建当前ON；部署产物及profile日志由runner维护，远端自有目录暂留供后续实验。
