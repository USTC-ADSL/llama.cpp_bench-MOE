# 36层resident A/F：完成交接

## 目标与实测结论

用户要求Qwen2.5-3B全部36层，CPU跑Attention、NPU跑FFN，选前轮已测最低能耗点，固定输入输出、不做激活交接，检查多层收益。已全部完成。最准确的报告口径见 `report.md`。

三组均为36层全部权重图常驻、循环A0F0..A35F35、同一层同一块固定输入、5s预热+约5s功率窗口、亮屏、decode1/KV256、双大核6–7、2线程、NPU NOM。

- mixed CPU1.4016GHz A / NPU NOM F：109.328ms、321.402mJ/完整36层pass、2.940W、46轮。
- 全NPU NOM，host2.6496GHz：223.211ms、386.863mJ、1.733W、23轮。
- 全NPU NOM，host1.4016GHz匹配对照：237.168ms、387.418mJ、1.634W、22轮。

相对原host频率全NPU节能16.92%、延迟降低51.02%；匹配host时节能17.04%、延迟降低53.90%。CPU频率全程固定，不做逐块调频；实际提交、同步、双后端驻留进入功率测量。无activation handoff，无跨块真实数据依赖，无embedding/head，不能称为真实token TBT。只是选定点比较，未对36层重扫DCVS、未测全CPU/GPU。

单层乘36与全层结果不一致，尚未剖析具体原因；不要直接归因为某个驱动/内存瓶颈。没有多次独立窗口/置信区间。

## 源码与结构

- `tools/dense-af-bench/dense-af-bench.cpp`：新增 `--resident`、`--ffn-device`；默认旧单块路径保留。
- `tools/dense-af-bench/resident-bench.h`：全部层block在vector中常驻，独立权重和输入，逐块同步；对每个block检查CPU参考，测量窗口不分配权重/图、不传递激活。
- `tools/dense-af-bench/run-resident-android.py`：生成72个旧CPU路径参考，3层resident CPU检查，然后3组全部层测量；CPU/device trap+host finally恢复设置，亮屏临时5h。
- `tools/dense-af-bench/summarize-resident.py`：独立积分、36层/72块/后端/频率/正确性检查，CSV/JSON与PNG/PDF。

`other` backend先声明、other_workers后声明、blocks最后声明；析构时先blocks，再解绑线程池，再backend，主CPU worker/backend寿命覆盖helper。只读审查无阻塞项。

## 验证、数据与部署

实际增量构建 `cmake --build build-android-opencl-hexagon -j 8 --target dense-af-bench` 成功。CMake因版本变化重新生成，随后8个增量步骤成功（base版本对象、共享库重链接、benchmark编译/链接）；DSP源码未改。

独立设备目录 `/data/local/tmp/dense-af-multilayer/`，新benchmark+匹配host库+真实ggml/src/ggml-hexagon/libggml-htp-v79.so。7文件sha256已核对，见deployment-hashes.json。旧dense-af-dvfs目录和模型未覆盖。

采集命令：

```sh
python3 tools/dense-af-bench/run-resident-android.py --serial 192.168.50.180:44159 --model /data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf --output task/handoff/handoff-2026-10-01-af-multilayer/results
python3 tools/dense-af-bench/summarize-resident.py task/handoff/handoff-2026-10-01-af-multilayer/results --output task/handoff/handoff-2026-10-01-af-multilayer/analysis
```

results中cpu-reference.log验证旧逐块路径，resident-cpu-check验证0/17/35常驻CPU与参考一致；三组正式结果均72块检查通过，混合最大relative RMS0.804%、全NPU1.308%。所有NPU初始core1248/HMX979.2MHz、getter0。能量积分、BOOTTIME、CPU频率校验通过。CV0.465/0.653/1.753%，漂移0.283/0.818/2.271%，温度30.0–30.1C。

analysis内summary.json/results.csv/resident-comparison.png/pdf为最终成果，图已目视检查。参考数据已从设备删除72文件+目录，保留可复现部署。CPU/GPU限频原值、3h超时已恢复，wake lock空，屏幕Awake，见final-device-state.txt及results/restored-state.json。

## Git与下一步

起点35881a896，所有本任务改动随独立最终检查点提交；最终ID用git log -1查看。用户未跟踪TODO.md不动。没有剩余必要设备测试，不需要重复测相同配置。

若用户追问收益原因，需要另做阶段计时/profile来定位全NPU多层非线性；当前数据只支持比较结果，不能解释具体瓶颈。若用户追问论文证据，称“多层固定输入无激活交换microbench实测收益”，不要跨用旧QNN数据或声称真实解码已节能17%。
