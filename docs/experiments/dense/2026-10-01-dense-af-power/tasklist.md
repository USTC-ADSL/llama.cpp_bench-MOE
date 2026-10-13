# 无线设备 A/F 能耗对比

用户要求少量测量，不反向复测。设备 `192.168.50.180:44159`，CPU/GPU/NPU；FA Attention 与完整 FFN；prefill 64/256，decode 1 token + 历史 KV 256。同一中间层（索引17），驻留权重与确定性输入，CPU全部8核。测整机电池侧能量，不宣称芯片独立功率。

- [x] 确认在线设备、既有模型、电池接口和测试范围
- [x] 实现短持续窗口与设备本地电池采样，增量构建和设备冒烟
- [x] 核实电流尺度，完成单轮18个窗口和空闲基线
- [x] 汇总功率、每块能耗和比值，清理设备临时参考，Git检查点

已确认：SM8750，CPU0-7，无充电。模型 `/data/local/tmp/models/Qwen2.5-3B-2K/ggml/weights.gguf`。root可读 `/sys/class/power_supply/battery/{current_now,voltage_now,charge_counter}`；空闲电流原始约70–130、电压4196000、电量3936000。电流疑为厂商mA，须用电量变化核对，不能照标准µA硬套。虚拟gauge虽有两个IC索引，但没有乘2证据，不擅自加倍。

最终核实：current_now是mA；实机活动电池节点oplus,batt_num=2，公开驱动返回单节最高电压，因此最终功率/能量估算乘2，原始trace未改写。详见battery-units.txt、report.md和handoff-01.md。正式18窗口通过误差、积分和无休眠核验；初始无wake lock诊断已隔离。
