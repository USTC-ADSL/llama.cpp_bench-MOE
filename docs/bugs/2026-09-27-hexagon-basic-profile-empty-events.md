# BASIC profiling 的空事件列表格式化

- 触发：设置 `GGML_HEXAGON_PROFILE=1` 后，初始化主动将PMU事件列表清空，再打印该列表。
- 根因：`ggml-hexagon.cpp::vec_to_str` 无条件对生成字符串调用 `pop_back()`；空vector产生空字符串，此调用属于未定义行为。本次是启用逐算子诊断前的源码检查发现，没有宣称已复现设备崩溃。
- 修改：仅当字符串非空时移除末尾逗号；非空PMU列表输出保持原样。
- 验证：增量编译 `cmake --build build-android-opencl-hexagon --target ggml-hexagon -j 4` 成功；fd8657d6上的128/768完整模型BASIC profiling各运行8次正式请求，成功取得逐算子数据，完整logits与旧原生路径逐字节相等。日志位于 `task/handoff/handoff-2026-09-27-device-timing/evidence/timing-{128,768}-npu-op-a.*`。
- 限制：本次未执行PMU自定义8事件模式；修改只影响空列表的日志格式化，不改变DSP电源请求或算子。
