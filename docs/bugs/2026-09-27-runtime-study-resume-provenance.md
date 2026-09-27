# 状态校准实验恢复时的证据身份检查

- 现象与触发条件：同名 tag 的旧 manifest 只要退出成功、结果存在且二进制相同，就可能被恢复逻辑复用，即使 prompt、模式或分配已改变。本轮首次采集未发现错误复用；这是独立审查发现的恢复路径缺陷。
- 根因：`tests/run-moe-runtime-state-study.py::run` 的恢复分支缺少输入与配置身份校验。
- 修改：同时核对 prompt 哈希、命令的 tokens/mode/reps、输出 tokens/requested_mode/样本数、状态记录、custom 执行开关、计划哈希，以及 live-auto 模型哈希和决策记录。
- 验证：28 个实际 Android 校准进程的本地 manifest 全部通过恢复检查，未重新执行推理；在临时目录分别注入错误 prompt、命令长度、结果长度、custom 开关、计划哈希，5 项均拒绝复用。Python 编译及 `git diff --check` 通过。
- 限制：此检查用于本阶段固定协议的未归档证据；归档后的重现应使用 `audit_results.py`，不能把其恢复逻辑当作任意历史实验的通用缓存。
