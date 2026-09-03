# Exp1 LMKD/OOM 日志误判

## 问题现象

4 GiB 限制实验的部分 `result.json` 将 `lmkd_oom_evidence` 标记为 `true`，但对应运行正常完成，或仅在加载超时后由 runner 主动终止。

## 触发条件

实验时间窗内只要出现任意 `out of memory` 文本，或者同一份过滤日志中同时包含 lowmemorykiller 关键词和 `llama-server`，旧逻辑就可能判为目标进程 OOM/LMKD 证据。实际日志包含：

- `lowmemorykiller ... device has enough memory ... disable killing`；
- 针对其他后台应用的 `am_kill`；
- vendor 组件打开节点失败的 `ENOMEM (Out of memory)`；
- 独立的 `llama-server` 资源监控行。

## 根本原因

旧实现对整段 logcat 分别执行宽泛正则。关键词可以来自不同日志行，也没有验证 kill 的目标进程，因此把内存观测、其他进程回收和目标进程实际被杀混为一类。

## 修改内容

新增逐行 `analyze_memory_logcat()`：

- 明确排除 `disable killing`；
- 只把同一行中同时出现 `llama-server` 和 `am_kill`、`oom-kill`、`killed process`，或 LMKD/lowmemorykiller 明确 kill 动作的记录视为目标 kill 证据；
- 单独统计非目标的系统内存压力/后台回收事件；
- 在结果中保存匹配到的目标 kill 原始行，便于审计。

使用修正后的函数重新解析了 12 份已保存的过滤日志，并更新对应 `result.json` 和 `summary.json`，无需重跑模型。

## 验证方法和结果

- `disable killing` 回归样例返回 `target_kill_evidence=false`；
- `lowmemorykiller: Killing 'llama-server'` 回归样例返回 `true`；
- 针对其他应用的 `am_kill` 回归样例返回 `false`；
- 12 份正式日志重新解析后，针对 `llama-server` 的 kill 证据均为 `false`；
- OpenCL run-01 保留 37 条针对其他应用的系统内存压力事件；
- `python3 -m py_compile` 和 `git diff --check` 通过。

## 限制与风险

厂商日志格式可能随系统版本变化。当前分类只接受明确、同行且包含目标进程名的 kill 证据；如果未来 LMKD 只记录 PID 而不记录进程名，需要结合实验 PID 和时间窗增加 PID 级匹配。
