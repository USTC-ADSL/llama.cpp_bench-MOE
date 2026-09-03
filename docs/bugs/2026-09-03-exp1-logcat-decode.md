# Exp1 logcat 非 UTF-8 解码失败

## 问题现象

4 GiB 限制实验的首个 CPU mmap 样本完成加载等待和设备清理后，运行器在读取 `adb logcat` 时抛出 `UnicodeDecodeError`，因此没有生成该次 `result.json`。

## 触发条件

设备 `3B661501LA000000` 的 `logcat -b all -d -v epoch` 输出中包含非 UTF-8 字节。宿主 Python 以 `text=True` 和默认严格错误处理解码整个输出时触发异常。

## 根本原因

`exp1_runner.py` 的通用 `run()` 使用文本模式收集子进程输出，但没有指定解码错误策略。实验日志来自 Android 的多个 ring buffer，不能假定所有 vendor 日志载荷都是有效 UTF-8。

## 修改内容

为通用子进程文本解码增加 `errors="replace"`。无效字节会替换为 Unicode replacement character，时间戳、LMKD/OOM 关键词和其他有效日志仍可继续解析。

同时增加加载等待时长和 cgroup 达到上限的明确结果字段，使达到 4 GiB 后持续回收直至超时的运行能够分类为 `cgroup_limit_timeout`。

## 验证方法和结果

- `python3 -m py_compile task/handoff/handoff-2026-09-01-exp1/exp1_runner.py` 通过；
- 使用产生非法 UTF-8 字节的子进程调用 `run()`，确认不再抛出异常并返回替换后的文本；
- 后续设备实验将再次读取同一设备的完整 logcat 并生成结构化结果，作为端到端回归验证。

## 限制与风险

替换策略会丢失非法字节的原始值，但本实验只依赖 ASCII 时间戳和 LMKD/OOM 关键词，不影响目标事件识别。原始 logcat 本身并未按字节永久保存；如后续需要分析 vendor 二进制载荷，应改为二进制采集后单独归档。
