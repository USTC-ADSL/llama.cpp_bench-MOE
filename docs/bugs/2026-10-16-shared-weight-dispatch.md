# 共享权重的非支持操作误入普通内核

## 现象与触发条件

新CPU共享Q4_0原型只在supports_op中限制MUL_MAT。调用方直接向backend提交DUP、非法view或不支持的matmul时，CPU普通graph/plan路径不查询supports_op，可把HTP tiled bytes当canonical Q4_0。GPU也需要在执行前拒绝共享权重的不支持操作。

## 原因和修复

CPU extra traits未匹配时会回落普通执行；supports_op是能力查询，不是所有直接执行接口的强制门禁。增加CPU graph/plan和OpenCL graph入口预检，追溯view来源，明确失败。增加输出维度、descriptor、索引上限和导入设备限制。原生kernel保留。

OpenCL profiling原来保存裸kernel句柄，新的共享weight handle会比backend先释放kernel。现让每个待处理event保留kernel引用，flush读取名字后释放，保证生命周期匹配。

## 验证

增量Android构建和设备命令见 `task/handoff/handoff-2026-10-16-shared-weight-layout/handoff-02.md`。fd8657d6上v9 edges三端各20组均通过；CPU/GPU直接DUP返回FAILED，view能力查询拒绝，正常共享及原生MUL_MAT的N=1/2/3/4/5/7/8/9/31/33通过独立数值参考。GPU启用event profiling跨多个handle释放/重建运行通过。

## 限制

实验API仍为受限2D Q4_0，调用者必须保持backend与owner生命周期及immutable契约。未支持任意view。功能回归通过不等于性能验收，CPU/GPU仍未达到5%目标。
