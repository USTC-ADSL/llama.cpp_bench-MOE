# Memory probes

这里保留两项仍有独立诊断价值的前置工具：

- `buffer-capacity-probe`：CPU/OpenCL/rpcmem/HTP/组合容量与驻留验证。
- `htp-map-latency-probe`：HLOS FastRPC 与 DSP `HAP_mmap2`/`HAP_munmap2` 延迟。

两个探针都链接公共 Memory Runtime，FastRPC session 按进程复用，rpcmem allocation 由 `HtpPrivateBuffer` 持有，fd 导出和失败回滚不再各自实现。保留旧库只有 `rpcmem_alloc` 时的有界回退。map latency 的计时仍直接包围预先解析的 map/unmap 函数，session、allocation 和 fd 校验不进入其计时；capacity 的 CPU mmap/OpenCL 极限探测仍保留所需的底层 API 探针。

V79 DSP RPC/skel 位于 `dsp/`。独立的 rpcmem/OpenCL dma-buf 字节读回 probe 已在 Expert Slot demo
正常后移除；不要在这里重新引入另一套 Expert Slot 生命周期实现。

```sh
MOE/tests/probes/build_probe.sh
MOE/tests/probes/run_probe.sh --smoke \
  --modes cpu,opencl,rpcmem,htp,combined
```

Expert Slot map decomposition 使用精确 Slot 容量，并分别保留 pinned/delayed 行：

```sh
MOE/build/android-raw/bin/htp-map-latency-probe \
  --slot-counts 1,4,8,16 --map-policy both \
  --warmup 5 --iterations 50 --output htp-map-decomposition.csv
```

`--sizes-mib` 保留给通用诊断，与 `--slot-counts` 互斥。CSV 的 Host 时间只覆盖
`fastrpc_mmap/munmap`，DSP 时间只覆盖 `HAP_mmap2/HAP_munmap2`；为 DSP probe 建立的最终
Host registration 不进入 Host 平均值。

完整容量搜索会制造明显内存压力，只有在 smoke 通过后才可显式添加
`--confirm-capacity-test`。运行结果默认写到 `MOE/results/buffer-capacity/`。
