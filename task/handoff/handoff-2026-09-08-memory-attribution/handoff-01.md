# 模型缓存页与隐藏容量核实

## 结论

fd8657d6 / K80 实机已证实模型文件缓存存在跨 memcg 归属，不再是推测。正式复核使用 MADV_RANDOM 的独立只读映射：mincore 选取已驻留页，触碰建立PTE，再用 /proc/self/pagemap → PFN → /proc/kpagecgroup 读取实际owner inode，用 /dev/memcg 目录inode定位。未迁移原有页charge，未清page cache，未复制模型，未改推理逻辑或系统策略。

| 正式复核阶段 | 模型驻留缓存 | 其中记在根组、不在实验组 |
|---|---:|---:|
| 加载前 | 2090.59 MiB（2.0416 GiB） | 2090.59 MiB，全部根组 |
| OpenCL ready、请求前 | 1882.39 MiB | 196.76 MiB |
| OpenCL结束后、删除实验组前 | 1882.39 MiB | 196.76 MiB |
| HTP ready、请求前 | 2111.12 MiB | 57.32 MiB |
| HTP结束后、删除实验组前 | 2028.95 MiB | 55.86 MiB |

正式根组inode=1，实验组5605，观察者组5649；观察者在正式五次扫描中均没有模型页归属。每次major_faults=0、PFN被屏蔽数=0、非present竞态数=0，扫描前后mincore驻留数相同，owner字节和等于起始驻留字节。观察者页表等内存约4MiB增量不冒充模型页。不是整个加载过程的逐时峰值追踪，页cache可回收且跨阶段会变化；不能把模型文件大小当实际驻留，也不能把root归属视为未计入全系统RAM。

初轮未设置MADV_RANDOM的数据保留在 evidence/ 根目录：加载前root约1995.75MiB，GPU ready root255.42MiB。初轮部分扫描发生1/9次major fault并触发预读，HTP观察者页charge最高约269MiB（非同时峰值）；该轮是方法诊断，不把它当低扰动正式测量。随后仅对观察者映射设置MADV_RANDOM重做，正式证据在 evidence/random/。GPU首轮与正式root数不同，说明缓存归属量不是稳定常数；不把该差异归因为单一因素。

## 可见隐藏容量

- 正式GPU pid4699的KGSL kernel_max=8653164544 bytes（8252.30MiB），账本全程peak=8619413504 bytes（8220.11MiB），差32.1875MiB。KGSL live kernel=8520040448、mapped=8519843840、unmapped=196608、imported_mem=0。不同计量口径/峰值时刻，仅作粗略容量对照，不能称为全部未记账物理内存的精确值。
- HTP进程fdinfo按inode去重得到17个DMA-BUF、总capacity=7872421888 bytes（7507.73MiB）；账本peak=7871344640（7506.70MiB），差1.02734MiB。此差同样不是所有DSP PD/驱动私有页总量，不能把两者相加。
- DSP已知另有512KiB DDR scratch；主栈32KiB、work栈16KiB×(n_hvx−1)、HMX栈16KiB及队列/描述符统一块。源码main.c:369-413、:552-554及work/dma/hmx-queue.c可核验。VTCM单列，不当DDR叠加。未获取完整DSP私有heap/PD物理页统计；未证明R256MiB是严格上界。

本轮没有看到可见GPU/RPCMEM容量之外的GiB级差额。驱动/DSP杂项与模型文件cache应分开估计。新事实使E0B的主要覆盖缺口定位到模型cache归属；尚不能仅凭当前私有cgroup+extra证明全部LLM RAM在4GiB+5%以内。正式运行是非约束预算的observe诊断，不是4GiB下实际越界复现。文件cache在推理时可能已无CPU映射但仍可回收，是否计入实验的LLM RAM需保持口径一致。

## 实施、证据与复现

新增 tests/model-page-charge.cpp、tests/run-memory-attribution.py；后者复用旧guarded与安全阈值，只新增健康状态内存诊断，不把计时当性能结果。实际执行两轮：

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ tests/model-page-charge.cpp -o build-android-opencl-hexagon/bin/model-page-charge
python3 tests/run-memory-attribution.py
# 添加MADV_RANDOM与默认random子目录后，重编译并执行同一命令。
```

再次复现须用新 `--tag <字母数字标签>`；脚本拒绝覆盖已有目录。模型仍为 /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf，文件8149865792 bytes。两轮固定128/32、无prefix缓存、全层-ngl99；正式GPU/HTP请求均完成且32token完全一致，无安全截断。先GPU再HTP，复用同一实验组，不能当独立启动缓存状态比较。产物push后逐一验证SHA256，OpenCL修复库保持70bac8ab6b0a692507e180e04ece6c9dc5e3e1d4bca4901416b832cbb666c20f。

证据包括cache-before/*-cache-live/*-cache-after.json（owner路径+原始数）、*-live-counters.txt.gz（cgroup.stat、KGSL、smaps_rollup、fdinfo、meminfo）、*-server.log.gz、*-request.json、hashes.json。主线程已用jq核验正式5次页面一致性及2个128/32请求token一致。128/32计时因插入观测，不进入性能表。

设备隔离目录 /data/local/tmp/llama-memory-attribution、/dev/memcg/llama-memory-attribution、/dev/memcg/llama-memory-observer 已清理，无server/probe进程；删除本次cgroup会正常把其剩余cache charge归并到父组，未改变用户既有组策略。模型和原部署保留。Git起点ff86522d5，本任务独立提交，不包含用户AGENTS.md/Exp1修改；git log -1可取最终检查点。E0B最终简报已追加本次实测事实，整体状态不虚改为PASS。
