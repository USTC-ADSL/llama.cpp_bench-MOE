# E0B 文件级缓存准备与工程收尾

## 最终结论

用户授权验证“仅回收目标GGUF缓存 → 进程先入私有cgroup → 加载/请求 → 归属复核”，通过后尝试收尾。六项全部通过，E0保留PASS，E0B最终报告改为限定流程的工程PASS：接受估计R和约5%误差，不声称严格逐物理页上界或所有后端路径完整覆盖。两份短报告仍并列于 docs/Motivation_Exp/E0_E0B_final/。

## 变更与安全边界

- 新 tests/model-cache-prepare.cpp：必须显式--evict，仅接受固定模型绝对路径及8149865792 bytes大小；只读打开文件，扫描其他进程maps/FD模型持有者，发现持有者或不可读proc条目即拒绝。mincore检查前后驻留，POSIX_FADV_DONTNEED只针对该文件；无文件内容写入、无模型副本、无drop_caches、无swap/SELinux/LMKD修改。回收残留>16MiB、fadvise错误或元数据变化均返回失败。本轮六次holders=0、unreadable=0、回收后驻留=0、inode84678及size/mtime不变。
- 新 tests/run-cache-closure.py：默认dry-run，--apply执行；每个case独立子cgroup，加载前进入；复用前轮MADV_RANDOM页归属探针、原guarded及安全监测；先保存证据后判断PASS，拒绝覆盖旧tag。模型准备在case组创建前，模型加载在进入组之后。其他程序不得在检查与运行之间打开模型，属于该流程的独占条件。
- 无runtime改动，OpenCL仍为252aa4396修复产物，hash见evidence/final/hashes.json。用户已有AGENTS.md/Exp1修改不纳入提交。

## 实测

| Case | 运行结果 | ready / after模型组外页 |
|---|---|---|
| cpu（off，宽松） | 128/32、cache_n0、exit0 | 0 / 0 |
| gpu-observe（宽松） | 25/25层、128/32、exit0 | 0 / 0 |
| gpu-enforce（宽松） | 同上 | 0 / 0 |
| htp-enforce（宽松） | 25/25层、128/32、exit0 | 0 / 0 |
| gpu-4g | budget_denied、exit1、cgroup failcnt0 | 无ready / 0 |
| htp-4g | 同上 | 无ready / 0 |

四个成功请求的32token逐个相同。十次页面扫描均PFN可读、无nonpresent竞态、majorfault0、扫描前后驻留相同；所有被检查的页仅归属对应case inode（5781/5825/5869/5913/5957/6001），不是观察者5737、父组或root。ready实际缓存CPU2530770944、GPU observe2039431168、GPU enforce1774886912、HTP2447884288 bytes。六次准备前缓存约6.45MiB～2.40GiB，准备后全部0；没有把fadvise返回0当作唯一成功依据。

成功请求charged12GiB、extra64GiB为非约束正确性/归属对照；CPU组峰值10432110592 bytes，故**没有声称CPU本轮在4GiB成功推理**。有限组charged768MiB+extra3072MiB+R_est256MiB=4GiB。GPU拒绝时used2348MiB/request1018.5546875MiB，HTP2075.80859375/1018.55078125MiB；两组cgroup峰值193617920/82825216 bytes，failcnt均0。预期拒绝是预算机制PASS，不是模型推理成功。

R_est256MiB保留为工程余量。全层GPU KGSL kernel_max8653164544与账本peak8619413504差32.1875MiB；HTP去重17个DMA-BUF capacity7872421888与账本peak7871344640差1.02734375MiB，均复核与前轮一致。加上已知DSP512KiB scratch、栈队列等作估计，不将不同口径峰值差等同全部隐藏物理内存。原先GiB级组外文件缓存已通过准备及实机归属检查消除；不要求为工程PASS补齐每个驱动私有字节。独立image、其他形状/模型/设备/并发、未清理热缓存换组、严格物理上界仍不在PASS范围。

## 命令、证据与后续

```sh
/home/miog/pzw/download/pzw/HeteroCompute/android-ndk-r27d/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android34-clang++ -std=c++17 -O2 -static-libstdc++ tests/model-cache-prepare.cpp -o build-android-opencl-hexagon/bin/model-cache-prepare
python3 tests/run-cache-closure.py --apply
python3 -m py_compile tests/run-cache-closure.py
```

初次编译使用`struct stat final`遇到C++解析问题，改变量名end_stat后成功；失败产物未部署。其余产物复用有效构建，没有全量重建。实际模型/库命令完整保存在各*-server.log.gz开头。未来重跑使用新--tag字母数字标签，且须满足独占模型的准备前提。

主线程最终用jq核验results.json六项有效、十次组外页0/majorfault0/稳定、四请求token完全一致；六份*-prepare.json全部残留0且metadata_unchanged。检查日志实际25/25层，检查新源码/报告diff、Python语法及git diff --check。所有测试无安全截断；观测过程插入扫描，因此不把本轮时序当性能/开销对照。

设备隔离目录/data/local/tmp/llama-cache-closure、父/子/dev/memcg/llama-cache-closure及观察组/dev/memcg/llama-cache-observer已清理；核对无server/prepare/probe进程、模型仍在。只清理本任务目录，旧部署保留。删除本次cgroup会正常归并其残余cache charge，下一次仍须重新执行准备。

Git起点3851cfaad，本任务独立提交，最终ID用git log -1查看；不push。报告交付及工程收尾已完成。不能直接复用旧run-e0-final.py的热缓存启动方式冒充本流程。若后续做性能对照，各后端统一准备，在同一cgroup内预热并完成perf测量，内存诊断与perf分开。
