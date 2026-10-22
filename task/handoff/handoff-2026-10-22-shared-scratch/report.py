#!/usr/bin/env python3
"""Render all paired storage comparisons, including regressions."""
import csv
import json
import statistics as st
from run import TASK


def main():
    s = json.loads((TASK / 'summary.json').read_text())
    assert s['complete']
    lines = ['# Shared expert 单份激活内存：Android 验证', '',
        '设备 Redmi K80 Pro（fd8657d6），Qwen1.5-MoE-A2.7B-Chat Q4_0。仅改变激活存储；冻结上轮 joint 调度模型、权重与计算 kernel，运行时仍按实际route决策，并逐项校验两种存储方式的route和分配结果一致。2K 暂缓。',
        '每个长度、每份文本、每种存储方式运行两个独立进程，每进程2次预热、6次计时；共32进程、192个计时请求，进程间冷却30秒，采用ABBA/BAAB顺序。保留全部样本。',
        '计时覆盖权重预放置后的单chunk完整prefill，包括dense、MoE、调度和最终logits，不包含冷启动加载。下表延迟为两进程中位数的平均；吞吐变化=T_private/T_common−1。', '',
        '## 完整模型性能', '',
        '| tokens | 输入 | private ms | common ms | 延迟变化 | 吞吐变化 |',
        '|---:|---|---:|---:|---:|---:|']
    rows = []
    for c in s['comparisons']:
        a, b = [c['modes'][m]['mean_process_median_ms'] for m in ('private', 'common')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {a:.3f} | {b:.3f} | {c['latency_change_percent']:+.2f}% | {c['throughput_change_percent']:+.2f}% |")
        rows.append(dict(tokens=c['tokens'], prompt=c['prompt'], private_ms=a, common_ms=b,
                         latency_change_percent=c['latency_change_percent'], throughput_change_percent=c['throughput_change_percent'], saved_bytes=c['saved_bytes']))
    with (TASK / 'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    lines += ['', '## 分阶段核验', '',
        '以下为每请求计时的算术均值，区别于上表的进程中位数。outside 是 total−expert_wall，包含 dense、callback 外围、分派前准备、归约等，不能将它单独解释为 dense 算子时间。表中gather统计包含分派前的决策与job准备；分支内部实际输入gather已包含在expert_wall中。后端并行时间不能相加。', '',
        '| tokens | 输入 | MoE private/common ms | outside private/common ms | gather private/common ms | combine private/common ms |',
        '|---:|---|---:|---:|---:|---:|']
    for c in s['comparisons']:
        modes = {m: [r for tag, r in s['runs'].items() if tag.startswith(f"{c['prompt']}-{c['tokens']}-{m}-")] for m in ('private', 'common')}
        cols = []
        for stage in ('expert_wall', 'outside_expert_wall', 'gather', 'combine'):
            cols.append(' / '.join(f"{st.mean(v for r in modes[m] for v in r['stages_ms'][stage]):.3f}" for m in modes))
        lines.append(f"| {c['tokens']} | {c['prompt']} | " + ' | '.join(cols) + ' |')
    lines += ['', '128/wetland进程级诊断（按实际执行顺序）：', '',
        '| 进程 | 延迟中位数 ms | MoE均值 ms | outside均值 ms | DSP cycles/us |',
        '|---|---:|---:|---:|---:|']
    for tag in ('wetland-128-private-0', 'wetland-128-common-1', 'wetland-128-common-2', 'wetland-128-private-3'):
        r = s['runs'][tag]
        lines.append(f"| {tag} | {r['median_ms']:.3f} | {st.mean(r['stages_ms']['expert_wall']):.3f} | {st.mean(r['stages_ms']['outside_expert_wall']):.3f} | {r['dsp_cycles_per_us']:.1f} |")
    lines += ['', '首个private进程的DSP cycles/time与其余进程不同，是执行频率状态变化的线索；该值为批次计数器之比，不是直接采样的全程硬件频率。两种存储方式未锁定频率，outside也有较大波动，因此无法将该组全部端到端差异归因于scratch。没有删除该进程或据此修正收益。']
    lines += ['', '八组MoE墙钟均值的变化均在约±0.5%内；本轮支持“节省激活分配容量且未观察到明显MoE执行损失”，不支持“固定的端到端加速”。common计时请求major faults均为0；private在512/foundry、1024/wetland、1024/foundry的每请求最大major faults分别为1、1、37，其余组为0。尤其不能将1024/foundry的端到端改善完全归因于scratch；原始计时和缺页样本均保留。']
    lines += ['', '## 激活分配容量', '',
        '统计CPU/NPU/GPU三个私有arena，加一份common backing；三端import view不重复计算。这里是分配容量，不是整进程峰值RSS或全系统实际驻留量。', '',
        '| tokens | private MiB | common MiB | 节省 MiB | 节省比例 |', '|---:|---:|---:|---:|---:|']
    for c in s['comparisons'][::2]:
        a, b = [c['modes'][m]['capacity_bytes'] / 2**20 for m in ('private', 'common')]
        lines.append(f"| {c['tokens']} | {a:.3f} | {b:.3f} | {a-b:.3f} | {100*(1-b/a):.2f}% |")
    lines += ['',
        '专家权重unique payload/allocation均保持7,474,348,032 B；没有复制三份权重。原三份shared scratch预算变为一份；NPU indexed full_input在shared由别端执行时仍保留在NPU私有区，以避免并发覆盖。common相对更早的仅NPU可执行shared配置仍多2/3/5/9 MiB，包含NPU输入预算与common padding。',
        '', '## 实现及安全边界', '',
        '1. HTP分配DMA backing，导出描述符；CPU使用host pointer view，GPU通过OpenCL DMA import创建view，NPU直接使用backing。三种后端对象对应同一份物理存储。所有view在dispatch前建立。',
        '2. 图中shared expert的root tensors单独进行生命周期布局；routed roots留在每端私有arena。shared每层只有一个owner，可跨层轮换。并发routed不能同时写同一地址。',
        '3. backend compute同步、分支join、host归约完成后，下一层才复用common scratch。若未来同一层多端分割shared、多个请求并发或层流水重叠，必须重新划分区域或增加同步，不能直接套用当前独占契约。',
        '4. 保持private为默认回退路径，通过MOE_SHARED_SCRATCH=common或runner的--shared-scratch common启用；计算kernel和库未修改。旧binary/源码备份见backup.json和归档。',
        '', '## 验证和限制', '',
        '- 定向probe覆盖9种长度（1/2/17/32/33/128/256/512/1024）、4种routed拓扑、3种shared owner、3轮变化输入，共324例。common/private输出逐位一致，三端view的host base一致，routed与common地址分离；warmup后不再分配job。',
        '- 32个完整模型进程的最终logits、各轮logits、实际route和shared/routed分配均一致；common owner、布局容量、权重总量及binary/model哈希检查通过。每次请求的缺页及进程内存字段保存在summary.json。',
        '- 单物理backing的结论来自分配/import实现与别名断言，并非全系统物理页追踪。/proc VmRSS/VmHWM不充分反映DMA驻留，不能用arena节省量直接宣称整进程峰值减少同样数值。',
        '- 本轮只对比同一joint调度器的两种存储方式，未重测纯NPU，不能把历史NPU数据拼接为同期收益。CPU expert仍为4线程且未限定小核，尚未验证1–2小核目标。',
        '- 本轮不把端到端波动等同于算子变化，亦未建立各算子≤5%的独立置信区间；完整模型只有两份输入，每方案两个独立进程。',
        '- routed arena仍按各端最坏4N routed pairs预留；尚未消除这部分保守预算。top-4 gather与shared重复预算是不同来源，本轮处理后者。', '']
    (TASK / 'results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
