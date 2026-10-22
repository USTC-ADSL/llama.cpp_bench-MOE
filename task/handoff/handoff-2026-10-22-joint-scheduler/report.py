#!/usr/bin/env python3
"""Render the completed experiment without filtering timing samples."""
import csv
import json
import statistics as st
import numpy as np
from run import TASK,OUT

def main():
    s=json.loads((TASK/'summary.json').read_text());assert s['complete']
    comparisons=s['comparisons'];runs=s['runs']
    lines=['# 长度感知 shared/routed 联合调度：128–1K 设备验证','',
        '设备 Redmi K80 Pro（fd8657d6），Qwen1.5-MoE-A2.7B-Chat Q4_0；Android 实测，2K 未运行。',
        '计时范围为权重预放置后的单chunk完整prefill，包括调度、dense、MoE与最终logits；不包含冷启动权重准备，也不是完整多token生成时间。',
        '每长度、每输入、每方案两进程，每进程两次 warmup 与六次 timed；48 进程、288 timed requests，所有样本保留。进程间冷却30秒，正反顺序平衡。延迟为两个进程中位数的平均。',
        '','## 完整模型结果','',
        '吞吐收益 = T_NPU / T_方案 − 1；不是延迟下降百分比。纯 NPU 指原生 HTP0 后端及其普通 CPU fallback。旧/新调度都使用单份 CPU4x8 权重、同一新版 binary 和未改变的 host/DSP 库。',
        '', '| tokens | 输入 | NPU ms | 旧调度 ms | 新调度 ms | 旧/NPU 吞吐收益 | 新/NPU 吞吐收益 | 新/旧吞吐收益 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    csvrows=[]
    for c in comparisons:
        n,o,j=[c['modes'][m]['mean_process_median_ms'] for m in ('npu','old','joint')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {n:.3f} | {o:.3f} | {j:.3f} | {c['old_gain_percent']:.2f}% | {c['joint_gain_percent']:.2f}% | {c['joint_over_old_percent']:.2f}% |")
        csvrows.append({'tokens':c['tokens'],'prompt':c['prompt'],'npu_ms':n,'old_ms':o,'joint_ms':j,
            'old_gain_percent':c['old_gain_percent'],'joint_gain_percent':c['joint_gain_percent'],
            'joint_over_old_percent':c['joint_over_old_percent'],'joint_latency_reduction_vs_npu_percent':c['joint_latency_reduction_vs_npu_percent']})
    with (TASK/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows)
    lines += ['', '## 128 的收益来源','',
        '以下对两份文本的所有计时请求、前23个完整token层取平均，排除最后N=1层。分支时间为 gather + compute；三个后端并行，不能将三列相加作为MoE墙钟时间。',
        '', '| 方案 | CPU ms/层 | NPU ms/层 | GPU ms/层 | CPU routed pairs | NPU routed pairs | GPU routed pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    branch_rows=[]
    for mode in ('old','joint'):
        layers=[]
        for tag in runs:
            if '-128-'+mode+'-' in tag:
                d=json.loads((OUT/(tag+'.json')).read_text())
                layers.extend(l for sample in d['samples'] for l in sample['layers'][:23])
        branch=[st.mean(l['backend_phases_us'][b][1]+l['backend_gather_us'][b] for l in layers)/1000 for b in range(3)]
        fractions=[sum(l['pairs'][b] for l in layers)/(len(layers)*512) for b in range(3)]
        lines.append(f"| {mode} | "+' | '.join(f'{v:.3f}' for v in branch)+' | '+' | '.join(f'{v*100:.2f}%' for v in fractions)+' |')
        branch_rows.append({'mode':mode,'branch_ms':branch,'routed_pair_fraction':fractions})
    lines += ['', '128 时新调度在前23层将 shared 分配到 GPU，减少其 routed 工作并将更多 routed pairs 交给 NPU；CPU/NPU/GPU分支时间更接近。这个对照同时改变长度代价、shared候选和context校正，不能把全部收益单独归因于某一项；未做这三项的完整消融。']
    (TASK/'branch128-summary.json').write_text(json.dumps(branch_rows,indent=2)+'\n')
    lines += ['', '## 实现与训练边界','',
        '1. 每个 MoE 层拿到真实 route 后统计60个专家的 token 数；以当前 nt、有效专家数和32-token tile 数预测各端工作量。没有预存验证请求的专家 ID 分配。',
        '2. shared 在 CPU/NPU/GPU 三端都可选；每个 shared 候选都重新平衡 routed 负载，优化 max(T_CPU,T_NPU,T_GPU)，使用 greedy + 有限 moves/swaps，总预算2048。它是有界启发式，不保证全局最优。',
        '3. 为 shared 增加长度/tile 特征支持；本轮整32长度训练只拟合 shared constant + nt，避免两项共线。原1K模型将 shared 视为常数的外推错误被修正。',
        '4. FFN训练输入来自独立 technical-1024 捕获的 token 前缀，并非重新完整执行四个长度的 capture。层0/12训练、6/18验证；192候选都通过逐token4%检查，8组留出层排序均选中已测候选最快者。',
        '   本次 profiling 对齐实际执行的 indexed HTP fused gather、NEON reduce、activation lifetime reuse 与 overlap；旧profiling使用host gather，存在执行路径不一致。分支模型目标为 gather + backend compute；其他host开销仍靠端到端实测检验。',
        '5. 完整 technical 输入额外运行四个长度，以全部层/请求的 observed/predicted 中位数校正每端倍率；运行时按 nt 线性插值。CPU倍率为1.200/1.012/0.970/1.439，取代固定1.5惩罚。此校正是离线训练，不是在线反馈；区间外只夹取端点，不代表已验证泛化。',
        '6. 候选在 wetland/foundry 执行前冻结；验证后未改模型。最后 N=1 层维持全CPU。192候选与technical校准用于训练/开发，wetland/foundry是本轮未用于拟合的完整模型验证文本（在此前实验中用过，并非全新语料）。',
        '', '## 开销与单份权重','',
        '新旧单份权重路径的 unique payload 与 unique allocation 均为 **7,474,348,032 B**。三端 alias 指向相同 CPU4x8 数据，未增加三份权重，也未改计算 kernel。',
        '开放额外 shared 后端增加激活 arena 的最坏情况容量；不能声称总内存零开销。下表是预分配 arena capacity 总和，不是物理驻留量或整进程峰值。',
        '', '| tokens | 旧 arena MiB | 新 arena MiB | 增量 MiB | 旧决策 ms/请求 | 新决策 ms/请求 |',
        '|---:|---:|---:|---:|---:|---:|']
    memory=[]
    for nt in (128,256,512,1024):
        modes={m:[r for t,r in runs.items() if f'-{nt}-{m}-' in t] for m in ('old','joint')}
        arena={m:sum(rs[0]['arena_capacity_bytes'])/2**20 for m,rs in modes.items()}
        decision={m:st.mean(v for r in rs for v in r['stages_ms']['decision']) for m,rs in modes.items()}
        memory.append({'tokens':nt,'old_mib':arena['old'],'joint_mib':arena['joint'],'delta_mib':arena['joint']-arena['old'],'decision_ms':decision})
        lines.append(f"| {nt} | {arena['old']:.3f} | {arena['joint']:.3f} | {arena['joint']-arena['old']:.3f} | {decision['old']:.3f} | {decision['joint']:.3f} |")
    lines += ['', '决策时间已计入端到端延迟（也包含在 gather 统计内，不能重复相加）。CPU expert 仍为4线程，未固定到小核；本轮没有验证用户最终1–2小核目标。',
        '', '## 数值与执行状态','']
    for c in comparisons:
        q=c['logits_vs_npu']['joint'];major=c['modes']['joint']['major_faults_range']
        lines.append(f"- {c['tokens']} / {c['prompt']}：新调度 vs NPU logits NRMSE {q['nrmse']*100:.3f}%，top1相同={q['top1_equal']}，top10重合{q['top10_overlap']}/10；新调度每请求major faults范围{major}。")
    lines += ['', '同模式/输入/长度的跨进程 logits 逐位一致，重复请求 ownership/route 一致，warmup 后 allocation 不增长。跨后端不是逐位一致；logits检查和top1不等于perplexity、生成质量或任务准确率验证。完整模型 logits NRMSE 与独立 FFN 逐token4%阈值是不同指标。',
        '', '## 证据与回退','',
        '- 原始计时、层级数据、数值和哈希：evidence/、summary.json、validation-runs.json、validation-frozen.json。',
        '- 独立训练/校准：fit-report.json、calibration-report.json、runtime-joint.json、runtime-joint-context.json。',
        '- 原 binary、host/DSP库和旧模型：evidence/backup；校准版本：evidence/freeze；正式版本：evidence/freeze-validation。',
        '- 旧 runtime-fast-cpu150 与 shared仅NPU的 plan 保留，默认配置未被覆盖。当前分支新增候选可显式选择。',
        '- 2K、训练区间之外长度、未见新语料、持续多请求温度变化、1–2小核和生成质量尚需独立验证。每条件只有两进程，不作为跨设备或所有负载的统计保证。本轮没有根据留出结果再调参。', '']
    (TASK/'results.md').write_text('\n'.join(lines))
    (TASK/'memory-summary.json').write_text(json.dumps(memory,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,prompt in zip(axes,('wetland','foundry')):
        rows=[c for c in comparisons if c['prompt']==prompt];x=np.arange(len(rows))
        ax.bar(x-.18,[r['old_gain_percent'] for r in rows],.36,label='Previous scheduler',color='#90a4ae')
        ax.bar(x+.18,[r['joint_gain_percent'] for r in rows],.36,label='Length-aware joint scheduler',color='#167e92')
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[str(r['tokens']) for r in rows]);ax.set_xlabel('Prefill tokens');ax.set_title(prompt)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Throughput gain over native NPU (%)');axes[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(TASK/'throughput-gain.svg');fig.savefig(TASK/'throughput-gain.png',dpi=180)
    svg=TASK/'throughput-gain.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
if __name__=='__main__':main()
