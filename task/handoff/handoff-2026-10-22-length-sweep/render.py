#!/usr/bin/env python3
"""Export the completed experiment as a table and a standalone scientific plot."""
import csv
import json
import statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TASK=Path(__file__).resolve().parent

def main():
    d=json.loads((TASK/'summary.json').read_text());assert d['complete']
    rows=d['comparisons']
    with (TASK/'comparison.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['tokens','prompt','npu_ms','hetero_ms','npu_tokens_s','hetero_tokens_s','throughput_gain_percent','pooled_median_gain_percent','logits_nrmse','top1_equal','top10_overlap'])
        for c in rows:
            n,h=[c['modes'][m] for m in ('npu','hetero')];q=c['cross_backend_logits']
            w.writerow([c['tokens'],c['prompt'],n['mean_process_median_ms'],h['mean_process_median_ms'],n['tokens_per_second'],h['tokens_per_second'],c['throughput_gain_percent'],c['pooled_median_gain_percent'],q['nrmse'],q['top1_equal'],q['top10_overlap']])
    fig,ax=plt.subplots(figsize=(7.1,3.9),layout='constrained')
    for prompt,color,marker in [('wetland','#1874b4','o'),('foundry','#d27820','s')]:
        data=[c for c in rows if c['prompt']==prompt]
        ax.plot([c['tokens'] for c in data],[c['throughput_gain_percent'] for c in data],marker=marker,color=color,label=prompt,linewidth=1.8)
    ax.axhline(0,color='#555555',linewidth=.8,linestyle='--');ax.set_xscale('log',base=2)
    ax.set_xticks([128,256,512,1024,2048],labels=['128','256','512','1024','2048'])
    ax.set_xlabel('Prefill input length (tokens)');ax.set_ylabel('Throughput gain over native NPU (%)')
    ax.set_title('Frozen cost model + CPU4x8 indexed gather, Redmi K80 Pro',fontsize=10)
    if any(c['tokens']==2048 and c['modes']['hetero']['major_faults_range'][1]>1000 for c in rows):
        ax.annotate('Heavy paging at 2048',xy=(2048,min(c['throughput_gain_percent'] for c in rows if c['tokens']==2048)),
            xytext=(570,-52),fontsize=9,arrowprops={'arrowstyle':'->','color':'#555555'})
    ax.grid(axis='y',alpha=.25);ax.legend(frameon=False)
    for ext in ('png','svg'):fig.savefig(TASK/f'throughput-gain.{ext}',dpi=180)
    svg=TASK/'throughput-gain.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)
    lines=['# 固定代价模型：128–2048 tokens 相对原生 NPU','',
        '设备fd8657d6 / Redmi K80 Pro；模型Qwen1.5-MoE-A2.7B Q4_0；执行代码基线5b32b8743。代价模型runtime-fast-cpu150保持冻结，异构启用单份CPU4x8权重和indexed DMA gather。','',
        '## 完整prefill性能','',
        '主统计先取每进程6样本的中位数，再平均同条件两个进程；吞吐提升=NPU耗时/异构耗时−1。所有240个计时样本保留。','',
        '| tokens | 输入 | NPU ms | 异构 ms | NPU token/s | 异构 token/s | 吞吐提升 | 总体中位数口径提升 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    for c in rows:
        n,h=[c['modes'][m] for m in ('npu','hetero')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {n['mean_process_median_ms']:.3f} | {h['mean_process_median_ms']:.3f} | {n['tokens_per_second']:.2f} | {h['tokens_per_second']:.2f} | {c['throughput_gain_percent']:+.2f}% | {c['pooled_median_gain_percent']:+.2f}% |")
    lines += ['', '![Throughput gain](throughput-gain.png)', '', '## 数值与运行状态','',
        '同输入、同长度、同后端配置的两个进程最终logits逐位一致。跨NPU/异构存在布局、算子和精度路径差异，下面是最终logits的有限数值核验，不代表完整生成质量评估。','',
        '| tokens | 输入 | logits NRMSE | cosine | top1一致 | top10重合 | NPU major faults范围 | 异构 major faults范围 |',
        '|---:|---|---:|---:|---|---:|---|---|']
    for c in rows:
        q=c['cross_backend_logits'];n,h=[c['modes'][m] for m in ('npu','hetero')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {100*q['nrmse']:.3f}% | {q['cosine']:.6f} | {q['top1_equal']} | {q['top10_overlap']}/10 | {n['major_faults_range']} | {h['major_faults_range']} |")
    if any(c['modes']['hetero']['major_faults_range'][1]>1000 for c in rows):
        lines += ['', '**分页压力：**存在异构计时请求超过1000次major faults；对应结果全部保留。这些长度的实测包含操作系统分页代价，不能解读成低缺页常驻条件下的纯算力或调度效率。首个2048异构进程的只读内存快照在evidence/n2048-wetland-hetero-1-memory-observation.json；该进程运行中做过额外只读状态查询。']
    lines += ['', '## 调度与内存','',
        '调度使用每层当前真实router IDs统计负载，进行整专家分配。没有重放预存route、重新拟合代价模型或按长度挑选策略。前23层routed专家三端均可用、shared固定NPU；最后层单token专家固定CPU。两输入使用相同availability。下表份额仅统计前23层routed token-expert pairs，不包括shared专家；CPU/NPU/GPU顺序固定。','',
        '| tokens | 输入 | Routed pair份额 CPU/NPU/GPU | 调度时间中位数 ms/请求 | activation arena MiB |',
        '|---:|---|---|---:|---:|']
    for c in rows:
        runs=[r for tag,r in d['runs'].items() if tag.startswith(f"n{c['tokens']}-{c['prompt']}-hetero-")]
        pairs=[sum(l[b] for l in runs[0]['pairs_by_layer'][:23]) for b in range(3)]
        shares='/'.join(f'{100*x/sum(pairs):.2f}%' for x in pairs)
        dec=st.median(x for r in runs for x in r['stages_ms']['decision'])
        arena=sum(runs[0]['arena_capacity_bytes'])/2**20
        lines.append(f"| {c['tokens']} | {c['prompt']} | {shares} | {dec:.3f} | {arena:.3f} |")
    lines += ['', '所有异构进程唯一权重payload/allocation均为7,474,348,032 B。arena是预留激活容量，不是总进程RSS。较长输入会增加KV和activation容量；不能由权重共享推断总内存固定。','',
        '## 协议、边界与复现','',
        '- 五档长度各两个输入，每输入4进程：wetland NPU/hetero/hetero/NPU，foundry反向；每进程2warmup+6timed，间隔30秒。',
        '- 两路径n_ctx=n_batch=n_ubatch=nt，单chunk，F16 KV、CPU outer4线程，异构expert4线程。前nt tokens逐项相同，跨长度严格前缀。',
        '- 计时包含完整prefill、attention/KV/router/专家、同步及最后token logits；加载、布局准备和计时前KV清空排除。没有decode。',
        '- 原生NPU为mode=npu，日志均核验25/25层offload到HTP0，保留框架CPU fallback。异构为mode=custom，共享权重、动态分配与输入融合。这是完整方案对比，不是只改变owner的因果消融。',
        '- 全程固定并核验binary/library/代价模型/输入hashes；GGUF核验路径、大小、mtime，未对8GB模型重新计算全文件hash。设备boot ID逐进程核验；温度、缺页、全部样本在run.json和报告内。没有调频/affinity设置。',
        '- 本实验考察同设备同模型下输入长度和两个文本的外推；不证明跨设备、跨模型或长时间热稳态泛化，也没有验证1–2小核约束。',
        '- 复算：`python3 task/handoff/handoff-2026-10-22-length-sweep/analyze.py`，导出：同目录`render.py`。`run.py`拒绝覆盖已有cohort；重跑应使用新目录/tag。',
        '- 原始结果、冻结binary/库/输入/代价模型及可用性计划归档见archive-record.json；CSV见comparison.csv，精确统计见summary.json。',
        '- 用户提出的历史128约40%收益核对，以及扫描结束后独立执行的原始输入四路径对照，见[128收益差异分析](attribution128.md)。补充对照不混入上述冻结代价模型长度扫描。','']
    (TASK/'results.md').write_text('\n'.join(lines))

if __name__=='__main__':main()
