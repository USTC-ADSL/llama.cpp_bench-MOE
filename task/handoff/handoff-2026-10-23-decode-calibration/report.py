#!/usr/bin/env python3
"""Render the frozen N=1 study; no NPU performance baseline or hidden filtering."""
import csv
import json
import statistics as st
from run import TASK


def main():
    s=json.loads((TASK/'summary.json').read_text());assert s['complete']
    development=json.loads((TASK/'development-validation.json').read_text())
    model=json.loads((TASK/'runtime-decode-table.json').read_text())
    simple=json.loads((TASK/'runtime-decode-ffn.json').read_text())
    wins=sum(c['throughput_gain_percent']>0 for c in s['comparisons'])
    lines=['# N=1 标定后 decode 对照','',
        f'四组汇总中 {wins} 组相对原生 CPU 获得正吞吐变化；逐进程波动见下表，正负百分点不是统计显著性结论。',
        '设备 fd8657d6 / Redmi K80 Pro，Qwen1.5-MoE-A2.7B-Chat Q4_0；纯 CPU 与采用 N=1 数量表的当前调度。单份 CPU4x8 权重，common scratch，CPU 均四线程，默认频率/affinity。2K 未测。',
        '128/1024 为 KV 前缀长度，decode 每步 N=1。每方案每输入两进程，每进程两轮 warmup、四轮计时，每轮32步 teacher-forced decode；共16进程、2048计时token。相同teacher输入，不回放expert IDs。',
        '计时包含完整 llama_decode + synchronize；加载、prefill重建和logits保存排除。wetland按CPU/current/current/CPU，foundry反序，进程间20秒冷却；所有慢样本和缺页样本保留。',
        '', '## 端到端结果','',
        '每进程取32步总时间中位数除以32，再平均两个进程。吞吐变化=T_CPU/T_current−1。',
        '', '| 前缀 | 输入 | CPU ms/token | 当前 ms/token | CPU token/s | 当前 token/s | 吞吐变化 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    rows=[]
    for c in s['comparisons']:
        cpu,current=[c['modes'][m]['ms_per_token'] for m in ('cpu','custom')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {cpu:.3f} | {current:.3f} | {1000/cpu:.2f} | {1000/current:.2f} | {c['throughput_gain_percent']:+.2f}% |")
        rows.append(dict(prefix=c['tokens'],prompt=c['prompt'],cpu_ms_per_token=cpu,current_ms_per_token=current,
            cpu_tokens_s=1000/cpu,current_tokens_s=1000/current,throughput_gain_percent=c['throughput_gain_percent']))
    with (TASK/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines+=['','| 前缀 | 输入 | CPU 两进程 ms/token | 当前两进程 ms/token |','|---:|---|---|---|']
    for c in s['comparisons']:
        lines.append(f"| {c['tokens']} | {c['prompt']} | "+' | '.join(' / '.join(f'{v:.3f}' for v in c['modes'][m]['process_ms_per_token']) for m in ('cpu','custom'))+' |')
    lines+=['','## 当前方案分配和开销','',
        '顺序CPU/NPU/GPU。阶段按24层求和，再对计时token取算术平均；decision已包含在prepare，branch可能并发，不能相加当作总墙钟。',
        '', '| 前缀/输入 | routed份额CPU/NPU/GPU | shared份额CPU/NPU/GPU | decision ms | prepare ms | MoE墙钟 ms | combine ms |',
        '|---|---|---|---:|---:|---:|---:|']
    branches=[]
    for c in s['comparisons']:
        rs=[r for tag,r in s['runs'].items() if tag.startswith(f"formal-{c['prompt']}-{c['tokens']}-custom-")]
        pairs=[sum(r['routed_pairs'][b] for r in rs) for b in range(3)]
        shared=[sum(r['shared_layers'][b] for r in rs) for b in range(3)]
        stages={k:st.mean(v for r in rs for v in r['stages_ms'][k]) for k in rs[0]['stages_ms']}
        b=dict(prefix=c['tokens'],prompt=c['prompt'],routed_fraction=[v/sum(pairs) for v in pairs],
            shared_fraction=[v/sum(shared) for v in shared],stages_ms=stages,
            backend_ms=[st.mean(v[i] for r in rs for v in r['backend_ms']) for i in range(3)],
            predicted_backend_ms=[st.mean(v[i] for r in rs for v in r['predicted_backend_ms']) for i in range(3)])
        branches.append(b)
        lines.append(f"| {c['tokens']}/{c['prompt']} | "+' | '.join(' / '.join(f'{v*100:.2f}%' for v in b[k]) for k in ('routed_fraction','shared_fraction'))+
            ' | '+' | '.join(f'{stages[k]:.3f}' for k in ('decision','prepare','expert_wall','combine'))+' |')
    (TASK/'branch-summary.json').write_text(json.dumps(branches,indent=2)+'\n')
    lines+=['','## 标定和 independent technical 验证','',
        '360候选、1800计时sample：technical capture第127/511行，层0/12训练、6/18用于开发留出检查；每候选两次warmup、五次计时。所有候选通过4%逐token数值检查，全部无major faults。',
        f"初始启动/每专家/shared三项模型只拟合分支时长，开发留出平均regret为{simple['validation']['mean_regret']*100:.2f}%；完整FFN包含非加性交互与dispatch/join，该模型被保留但未用于正式对照。",
        f"最终模型直接保存15种三端数量分配×3种shared位置的完整FFN墙钟代价，按四个训练上下文的候选中位数再取中位数；留出平均regret为{model['validation']['mean_regret']*100:.2f}%，总耗时MAPE为{model['validation']['total_mape']*100:.2f}%。",
        '开发留出数据已在模型结构迭代中查看，不称为完全未见测试集。wetland/foundry未参与拟合。表只索引数量/shared位置，不索引expert ID、layer、prompt；对当前四expert同尺寸假设泛化，其他模型/CPU线程配置需重标定。',
        '运行时枚举最多81个四expert assignment，每个可行数量组合只评估一次，最多45次；驻留限制保持，N>1使用原模型。独立N=1模型也用于实际单行prefill层，最后层CPU-only约束不变。',
        '', '| technical前缀 | 旧调度 ms/token | N=1调度 ms/token | 吞吐变化 |', '|---:|---:|---:|---:|']
    for d in development['comparisons']:
        lines.append(f"| {d['prefix']} | {d['modes']['old']['ms_per_token']:.3f} | {d['modes']['n1']['ms_per_token']:.3f} | {d['throughput_vs_old_percent']:+.2f}% |")
    lines+=['','technical每配置只有一次进程、两轮计时，作为模型/执行契约验证与性能诊断，不替代正式均衡CPU对照。prefill末logits旧/新逐位一致；各路径重复decode稳定。',
        '', '## 数值、内存及限制','',
        '| 前缀/输入 | CPU/current top1一致 | logits平均/最大NRMSE | CPU major faults/步 | 当前major faults/步 |',
        '|---|---:|---:|---|---|']
    for c in s['comparisons']:
        e=c['logits_vs_cpu']
        lines.append(f"| {c['tokens']}/{c['prompt']} | {e['top1_equal_positions']}/{e['positions']} | {100*e['mean_nrmse']:.3f}% / {100*e['max_nrmse']:.3f}% | {c['modes']['cpu']['major_faults_range']} | {c['modes']['custom']['major_faults_range']} |")
    lines+=['',
        '- 同路径跨重复/进程logits逐位一致；跨路径差异不等同于生成质量评估。使用固定32步teacher tokens，不是自然停止的自由生成。',
        '- unique expert weight payload保持7,474,348,032 B，common单物理scratch/唯一owner、容量与预热后graph allocation稳定检查通过。decode仍保留prefill预留arena和缓存，未做phase内存释放。',
        '- 全CPU的expert分配属于custom共享布局路径，dense/attention仍沿用其既有后端；不等同于原生CPU整图，也没有实现保留同一KV的原生CPU整图迁移。不能将CPU/current总差异全归因于专家调度。',
        '- 本轮不修改计算kernel；原执行器、profile版本执行器和原模型均保留。CPU仍四线程，未验证1–2小核；没有固定CPU affinity或频率。',
        '- 复算：analyze.py formal；report.py。协议、输入和binary/model hashes见protocol.json、formal-frozen.json，归档见archive-record.json。','']
    (TASK/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
