#!/usr/bin/env python3
"""Render decode performance and scope without conflating prefill or GPU use."""
import csv
import json
import statistics as st
from run import TASK


def main():
    s=json.loads((TASK/'summary.json').read_text());assert s['complete']
    wins=sum(c['throughput_gain_percent']>0 for c in s['comparisons'])
    outcome=('按预先规定的组汇总口径，四组对照均未观察到当前调度相对纯CPU的decode吞吐收益。' if wins==0 else f'按组汇总，四组对照中{wins}组观察到当前调度相对纯CPU的decode吞吐收益；其余组仍由CPU领先。')
    lines=['# 当前joint调度的连续decode设备验证','',outcome,
        '设备fd8657d6 / Redmi K80 Pro，Qwen1.5-MoE-A2.7B-Chat Q4_0；用户指定仅比较原生CPU后端与当前joint调度、单份CPU4x8权重和common shared scratch。CPU基线和异构外层CPU、expert CPU均4线程，未固定affinity。2K未测。',
        '两份模板化英文输入wetland/foundry，前缀长度128/1024；每组使用NPU诊断生成的同一32-token continuation，固定输入token但不回放expert IDs，各路径实时执行router。所有层decode计算批量都是N=1，prefix长度只改变KV上下文。',
        '每方案每组两个独立进程，进程内两次sequence warmup、四次timed sequence，每序列32步；16个正式进程、64个计时序列、2048个计时token。wetland按CPU/current/current/CPU，foundry反序，进程间20秒冷却。',
        '计时为完整单token llama_decode + synchronize，包含真实路由、调度、专家、dense、attention和同步；加载、prefill重建、logits保存检查及报告序列化排除。权重和prefill arena保留，无phase migration。',
        '', '## 端到端结果','',
        '每进程取32步序列总时间的中位数除以32，再平均两个进程；吞吐收益=T_CPU/T_current−1，延迟下降=1−T_current/T_CPU。没有删除慢样本或缺页样本。',
        '', '| KV前缀 | 输入 | CPU ms/token | 当前 ms/token | CPU token/s | 当前 token/s | 吞吐收益 | 延迟下降 |',
        '|---:|---|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for c in s['comparisons']:
        n,j=[c['modes'][m]['ms_per_token'] for m in ('cpu','custom')]
        lines.append(f"| {c['tokens']} | {c['prompt']} | {n:.3f} | {j:.3f} | {1000/n:.2f} | {1000/j:.2f} | {c['throughput_gain_percent']:+.2f}% | {c['latency_reduction_percent']:+.2f}% |")
        rows.append(dict(prefix=c['tokens'],prompt=c['prompt'],cpu_ms_per_token=n,current_ms_per_token=j,cpu_tokens_s=1000/n,current_tokens_s=1000/j,
                         throughput_gain_percent=c['throughput_gain_percent'],latency_reduction_percent=c['latency_reduction_percent']))
    with (TASK/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    lines+=['','进程间波动如下；每组只有两个独立进程，几百分点差异不能直接作为统计显著性结论。','',
        '| 前缀 | 输入 | CPU两个进程 ms/token | 当前两个进程 ms/token |', '|---:|---|---:|---:|']
    for c in s['comparisons']:
        lines.append(f"| {c['tokens']} | {c['prompt']} | "+' | '.join(' / '.join(f'{v:.3f}' for v in c['modes'][m]['process_ms_per_token']) for m in ('cpu','custom'))+' |')
    lines+=['','## 当前方案分阶段统计','',
        '下表为所有计时decode token的算术均值，阶段统计跨24层求和。decision已包含在prepare中，不可重复相加；expert_wall包含并发分支输入gather与compute。outside=总时间−expert_wall，包含其他阶段，不能单独当作dense kernel时间。',
        '', '| 前缀 | 输入 | 决策 ms/token | prepare ms/token | MoE墙钟 ms/token | 归约 ms/token | outside ms/token |',
        '|---:|---|---:|---:|---:|---:|---:|']
    branches=[]
    for c in s['comparisons']:
        rs=[r for t,r in s['runs'].items() if t.startswith(f"formal-{c['prompt']}-{c['tokens']}-custom-")]
        vals={k:st.mean(v for r in rs for v in r['stages_ms'][k]) for k in rs[0]['stages_ms']}
        lines.append(f"| {c['tokens']} | {c['prompt']} | "+' | '.join(f'{vals[k]:.3f}' for k in ('decision','prepare','expert_wall','combine','outside_expert_wall'))+' |')
        pairs=[sum(r['routed_pairs'][b] for r in rs) for b in range(3)]
        shared=[sum(r['shared_layers'][b] for r in rs) for b in range(3)]
        times=[st.mean(v[b] for r in rs for v in r['backend_ms']) for b in range(3)]
        predicted=[st.mean(v[b] for r in rs for v in r['predicted_backend_ms']) for b in range(3)]
        branches.append(dict(tokens=c['tokens'],prompt=c['prompt'],backend_ms_per_token=times,
                             predicted_backend_ms_per_token=predicted,
                             routed_pair_fraction=[v/sum(pairs) for v in pairs],shared_layer_fraction=[v/sum(shared) for v in shared],
                             dispatch_residual_ms=vals['dispatch_residual'],graph_allocations=[r['allocations'][0] for r in rs]))
    lines+=['','## 实际分配与分支时间','',
        '顺序固定为CPU/NPU/GPU。分支时间为各层gather+backend compute之和，各端可能并发，不能将三列相加当作MoE墙钟。份额包含最后一层的CPU-only resident约束。',
        '', '| 前缀 | 输入 | Routed pair份额 CPU/NPU/GPU | shared层份额 CPU/NPU/GPU | 分支 ms/token CPU/NPU/GPU |',
        '|---:|---|---|---|---|']
    for b in branches:
        fields=[' / '.join(f'{v*100:.2f}%' for v in b[k]) for k in ('routed_pair_fraction','shared_layer_fraction')]
        fields.append(' / '.join(f'{v:.3f}' for v in b['backend_ms_per_token']))
        lines.append(f"| {b['tokens']} | {b['prompt']} | "+' | '.join(fields)+' |')
    (TASK/'branch-summary.json').write_text(json.dumps(branches,indent=2)+'\n')
    lines+=['','当前N=1代价预测与观测（跨24层求和，分支计时定义同上）：','',
        '| 前缀 | 输入 | CPU预测/实测 ms/token | NPU预测/实测 ms/token | GPU预测/实测 ms/token |',
        '|---:|---|---:|---:|---:|']
    for b in branches:
        lines.append(f"| {b['tokens']} | {b['prompt']} | "+' | '.join(f"{b['predicted_backend_ms_per_token'][i]:.3f} / {b['backend_ms_per_token'][i]:.3f}" for i in range(3))+' |')
    lines+=['','代价预测是当前prefill模型在N=1的外推。预测与实际分支时间不匹配可用于定位下一轮decode标定，但本轮没有测所有替代分配，因此不能声称已经找到了decode最优分配。总延迟还包含不同dense/attention执行路径，不能把CPU/current全部差异归结为专家调度。']
    lines+=['','## 数值和缺页','',
        '各方案内部跨重复、跨进程33个位置logits逐位一致；跨方案表中比较包括prefill末位置与32个decode位置，报告全部位置，不等同于生成质量评估。',
        '', '| 前缀 | 输入 | 相对CPU平均/最大logits NRMSE | top1相同位置 | 最小top10重合 | CPU major faults/步 | 当前 major faults/步 |',
        '|---:|---|---:|---:|---:|---|---|']
    for c in s['comparisons']:
        e=c['logits_vs_cpu']
        lines.append(f"| {c['tokens']} | {c['prompt']} | {100*e['mean_nrmse']:.3f}% / {100*e['max_nrmse']:.3f}% | {e['top1_equal_positions']}/{e['positions']} | {e['min_top10_overlap']}/10 | {c['modes']['cpu']['major_faults_range']} | {c['modes']['custom']['major_faults_range']} |")
    lines+=['','## 实现、适用范围及复现','',
        '- decode接入只放宽fullmodel-oracle中single_layout的quality_steps入口禁令。已有路径支持N=1、按counts缓存graph、单shared写入者和同步后scratch复用；计算kernel、joint模型、搜索预算与resident计划未改。native fixed decode graph仍与当前route search互斥。',
        '- 纯CPU预检暴露了CPU_SHARED_Q4_0对模型加载器零大小探测buffer解引用空descriptor的问题，已修复能力检查并增加Android回归。CPU_REPACK保持启用。失败日志保留，正式CPU/current使用相同修复后的库；修复前后有效shared路径的33个位置logits逐位一致。详见docs/bugs/2026-10-23-cpu-shared-weight-probe-null-descriptor.md。',
        '- 定向128前缀common/private比较33个位置逐位一致，route/placement一致；正式测试每步24层、每层4 routed pairs，预热后不再增加执行图，common容量与唯一owner检查通过。原binary、库和源码备份在evidence/backup。',
        '- 代价模型只在128–1024 prefill校准，N=1使用公式外推和128端点context scale；此次是当前策略外推的测量，不是decode专用模型。实际每层重新搜索，没有跨token缓存决策；缓存的是执行graph。',
        '- 最后一层resident仍只允许CPU；前23层三端权重均可选。若实际GPU份额为0，不可把结果描述为三端同时加速。',
        '- 原生CPU与当前方案同时存在专家布局、dense/attention后端、执行方式和调度差异，因此端到端差异属于整个方案，不能仅归因于调度搜索算法；本轮未做单份布局下固定CPU的独立消融。NPU诊断仅生成固定teacher tokens，不作为正式性能基线。',
        '- CPU仍四线程，未验证1–2小核目标；teacher-forced固定长度包含可能的EOG token，不是自然停止的自由生成测试；两份模板化英文文本不代表广泛任务泛化。',
        '- decode期间仍保留按prefill长度预留的三个routed arena、common shared scratch及已缓存graph；不是N=1专用内存预算。若1K有major faults，其实测包含分页代价，本轮未筛掉或修正对应token。',
        '- 所有测量均在Android，未锁定频率或CPU affinity。各进程原始逐token数据、温度、major faults、输入与binary哈希在evidence及summary.json，未将prefill时间混入decode。',
        '- 复算：python3 task/handoff/handoff-2026-10-23-decode-scheduler/analyze.py；然后运行同目录report.py。run.py pilot/teachers准备输入并验证，正式CPU/current入口为formal.py，protocol.json记录用户澄清后的范围；不使用run.py中早期未执行的NPU formal入口。归档见archive-record.json。','']
    (TASK/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
