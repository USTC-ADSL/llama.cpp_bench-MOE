#!/usr/bin/env python3
"""Validate continuous decode and compare all formal samples without filtering."""
import argparse
import json
import pathlib
import statistics as st
import numpy as np
from run import TASK, ROOT, OUT, JOINT, sha


def read(tag):
    return json.loads((OUT / (tag + '.json')).read_text())


def logits(tag):
    a = np.fromfile(OUT / (tag + '.json.decode.bin'), dtype=np.float32)
    assert a.size % 33 == 0 and a.size > 0
    a = a.reshape(33, -1)
    assert np.isfinite(a).all()
    return a


def error(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    nrmse = np.sqrt(np.mean((a-b)**2, axis=1)) / np.sqrt(np.mean(b*b, axis=1))
    top1 = np.argmax(a, axis=1) == np.argmax(b, axis=1)
    top10 = [len(set(np.argpartition(x,-10)[-10:]) & set(np.argpartition(y,-10)[-10:])) for x,y in zip(a,b)]
    return dict(nrmse_by_position=nrmse.tolist(), max_nrmse=float(nrmse.max()),
                mean_nrmse=float(nrmse.mean()), top1_equal_positions=int(top1.sum()),
                positions=len(top1), top10_overlap_by_position=top10, min_top10_overlap=min(top10))


def validate(tag, nt, custom, frozen, reps):
    d = read(tag); rec = json.loads((OUT / (tag + '.run.json')).read_text())
    assert rec['exit'] == 0 and rec['model'].strip() == frozen['model_stat']
    expected = {pathlib.Path(p).name:h for p,h in frozen['sha256'].items() if p.startswith('build-android')}
    actual = {pathlib.Path(line.split()[1]).name:line.split()[0] for line in rec['hashes'].splitlines() if not line.endswith('/prompt.txt')}
    assert actual == expected
    assert d['tokens'] == nt and d['prefill_chunks'] == 1
    db = d['decode_benchmark']; assert db['warmup_sequences'] == 2 and len(db['teacher_tokens']) == 32
    assert len(db['samples']) == reps and db['all_repeat_logits_bitwise_equal']
    r = dict(process_median_ms_per_token=db['sequence_timing']['median_us']/32000,
             sequence_ms_per_token=[s['total_us']/32000 for s in db['samples']],
             token_ms=[], major_faults=[], stages_ms={k:[] for k in ('decision','prepare','expert_wall','combine','outside_expert_wall','dispatch_residual')},
             backend_ms=[], routed_pairs=[0,0,0], shared_layers=[0,0,0], evaluations=[], allocations=[],
             process_memory=d['process_memory'], arena_capacity_bytes=d['arena_capacity_bytes'],
             shared_scratch_bytes=d['shared_scratch_capacity_bytes'], sha256=sha(OUT / (tag+'.json')))
    signature = None
    for seq in db['samples']:
        assert len(seq['steps']) == 32
        assert abs(sum(s['total_us'] for s in seq['steps'])-seq['total_us']) < .01
        r['allocations'].append(seq['expert_graph_allocations'])
        sig = []
        for i,s in enumerate(seq['steps']):
            assert s['step'] == i
            r['token_ms'].append(s['total_us']/1000); r['major_faults'].append(s['major_faults'])
            if not custom:
                continue
            ls = s['layers']; assert len(ls) == 24
            assert [l['layer'] for l in ls] == list(range(24))
            assert all(l['tokens'] == 1 and sum(l['pairs']) == 4 for l in ls)
            assert all(sum(l['expert_token_counts']) == 4 for l in ls)
            sig.append([(l['expert_token_counts'],l['scheduling']['owner'],l['scheduling']['shared']) for l in ls])
            for l in ls:
                sc = l['scheduling']; assert l['shared_scratch_owner'] == sc['shared']
                assert 0 < l['shared_scratch_layout_bytes'] <= d['shared_scratch_capacity_bytes']
                assert all(x<=y for x,y in zip(l['routed_layout_bytes'], d['arena_capacity_bytes']))
                for b in range(3): r['routed_pairs'][b] += l['pairs'][b]
                r['shared_layers'][sc['shared']] += 1
                r['evaluations'].append(sc['evaluations'])
            assert ls[-1]['scheduling']['shared'] == 0 and ls[-1]['pairs'] == [4,0,0]
            for stage,key in (('decision','decision_us'),('prepare','gather_us'),('expert_wall','expert_wall_us'),('combine','combine_us')):
                r['stages_ms'][stage].append(sum(l['scheduling'][key] if stage=='decision' else l[key] for l in ls)/1000)
            r['stages_ms']['outside_expert_wall'].append(s['total_us']/1000-r['stages_ms']['expert_wall'][-1])
            r['backend_ms'].append([sum(l['backend_phases_us'][b][1]+l['backend_gather_us'][b] for l in ls)/1000 for b in range(3)])
            r['stages_ms']['dispatch_residual'].append(sum(l['expert_wall_us']-max(l['backend_phases_us'][b][1]+l['backend_gather_us'][b] for b in range(3)) for l in ls)/1000)
        if custom:
            if signature is None: signature = sig
            assert sig == signature, (tag, 'routes or decisions differ between repetitions')
    assert len(set(r['allocations'])) == 1
    if custom:
        assert d['current_route_scheduler'] and d['expert_cpu_threads'] == 4
        assert rec['route_model_sha256'] == sha(JOINT/'runtime-joint-context.json')
        assert d['shared_scratch_physical_backings'] == 1
        assert sum(p['unique_weight_payload_bytes'] for p in d['placements']) == 7474348032
    r['major_faults_range'] = [min(r['major_faults']), max(r['major_faults'])]
    return d, r, signature


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--partial', action='store_true'); parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    if args.pilot:
        a,b = [read('pilot-wetland-128-'+m) for m in ('common','private')]
        assert np.array_equal(logits('pilot-wetland-128-common'),logits('pilot-wetland-128-private'))
        assert a['decode_benchmark']['teacher_tokens'] == b['decode_benchmark']['teacher_tokens']
        assert all(d['decode_benchmark']['all_repeat_logits_bitwise_equal'] for d in (a,b))
        f = json.loads((TASK/'pilot-frozen.json').read_text())
        validate('pilot-wetland-128-common',128,True,f,2)
        sig=lambda d:[[(l['expert_token_counts'],l['scheduling']['owner'],l['scheduling']['shared']) for l in s['layers']] for s in d['decode_benchmark']['samples'][0]['steps']]
        assert sig(a)==sig(b)
        result=dict(positions=33,steps=32,repetitions=2,private_common_logits_bitwise_equal=True,routes_and_placement_equal=True,
                    prefix_and_decode_reused=True,common_warmup_allocations=a['decode_benchmark']['samples'][0]['expert_graph_allocations'])
        (TASK/'pilot-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result);return
    records = [r for r in json.loads((TASK/'runs.json').read_text()) if r['tag'].startswith('formal-')]
    frozen = json.loads((TASK/'formal-frozen.json').read_text()); summary=dict(complete=False,runs={},comparisons=[])
    for nt in (128,1024):
        for prompt in ('wetland','foundry'):
            group=[r for r in records if r['tokens']==nt and r['prompt']==prompt and r['exit']==0]
            if len(group)!=4: assert args.partial;continue
            teacher=json.loads((TASK/f'teacher-{prompt}-{nt}.json').read_text());ids=None;modes={}; arrays={}
            for mode in ('npu','custom'):
                entries=[];ref=None;sig=None
                for row in [r for r in group if r['mode']==mode]:
                    tag=row['tag'];d,r,current_sig=validate(tag,nt,mode=='custom',frozen,4)
                    assert d['decode_benchmark']['teacher_tokens']==teacher
                    assert d['quality']['teacher_forced']
                    if ids is None:ids=d['token_ids']
                    assert ids==d['token_ids']
                    a=logits(tag)
                    if ref is None:ref=a;sig=current_sig
                    assert np.array_equal(a,ref), (tag,'logits differ across processes')
                    assert sig==current_sig
                    summary['runs'][tag]=r;entries.append(r)
                assert len(entries)==2;arrays[mode]=ref
                modes[mode]=dict(ms_per_token=st.mean(r['process_median_ms_per_token'] for r in entries),
                                process_ms_per_token=[r['process_median_ms_per_token'] for r in entries],
                                p95_token_ms=float(np.percentile([v for r in entries for v in r['token_ms']],95)),
                                major_faults_range=[min(v for r in entries for v in r['major_faults']),max(v for r in entries for v in r['major_faults'])])
            n,c=[modes[m]['ms_per_token'] for m in ('npu','custom')]
            result=dict(tokens=nt,prompt=prompt,modes=modes,throughput_gain_percent=100*(n/c-1),
                        latency_reduction_percent=100*(1-c/n),logits_vs_npu=error(arrays['custom'],arrays['npu']))
            summary['comparisons'].append(result)
            print(nt,prompt,'NPU/current ms/token',round(n,3),round(c,3),'throughput%',round(result['throughput_gain_percent'],2),flush=True)
    summary['complete']=len(records)==16 and len(summary['comparisons'])==4 and all(r['exit']==0 for r in records)
    (TASK/('summary-partial.json' if args.partial else 'summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
    if not args.partial:assert summary['complete']


if __name__=='__main__':main()
