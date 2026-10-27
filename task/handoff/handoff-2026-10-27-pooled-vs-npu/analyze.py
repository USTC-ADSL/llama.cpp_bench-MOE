#!/usr/bin/env python3
"""Check all runs and summarize process medians and order-balanced comparisons."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
import math

TASK = Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
BASELINE = ROOT / 'task/handoff/handoff-2026-10-26-fd-2k-memory/evidence'


def thermal(text):
    status = re.search(r'^Thermal Status: (\d+)', text, re.M)
    current = text.split('Current temperatures from HAL:', 1)[1].split('Current cooling devices', 1)[0]
    temps = {name: float(value) for value, name in re.findall(
        r'Temperature\{mValue=([\d.-]+), mType=\d+, mName=([^,]+), mStatus=\d+\}', current)}
    return {'status': int(status[1]) if status else None, 'current_celsius': temps}


def main():
    frozen = json.loads((TASK / 'frozen.json').read_text())
    runs = json.loads((TASK / 'runs.json').read_text())
    assert len(runs) == 8
    rows, events, outputs = [], {}, {}
    for run in runs:
        tag, prompt, mode = run['tag'], run['prompt'], run['mode']
        report = json.loads((OUT / (tag + '.json')).read_text())
        manifest = json.loads((OUT / (tag + '.run.json')).read_text())
        ref_tag = f'n2048-{prompt}-' + ('npu' if mode == 'npu' else 'hetero')
        reference = json.loads((BASELINE / (ref_tag + '.json')).read_text())
        assert run['exit'] == manifest['exit'] == 0 and run['boot_after'] == frozen['boot_id']
        assert manifest['model'] == frozen['model_stat']
        assert report['tokens'] == report['prefill_chunk'] == 2048 and report['prefill_chunks'] == 1
        assert len(report['samples']) == 6 and report['process_identity']['uid'] == 2000
        assert report['mode'] == ('npu' if mode == 'npu' else 'custom')
        assert report['token_ids'] == reference['token_ids'] and report['top1'] == reference['top1']
        hashes = {Path(s.split()[-1]).name: s.split()[0] for s in manifest['hashes'].splitlines()}
        for name, digest in frozen['hashes'].items():
            if Path(name).name in ('moe-pooled-activation-probe', 'test-shared-activation-layout'):
                continue
            assert hashes[Path(name).name] == digest
        logits = (OUT / (tag + '.json.logits.bin')).read_bytes()
        assert logits == (BASELINE / (ref_tag + '.json.logits.bin')).read_bytes()
        outputs[prompt, mode] = logits
        if mode == 'pooled':
            assert report['all_round_logits_bitwise_equal'] and report['activation_arena_policy'] == 'pooled'
            assert sum(report['activation_arena_bytes']) + report['shared_scratch_capacity_bytes'] == 464617856
            assert len(reference['samples']) == len(report['samples'])
            for sample, old_sample in zip(report['samples'], reference['samples']):
                assert len(sample['layers']) == len(old_sample['layers'])
                for layer, old_layer in zip(sample['layers'], old_sample['layers']):
                    assert layer['expert_token_counts'] == old_layer['expert_token_counts']
                    assert layer['scheduling']['owner'] == old_layer['scheduling']['owner']
                    assert layer['scheduling']['shared'] == old_layer['scheduling']['shared']
                    pool = layer['pooled_arena']; end = 0
                    assert len(pool['partition_offsets']) == len(pool['partition_bytes']) == len(pool['layout_bytes'])
                    for offset, size, actual in zip(pool['partition_offsets'], pool['partition_bytes'], pool['layout_bytes']):
                        assert actual <= size
                        if size:
                            assert offset >= end
                            end = offset + size
                    assert end == pool['reserved_bytes'] and end <= pool['capacity_bytes']
        observations = []
        for line in (OUT / (tag + '-memory.jsonl')).read_text().splitlines():
            sample = json.loads(line)
            values = {k: int(v) for k, v in re.findall(r'^(MemAvailable|VmSwap):\s+(\d+)', sample['text'], re.M)}
            if 'MemAvailable' in values:
                observations.append(values)
        matched = []
        for kind in ('dmesg', 'logcat'):
            def read(phase):
                return gzip.decompress((OUT / f'{tag}-{phase}-{kind}.txt.gz').read_bytes()).decode().splitlines()
            before = set(read('before'))
            current = [s for s in read('after') if s not in before and re.search(
                r'out of memory|oom-kill|oom_reaper|invoked oom|Killed process|lowmemorykiller: Kill |ENOMEM|allocation fail|dspqueue_read failed|kernel panic', s, re.I)]
            events[tag + '-' + kind] = current
            matched += current
        times = [s['total_us'] / 1000 for s in report['samples']]
        rows.append({'tag': tag, 'prompt': prompt, 'mode': mode, 'samples_ms': times,
            'median_ms': statistics.median(times), 'major_faults': [s['major_faults'] for s in report['samples']],
            'min_available_mib': min(v['MemAvailable'] for v in observations) / 1024,
            'max_process_swap_mib': max(v.get('VmSwap', 0) for v in observations) / 1024,
            'kgsl_oom_lines': sum('kgsl: out of memory' in s for s in matched),
            'lmkd_kill_lines': sum('lowmemorykiller: Kill ' in s for s in matched),
            'thermal_before': thermal(manifest['before_thermal']), 'thermal_after': thermal(manifest['after_thermal']),
            'logits_sha256': hashlib.sha256(logits).hexdigest(), 'bitwise_equal_to_same_path_baseline': True})
    comparisons = []
    for prompt in ('wetland', 'foundry'):
        by_mode = {}
        selected = [r for r in rows if r['prompt'] == prompt]
        for mode in ('npu', 'pooled'):
            group = [r for r in selected if r['mode'] == mode]
            assert len(group) == 2
            times = [v for r in group for v in r['samples_ms']]
            by_mode[mode] = {'process_medians_ms': [r['median_ms'] for r in group],
                'median_of_12_samples_ms': statistics.median(times), 'sample_range_ms': [min(times), max(times)]}
        pair_ratios = []
        for pair in (selected[:2], selected[2:]):
            med = {r['mode']: r['median_ms'] for r in pair}
            pair_ratios.append(med['npu'] / med['pooled'])
        a = [v[0] for v in struct.iter_unpack('<f', outputs[prompt, 'npu'])]
        b = [v[0] for v in struct.iter_unpack('<f', outputs[prompt, 'pooled'])]
        assert len(a) == len(b) and all(math.isfinite(v) for v in a + b)
        ratio = by_mode['npu']['median_of_12_samples_ms'] / by_mode['pooled']['median_of_12_samples_ms']
        comparisons.append({'prompt': prompt, **by_mode, 'speedup_npu_over_pooled': ratio,
            'latency_reduction_percent': (1 - 1 / ratio) * 100, 'pair_speedups_in_order': pair_ratios,
            'cross_path_logits_nrmse': math.sqrt(sum((x-y)**2 for x,y in zip(a,b)) / sum(x*x for x in a)),
            'cross_path_logits_cosine': sum(x*y for x,y in zip(a,b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))})
    (TASK / 'summary.json').write_text(json.dumps({'runs': rows, 'comparisons': comparisons}, indent=2) + '\n')
    (TASK / 'memory-events.json').write_text(json.dumps(events, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
