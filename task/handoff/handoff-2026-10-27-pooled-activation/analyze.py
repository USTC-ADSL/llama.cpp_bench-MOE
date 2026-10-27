#!/usr/bin/env python3
"""Verify pooled memory bounds, unchanged routes/logits, and report all memory events."""
import gzip
import hashlib
import json
import pathlib
import re
import statistics

TASK = pathlib.Path(__file__).resolve().parent
ROOT = TASK.parents[2]
OUT = TASK / 'evidence'
OLD = ROOT / 'task/handoff/handoff-2026-10-26-fd-2k-memory/evidence'


def main():
    frozen = json.loads((TASK / 'frozen.json').read_text())
    rows, events = [], {}
    for run in json.loads((TASK / 'runs.json').read_text()):
        tag, prompt = run['tag'], run['prompt']
        report = json.loads((OUT / (tag + '.json')).read_text())
        manifest = json.loads((OUT / (tag + '.run.json')).read_text())
        reference = json.loads((OLD / f'n2048-{prompt}-hetero.json').read_text())
        assert run['exit'] == manifest['exit'] == 0 and run['boot_after'] == frozen['boot_id']
        assert report['tokens'] == report['prefill_chunk'] == 2048 and len(report['samples']) == 6
        assert report['process_identity']['uid'] == 2000 and report['all_round_logits_bitwise_equal']
        assert report['token_ids'] == reference['token_ids']
        logits = OUT / (tag + '.json.logits.bin')
        assert logits.read_bytes() == (OLD / f'n2048-{prompt}-hetero.json.logits.bin').read_bytes()
        deployed = {s.split()[-1].split('/')[-1]: s.split()[0] for s in manifest['hashes'].splitlines()}
        for name, digest in frozen['hashes'].items():
            if pathlib.Path(name).name in deployed:
                assert deployed[pathlib.Path(name).name] == digest
        for sample, old_sample in zip(report['samples'], reference['samples']):
            for layer, old_layer in zip(sample['layers'], old_sample['layers']):
                assert layer['expert_token_counts'] == old_layer['expert_token_counts']
                assert layer['scheduling']['owner'] == old_layer['scheduling']['owner']
                assert layer['scheduling']['shared'] == old_layer['scheduling']['shared']
                if run['mode'] == 'pooled':
                    pool = layer['pooled_arena']; end = 0
                    for offset, size, actual in zip(pool['partition_offsets'], pool['partition_bytes'], pool['layout_bytes']):
                        if size:
                            assert offset >= end and actual <= size
                            end = offset + size
                    assert end == pool['reserved_bytes'] and end <= pool['capacity_bytes']
        arena = sum(report['activation_arena_bytes']) + report['shared_scratch_capacity_bytes']
        expected = 464617856 if run['mode'] == 'pooled' else 1009876992
        assert arena == expected
        observations = []
        for line in (OUT / (tag + '-memory.jsonl')).read_text().splitlines():
            values = {key: int(value) for key, value in re.findall(
                r'^(MemAvailable|SwapTotal|SwapFree|VmRSS|VmHWM|VmSwap):\s+(\d+)', json.loads(line)['text'], re.M)}
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
        rows.append({'tag': tag, 'policy': run['mode'], 'prompt': prompt, 'exit': 0,
            'samples_ms': times, 'median_ms': statistics.median(times),
            'tokens_per_second': 2048000 / statistics.median(times),
            'major_faults': [s['major_faults'] for s in report['samples']],
            'arena_bytes': arena, 'arena_mib': arena / 1048576,
            'unique_weight_bytes': sum(p['unique_weight_allocation_bytes'] for p in report['placements']),
            'bitwise_logits_equal_to_baseline': True, 'routes_and_owners_equal_to_baseline': True,
            'logits_sha256': hashlib.sha256(logits.read_bytes()).hexdigest(),
            'min_system_available_mib': min(v['MemAvailable'] for v in observations) / 1024,
            'max_process_swap_mib': max(v.get('VmSwap', 0) for v in observations) / 1024,
            'kgsl_oom_lines': sum('kgsl: out of memory' in s for s in matched),
            'lmkd_kill_lines': sum('lowmemorykiller: Kill ' in s for s in matched),
            'boot_unchanged': True})
    (TASK / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (TASK / 'memory-events.json').write_text(json.dumps(events, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
