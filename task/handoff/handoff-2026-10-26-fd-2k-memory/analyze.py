#!/usr/bin/env python3
"""Summarize completed or failed 2K runs without overwriting raw evidence."""
import gzip
import json
import pathlib
import re
import statistics

TASK = pathlib.Path(__file__).resolve().parent
OUT = TASK / 'evidence'
EVENT = re.compile(r'out of memory|oom-kill|oom_reaper|invoked oom|Killed process|lowmemorykiller|lmkd.*kill|Kill.*fullmodel|ENOMEM|allocation fail|alloc.*fail|dspqueue_read failed|kernel panic', re.I)


def text_snapshot(tag, phase, kind):
    path = OUT / f'{tag}-{phase}-{kind}.txt.gz'
    return gzip.decompress(path.read_bytes()).decode(errors='replace') if path.exists() else ''


def main():
    rows = []
    events = {}
    for run in json.loads((TASK / 'runs.json').read_text()):
        tag = run['tag']
        row = {k: run[k] for k in ('tag', 'mode', 'prompt', 'exit', 'boot_after')}
        path = OUT / f'{tag}.json'
        if path.exists():
            report = json.loads(path.read_text())
            samples = report['samples']
            assert len(samples) == 6
            row.update(samples_ms=[s['total_us'] / 1000 for s in samples],
                       median_ms=statistics.median(s['total_us'] / 1000 for s in samples),
                       major_faults=[s['major_faults'] for s in samples],
                       minor_faults=[s['minor_faults'] for s in samples],
                       process_memory=report.get('process_memory'),
                       process_identity=report.get('process_identity'),
                       tokens=report.get('tokens'), prefill_chunk=report.get('prefill_chunk'),
                       top1=report.get('top1'),
                       arena_bytes=sum(report.get('activation_arena_bytes', [])),
                       unique_weight_bytes=sum(p.get('unique_weight_allocation_bytes', 0) for p in report.get('placements', [])))
            row['tokens_per_second'] = 2048 * 1000 / row['median_ms']
        sampled = []
        for line in (OUT / f'{tag}-memory.jsonl').read_text().splitlines():
            observation = json.loads(line)
            values = {key: int(value) for key, value in re.findall(r'^(MemAvailable|SwapTotal|SwapFree|VmRSS|VmHWM|VmSwap):\s+(\d+)', observation['text'], re.M)}
            if values:
                sampled.append(values)
        row['observation_count'] = len(sampled)
        row['min_system_available_kib'] = min(s['MemAvailable'] for s in sampled)
        row['max_system_swap_used_kib'] = max(s['SwapTotal'] - s['SwapFree'] for s in sampled)
        row['max_sampled_process_rss_kib'] = max(s.get('VmRSS', 0) for s in sampled)
        row['max_sampled_process_swap_kib'] = max(s.get('VmSwap', 0) for s in sampled)
        for kind in ('dmesg', 'logcat'):
            before = set(text_snapshot(tag, 'before', kind).splitlines())
            events[f'{tag}-{kind}'] = [line for line in text_snapshot(tag, 'after', kind).splitlines()
                                      if line not in before and EVENT.search(line)]
        rows.append(row)
    (TASK / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    (TASK / 'memory-events.json').write_text(json.dumps(events, indent=2) + '\n')
    print(json.dumps(rows, indent=2))
    print('Memory-event matches:', {k: len(v) for k, v in events.items()})


if __name__ == '__main__':
    main()
