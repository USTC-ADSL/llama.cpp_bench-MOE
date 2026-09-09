#!/usr/bin/env python3
"""Audit retained E1-B heavy pairs; optional read-only device export collection."""
import argparse
import csv
import importlib.util
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/Motivation_Exp/E1-B/evidence/heavy'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def collect_exports():
    # No UI actions, no re-execution of a task, no replacement of initial checks.
    spec = importlib.util.spec_from_file_location('heavy', ROOT / 'tests/run-e1b-heavy.py')
    heavy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(heavy)
    assert heavy.e.cmd(['adb', '-s', heavy.e.SERIAL, 'get-state']).stdout.strip() == 'device'
    for folder in sorted(OUT.glob('paired-image-*')):
        supplement = folder / 'export-followup'
        supplement.mkdir(exist_ok=False)
        before = read(folder / 'process-before.json')
        result = heavy.export_evidence(supplement, before[0]['pid'])
        initial = read(folder / 'export-check.json')
        assert not result['device_path'] or result['device_path'] == initial['device_path']
        print(folder.name, result.get('success'), result.get('content_valid'), flush=True)


def frame_latency(raw, start_us):
    header = None
    valid = []
    for line in raw.splitlines():
        if line.startswith('Flags,'):
            header = line.strip(',').split(',')
        elif header and re.match(r'^\d+,', line):
            values = line.strip(',').split(',')
            if len(values) != len(header):
                continue
            row = dict(zip(header, map(int, values)))
            if (row['Flags'] == 0 and start_us * 1000 <= row['IntendedVsync']
                    <= row['FrameCompleted'] < 2**63 - 1):
                valid.append((row['FrameCompleted'] - start_us * 1000) / 1e6)
    return min(valid) if valid else None


def analyze():
    reference = read(OUT.parent / 'completion/screen-B0-1/response.json')['tokens']
    rows = []
    for task in ['image', 'pdf']:
        for rep in range(1, 4):
            for condition in ['noLLM', 'LLM']:
                folder = OUT / f'paired-{task}-{condition}-{rep}'
                result = read(folder / 'result.json')
                app = read(folder / 'app-result.json')
                before = read(folder / 'process-before.json')
                row = dict(run=folder.name, task=task, rep=rep, condition=condition,
                           inference_status=result['inference_status'],
                           state_preserved=app['state']['preserved'],
                           original_continuation_ok=result['continuation_ok'],
                           continuation_ok=app.get('continuation_ok'),
                           pid_preserved=before[0]['pid'] in [x['pid'] for x in app['process_at_return']],
                           first_frame_ms=frame_latency(read(folder / 'graphics.json')['raw'], app['switch']['device_start_us']),
                           recovery_upper_ms=app['recovery_upper_ms'])
                if row['first_frame_ms'] is not None:
                    assert abs(row['first_frame_ms'] - result['first_frame_ms']) < .001
                if task == 'image':
                    check = folder / 'export-followup/export-check.json'
                    check = check if check.exists() and read(check).get('content_valid') else folder / 'export-check.json'
                    export = read(check)
                    row['continuation_ok'] = True if export.get('success') and export.get('content_valid') else None
                    row['export_evidence'] = str(check.relative_to(OUT))
                    starts = [float(m[1]) for s in export['logs'] if (m := re.match(r'\s*([\d.]+).*\[Export\] start', s))]
                    ends = [float(m[1]) for s in export['logs'] if (m := re.match(r'\s*([\d.]+).*onExported\(\) called with: success = \[true\]', s))]
                    row['export_duration_ms'] = (ends[-1] - starts[-1]) * 1000 if starts and ends else None
                if condition == 'LLM':
                    response = read(folder / 'response.json')
                    events = [json.loads(s) for s in (folder / 'events.jsonl').read_text().splitlines()]
                    ready = [x for x in events if x['event'] == 'token_ready']
                    assert result['inference_status'] == 'success'
                    assert response['tokens_evaluated'] == 512 and response['tokens_predicted'] == 128 and response['timings']['cache_n'] == 0
                    assert len(ready) == 128 and [x['value'] for x in ready] == response['tokens'] == reference
                    assert not any(x['overflow'] for x in events)
                    t0 = next(x['t_us'] for x in events if x['event'] == 'request_received')
                    t1 = next(x['t_us'] for x in events if x['event'] == 'request_end')
                    row['return_during_inference'] = t0 < app['switch']['device_start_us'] < t1
                    row['request_ms'] = (t1 - t0) / 1000
                    row['load_wall_s'] = result['load_wall_s']
                    assert row['return_during_inference']
                    if task == 'image':
                        row['save_during_inference'] = t0 < app['save_action_us'] < t1
                        row['export_start_during_inference'] = bool(starts and t0 < starts[-1] * 1e6 < t1)
                        row['export_inference_overlap_ms'] = max(0, min(t1 / 1e6, ends[-1]) - max(t0 / 1e6, starts[-1])) * 1000 if starts and ends else None
                        assert row['save_during_inference']
                memory = list(csv.DictReader((folder / 'memory.csv').open()))
                assert not (folder / 'stop_reason').exists(), folder
                assert memory and all(x['pss_kb'] == '-1' and x['cgroup_usage_bytes'] == '-1' for x in memory)
                row['min_memavailable_kib'] = min(int(x['memavailable_kb']) for x in memory)
                row['max_thermal_status'] = max(int(x['thermal_status']) for x in memory)
                row['status'] = 'fail' if not row['state_preserved'] or row['continuation_ok'] is False else 'indeterminate' if row['continuation_ok'] is None or row['first_frame_ms'] is None else 'pass'
                rows.append(row)
    for row in rows:
        if row['condition'] != 'LLM':
            continue
        control = next(x for x in rows if x['task'] == row['task'] and x['rep'] == row['rep'] and x['condition'] == 'noLLM')
        value, base = row['first_frame_ms'], control['first_frame_ms']
        if value is not None and base is not None:
            row['delta_ms'] = value - base
            row['delta_pct'] = (value / base - 1) * 100
            row['frame_degraded'] = value - base >= 500 and value >= base * 1.5
            row['extra_reload'] = control['pid_preserved'] and not row['pid_preserved']
            if row['status'] == 'pass' and (row['frame_degraded'] or row['extra_reload']):
                row['status'] = 'degraded'
    save(OUT / 'analysis.json', rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collect-exports', action='store_true')
    args = parser.parse_args()
    collect_exports() if args.collect_exports else analyze()
