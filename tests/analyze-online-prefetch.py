#!/usr/bin/env python3
"""Compare complete first-prefill results and summarize causal preparation overlap."""
import argparse
import hashlib
import json
import pathlib
import struct


def summarize(path):
    d = json.loads(path.read_text())
    logits = path.with_suffix('.logits.bin').read_bytes()
    row = {'file': str(path), 'mode': d['mode'], 'prefill_ms': d['total_us'] / 1000,
           'tokens_per_second': d['tokens_per_second'],
           'restore_plus_prefill_ms': d['restore_plus_prefill_us'] / 1000,
           'logits_sha256': hashlib.sha256(logits).hexdigest(),
           'maxrss_mib': d['maxrss_kib'] / 1024}
    if 'layers' in d:
        layers = d['layers']
        row['branch_sum_ms'] = [sum(l['branches'][b]['total_us'] for l in layers) / 1000 for b in range(3)]
        row['ffn_sum_ms'] = sum(l['total_us'] for l in layers) / 1000
        row['outside_ffn_ms'] = row['prefill_ms'] - row['ffn_sum_ms']
        row['raw_scoped_extra_mib'] = d['accounted_extra_allowance_bytes'] / 2**20
    if 'prefetch' in d:
        p = d['prefetch']; records = p['preparations']; layers = d['layers']
        prep = sum(r['total_us'] for r in records)
        overlap = 0
        for r in records:
            # Union of current routed CPU/NPU wall intervals, never sum overlapping branches.
            intervals = sorted((max(r['start_us'], b['start_us']), min(r['end_us'], b['end_us']))
                               for b in layers[r['source_layer']]['branches'][:2] if b['total_us'])
            end = float('-inf')
            for lo, hi in intervals:
                overlap += max(0, hi - max(lo, end)); end = max(end, hi)
        useful = sum(len(l['npu_cached_experts']) for l in layers)
        eligible = sum(sum(layers[r['target_layer']]['counts'][e] > 0 and
                          layers[r['target_layer']]['placement']['owner'][e] == 1 for e in r['ids'])
                       for r in records)
        row['prefetch'] = {'capacity': p['capacity'], 'prepared_experts': sum(len(r['ids']) for r in records),
            'mixed_graph': p.get('mixed_graph', False), 'batch_pack': p.get('batch_pack', False),
            'persistent_source_views': p.get('persistent_source_views', False),
            'source_view_setup_ms': p.get('source_view_setup_us', 0) / 1000,
            'source_import_ms': sum(r['conversion_start_us'] - r['start_us'] for r in records) / 1000,
            'eligible_predictions': eligible, 'used_experts': useful,
            'ready_layers': sum(l['cache_ready'] for l in layers), 'target_layers': len(records),
            'preparation_ms': prep / 1000, 'current_routed_wall_overlap_ms': overlap / 1000,
            'current_routed_wall_overlap_fraction': overlap / prep if prep else 0,
            'gpu_join_ms': sum(l['branches'][2].get('previous_prefetch_wait_us', 0) for l in layers) / 1000,
            'target_mib': p['target_bytes'] / 2**20,
            'alias_adjusted_scoped_extra_mib': p['allowance_minus_known_source_alias_bytes'] / 2**20}
    return row, d, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='+', type=pathlib.Path)
    parser.add_argument('--reference', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path)
    args = parser.parse_args()
    reference = summarize(args.reference) if args.reference else None
    rows = []
    for path in args.results:
        row, d, logits = summarize(path)
        if reference:
            rr, rd, rl = reference
            assert d['tokens'] == rd['tokens'], 'different tokenized inputs'
            row['versus_reference'] = {'file': str(args.reference),
                'throughput_change_pct': 100 * (rd['total_us'] / d['total_us'] - 1),
                'logits_byte_exact': logits == rl}
            if 'layers' in rd and 'layers' in d:
                row['versus_reference']['routes_placement_exact'] = len(d['layers']) == len(rd['layers']) and all(
                    x['counts'] == y['counts'] and x['placement'] == y['placement']
                    for x, y in zip(d['layers'], rd['layers']))
            assert len(logits) == len(rl)
            x = struct.unpack('<' + 'f' * (len(logits) // 4), logits)
            y = struct.unpack('<' + 'f' * (len(rl) // 4), rl)
            row['versus_reference']['logits_max_abs'] = max(abs(a-b) for a,b in zip(x,y))
        rows.append(row)
    result = json.dumps({'runs': rows, 'note': 'Wall overlap proves concurrency, not isolated kernel utilization or causal speedup. Scoped memory excludes untracked driver/dense/KV allocations.'}, indent=2) + '\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
