#!/usr/bin/env python3
"""Reconcile archived Expert timings with same-device model/layer timings."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H = HERE.parent


def read(relative):
    return json.loads((H / relative).read_text())


def hashes(text):
    return {Path(line.split()[1]).name: line.split()[0]
            for line in text.splitlines() if line.strip()}


def main():
    micro = read('handoff-2026-09-12-expert-compute/evidence/summary.json')
    model = read('handoff-2026-09-14-k80-prefill/evidence/summary.json')
    e2e = read('handoff-2026-09-23-expert-e2e/evidence/summary.json')
    cpu_us = micro['backends']['cpu']['64']['timings']['full']['median_us']/64
    hmx_us = micro['backends']['hmx']['64']['timings']['full']['median_us']/64
    old_hashes = hashes(micro['binary_hashes'])
    full_hashes = hashes((H/'handoff-2026-09-14-k80-prefill/evidence/host.sha256').read_text())
    for lib in ('libggml-cpu.so', 'libggml-hexagon.so', 'libggml-htp-v79.so'):
        assert old_hashes[lib] == full_hashes[lib]
    result = {
        'device': 'fd8657d6',
        'model_sha256': micro['model_sha256'],
        'metadata_verified_from_host_gguf': {
            'layers': 24, 'experts': 60, 'topk': 4,
            'routed_width': 1408, 'shared_width': 5632},
        'statistics_note': 'Micro/layer values are pooled medians; model values are sample means. Independent medians need not add. Mean throughput is not the inverse of mean latency.',
        'library_differences_micro_vs_model': {
            name: [digest, full_hashes[name]] for name, digest in old_hashes.items()
            if name.endswith('.so') and name in full_hashes and digest != full_hashes[name]},
        'micro_full_ffn_ms': {
            mode: {t: data['timings']['full']['median_us']/1000
                   for t, data in micro['backends'][mode].items() if data['status'] == 'ok'}
            for mode in ('cpu', 'opencl', 'hmx', 'hvx')},
        'full_model': [], 'first_layer_routing': [],
    }
    for row in model['rows']:
        selected = (row['threads'] == 4 and
                    ((row['ubatch'] == 128 and (row['prompt'] == 128 or row['generation'] == 128)) or
                     (row['ubatch'] == 512 and row['prompt'] == 512)))
        if not selected:
            continue
        raw = read(f"handoff-2026-09-14-k80-prefill/evidence/{row['run']}.json")
        raw = next(x for x in raw if x['n_threads'] == 4 and
                   x['n_prompt'] == row['prompt'] and x['n_gen'] == row['generation'])
        out = dict(row, mean_time_ms=raw['avg_ns']/1e6,
                   mean_time_per_token_ms=raw['avg_ns']/1e6/(row['prompt']+row['generation']))
        if row['backend'] == 'cpu' and row['prompt'] == 128:
            # Rough FFN-equivalent work, not a fitted model or a timing attribution.
            # The final layer FFN keeps only the last output token in this benchmark.
            estimate = (23*128+1)*(4+5632/1408)*cpu_us/1000
            out['rough_ffn_equivalent_ms'] = estimate
            out['rough_error_vs_total_mean_fraction'] = estimate/out['mean_time_ms']-1
        if row['backend'] == 'cpu' and row['generation'] == 128:
            out['rough_ffn_equivalent_ms_per_token'] = 24*8*result['micro_full_ffn_ms']['cpu']['1']
        result['full_model'].append(out)
    for nt in (64, 128):
        relative = f'handoff-2026-09-22-threshold-routing/evidence/summary-p{nt}-t8.json'
        s = read(relative)
        case = next(x for x in s['cases'] if x['layers'] == 1)
        hist = next(x['route']['histogram'] for x in case['validation']
                    if x['policy'] == 'npu_native' and x.get('layer') == 0)
        assert len(hist) == 60 and sum(hist) == nt*4
        base, mixed = (case['policies'][p] for p in ('npu_native', 'mixed_parallel'))
        cpu_pairs = sum(x for x in hist if x <= 8)
        out = {
            'source': relative, 'input_tokens': nt, 'pairs': sum(hist),
            'active_experts': sum(x > 0 for x in hist), 'max_tokens_per_expert': max(hist),
            'hmx_32row_padded_rows_per_projection': sum((x+31)//32*32 for x in hist),
            'npu_compute_ms': base['npu_upload_compute_readback_median_ms'][1],
            'npu_wall_ms': base['phase_median_ms']['experts_wall_us'],
            'invalid_uniform_T64_npu_extrapolation_ms': sum(hist)*hmx_us/1000,
            'cpu_assigned_experts': sum(0 < x <= 8 for x in hist),
            'cpu_assigned_pairs': cpu_pairs,
            'rough_cpu_work_only_ms': cpu_pairs*cpu_us/1000,
            'mixed_cpu_compute_ms': mixed['cpu_upload_compute_readback_median_ms'][1],
            'mixed_npu_compute_ms': mixed['npu_upload_compute_readback_median_ms'][1],
            'mixed_expert_wall_ms': mixed['phase_median_ms']['experts_wall_us'],
            'baseline_layer_ms': base['median_ms'], 'mixed_layer_ms': mixed['median_ms'],
            'expert_wall_delta_ms': mixed['phase_median_ms']['experts_wall_us']-base['phase_median_ms']['experts_wall_us'],
            'layer_delta_ms': mixed['median_ms']-base['median_ms'],
        }
        result['first_layer_routing'].append(out)
    result['same_graph_2_over_1_E2E_ratio'] = {
        mode: v['cases'][1]['total_ms']['median']/v['cases'][0]['total_ms']['median']
        for mode, v in e2e['modes'].items()}
    result['full_model_profile_counts'] = model['expert_profile_counts']
    (HERE/'history-analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
