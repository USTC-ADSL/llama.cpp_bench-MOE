#!/usr/bin/env python3
"""One independent full-model context correction; never reads held-out runs."""
import hashlib
import json
import pathlib
import numpy as np
TASK=pathlib.Path(__file__).resolve().parent
def main():
    model=json.loads((TASK/'runtime-joint.json').read_text());knots=[];sources={};rows=[]
    for nt in (128,256,512,1024):
        p=TASK/f'evidence/calibration-technical-{nt}-joint-0.json';d=json.loads(p.read_text())
        sources[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
        layers=[l for s in d['samples'] for l in s['layers'] if l['tokens']!=1]
        scale=[float(np.median([(l['backend_phases_us'][b][1]+l['backend_gather_us'][b])/l['scheduling']['predicted_backend_us'][b] for l in layers])) for b in range(3)]
        assert all(.25<v<4 for v in scale)
        knots.append({'tokens':nt,'backends':scale})
        rows.append({'tokens':nt,'median_ms':d['timing']['median_us']/1000,
            'major_faults':[s['major_faults'] for s in d['samples']],
            'observed_over_predicted_median':scale,
            'decision_ms_mean':float(np.mean([sum(l['scheduling']['decision_us'] for l in s['layers'])/1000 for s in d['samples']])),
            'unique_weight_payload_bytes':sum(p['unique_weight_payload_bytes'] for p in d['placements']),
            'arena_capacity_bytes':d['arena_capacity_bytes'],'all_round_logits_bitwise_equal':d['all_round_logits_bitwise_equal']})
    model['context_scales']=knots
    model['context_calibration']={'source_sha256':sources,'interpolation':'linear in tokens; clamp at endpoints',
        'scope':'one fixed technical training text, six requests per length; no expert IDs; no online updates',
        'validated_domain':[128,1024]}
    dest=TASK/'runtime-joint-context.json';assert not dest.exists()
    dest.write_text(json.dumps(model,indent=2)+'\n')
    (TASK/'calibration-report.json').write_text(json.dumps({'rows':rows,
        'decision':'CPU underprediction is shape-dependent (1K ratio 1.44 vs 0.97 at512). Freeze one length-interpolated backend correction using all calibration samples. No online feedback or held-out tuning.'},indent=2)+'\n')
    print(json.dumps(knots,indent=2))
if __name__=='__main__':main()
