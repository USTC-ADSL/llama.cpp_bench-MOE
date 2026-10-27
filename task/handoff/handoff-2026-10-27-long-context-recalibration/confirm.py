#!/usr/bin/env python3
"""Symmetric same-build old/new/NPU comparison after freezing calibration."""
import argparse
import json
from pathlib import Path
import time
import run as driver

TASK=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokens',type=int,choices=[2048,4096],required=True);a=p.parse_args()
    model=TASK/'runtime-calibrated.json'
    digest=driver.base.sha(model)
    freeze=TASK/'evaluation-freeze.json'
    if freeze.exists():
        assert json.loads(freeze.read_text())['model_sha256']==digest
    else:
        freeze.write_text(json.dumps({'model_sha256':digest,'model':str(model),'tokens':[2048,4096],
            'sequence':{'wetland':['npu','old','new','new','old','npu'],
                        'foundry':['new','npu','old','old','npu','new']},
            'warmups':2,'reps':6,'interval_seconds':30,
            'note':'Freeze before evaluation; no model edits using wetland/foundry results'},indent=2)+'\n')
    sequence=[(prompt,policy) for prompt,policies in [('wetland',['npu','old','new','new','old','npu']),
        ('foundry',['new','npu','old','old','npu','new'])] for policy in policies]
    for index,(prompt,policy) in enumerate(sequence,1):
        assert driver.base.sha(model)==digest
        tag=f'eval-n{a.tokens}-{index:02d}-{prompt}-{policy}'
        report=driver.fullmodel(a.tokens,prompt,policy,tag,6,str(model))
        assert report['tokens']==report['prefill_chunk']==a.tokens and report['prefill_chunks']==1
        assert len(report['samples'])==6 and report['process_identity']['uid']==2000
        print('RESULT',tag,report['timing'],flush=True)
        if index!=len(sequence):time.sleep(30)


if __name__=='__main__':main()
