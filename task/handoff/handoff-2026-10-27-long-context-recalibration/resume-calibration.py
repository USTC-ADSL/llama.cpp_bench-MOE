#!/usr/bin/env python3
"""Continue only unstarted profiling after a pre-launch busy-device guard."""
import gzip
import json
import run as driver

driver.check()
tag='technical-1536-profile'
assert not list(driver.OUT.glob(tag+'*'))
capture=driver.OUT/'technical-1536-capture.json'
assert capture.is_file()
report=driver.execute(tag,'profile',1536,['--reference','technical-1536-capture',
    '--profile-layers','0,12,22','--profile-candidates',str(driver.TASK/'inputs/candidates-1536.json'),
    '--shared-weights','cpu4x8','--shared-gather','htp','--shared-fused-gather',
    '--shared-reduce','neon','--shared-arena','pooled'])
assert len(report['layers'])==3
for layer in report['layers']:
    assert len(layer['candidates'])==18
    for c in layer['candidates']:
        assert c['check']['passes_4percent_per_token'] and len(c['samples'])==5
for il in driver.LAYERS:
    path=driver.OUT/f'technical-1536-capture.json.layer{il}.bin'
    compressed=path.with_suffix(path.suffix+'.gz')
    compressed.write_bytes(gzip.compress(path.read_bytes(),compresslevel=1,mtime=0))
    assert gzip.decompress(compressed.read_bytes())==path.read_bytes()
    path.unlink();driver.adb('shell','rm',driver.REMOTE+'/'+path.name)
print('VERIFIED profile 1536 54 candidate-layer cases',flush=True)
driver.calibration([3072])
