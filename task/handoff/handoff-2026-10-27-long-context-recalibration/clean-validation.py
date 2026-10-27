#!/usr/bin/env python3
"""Repeat only contaminated 1536 holdout after user confirmed exclusive device access."""
import gzip
import re
import run as driver

driver.check()
mem=driver.adb('shell','cat /proc/meminfo').stdout
assert int(re.search(r'MemAvailable:\s+(\d+)',mem)[1])>8*1024*1024
capture='technical-1536-capture'
assert driver.adb('shell','sha256sum',driver.REMOTE+'/'+capture+'.json').stdout.split()[0]==driver.base.sha(driver.OUT/(capture+'.json'))
restore=driver.OUT/'restore-1536';restore.mkdir()
for il in driver.LAYERS:
    name=f'{capture}.json.layer{il}.bin'
    path=restore/name;path.write_bytes(gzip.decompress((driver.OUT/(name+'.gz')).read_bytes()))
    driver.adb('push',str(path),driver.REMOTE+'/'+name)
    assert driver.adb('shell','sha256sum',driver.REMOTE+'/'+name).stdout.split()[0]==driver.base.sha(path)
report=driver.execute('technical-1536-profile-clean','profile',1536,['--reference',capture,
    '--profile-layers','0,12,22','--profile-candidates',str(driver.TASK/'inputs/candidates-1536.json'),
    '--shared-weights','cpu4x8','--shared-gather','htp','--shared-fused-gather',
    '--shared-reduce','neon','--shared-arena','pooled'])
assert len(report['layers'])==3
for layer in report['layers']:
    assert len(layer['candidates'])==18
    for c in layer['candidates']:
        assert c['check']['passes_4percent_per_token'] and len(c['samples'])==5
for path in restore.iterdir():
    driver.adb('shell','rm',driver.REMOTE+'/'+path.name);path.unlink()
restore.rmdir()
print('VERIFIED uncontaminated 1536 holdout; original concurrent run retained',flush=True)
