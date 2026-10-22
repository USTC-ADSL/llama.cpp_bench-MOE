#!/usr/bin/env python3
"""Remove only this experiment's device directory after verified archival."""
import json
import pathlib
from run import TASK,REMOTE,adb
from archive import sha
def main():
    frozen=json.loads((TASK/'validation-frozen.json').read_text())
    archived=json.loads((TASK/'precleanup-archive-record.json').read_text())
    assert archived['all_member_hashes_verified'] and sha(pathlib.Path(archived['path']))==archived['sha256']
    assert adb('get-state')=='device'
    assert not adb('shell','pidof fullmodel-oracle || true')
    assert adb('shell','cat /proc/sys/kernel/random/boot_id')==frozen['boot_id']
    stat_command="stat -c '%n %s %Y' /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf"
    before=adb('shell',stat_command);assert before==frozen['model_stat']
    assert REMOTE=='/data/local/tmp/llama-joint-scheduler-20261022'
    size=adb('shell','du','-sk',REMOTE)
    adb('shell','rm','-rf',REMOTE)
    assert adb('shell','test ! -e '+REMOTE+' && echo absent')=='absent'
    after=adb('shell',stat_command);assert before==after
    (TASK/'cleanup-record.json').write_text(json.dumps({'remote_removed':REMOTE,'remote_size_before':size,
        'model_stat_before':before,'model_stat_after':after,'boot_id':frozen['boot_id'],
        'precleanup_archive_sha256':archived['sha256'],'no_fullmodel_process':True},indent=2)+'\n')
    print('Removed experiment directory; model unchanged.')
if __name__=='__main__':main()
