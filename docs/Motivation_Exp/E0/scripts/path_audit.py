#!/usr/bin/env python3
"""Reduce detailed scheduler/FastRPC logs without treating placement as execution."""
import collections
import gzip
import json
import pathlib
import re
OUT=pathlib.Path(__file__).resolve().parents[1]
def read(path):
    if path.exists(): return path.read_text(errors='replace')
    return gzip.open(str(path)+'.gz','rt',errors='replace').read()
def main():
    results={}
    for name in ['cpu-profile','opencl-profile','htp-profile']:
        folder=OUT/'runs'/name
        if not (folder/'server.log').exists() and not (folder/'server.log.gz').exists(): continue
        log=read(folder/'server.log')
        request_start=next((i for i,l in enumerate(log.splitlines(),1) if 'new prompt,' in l and 'task.n_tokens = 128' in l),None)
        request_log='\n'.join(log.splitlines()[request_start-1:]) if request_start else ''
        groups=collections.defaultdict(collections.Counter); examples={}
        for number,line in enumerate(log.splitlines(),1):
            # Exclude worst-case graph reservation at model/context startup.
            if request_start is None or number<request_start: continue
            m=re.search(r'node #\s*\d+ \(\s*([^)]*)\):\s*(\S+)\s+\([^)]*\) \[\s*(\S+)',line)
            if not m: continue
            op,tensor,backend=m.groups()
            if 'shexp' in tensor or 'she' in tensor: group='shared_experts'
            elif 'exps' in tensor or 'MUL_MAT_ID' in op: group='routed_experts'
            elif any(s in tensor for s in ['attn','Qcur','Kcur','Vcur','kq','KQ','rope']): group='attention'
            else: continue
            groups[group][backend]+=1
            examples.setdefault((group,backend),{'line':number,'text':line})
        mapped={}; map_peak=0; map_events=0
        kernel_counts=collections.Counter()
        group_kernels=collections.defaultdict(collections.Counter)
        for number,line in enumerate(log.splitlines(),1):
            m=re.search(r'(HTP\S+) (unmapped|mapped) buffer: base (\S+) size (\d+) fd (\d+)',line)
            if m:
                sess,kind,base,size,fd=m.groups(); key=(sess,base,fd)
                if kind=='mapped': mapped[key]=int(size); map_events+=1
                else: mapped.pop(key,None)
                map_peak=max(map_peak,sum(mapped.values()))
            # Keep lifetime mappings, but exclude the two-token startup capability
            # probe from completed request-kernel and fusion statistics.
            if request_start is None or number<request_start: continue
            if 'profile-op ' in line:
                k=re.search(r'\|(h(?:mx|vx)-[^ |]+)',line)
                if k:
                    kernel_counts[k[1]]+=1
                    if 'MUL_MAT_ID' in line: group='routed_experts'
                    elif 'shexp' in line: group='shared_experts'
                    elif 'attn' in line or 'FLASH_ATTN' in line: group='attention'
                    else: group='other'
                    group_kernels[group][k[1]]+=1
        result={
            'request_start_line':request_start,
            'scheduler_scope':'assignments after actual 128-token request starts; excludes load-time graph reservation; graph reuse means node counts are not operation invocation counts',
            'scheduler_node_counts':{g:dict(c) for g,c in groups.items()},
            'examples':[{ 'group':g,'backend':b,**v } for (g,b),v in examples.items()],
            'htp_kernel_completed_profile_counts':dict(kernel_counts),
            'htp_profile_scope':'actual 128-token request only; excludes startup sequence-removal probe',
            'htp_group_kernel_completed_profile_counts':{g:dict(c) for g,c in group_kernels.items()},
            'htp_MUL_MAT_ID_NX_fusion_log_hits':len(re.findall(r'fused MUL_MAT_ID_NX',request_log)),
            'htp_MUL_MAT_ID_NX_completed_profile_count':len(re.findall(r'profile-op MUL_MAT_ID_NX\|',request_log)),
            'htp_MUL_MAT_ID_NX_lifetime_completed_profile_count':len(re.findall(r'profile-op MUL_MAT_ID_NX\|',log)),
            'htp_mapping_peak_bytes':map_peak if map_events else None,
            'mapping_definition':'peak sum of distinct simultaneously logged HTP mappings; virtual mapped capacity, not total RAM or exclusive physical ownership',
            'model_buffers':re.findall(r'.*model buffer size.*',log),
            'kv_evidence':re.findall(r'.*(?:KV.*buffer size|K \(f16\)|V \(f16\)|n_ctx_per_seq|n_batch\s*=|n_ubatch\s*=).*',log),
            'source_log':str((folder/'server.log').relative_to(OUT)),
        }
        results[name]=result
    (OUT/'path-audit.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:{x:v[x] for x in ['scheduler_node_counts','htp_kernel_completed_profile_counts','htp_MUL_MAT_ID_NX_fusion_log_hits','htp_mapping_peak_bytes']} for k,v in results.items()},indent=2))
if __name__=='__main__': main()
