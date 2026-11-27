#!/usr/bin/env python3
"""Fail closed if the selected routed row was never captured. Metrics are diagnostic, not a parity gate."""
import json, math, sys
from pathlib import Path
KEYS={'cpu_act_vs_gpu_act','cpu_down_vs_gpu_down','cpu_cross_vs_gpu_down','cpu_grouped_vs_cpu_pair','float_act_vs_gpu_act','float_cross_vs_gpu_down','cpu_gate_vs_float_gate','cpu_up_vs_float_up'}
def check(lines, target):
    records=[json.loads(s.split('PS5_VENUS_TRACE ',1)[1]) for s in lines if 'PS5_VENUS_TRACE ' in s]
    if len(records)!=1: raise ValueError(f'expected exactly one trace record, got {len(records)}')
    r=records[0]
    if r['schema']!=1 or tuple(r[k] for k in ('layer','position','rank'))!=target: raise ValueError('wrong trace target')
    if set(r['comparisons'])!=KEYS: raise ValueError('missing comparisons')
    for k,m in r['comparisons'].items():
        if m['finite'] is not True or m['n']<=0: raise ValueError(f'nonfinite or empty {k}')
        for f in ('max_abs','rmse','reference_max_abs','relative_max','relative_l2'):
            x=m[f]
            if x is None and f.startswith('relative_'): continue
            if not isinstance(x,(float,int)) or not math.isfinite(x) or x<0: raise ValueError(f'invalid {k}/{f}')
    return r
if __name__=='__main__':
    r=check(Path(sys.argv[1]).read_text().splitlines(),tuple(map(int,sys.argv[2].split(':'))))
    print(json.dumps(r['comparisons'],indent=2)); print('PASS: actual target captured; no hybrid parity claim')
