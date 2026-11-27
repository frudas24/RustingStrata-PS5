#!/usr/bin/env python3
"""Validate equivalent model/settings/workloads, positive finite rates and candidate dispatch."""
import json, math, pathlib, statistics, sys
root=pathlib.Path(sys.argv[1]); data={}; reference=None; prompts=None
expected=set(map(int,sys.argv[2].split(','))) if len(sys.argv)>2 else None
for wg in (0,256,512,1024):
    data[wg]={}
    for rep in 'ab':
        arm=f'{wg}{rep}'; d=json.loads((root/f'{arm}.json').read_text())
        assert d['schema']=='strata.bench.v1',arm
        meta=tuple(d[k] for k in ('backend','device','model','ctx','chunk','kv','mtp','tg','reps'))
        if reference is None: reference=meta
        assert meta==reference, f'{arm}: incompatible settings'
        assert 'vulkan' in str(d['backend']).lower(),f'{arm}: not Vulkan'
        if wg: assert f'PS5 R5 attention dispatch: attn_prefill_ps5_{wg}' in (root/f'{arm}.log').read_text(),f'{arm}: R5 not exercised'
        rows=d['results']; p={x['pp'] for x in rows}; assert p and len(p)==len(rows),arm
        if expected is not None: assert p==expected, f'{arm}: did not measure all requested prompts'
        if prompts is None: prompts=p
        assert p==prompts, f'{arm}: mismatched prompt lengths'
        for row in rows:
            vals=[float(row[k]['mean']) for k in ('prefill_tok_s','decode_tok_s')]
            assert all(math.isfinite(v) and v>0 for v in vals),f'{arm}: invalid rate'
            data[wg].setdefault(row['pp'],[]).append(vals)
print('wg       pp     prefill tok/s     change     decode tok/s     change')
for pp in sorted(prompts):
    base=[statistics.mean(v) for v in zip(*data[0][pp])]
    for wg in data:
        p,d=map(statistics.mean,zip(*data[wg][pp]))
        print(f'{wg:4} {pp:8} {p:17.2f} {100*(p/base[0]-1):+9.1f}% {d:16.2f} {100*(d/base[1]-1):+9.1f}%')
