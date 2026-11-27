#!/usr/bin/env python3
"""Compare full logits across identical inputs. A 5%-of-range gate is provisional, not bit parity for hybrid."""
import json, math, pathlib, struct, sys
root=pathlib.Path(sys.argv[1]); modes=('off','gpu','half','cpu','auto')
ref=None
for mode in modes:
    data=json.loads((root/(mode+'.json')).read_text()); assert data['mode']==mode
    trials,cpu,mixed=data['stats']; vals=data['logits']
    assert len(vals)==12 and all(v and all(math.isfinite(x) for x in v) for v in vals)
    if mode=='off':
        assert [trials,cpu,mixed]==[0,0,0]; ref=vals; continue
    assert trials>0 and (cpu==0 if mode=='gpu' else cpu>0)
    if mode=='half': assert mixed>0
    worst=0.
    for a,b in zip(vals,ref):
        assert len(a)==len(b)
        if mode=='gpu': assert all(struct.pack('<f',x)==struct.pack('<f',y) for x,y in zip(a,b)), 'GPU-only sync changed bits'
        span=max(b)-min(b); assert span>0
        diff=max(abs(x-y) for x,y in zip(a,b)); worst=max(worst,diff/span)
        assert diff<=.05*span, (mode,diff,span)
        best=max(range(len(a)),key=a.__getitem__)
        assert b[best]>=max(b)-2*diff
    print(f'PASS {mode}: {trials} trials, {cpu} CPU rows, {mixed} mixed; worst relative max drift={worst:.6f}')
