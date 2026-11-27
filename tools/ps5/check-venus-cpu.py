#!/usr/bin/env python3
"""Model gate: fixed teacher tokens, full logits, top-1 and short greedy exactness."""
import json, math, sys
from pathlib import Path

def check(base, candidate):
    if base['greedy']!=candidate['greedy']: raise ValueError('greedy generation differs')
    a,b=base['logits'],candidate['logits']
    if not a or len(a)!=len(b): raise ValueError('trace count differs')
    worst=0.
    for i,(x,y) in enumerate(zip(a,b)):
        if not x or len(x)!=len(y) or not all(math.isfinite(z) for z in x+y): raise ValueError('invalid logits')
        if max(range(len(x)),key=x.__getitem__)!=max(range(len(y)),key=y.__getitem__): raise ValueError(f'top1 differs at {i}')
        d=max(abs(u-v) for u,v in zip(x,y)); worst=max(worst,d)
        if d>.05: raise ValueError(f'logit max abs drift {d} at {i} exceeds 0.05')
    return worst
if __name__=='__main__':
    a,b=(json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    print(f'PASS CPU model: {len(a["logits"])} fixed-token positions; max abs logit drift {check(a,b):.8g}; greedy identical')
