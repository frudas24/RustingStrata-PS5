#!/usr/bin/env python3
"""Payload-only estimate for uniform full-length K/V layers (not a model residency guarantee)."""
import argparse
p=argparse.ArgumentParser(description=__doc__)
for name in ('ctx','layers','kv-heads','head-dim'): p.add_argument('--'+name,type=int,required=True)
a=p.parse_args()
if min(a.ctx,a.layers,a.kv_heads,a.head_dim)<=0 or a.head_dim%32: p.error('positive dimensions and head-dim divisible by 32 required')
n=2*a.ctx*a.layers*a.kv_heads*a.head_dim
for mode,num,den in [('f16',2,1),('q8',17,16),('q4',9,16),('q2',3,8)]:
    b=n*num//den
    print(f'{mode:4} {b:14,d} bytes  {b/(1<<30):8.3f} GiB')
print('Excludes weights, activations, scratch, driver, alignment and OS. Hybrid/sliding layers require per-layer accounting; this engine may still allocate full-length caches.')
