#!/usr/bin/env python3
"""Payload-only estimate for uniform full-length K/V layers (not a model residency guarantee)."""
import argparse
p=argparse.ArgumentParser(description=__doc__)
for name in ('ctx','layers','kv-heads','head-dim'): p.add_argument('--'+name,type=int,required=True)
modes={'f16':(2,1),'q8':(17,16),'q4':(9,16),'q2':(3,8)}
p.add_argument('--k',choices=modes,help='independent K format (requires --v)')
p.add_argument('--v',choices=modes,help='independent V format (requires --k)')
a=p.parse_args()
if (a.k is None)!=(a.v is None): p.error('--k and --v must be specified together')
if min(a.ctx,a.layers,a.kv_heads,a.head_dim)<=0 or a.head_dim%32: p.error('positive dimensions and head-dim divisible by 32 required')
n=2*a.ctx*a.layers*a.kv_heads*a.head_dim
for mode,(num,den) in modes.items():
    b=n*num//den
    print(f'{mode:4} {b:14,d} bytes  {b/(1<<30):8.3f} GiB')
if a.k is not None:
    kn,kd=modes[a.k]; vn,vd=modes[a.v]
    kb=n//2*kn//kd; vb=n//2*vn//vd
    print(f'K={a.k} V={a.v}: {kb+vb:,} bytes ({(kb+vb)/(1<<30):.3f} GiB); K={kb:,} V={vb:,} bytes')
print('Excludes weights, activations, scratch, driver, alignment and OS. Hybrid/sliding layers require per-layer accounting; this engine may still allocate full-length caches.')
