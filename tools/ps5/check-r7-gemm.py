#!/usr/bin/env python3
"""Independent SM2 dense output geometry and LDS arithmetic (not numerical shader parity)."""
from collections import Counter
import random

# Each wave computes 16 positions by 32 weight rows using two 16-column fragments.
coords = [(16*(w&3)+(lane>>2)+8*(i>>2),
           32*(w>>2)+16*nt+2*(lane&3)+(i&1)+8*((i>>1)&1))
          for w in range(8) for lane in range(32) for nt in range(2) for i in range(8)]
assert len(coords)==4096 and len(set(coords))==4096
assert set(coords)=={(r,c) for r in range(64) for c in range(64)}
# Clip final tiles exactly as shader does; check coverage with nonmultiple dimensions.
rng=random.Random(20261127)
for _ in range(256):
    rows, nd = rng.randint(1,193), rng.randint(1,385)
    counts=Counter((m+r,n+c) for m in range(0,nd,64) for n in range(0,rows,64)
                   for r,c in coords if m+r<nd and n+c<rows)
    assert len(counts)==rows*nd and max(counts.values())==1
for xm in (64,128):
    for pw in (20,36):
        # xq,wq,xd,xsum,wsc,wmn
        lds=4*(2*xm*20+2*64*pw+2*xm*2+2*xm*4+2*64*4*2)
        assert lds=={(64,20):27648,(64,36):35840,(128,20):40960,(128,36):49152}[xm,pw]
        # Last row, last word in the X ping-pong buffer.
        assert ((1*xm+xm-1)*20+15) < 2*xm*20
print('PASS: 4096 unique outputs, 256 clipped shapes, four LDS bounds; physical parity pending')
