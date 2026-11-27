#!/usr/bin/env python3
"""CPU-only verification of dispatch geometry, logical lane mapping and LDS bounds; not shader execution."""
from collections import Counter
cases = 0
for wpb in (2,4,8):
    for rows in list(range(1,130))+[255,256,257,4095,4096,4097,32769]:
        for rpw in (1,2,3,4,8,16):
            n = (rows+wpb*rpw-1)//(wpb*rpw)
            got = Counter(r for group in range(n) for w in range(wpb)
                          for r in range(group*wpb*rpw+w,min(rows,(group+1)*wpb*rpw),wpb))
            assert len(got)==rows and all(got[i]==1 for i in range(rows))
            cases += 1
        # Dual matrices; relative rows are never lost at the boundary.
        for split in (0,1,rows//2,rows):
            got=[(r>=split,r-split if r>=split else r) for g in range((rows+wpb-1)//wpb) for w in range(wpb) if (r:=g*wpb+w)<rows]
            assert got==[(i>=split,i-split if i>=split else i) for i in range(rows)]
            cases += 1
print(f'PASS: {cases} GEMV/MoE dispatch and split cases')
# Small GEMM mapping: every group covers 32 token positions x 64 output rows; full 128x64 mapping as reference.
for xm in (32,128):
    pairs=Counter()
    for w in range(8):
        bm,bn=(16*(w&1),16*(w>>1)) if xm==32 else (32*(w>>1),32*(w&1))
        for mt in range(1 if xm==32 else 2):
            for nt in range(1 if xm==32 else 2):
                for lane in range(32):
                    g,t=divmod(lane,4)
                    for i in range(8):
                        r=bm+16*mt+g+8*(i>>2); n=bn+16*nt+2*t+(i&1)+8*((i>>1)&1)
                        assert 0<=r<xm and 0<=n<64
                        pairs[r,n]+=1
    assert len(pairs)==xm*64 and set(pairs.values())=={1}
    for n in range(1,66):
        for rows in (1,19,63,64,65,129):
            result=Counter((m*xm+r,b*64+c) for m in range((n+xm-1)//xm) for b in range((rows+63)//64) for r,c in pairs if m*xm+r<n and b*64+c<rows)
            assert len(result)==n*rows and set(result.values())=={1}
print('PASS: GEMM 32/128 tile bijections, 780 tail cases')
# Buffer sizes from gemmi.glsl declarations, in bytes (the parser gate checks compiled SPIR-V on target).
for pw in (20,36):
    def lds(xm): return 4*(2*xm*20+2*64*pw+2*xm*2+2*xm*4+2*64*4+2*64*4)
    old,new=lds(128),lds(32)
    assert (old,new)==((40960,20992) if pw==20 else (49152,29184))
    print(f'LDS PW={pw}: {old} -> {new} bytes; reduction {(1-new/old)*100:.2f}%')
print('NOTE: CPU geometry only; GLSL/SPIR-V validation and GPU parity require target toolchain.')
