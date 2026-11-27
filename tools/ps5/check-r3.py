#!/usr/bin/env python3
"""Independent integer geometry model, not execution of Rust or Vulkan."""
import random
MAX = (1 << 64) - 1

def checked(x):
    return x if 0 <= x <= MAX else None

def window(base, file_len, offset, n, a):
    if a < 16 or a & (a-1) or n < 65536 or n % 4: return None
    p = checked(base+offset)
    if p is None or p % 16: return None
    end = checked(p+n)
    if end is None or checked(end+a-1) is None or checked(base+file_len) is None: return None
    lo, hi = p & ~(a-1), (end+a-1) & ~(a-1)
    if lo < base or hi > base+file_len: return None
    return lo-base, hi-lo, p-lo

rng = random.Random(20261126)
accepted=0
for i in range(100000):
    a=rng.choice([16,64,4096,65536,2**21]); base=rng.randrange(1,8192)*4096
    size=rng.randrange(1,2**22); off=rng.randrange(0,2**22)//16*16
    n=rng.randrange(0,2**21)//4*4
    actual=window(base,size,off,n,a)
    # Independent division-based oracle
    lo=((base+off)//a)*a; hi=((base+off+n+a-1)//a)*a
    valid=n>=65536 and base<=lo and hi<=base+size
    assert (actual is not None)==valid
    if actual:
        start,length,delta=actual; accepted+=1
        assert start+delta==off and delta+n<=length and start+length<=size
        assert (base+start)%a==0 and length%a==0 and (base+off)%16==0
assert window(MAX-15,65536,32,65536,4096) is None
assert window(0x10000,65568,32,65536,4096) is None
assert window(0x10000,0x40000,1,65536,4096) is None
assert window(0x10000,0x40000,32,65535,4096) is None
assert window(0x10000,0x40000,32,65536,4096)==(0,69632,32)
print(f'PASS: 100000 independent window geometry cases ({accepted} eligible), plus 5 boundaries')
