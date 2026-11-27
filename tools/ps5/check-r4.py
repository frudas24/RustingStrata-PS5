#!/usr/bin/env python3
"""Independent CPU model of R4 policy/partition and scatter geometry, not execution of Rust/Vulkan."""
import random
rng=random.Random(20261126)
for case in range(20000):
    n=rng.randrange(1,129); tokens=rng.randrange(1,129); k=rng.randrange(1,min(n,8)+1)
    ids=[e for _ in range(tokens) for e in rng.sample(range(n),k)]
    counts=[0]*n
    for e in ids: counts[e]+=1
    off=[0]
    for c in counts: off.append(off[-1]+c)
    missed=[e for e,c in enumerate(counts) if c and rng.randrange(3)]
    resident=[e for e,c in enumerate(counts) if c and e not in missed]
    max_rows=rng.randrange(1,33); ne=rng.choice([512,1024,2048,4096]); ff=rng.choice([256,512,1024,2048]); budget=rng.randrange(1,65)<<20
    input_bytes=8*tokens*ne; per_pair=64*(ne+ff)+512
    eligible=[]; room=max(0,(budget-input_bytes)//per_pair)
    for e in sorted((e for e in missed if counts[e]<=max_rows),key=lambda e:(counts[e],e)):
        if counts[e]>room: break
        eligible.append(e); room-=counts[e]
    for arm in (0,1,2):
        cpu=sorted(eligible[:[0,(len(eligible)+1)//2,len(eligible)][arm]])
        gpu=resident+[e for e in missed if e not in cpu]
        if cpu: assert input_bytes+sum(counts[e] for e in cpu)*per_pair<=budget
        # Fill slot rows from distinct partitions. Scatter must restore the original slot index.
        out=[None]*len(ids)
        for device,es in [('cpu',cpu),('gpu',gpu)]:
            for e in es:
                for slot in range(off[e],off[e+1]):
                    assert out[slot] is None
                    out[slot]=(e,device)
        assert all(out[i][0]==e for e in range(n) for i in range(off[e],off[e+1]))
        assert all(e in gpu for e in resident)
print('PASS: 20,000 randomized router/cache/budget cases; 60,000 partitions and scatter checks')
# Independent convergence check of EWMA/exploration rule.
visits=0; avg=[None]*3; seen=[0]*3
for i in range(1024):
    a=next((a for a,v in enumerate(avg) if v is None),None)
    if a is None:
        if visits%16==15: a=visits//16%3
        else:
            a=0
            for j in (1,2):
                if avg[j]<.9*avg[a]: a=j
    visits+=1; seen[a]+=1; ns=[100,45,70][a]
    avg[a]=ns if avg[a] is None else .8*avg[a]+.2*ns
assert seen[1]>900 and min(seen)>0
print('PASS: deterministic exploration and EWMA convergence (1,024 synthetic observations)')
