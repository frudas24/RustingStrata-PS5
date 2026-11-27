#!/usr/bin/env python3
"""Independent geometry/reference checks, not execution of Rust or GLSL."""
import random, struct, math
r=random.Random(20261126)
for _ in range(6000):
    t=r.randint(1,40); g=r.randint(1,40); nkv=r.randint(1,4); tasks=r.choice((8,16,32))
    hper=min(g,tasks); bq=tasks//hper; hg=(g+hper-1)//hper
    seen=set()
    for x in range((t+bq-1)//bq):
        for y in range(nkv*hg):
            kvh,hgroup=divmod(y,hg)
            for task in range(tasks):
                hl,ql=divmod(task,bq); q=x*bq+ql; h=hgroup*hper+hl
                if q<t and hl<hper and h<g:
                    coord=q,kvh*g+h
                    assert coord not in seen; seen.add(coord)
    assert len(seen)==t*g*nkv
for _ in range(100000):
    pos=r.randrange(0,131072); q0=r.randrange(0,512); n=r.randrange(1,33); off=r.randrange(0,512)
    win=r.choice((0,1,2,31,32,33,127,128,129,r.randrange(1,140000)))
    hd=r.choice((64,128,256)); wg=r.choice((256,512,1024)); bk=wg*8//hd
    earliest=pos+off+q0; last=earliest+n-1
    start=max(0,earliest-win+1) if win else 0; tile_start=start//bk*bk
    assert tile_start<=start and start-tile_start<bk
    for q in (earliest,last):
        live=max(0,q-win+1) if win else 0
        assert max(tile_start,live)==live
        if tile_start: assert win and tile_start-1+win<=q
for wg in (256,512,1024):
    for hd in (64,128,256):
        elems=wg*8; seen=set()
        for tid in range(wg):
            for i in range(2*tid,elems,2*wg):
                assert i//hd==(i+1)//hd and i//32==(i+1)//32
                assert i not in seen and i+1 not in seen; seen.update((i,i+1))
        assert len(seen)==elems
def half(v): return struct.unpack('<e',struct.pack('<e',v))[0]
def bits(v): return struct.pack('<e',v)
for mode in ('q8','q4','q2'):
    for _ in range(20000):
        hd=r.choice((64,128,256)); stride={'q8':hd+hd//16,'q4':hd//2+hd//16,'q2':hd//4+hd//8}[mode]
        payload={'q8':hd,'q4':hd//2,'q2':hd//4}[mode]
        scales=[half(math.ldexp(r.uniform(0.5,1),r.randrange(-20,4))) for _ in range(hd//32)]
        mins=[half(r.uniform(-2,2)) for _ in scales]
        row=bytearray(r.randbytes(payload))+b''.join(map(bits,scales))+(b''.join(map(bits,mins)) if mode=='q2' else b'')
        assert len(row)==stride
        w=r.randrange(hd//2)*2; d=struct.unpack_from('<e',row,payload+w//32*2)[0]
        mn=struct.unpack_from('<e',row,payload+hd//16+w//32*2)[0] if mode=='q2' else 0
        def scalar(j):
            if mode=='q8': q=struct.unpack_from('<b',row,j)[0]
            elif mode=='q4': q=((row[j//2]>>(4*(j%2)))&15)-8
            else: q=(row[j//4]>>(2*(j%4)))&3
            return bits(q*d+mn if mode=='q2' else q*d)
        if mode=='q8': a,b=struct.unpack_from('<bb',row,w)
        elif mode=='q4': z=row[w//2]; a,b=(z&15)-8,(z>>4)-8
        else: z=(row[w//4]>>(2*(w&2)))&15; a,b=z&3,z>>2
        assert scalar(w)+scalar(w+1)==bits(a*d+mn if mode=='q2' else a*d)+bits(b*d+mn if mode=='q2' else b*d)
print('PASS: 6,000 dispatch geometries; 100,000 window bounds; 9 pair geometries; 60,000 reference quantized pairs')
