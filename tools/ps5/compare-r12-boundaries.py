#!/usr/bin/env python3
"""Full-chunk, raw-FP32 boundary comparison after the independent R10 controls pass."""
import argparse
import json
import math
from pathlib import Path
import struct

STAGES=('layer_input','mixer_norm','mixer_residual','ffn_norm','layer_output')
CONTROLS=('sham','replay-shadow')
CANDIDATES=('contract','replay-contract')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read(path):
    rows={}
    with path.open() as f:
        for line in f:
            if not line.startswith('r12_boundary,'):
                continue
            _,layer,pos,stage,tokens,width,data=line.split(',',6)
            layer,pos,tokens,width=map(int,(layer,pos,tokens,width))
            key=(pos,layer,stage)
            require(key not in rows and stage in STAGES,'duplicate/unknown boundary')
            values=json.loads(data)
            require(0<tokens*width<=262144 and tokens>0 and width>0 and len(values)==tokens*width,'boundary shape')
            require(all(type(v) is int and 0<=v<=0xffffffff and v&0x7f800000!=0x7f800000 for v in values),'invalid FP32 bits')
            rows[key]=(tokens,width,values)
    require(rows,'missing boundaries')
    return rows


def compare(ref, other):
    require(list(ref)==list(other),'boundary coverage/order changed')
    out=[]
    for key,(tokens,width,a) in ref.items():
        t,w,b=other[key]
        require((tokens,width)==(t,w),'boundary shape changed')
        changed=0; maximum=0.; token=None
        for i,(x,y) in enumerate(zip(a,b,strict=True)):
            if x!=y:
                changed+=1
                d=abs(struct.unpack('<f',struct.pack('<I',x))[0]-struct.unpack('<f',struct.pack('<I',y))[0])
                if d>maximum: maximum=d;token=i//width
        out.append(dict(pos=key[0],layer=key[1],stage=key[2],changed=changed,max_abs=maximum,worst_token=token))
    return out


def evaluate(root):
    # Require successful prior R10 comparison (including raw GPU/Shadow/Sham controls).
    prior=json.loads((root/'summary.json').read_text())
    require(prior.get('production_promotion') is False and set(prior['arms'])=={'shadow','sham','replay-shadow','contract','replay-contract'},'R10 summary missing')
    ref=read(root/'shadow.log'); keys=list(ref)
    pos,layer,_=keys[0]
    meta=json.loads((root/'gpu.json').read_text())['r10']
    # R10 fixture reports all MoE layers; Qwen3.6 uses a MoE in each traced layer.
    end=min(layer+8,max(meta['layers']))
    require(keys==[(pos,l,s) for l in range(layer,end+1) for s in STAGES],'incomplete reference stage coverage')
    require(all(t==len(meta['tokens'][pos:pos+meta['chunk']]) for t,_,_ in ref.values()),'trace prompt shape differs')
    for arm in CONTROLS:
        require(not any(r['changed'] for r in compare(ref,read(root/f'{arm}.log'))),arm+' changed raw boundary bits')
    result={}
    for arm in CANDIDATES:
        rows=compare(ref,read(root/f'{arm}.log'))
        require(not any(r['changed'] for r in rows[:4]),arm+' changed before intervention')
        result[arm]=rows
    return {'production_promotion':False,'controls':'bitwise identical','scope':'all tokens, intervention chunk only', 'arms':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    result=evaluate(a.root)
    (a.root/'r12-boundaries.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('R12 full-chunk boundary controls PASS; no production promotion.')

if __name__=='__main__':main()
