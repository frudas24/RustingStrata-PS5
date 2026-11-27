#!/usr/bin/env python3
"""Full-chunk, raw-FP32 boundary comparison after the independent R10 controls pass."""
import argparse
import importlib.util
import json
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
        changed=0; maximum=0.; token=None; changed_tokens=set()
        for i,(x,y) in enumerate(zip(a,b,strict=True)):
            if x!=y:
                changed+=1; changed_tokens.add(i//width)
                if token is None: token=i//width
                d=abs(struct.unpack('<f',struct.pack('<I',x))[0]-struct.unpack('<f',struct.pack('<I',y))[0])
                if d>maximum: maximum=d;token=i//width
        out.append(dict(pos=key[0],layer=key[1],stage=key[2],changed=changed,changed_tokens=sorted(changed_tokens),max_abs=maximum,worst_token=token))
    return out


def bind_to_existing(r10, root, arm, rows, target):
    # Tie each full-chunk capture to the independently logged R9/R10 tracked row.
    tok=target['token']
    def fpbits(values):
        return [struct.unpack('<I',struct.pack('<f',v))[0] for v in values]
    def check(key,values):
        if key not in rows: return
        t,w,raw=rows[key]
        require(tok<t and len(values)==w and raw[tok*w:(tok+1)*w]==fpbits(values),
                arm+' boundary disagrees with independent tracked row '+str(key))
    for route in r10.read_routes(root,arm):
        r=route['effective'];check((r['pos'],r['layer'],'ffn_norm'),route['hidden'])
    for layer,pos,vec in r10.layers(root,arm):
        check((pos,layer,'layer_output'),vec)
    _,combine,_=r10.r9.read_arm(root,arm,1,r10.ARMS[arm]); coord,vec=combine[0]
    check((coord[1],coord[0],'layer_output'),vec)


def evaluate(root):
    # Revalidate the R10 controls rather than trusting a possibly stale summary.
    spec=importlib.util.spec_from_file_location('r10',Path(__file__).with_name('compare-r10.py'))
    r10=importlib.util.module_from_spec(spec);spec.loader.exec_module(r10)
    prior=r10.evaluate(root)
    fixed,_,_=r10.r9.read_arm(root,'shadow',1)
    target=fixed[0]
    require(prior.get('production_promotion') is False and set(prior['arms'])=={'shadow','sham','replay-shadow','contract','replay-contract'},'R10 summary missing')
    ref=read(root/'shadow.log'); keys=list(ref)
    pos,layer=target['pos'],target['layer']
    width=len(target['hidden'])
    meta=json.loads((root/'gpu.json').read_text())['r10']
    # R10 fixture reports all MoE layers; Qwen3.6 uses a MoE in each traced layer.
    end=min(layer+8,max(meta['layers']))
    require(keys==[(pos,l,s) for l in range(layer,end+1) for s in STAGES],'incomplete reference stage coverage')
    require(all((t,w)==(target['chunk'],width) for t,w,_ in ref.values()),'trace prompt shape differs')
    bind_to_existing(r10,root,'shadow',ref,target)
    for arm in CONTROLS:
        captured=read(root/f'{arm}.log')
        require(not any(r['changed'] for r in compare(ref,captured)),arm+' changed raw boundary bits')
        bind_to_existing(r10,root,arm,captured,target)
    result={}
    initial={}
    for arm in CANDIDATES:
        captured=read(root/f'{arm}.log')
        rows=compare(ref,captured)
        require(not any(r['changed'] for r in rows[:4]),arm+' changed before intervention')
        key=(pos,layer,'layer_output')
        a,b=ref[key][2],captured[key][2]
        start=target['token']*width;end=start+width
        require(a[:start]==b[:start] and a[end:]==b[end:],arm+' changed an unselected token at intervention')
        bind_to_existing(r10,root,arm,captured,target)
        initial[arm]=b
        result[arm]=rows
    require(initial['contract']==initial['replay-contract'],'initial full chunk differs under replay')
    return {'production_promotion':False,'controls':'bitwise identical','scope':'all tokens, intervention chunk only', 'arms':result}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    result=evaluate(a.root)
    (a.root/'r12-boundaries.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('R12 full-chunk boundary controls PASS; no production promotion.')

if __name__=='__main__':main()
