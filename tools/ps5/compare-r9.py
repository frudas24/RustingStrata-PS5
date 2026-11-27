#!/usr/bin/env python3
"""Matched-row causal replay; never promotes the hybrid scheduler."""
import importlib.util, json, math, pathlib, struct, sys
spec=importlib.util.spec_from_file_location('r8', pathlib.Path(__file__).with_name('compare-r8.py'))
r8=importlib.util.module_from_spec(spec); spec.loader.exec_module(r8)
STAGES=('r9_gate_full','r9_up_full','r9_activation_gpu_inputs','r9_activation_own',
        'r9_down_gpu_activation','r9_down_cpu_silu','r9_down_own','r9_contract_self')
FIELDS=('hidden','gpu_down','weights')
def require(ok, msg):
    if not ok: raise ValueError(msg)
def bits(a,b):
    return len(a)==len(b) and all(struct.pack('<f',x)==struct.pack('<f',y) for x,y in zip(a,b))
def finite(a):
    return isinstance(a,list) and bool(a) and all(math.isfinite(v) for v in a)
def read_arm(root, arm):
    raw=(root/(arm+'.log')).read_text().splitlines()
    fixed=[json.loads(x.split(',',1)[1]) for x in raw if x.startswith('r9_fixed,')]
    combine=[(tuple(map(int,p[1:4])),json.loads(p[4])) for x in raw if x.startswith('r9_combine,') for p in [x.split(',',4)]]
    writes=[x.split(',')[1:] for x in raw if x.startswith('r9_write,')]
    stages={k:[] for k in STAGES}
    for x in raw:
        if not x.startswith('r7_r4_stage,'): continue
        p=x.split(',')
        if len(p)==11 and p[6] in stages:
            n, different, err, denom=int(p[7]),int(p[8]),float(p[9]),float(p[10])
            require(n>0 and 0<=different<=n and math.isfinite(err) and math.isfinite(denom) and min(err,denom)>=0,'invalid stage '+p[6])
            stages[p[6]].append(dict(n=n,different=different,max_abs=err,denom=denom))
    require(len(fixed)==len(combine)==6,arm+' must exercise all six prompts once')
    require(all(len(v)==6 for v in stages.values()),arm+' incomplete stages')
    require(all(v['different']==0 for v in stages['r9_contract_self']),arm+' inconsistent contract decomposition')
    require(len(writes)==(0 if arm=='shadow' else 6),arm+' wrong number of writes')
    for i,(f,(key,vec)) in enumerate(zip(fixed,combine)):
        require(key==(f['layer'],f['pos'],f['token']) and finite(vec),arm+' invalid combine boundary')
        require(all(finite(f[k]) for k in FIELDS),arm+' invalid fixed vector')
        require(len(f['gpu_down'])==len(vec),arm+' wrong combine width')
        if arm!='shadow':
            w=writes[i]; require(w==[arm,str(f['layer']),str(f['pos']),str(f['slot']),str(len(vec))],arm+' wrong write identity')
    return fixed,combine,stages

def same_fixed(a,b):
    return a.keys()==b.keys() and all(bits(a[k],b[k]) if k in FIELDS else a[k]==b[k] for k in a)

def evaluate(root):
    root=pathlib.Path(root)
    logits={name:r8.load(root,name,'gpu',False) for name in ('gpu','shadow','legacy','contract')}
    require(all(bits(a,b) for a,b in zip(logits['gpu'],logits['shadow'])), 'shadow changed GPU logits')
    f,c,s=read_arm(root,'shadow')
    result={'unique_reference_logits':len({tuple(v) for v in logits['gpu']}),'matched_cases':len(f),'stages':s,'arms':{},'production_promotion':False}
    for name in ('legacy','contract'):
        ff,cc,ss=read_arm(root,name)
        require(all(same_fixed(a,b) for a,b in zip(f,ff)),name+' fixed input/route/GPU row/weights changed')
        require(ss==s,name+' local diagnostic metrics changed before replacement')
        combined=[max(abs(x-y) for x,y in zip(a,b,strict=True)) for (_,a),(_,b) in zip(c,cc)]
        result['arms'][name]={'combine_max_abs':combined,'logits':[r8.metrics(a,b) for a,b in zip(logits['gpu'],logits[name])]}
    return result
if __name__=='__main__':
    root=pathlib.Path(sys.argv[1]); result=evaluate(root)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['arms'],indent=2)); print('Matched causal replay complete; never promotes R4.')
