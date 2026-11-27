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
    return isinstance(a,list) and bool(a) and all(type(v) in (int,float) and math.isfinite(v) for v in a)
def read_arm(root, arm, expected=6, write_mode=None):
    mode=arm if write_mode is None else write_mode
    # R12 boundary lines can be megabytes each; R9 needs none of them.
    with (root/(arm+'.log')).open() as log:
        raw=[line.rstrip('\n') for line in log if line.startswith(('r9_fixed,','r9_combine,','r9_write,','r7_r4_stage,'))]
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
            stages[p[6]].append(dict(coordinates=list(map(int,p[1:6])),n=n,different=different,max_abs=err,denom=denom))
    require(len(fixed)==len(combine)==expected,arm+' wrong number of matched cases')
    require(all(len(v)==expected for v in stages.values()),arm+' incomplete stages')
    require(all(v['different']==0 for v in stages['r9_contract_self']),arm+' inconsistent contract decomposition')
    require(len(writes)==(0 if mode=='shadow' else expected),arm+' wrong number of writes')
    for i,(f,(key,vec)) in enumerate(zip(fixed,combine)):
        require(key==(f['layer'],f['pos'],f['token']) and finite(vec),arm+' invalid combine boundary')
        require(all(finite(f[k]) for k in FIELDS),arm+' invalid fixed vector')
        require(len(f['gpu_down'])==len(vec)==len(f['hidden']),arm+' wrong combine width')
        chunk, token, rank, expert, slot = (f[k] for k in ('chunk','token','rank','expert','slot'))
        ids, k = f['ids'], len(f['weights'])
        require(all(type(v) is int and v>=0 for v in (chunk,token,rank,expert,slot,f['layer'],f['pos']))
                and chunk>0 and token<chunk and rank<k and len(ids)==chunk*k
                and all(type(v) is int and v>=0 for v in ids),arm+' invalid fixed route shape')
        route=token*k+rank
        require(ids[route]==expert and slot==sum(e<expert for e in ids)+sum(e==expert for e in ids[:route]),arm+' expert/slot mismatch')
        coordinates=[f['layer'],chunk,expert,slot,token]
        require(all(rows[i]['coordinates']==coordinates for rows in stages.values()),arm+' stage coordinates do not match intervention')
        activation_width=stages['r9_gate_full'][i]['n']
        require(all(stages[name][i]['n']==activation_width for name in STAGES[:4])
                and all(stages[name][i]['n']==len(vec) for name in STAGES[4:]),arm+' stage width mismatch')
        if mode!='shadow':
            w=writes[i]; require(w==[mode,str(f['layer']),str(f['pos']),str(f['slot']),str(len(vec))],arm+' wrong write identity')
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
