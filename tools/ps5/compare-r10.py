#!/usr/bin/env python3
"""Strict one-prompt frozen-route counterfactual. Never promotes R4."""
import importlib.util, json, math, pathlib, sys
spec=importlib.util.spec_from_file_location('r9',pathlib.Path(__file__).with_name('compare-r9.py'))
r9=importlib.util.module_from_spec(spec); spec.loader.exec_module(r9)
require=r9.require; bits=r9.bits
ARMS={'shadow':'shadow','sham':'sham','replay-shadow':'shadow','contract':'contract','replay-contract':'contract'}

def key(r): return r['pos'],r['layer']
def same(a,b):
    return a.keys()==b.keys() and all(bits(a[k],b[k]) if k=='weights' else a[k]==b[k] for k in a)
def read_routes(root, arm):
    with (root/(arm+'.log')).open() as f:
        rows=[json.loads(s.split(',',1)[1]) for s in f if s.startswith('r10_route,')]
    require(rows and len({key(x['effective']) for x in rows})==len(rows),arm+' missing or duplicate routes')
    mode='replay' if arm.startswith('replay-') else 'record'
    for row in rows:
        require(row['mode']==mode,arm+' wrong route mode')
        a,b=row['natural'],row['effective']
        for r in (a,b):
            k=r['top_k']; n=len(r['tokens']); ids=r['ids']; weights=r['weights']
            require(k>0 and n>0 and len(ids)==len(weights)==n*k and all(isinstance(x,int) and x>=0 for x in ids) and all(math.isfinite(x) and x>=0 for x in weights),arm+' invalid route')
            require(all(len(set(ids[i:i+k]))==k for i in range(0,len(ids),k)),arm+' duplicate expert')
        require(all(a[k]==b[k] for k in ('layer','pos','tokens','top_k')),arm+' route coordinate mismatch')
        require(len(row['raw_cutoff_gaps'])==len(a['tokens']) and all(x is None or (math.isfinite(x) and x>=0) for x in row['raw_cutoff_gaps']),arm+' invalid gap')
        if mode=='record': require(same(a,b),arm+' record modified route')
    return rows

def logits(root,arm):
    d=json.loads((root/(arm+'.json')).read_text()); t,c,m=d['stats']; v=d['logits']
    require(d['mode']=='gpu' and t>0 and c==m==0 and len(v)==2 and all(r9.finite(x) for x in v),arm+' invalid logits/coverage')
    return v

def layers(root,arm):
    with (root/(arm+'.log')).open() as f:
        rows=[(int(p[1]),int(p[2]),json.loads(p[3])) for s in f if s.startswith('r9_layer_out,') for p in [s.split(',',3)]]
    require(rows and len({(pos,layer) for layer,pos,_ in rows})==len(rows) and all(r9.finite(v) for _,_,v in rows),arm+' invalid layer outputs')
    return rows

def tape(root):
    rows=read_routes(root,'shadow'); out=root/'routes.json'
    out.write_text(json.dumps([x['effective'] for x in rows],allow_nan=False)+'\n')

def evaluate(root):
    baseline=logits(root,'gpu'); fixed,combine,stages=r9.read_arm(root,'shadow',1)
    routes=read_routes(root,'shadow'); ref=[x['effective'] for x in routes]; lay=layers(root,'shadow')
    target=key(fixed[0]); result={'production_promotion':False,'scope':'prefill routes only; fixed-token decode routes naturally','arms':{}}
    require(len(lay)==len(ref) and [(pos,layer) for layer,pos,_ in lay]==[key(r) for r in ref],'reference trace coverage mismatch')
    require(all(bits(a,b) for a,b in zip(baseline,logits(root,'shadow'),strict=True)),'shadow changed GPU')
    stored=json.loads((root/'routes.json').read_text()); require(len(stored)==len(ref) and all(same(a,b) for a,b in zip(stored,ref,strict=True)),'tape does not match shadow')
    for arm,mode in ARMS.items():
        f,c,s=r9.read_arm(root,arm,1,mode); require(r9.same_fixed(fixed[0],f[0]) and s==stages,arm+' changed intervention input')
        rr=read_routes(root,arm); ll=layers(root,arm); vv=logits(root,arm)
        require(len(rr)==len(ref)==len(ll),arm+' incomplete trace')
        control=mode in ('shadow','sham'); changed=[]; residual=[]
        for a,b,lc,la in zip(ref,rr,lay,ll,strict=True):
            n,e=b['natural'],b['effective']; require(all(n[k]==a[k] for k in ('layer','pos','tokens','top_k')),arm+' coordinate changed')
            require(lc[:2]==la[:2] and (la[1],la[0])==key(a),arm+' layer order changed')
            if arm.startswith('replay-') or control: require(same(a,e),arm+' effective route changed')
            if key(a)<=target or control: require(same(a,n),arm+' natural route changed before intervention/control')
            k=a['top_k']; tokens=[i for i in range(len(a['tokens'])) if a['ids'][i*k:(i+1)*k]!=n['ids'][i*k:(i+1)*k]]
            if tokens: changed.append({'pos':a['pos'],'layer':a['layer'],'tokens':tokens,'baseline_raw_gaps':[routes[len(residual)]['raw_cutoff_gaps'][i] for i in tokens], 'candidate_raw_gaps':[b['raw_cutoff_gaps'][i] for i in tokens]})
            d=max(abs(x-y) for x,y in zip(lc[2],la[2],strict=True)); residual.append({'pos':a['pos'],'layer':a['layer'],'max_abs':d})
            if control: require(bits(lc[2],la[2]),arm+' changed layer output')
        if control: require(all(bits(a,b) for a,b in zip(baseline,vv,strict=True)) and bits(combine[0][1],c[0][1]),arm+' changed GPU output')
        result['arms'][arm]={'natural_route_changes':changed,'layer_residuals':residual,'logits':[r9.r8.metrics(a,b) for a,b in zip(baseline,vv,strict=True)]}
    return result

if __name__=='__main__':
    root=pathlib.Path(sys.argv[2]); action=sys.argv[1]
    if action=='tape': tape(root)
    elif action=='compare':
        result=evaluate(root); (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({a:r['logits'] for a,r in result['arms'].items()},indent=2)); print('R10 controls PASS; no production promotion.')
    else: raise SystemExit('expected tape or compare')
