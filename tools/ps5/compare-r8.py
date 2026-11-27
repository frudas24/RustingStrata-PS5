#!/usr/bin/env python3
"""Compare complete logits of three immutable R4 arms, plus instrumentation control. No promotion."""
import json, math, pathlib, struct, sys

def require(ok, msg):
    if not ok: raise ValueError(msg)

def load(root, name, mode, cpu):
    d=json.loads((root/(name+'.json')).read_text())
    require(d['mode']==mode, name+' wrong mode')
    trials, rows, mixed=d['stats']
    require(trials>0 and ((rows>0 and mixed>0) if cpu else rows==mixed==0), name+' missing coverage')
    vals=d['logits']; require(len(vals)==12 and all(v and all(math.isfinite(x) for x in v) for v in vals),name+' invalid logits')
    return vals

def metrics(a,b):
    require(len(a)==len(b),'logit length mismatch')
    span=max(a)-min(a); require(span>0,'zero baseline range')
    n=len(a); diff=max(abs(x-y) for x,y in zip(a,b))
    top=lambda v: sorted(range(n),key=lambda i:(-v[i],i))[:5]
    ma,mb=max(a),max(b); pa=[math.exp(x-ma) for x in a]; za=sum(pa); zb=sum(math.exp(x-mb) for x in b)
    kl=sum(p*(x-y) for p,x,y in zip(pa,a,b))/za + mb-ma+math.log(zb/za)
    return dict(relative_max=diff/span, max_abs=diff, argmax=top(a)[0]==top(b)[0], top5=top(a)==top(b), kl=max(0.,kl))

def evaluate(root):
    root=pathlib.Path(root)
    gpu=load(root,'gpu','gpu',False); legacy=load(root,'legacy','half',True); new=load(root,'contract','half',True)
    require('PS5 R8 exercised:' in (root/'contract.log').read_text(),'candidate did not execute')
    require('PS5 R8 exercised:' not in (root/'legacy.log').read_text(),'legacy contaminated')
    probe=load(root,'probe','gpu',False)
    require(all(struct.pack('<f',x)==struct.pack('<f',y) for a,b in zip(gpu,probe) for x,y in zip(a,b,strict=True)),'probe changed GPU logits')
    plog=(root/'probe.log').read_text()
    for stage in ('gate','up','down'):
        require(f',{stage}_contract_oracle,' in plog, 'missing probe stage '+stage)
    results={name:[metrics(a,b) for a,b in zip(gpu,arm)] for name,arm in [('legacy',legacy),('contract',new)]}
    summary={name:{'worst_relative':max(v['relative_max'] for v in rows),'argmax_matches':sum(v['argmax'] for v in rows),'top5_matches':sum(v['top5'] for v in rows),'max_kl':max(v['kl'] for v in rows),'range_pass':sum(v['relative_max']<=.05 for v in rows)} for name,rows in results.items()}
    c=summary['contract']; l=summary['legacy']; passed=c['range_pass']==12 and c['argmax_matches']==12 and c['top5_matches']==12 and c['max_kl']<=.01
    return dict(summary=summary, vectors=results, reference_distinct_vectors=len({tuple(v) for v in gpu}), hypothesis_improved=c['worst_relative']<l['worst_relative'] and c['max_kl']<l['max_kl'], strict_diagnostic_gate=passed, production_promotion=False)

if __name__=='__main__':
    root=pathlib.Path(sys.argv[1]); result=evaluate(root)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['summary'],indent=2)); print('Diagnostic gate:',result['strict_diagnostic_gate'],'(never promotes production)')
    sys.exit(0 if result['strict_diagnostic_gate'] else 1)
