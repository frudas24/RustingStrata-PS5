#!/usr/bin/env python3
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); names = ('r1','r2-64','r2-128','r2-small','r2-combined-64','r2-combined-128')
base = {}; failed = False
print('case                 pp    prefill tok/s     delta     decode tok/s     delta')
for name in names:
    try:
        rows = json.loads((root / (name + '.json')).read_text())['results']
        if not rows: raise ValueError('empty results')
        for r in rows:
            pp = r['pp']; p = float(r['prefill_tok_s']['mean']); t = float(r['decode_tok_s']['mean'])
            if name == 'r1': base[pp] = (p,t)
            bp,bt = base.get(pp,(0,0))
            delta = lambda x,b: f'{100*(x/b-1):+7.1f}%' if b>0 else '      n/a'
            print(f'{name:20} {pp:5} {p:15.2f} {delta(p,bp):>9} {t:16.2f} {delta(t,bt):>9}')
    except (OSError, ValueError, TypeError, KeyError) as e:
        print(f'{name}: FAILED / {e}', file=sys.stderr); failed = True
sys.exit(int(failed))
