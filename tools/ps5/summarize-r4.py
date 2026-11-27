#!/usr/bin/env python3
"""Reject inert experiments; compare off vs timing-control, half, auto. No claim of speedup without samples."""
import json, math, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
results = {}
for name in ('off1','gpu','half','auto1','auto2','off2'):
    log = (root / (name+'.log')).read_text()
    trials = [tuple(map(int,x)) for x in re.findall(r'PS5 R4 trial: layer=\d+ tokens=\d+ arm=(\d+) cpu_rows=(\d+) gpu_rows=(\d+) ns=(\d+)',log)]
    cpu = sum(x[1] for x in trials); mixed = sum(x[1]>0 and x[2]>0 for x in trials)
    if name.startswith('off'):
        if trials: sys.exit(f'{name}: unexpected R4 activity')
    elif not trials or any(a not in (0,1,2) or n <= 0 or c+g <= 0 for a,c,g,n in trials): sys.exit(f'{name}: no valid trials')
    elif name == 'gpu' and any(a or c for a,c,_,_ in trials): sys.exit('GPU control unexpectedly used CPU')
    elif name != 'gpu' and (not cpu or not mixed): sys.exit(f'{name}: no actual mixed CPU/GPU layer; increase max rows, reduce expert budget or use supported MoE')
    print(f'{name}: trials={len(trials)} cpu_rows={cpu} mixed={mixed}')
    data = json.loads((root / (name+'.json')).read_text())['results']
    rows = {}
    for r in data:
        pp = int(r['pp']); p = float(r['prefill_tok_s']['mean']); d = float(r['decode_tok_s']['mean'])
        if pp <= 0 or pp in rows or not all(math.isfinite(x) and x>0 for x in (p,d)): sys.exit(f'{name}: invalid benchmark record')
        rows[pp] = (p,d)
    if not rows or (results and set(rows)!=set(results['off1'])): sys.exit(f'{name}: mismatched pp records')
    results[name] = rows
print('pp | off pp/s | gpu control pp/s | half pp/s | auto pp/s | auto delta | off tg/s | auto tg/s')
for pp in sorted(results['off1']):
    b = [statistics.mean([results[n][pp][j] for n in ('off1','off2')]) for j in (0,1)]
    a = [statistics.mean([results[n][pp][j] for n in ('auto1','auto2')]) for j in (0,1)]
    print(f'{pp:5} {b[0]:10.2f} {results["gpu"][pp][0]:12.2f} {results["half"][pp][0]:12.2f} {a[0]:12.2f} {100*(a[0]/b[0]-1):+8.2f}% {b[1]:10.2f} {a[1]:10.2f}')
