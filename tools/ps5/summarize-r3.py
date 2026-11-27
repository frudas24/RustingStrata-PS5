#!/usr/bin/env python3
import json, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
groups = {'base': ('base1','base2'), 'import': ('import1','import2')}
results = {}
for group, names in groups.items():
    for name in names:
        log = (root / (name+'.log')).read_text()
        nums = [int(n) for n in re.findall(r'PS5 R3: imported_bytes=(\d+)', log)]
        if group == 'import' and not any(nums):
            sys.exit(f'{name}: NO ACTUAL IMPORT — not a zero-copy benchmark')
        if group == 'base' and any(nums):
            sys.exit(f'{name}: baseline unexpectedly imported weights')
        print(f'{name}: cumulative imported payload {sum(nums)} bytes')
        rows = json.loads((root / (name+'.json')).read_text())['results']
        assert rows, name
        for r in rows:
            p, d = float(r['prefill_tok_s']['mean']), float(r['decode_tok_s']['mean'])
            assert p > 0 and d > 0, (name,r)
            results.setdefault((group,r['pp']), []).append((p,d))
print('pp     baseline prefill    import prefill     delta    baseline decode    import decode     delta')
for pp in sorted({pp for _,pp in results}):
    a,b = results['base',pp],results['import',pp]
    assert len(a)==len(b)==2, pp
    bp,bd = map(statistics.mean,zip(*a)); ip,id_ = map(statistics.mean,zip(*b))
    print(f'{pp:5} {bp:16.2f} {ip:17.2f} {100*(ip/bp-1):+8.2f}% {bd:16.2f} {id_:16.2f} {100*(id_/bd-1):+8.2f}%')
