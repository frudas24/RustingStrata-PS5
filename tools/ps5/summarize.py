#!/usr/bin/env python3
"""Print per-context throughput. Never treat a missing/failed case as zero tok/s."""
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); cases = ['upstream', 'control', 'guarded', 'gtt', 'mapped', 'combined']
print('case         pp     prefill tok/s     decode tok/s')
for name in cases:
    f = root / (name + '.json')
    if not f.exists():
        if (root / (name + '.log')).exists(): print(f'{name:10} FAILED / no JSON')
        continue
    try:
        d = json.loads(f.read_text())
        for r in d['results']:
            print(f"{name:10} {r['pp']:6} {r['prefill_tok_s']['mean']:17.2f} {r['decode_tok_s']['mean']:16.2f}")
    except (KeyError, ValueError, TypeError) as e:
        print(f'{name:10} unreadable benchmark: {e}')
