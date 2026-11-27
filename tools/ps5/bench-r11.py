#!/usr/bin/env python3
"""R11 Qwen MoE candidate A/B/B/A; never promotes a production profile."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location('r6', Path(__file__).with_name('autotune-r6.py'))
r6 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r6)


def compare(runs, mark):
    a1, b1, b2, a2 = runs
    for run, expected in zip(runs, (False, True, True, False)):
        r6.require((mark in run['evidence']['marks']) == expected,
                   'R11 dispatch absent or leaked into control')
    r6.stable(a1['rates'], a2['rates'])
    r6.stable(b1['rates'], b2['rates'])
    for run in (b1, b2, a2):
        r6.match_signatures(a1['signatures'], run['signatures'])
    a, b = r6.average([a1, a2]), r6.average([b1, b2])
    score, worst = r6.metrics(b, a)
    worst = min(worst, r6.metrics(b1['rates'], a1['rates'])[1], r6.metrics(b2['rates'], a2['rates'])[1])
    import math, statistics
    phases={name:math.exp(statistics.mean(math.log(b[pp][i]/a[pp][i]) for pp in a)) for i,name in enumerate(('prefill','decode'))}
    return dict(score=score, worst=worst, phase_speedups=phases, baseline=a, candidate=b, accepted=score >= 1.03 and worst >= .95)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate', choices=('iq32','gemv1','gemv2','gemv4','gemv8','down1','down2','down4','down8'), required=True)
    p.add_argument('--profile-only', action='store_true', help='separate timestamps; never score or promote')
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--bin', type=Path, default=r6.DEFAULT_BIN)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--ctx', type=int, default=8192)
    p.add_argument('--pp', default='128,512,2048,4096')
    p.add_argument('--tg', type=int, default=32)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--min-ram-mb', type=int, default=512)
    p.add_argument('--threads', type=int, choices=(4,6,8,16), default=6)
    p.add_argument('--chunk', type=int, choices=(128,256,512,1024), default=512)
    p.add_argument('--kv', choices=('f16','q8','q4','q2'), default='q8')
    a = p.parse_args()
    s = r6.settings_ok(dict(ctx=a.ctx, pp=list(map(int,a.pp.split(','))), tg=a.tg, reps=a.reps, timeout=a.timeout, min_ram_mb=a.min_ram_mb, kv=a.kv))
    k = r6.knobs_ok({**r6.BASE, 'threads':a.threads, 'chunk':a.chunk})
    session = r6.Session(a.model, a.bin, s, a.out)
    ident = r6.identity(a.model, a.bin)
    r6.atomic_json(session.out/'identity.json', ident)
    if a.candidate == 'iq32': key, value, mark = 'STRATA_PS5_R11_IQ32', '1', 'gemm_f_dp_ps5_iq32'
    elif a.candidate.startswith('gemv'): key, value, mark = 'STRATA_PS5_R11_GEMV_RPW', a.candidate[-1], 'r11_gemv_rpw'
    else: key, value, mark = 'STRATA_PS5_R11_DOWN_RPW', a.candidate[-1], 'r11_down_rpw'
    reports = {}
    try:
        for task in ('prose','holdout'):
            runs = []
            for i, tile in enumerate((0,1) if a.profile_only else (0,1,1,0), 1):
                stem = f'{task}-{i}-tile{tile}'
                log, result = (session.out/(stem+x) for x in ('.log','.json'))
                env = r6.child_env(k)
                env[key] = value if tile else '0'
                if a.profile_only: env.update(STRATA_VK_PROF='1', STRATA_PS5_R11_PROFILE='1')
                cmd = [str(session.binary),'bench',*r6.engine_args(a.model,k,s),'--pp',a.pp,'--tg',str(s['tg']),'--reps',str(s['reps']),'--json-out',str(result)]
                if task == 'holdout': cmd += ['--prompt-file',str(session.holdout)]
                r6.atomic_json(session.out/(stem+'-request.json'), dict(cmd=cmd,knobs=k,candidate=a.candidate,enabled=bool(tile),profile_only=a.profile_only))
                print(stem, flush=True)
                memory = r6.run_child(cmd,env,log,s['timeout'],s['min_ram_mb'])
                rates, signatures = r6.checked_rates(json.loads(result.read_text()),k,s,a.model)
                r6.match_signatures(signatures,signatures)
                evidence = r6.evidence(log.read_text(errors='replace'),k)
                r6.require((mark in evidence['marks'])==bool(tile), 'candidate dispatch absent or leaked')
                run = dict(rates=rates,signatures=signatures,evidence=evidence,memory=memory)
                r6.atomic_json(session.out/(stem+'-checked.json'),run)
                runs.append(run)
            reports[task] = dict(profile_only=True, accepted=False) if a.profile_only else compare(runs, mark)
        r6.require(r6.identity(a.model,a.bin)==ident, 'model, binary or hardware identity changed')
        if a.profile_only:
            r6.atomic_json(session.out/'summary.json',dict(profile_only=True,accepted=False,reports=reports))
            print('Profiling only: no performance acceptance'); return 0
        accepted = all(x['accepted'] for x in reports.values())
        r6.atomic_json(session.out/'summary.json',dict(accepted=accepted,reports=reports))
        print('PASS: research candidate meets gates' if accepted else 'REJECT: no qualified improvement')
        return 0 if accepted else 1
    except (r6.Reject, KeyError, TypeError, ValueError, OSError, IndexError) as exc:
        r6.atomic_json(session.out/'summary.json',dict(accepted=False,error=str(exc),reports=reports))
        print(f'REJECT: {exc}',file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(main())
