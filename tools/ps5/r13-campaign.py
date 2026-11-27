#!/usr/bin/env python3
"""R13 physical A/B/B/A and quality campaign, with exact dispatch gates.
No profile promotion; all measurements use R11 IQ32 and explicit R12 tiles.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r6 = load('r6', 'autotune-r6.py')
quality = load('quality', 'quality-r11.py')
IQ2 = 'gemm_f_dp_ps5_iq32'
INT = {0: 'gemmi_dp_ps5', 32: 'gemmi_dp_ps5_iq32', 64: 'gemmi_dp_ps5_iq64'}
TYPES = {'r13_lut_iq3xxs', 'r13_lut_iq4xs'}
ROUTED = {t: {f'r13_int_{tag}_{fmt}' for fmt in ('iq3xxs', 'iq4xs')}
          for t, tag in ((0, 'base'), (32, '32'), (64, '64'))}
FALLBACK = {'r13_int_f16_iq3xxs', 'r13_int_f16_iq4xs'}
KNOWN = {IQ2, IQ2 + '_lut', *TYPES, INT[32], INT[64], *(x + '_lut' for x in INT.values()),
         *set().union(*ROUTED.values()), *FALLBACK}
# GTT residence is the only other R6 mark permitted by these fixed workloads.
# Reject unknown/other experimental markers rather than silently ignoring them.
ALLOWED_OTHER = {'gtt_nonlocal'}


def expected(tile, lut, test='model'):
    marks = set()
    if test in ('iq2', 'model'):
        marks.add(IQ2 + ('_lut' if lut in ('iq2', 'all') else ''))
    if test in ('int', 'model'):
        if lut in ('int', 'all'):
            marks.update({INT[tile] + '_lut', *TYPES})
        else:
            marks.update(ROUTED[tile])
            if tile:
                marks.add(INT[tile])
    return marks


def check_marks(log, tile, lut, test='model'):
    all_marks = r6.dispatch_marks(log)
    r6.require(not all_marks - KNOWN - ALLOWED_OTHER, f'unexpected dispatch markers: {sorted(all_marks - KNOWN - ALLOWED_OTHER)}')
    found = all_marks & KNOWN
    want = expected(tile, lut, test)
    r6.require(found == want, f'dispatch identity: expected {sorted(want)}, got {sorted(found)}')
    return sorted(found)


def check_lds(log):
    expected = {('IQ2', ''): 4096}
    expected.update({(str(t), k): 1024 for t in (18, 23) for k in
                     ('gemmi_dp_ps5_lut', 'gemmi_dp_ps5_iq32_lut', 'gemmi_dp_ps5_iq64_lut')})
    seen = set()
    for line in log.splitlines():
        if not line.startswith('R13 LDS '):
            continue
        fields = line.split()
        if len(fields) == 5 and fields[2] == 'IQ2':
            key, numbers = ('IQ2', ''), fields[3:]
        elif len(fields) == 6:
            key, numbers = tuple(fields[2:4]), fields[4:]
        else:
            raise r6.Reject('malformed LDS record')
        r6.require(key in expected and key not in seen, f'unexpected/duplicate LDS record: {key}')
        base, new = map(int, numbers)
        r6.require(0 < base and base + expected[key] <= new <= 65536, f'invalid LDS budget: {key}')
        seen.add(key)
    r6.require(seen == set(expected), 'missing LDS records')
    return len(seen)


def arm(base_tile, tile, lut, candidate):
    # A winning integer tile remains independently testable when every LUT loses.
    # Distinct filenames avoid overwriting the two quality arms when both use off.
    if not candidate: return base_tile, 'off', 'off'
    return tile, lut, 'candidate-off' if lut == 'off' else lut


def flags(tile, lut):
    return dict(STRATA_PS5_R11_IQ32='1', STRATA_PS5_R12_IQ_INT_TILE=str(tile), STRATA_PS5_R13_LUT=lut)


def profile_summary(text):
    """Allocation totals are cumulative, not live residency. CPU copy times exclude fences."""
    from collections import Counter, defaultdict
    allocations=defaultdict(lambda: dict(count=0,requested=0,allocated=0))
    copies=defaultdict(lambda: dict(count=0,bytes=0,ns=0))
    for role,requested,allocated,mt,heap,fl,cached in re.findall(r'^r13_mem_alloc,(\w+),requested=(\d+),allocated=(\d+),type=(\d+),heap=(\d+),flags=(0x[0-9a-f]+),host_cached=(true|false)$',text,re.M):
        row=allocations[f'{role}:type{mt}:heap{heap}:{fl}:cached={cached}'];row['count']+=1;row['requested']+=int(requested);row['allocated']+=int(allocated)
    for role,n,ns in re.findall(r'^r13_mem_copy,(\w+),bytes=(\d+),ns=(\d+)$',text,re.M):
        row=copies[role];row['count']+=1;row['bytes']+=int(n);row['ns']+=int(ns)
    q8=Counter(re.findall(r'^r13_q8,(\w+),rows=\d+,cols=\d+,stride=\d+,type=([^\s]+)$',text,re.M))
    r6.require(bool(allocations) and bool(copies) and bool(q8), 'missing diagnostic allocation/copy/Q8 coverage')
    return dict(allocations=dict(allocations),host_copies=dict(copies),q8={':'.join(k):v for k,v in sorted(q8.items())})


def compare(runs):
    a1, b1, b2, a2 = runs
    r6.stable(a1['rates'], a2['rates'])
    r6.stable(b1['rates'], b2['rates'])
    for run in runs[1:]:
        r6.match_signatures(a1['signatures'], run['signatures'])
    a, b = r6.average([a1, a2]), r6.average([b1, b2])
    score, worst = r6.metrics(b, a)
    pairs = [r6.metrics(b1['rates'], a1['rates']), r6.metrics(b2['rates'], a2['rates'])]
    worst = min(worst, *(p[1] for p in pairs))
    paired_scores = [p[0] for p in pairs]
    import math, statistics
    phases = {name: math.exp(statistics.mean(math.log(b[pp][i] / a[pp][i]) for pp in a)) for i, name in enumerate(('prefill', 'decode'))}
    return dict(score=score, worst=worst, paired_scores=paired_scores, phase_speedups=phases, baseline=a, candidate=b,
                accepted=score >= 1.03 and min(paired_scores) >= 1.03 and worst >= .95)


def run_campaign(a):
    s = r6.settings_ok(dict(ctx=a.ctx, pp=list(map(int, a.pp.split(','))), tg=a.tg, reps=a.reps,
                            timeout=a.timeout, min_ram_mb=a.min_ram_mb, kv=a.kv))
    k = r6.knobs_ok({**r6.BASE, 'threads': a.threads, 'chunk': a.chunk})
    session = r6.Session(a.model, a.bin, s, a.out)
    ident = r6.identity(a.model, a.bin)
    holdout_id = r6.file_id(session.holdout)
    r6.atomic_json(session.out/'identity.json', dict(environment=ident, holdout=holdout_id))
    reports = {}
    base_tile = a.tile if a.base_tile is None else a.base_tile
    try:
        r6.require(a.profile_only or a.lut != 'off' or base_tile != a.tile, 'no experimental change: use a different --base-tile with --lut off')
        if a.corpus:
            r6.require(128 <= a.ctx_len <= a.ctx and a.chunks > 0, 'invalid quality settings')
            corpus = r6.file_id(a.corpus)
            hashes = []
            for candidate in (False, True):
                tile, lut, stem = arm(base_tile, a.tile, a.lut, candidate)
                env = r6.child_env(k); env.update(flags(tile, lut))
                dump, log = session.out/f'{stem}.bin', session.out/f'{stem}.log'
                cmd = [str(session.binary), 'perplexity', *r6.engine_args(a.model,k,s),'--file',str(a.corpus.resolve()),'--ctx-len',str(a.ctx_len),'--chunks',str(a.chunks),'--kld-base',str(dump)]
                r6.atomic_json(session.out/f'{stem}-request.json',dict(cmd=cmd,flags=flags(tile,lut)))
                r6.require(corpus==r6.file_id(a.corpus), 'corpus changed before quality arm')
                memory=r6.run_child(cmd,env,log,s['timeout'],s['min_ram_mb'])
                r6.require(corpus==r6.file_id(a.corpus), 'corpus changed during quality arm')
                text=log.read_text(errors='replace'); r6.evidence(text,k); check_marks(text,tile,lut)
                quality.check_dump(dump,text,a.ctx_len,a.chunks)
                hashes.append(r6.digest(dump))
                r6.atomic_json(session.out/f'{stem}-memory.json',memory)
            r6.require(ident==r6.identity(a.model,a.bin) and corpus==r6.file_id(a.corpus),'identity changed')
            same=hashes[0]==hashes[1]
            r6.atomic_json(session.out/'summary.json',dict(encoded_identity=same,hashes=hashes,identity=ident,corpus=corpus,base_tile=base_tile,tile=a.tile,lut=a.lut,production_promotion=False))
            return 0 if same else 1
        for task in ('prose','holdout'):
            runs=[]
            for i,candidate in enumerate((False,True) if a.profile_only else (False,True,True,False),1):
                tile,lut,label=arm(base_tile,a.tile,a.lut,candidate)
                stem=f'{task}-{i}-{label}'
                log,result=(session.out/(stem+ext) for ext in ('.log','.json'))
                env=r6.child_env(k);env.update(flags(tile,lut))
                if a.profile_only:
                    env.update(STRATA_VK_PROF='1',STRATA_VK_STATS='1',STRATA_PS5_R11_PROFILE='1',STRATA_PS5_R13_MEM_TRACE='1',STRATA_PS5_R13_QUANT_TRACE='1')
                cmd=[str(session.binary),'bench',*r6.engine_args(a.model,k,s),'--pp',a.pp,'--tg',str(s['tg']),'--reps',str(s['reps']),'--json-out',str(result)]
                if task=='holdout': cmd+=['--prompt-file',str(session.holdout)]
                r6.atomic_json(session.out/(stem+'-request.json'),dict(cmd=cmd,knobs=k,flags=flags(tile,lut),profile_only=a.profile_only))
                r6.require(r6.file_id(session.holdout)==holdout_id, 'holdout changed before measurement')
                print(stem,flush=True)
                memory=r6.run_child(cmd,env,log,s['timeout'],s['min_ram_mb'])
                r6.require(r6.file_id(session.holdout)==holdout_id, 'holdout changed during measurement')
                rates,signatures=r6.checked_rates(json.loads(result.read_text()),k,s,a.model)
                r6.match_signatures(signatures,signatures)
                text=log.read_text(errors='replace');evidence=r6.evidence(text,k);check_marks(text,tile,lut)
                run=dict(rates=rates,signatures=signatures,evidence=evidence,memory=memory)
                if a.profile_only: run['diagnostics']=profile_summary(text)
                r6.atomic_json(session.out/(stem+'-checked.json'),run);runs.append(run)
            reports[task]=dict(profile_only=True,accepted=False) if a.profile_only else compare(runs)
        r6.require(r6.identity(a.model,a.bin)==ident and r6.file_id(session.holdout)==holdout_id,'identity changed')
        accepted=not a.profile_only and all(x['accepted'] for x in reports.values())
        r6.atomic_json(session.out/'summary.json',dict(accepted=accepted,profile_only=a.profile_only,base_tile=base_tile,tile=a.tile,lut=a.lut,reports=reports,production_promotion=False))
        return 0 if accepted or a.profile_only else 1
    except (r6.Reject, quality.r6.Reject, KeyError,TypeError,ValueError,OSError,IndexError) as exc:
        r6.atomic_json(session.out/'summary.json',dict(accepted=False,error=str(exc),reports=reports,production_promotion=False))
        print(f'REJECT: {exc}',file=sys.stderr);return 1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    m=sub.add_parser('marks');m.add_argument('--log',type=Path,required=True);m.add_argument('--test',choices=('iq2','int','model'),required=True)
    ld=sub.add_parser('lds');ld.add_argument('--log',type=Path,required=True)
    b=sub.add_parser('run');b.add_argument('--model',type=Path,required=True);b.add_argument('--bin',type=Path,default=r6.DEFAULT_BIN);b.add_argument('--out',type=Path,required=True)
    for x in (m,b):
        x.add_argument('--tile',type=int,choices=(0,32,64),default=0);x.add_argument('--lut',choices=('off','iq2','int','all'),required=True)
    b.add_argument('--base-tile',type=int,choices=(0,32,64),help='default: same as candidate tile; set 0 for final combined vs R11')
    b.add_argument('--ctx',type=int,default=8192);b.add_argument('--pp',default='128,512,2048,4096');b.add_argument('--tg',type=int,default=32);b.add_argument('--reps',type=int,default=3)
    b.add_argument('--threads',type=int,choices=(4,6,8,16),default=6);b.add_argument('--chunk',type=int,choices=(128,256,512,1024),default=512);b.add_argument('--kv',choices=('f16','q8','q4','q2'),default='q8')
    b.add_argument('--timeout',type=int,default=5400);b.add_argument('--min-ram-mb',type=int,default=1024)
    group=b.add_mutually_exclusive_group();group.add_argument('--profile-only',action='store_true');group.add_argument('--corpus',type=Path)
    b.add_argument('--ctx-len',type=int,default=512);b.add_argument('--chunks',type=int,default=8)
    a=p.parse_args()
    if a.action=='lds':
        try: print('PASS LDS:',check_lds(a.log.read_text()));return 0
        except (r6.Reject,OSError,ValueError) as exc: print(f'REJECT: {exc}',file=sys.stderr);return 1
    if a.action=='marks':
        try: print('PASS dispatch:',check_marks(a.log.read_text(),a.tile,a.lut,a.test));return 0
        except (r6.Reject,OSError) as exc: print(f'REJECT: {exc}',file=sys.stderr);return 1
    return run_campaign(a)


if __name__=='__main__': sys.exit(main())
