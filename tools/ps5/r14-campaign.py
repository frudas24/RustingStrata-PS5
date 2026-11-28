#!/usr/bin/env python3
"""R14: controlled dense integer tiles and fixed-work long-context validation.
Never promotes a profile. Natural generation and full-logit readback are not scored.
"""
import argparse
from array import array
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r6 = load('r6', 'autotune-r6.py')
r13 = load('r13', 'r13-campaign.py')
quality = load('quality', 'quality-r11.py')
DENSE = {'r14_dense_q5', 'r14_dense_q6', 'gemmi_dp_ps5_32', 'gemmi_dp_ps5_64'}


def dense_marks(tile, types):
    r6.require(tile in (0, 32, 64) and types in ('q5', 'q6', 'both'), 'invalid R14 selector')
    if not tile:
        return set()
    return {f'gemmi_dp_ps5_{tile}', *(f'r14_dense_{t}' for t in ('q5', 'q6') if types in (t, 'both'))}


def check_marks(log, tile, types, iq=None):
    want = dense_marks(tile, types)
    if iq is not None:
        want |= r13.expected(iq, 'off')
    got = r6.dispatch_marks(log) - {'gtt_nonlocal'}
    r6.require(got == want, f'dispatch identity: expected {sorted(want)}, got {sorted(got)}')


def fixed_record(record, k, s, model, mode):
    r6.require(record.get('r14_decode') == mode, 'wrong decode mode')
    nv = record.get('r14_vocab')
    r6.require(type(nv) is int and 5 <= nv <= 2**32, 'invalid vocabulary')
    rows = len(s['pp'])*s['reps']*(s['tg']+1) if mode == 'fixed_logits' else 0
    r6.require(type(record.get('r14_logit_rows')) is int and record['r14_logit_rows'] == rows, 'wrong logit row count')
    rates, sig = r6.checked_rates(record, k, s, model, decode_mode=mode)
    for entry in record['r6']['rows']:
        pp=entry['pp']; inputs=entry.get('input_ids')
        r6.require(isinstance(inputs,list) and len(inputs)==pp and all(type(x) is int and 0<=x<nv for x in inputs),'invalid fixed prefill inputs')
        data=array('I',inputs)
        r6.require(data.itemsize==4,'unsupported token representation')
        if sys.byteorder!='little':data.byteswap()
        prompt_hash=hashlib.sha256(data.tobytes()).hexdigest()
        for sample in sig[pp]:
            sample['input_sha256']=prompt_hash
            pred = sample.get('predictions')
            r6.require(isinstance(pred, list) and len(pred) == s['tg'], 'missing forced-step predictions')
            r6.require(all(type(x) is int and 0 <= x < nv for x in pred+sample['tokens']+[x[0] for x in sample['top5']]), 'token outside vocabulary')
    return rates, sig, nv, rows


def match(ref, cur, fixed=True):
    r6.match_signatures(ref, cur)
    if fixed:
        for pp in ref:
            for a in ref[pp]:
                for b in cur[pp]:
                    r6.require(a['input_sha256']==b['input_sha256'], f'prefill token sequence divergence at pp={pp}')
                    r6.require(a['predictions'] == b['predictions'], f'forced-step argmax divergence at pp={pp}')


def check_raw(path, sig, nv, rows):
    r6.require(path.stat().st_size == nv*rows*4, 'truncated or oversized FP32 dump')
    count = 0
    with path.open('rb') as f:
        for samples in sig.values():
            for sample in samples:
                # One row after prefill, then one row after each forced token.
                for token in [sample['top5'][0][0], *sample['predictions']]:
                    data = array('f'); data.frombytes(f.read(nv*4))
                    if sys.byteorder != 'little': data.byteswap()
                    r6.require(len(data)==nv and all(map(math.isfinite,data)), 'invalid raw logits')
                    # Rust total_cmp distinguishes +0 from -0; the strict byte gate
                    # handles this distinction. Accept either signed zero for argmax.
                    r6.require(data[token]==max(data), 'FP32 dump disagrees with prediction')
                    count += 1
        r6.require(count == rows and not f.read(1), 'FP32 dump coverage mismatch')
    return r6.digest(path)


def profile_report(text):
    diagnostics=r13.profile_summary(text)
    boundary=list(re.finditer(r'^r7_load (\{[^\n]+\})$',text,re.M))
    r6.require(len(boundary)==1,'missing or ambiguous load boundary')
    load=json.loads(boundary[0][1]).get('load_ns')
    r6.require(type(load) is int and load>0,'invalid load time')
    def copies(segment):
        out={}
        for role,n,ns in re.findall(r'^r13_mem_copy,(\w+),bytes=(\d+),ns=(\d+)$',segment,re.M):
            row=out.setdefault(role,dict(count=0,bytes=0,ns=0));row['count']+=1;row['bytes']+=int(n);row['ns']+=int(ns)
        return out
    kernels={}
    for name,ms,pct,count,us in re.findall(r'^prof (\S+)\s+([\d.]+) ms\s+([\d.]+)%\s+(\d+) calls\s+([\d.]+) us$',text,re.M):
        r6.require(name not in kernels,'ambiguous multiple GPU profiles')
        kernels[name]=dict(ms=float(ms),percent=float(pct),calls=int(count),us_per_call=float(us))
    r6.require(bool(kernels),'missing GPU kernel profile')
    return dict(load_ns=load,host_copies_before_load_marker=copies(text[:boundary[0].start()]),
                host_copies_after_load_marker=copies(text[boundary[0].end():]),gpu_intervals=kernels,**diagnostics)


def compare(runs):
    a1,b1,b2,a2=runs
    for arm in runs:
        match(a1['signatures'],arm['signatures'])
        r6.require(arm['vocab']==a1['vocab'],'vocabulary changed')
    r6.stable(a1['rates'],a2['rates']); r6.stable(b1['rates'],b2['rates'])
    a,b=r6.average([a1,a2]),r6.average([b1,b2])
    score,worst=r6.metrics(b,a)
    pairs=[r6.metrics(b1['rates'],a1['rates']),r6.metrics(b2['rates'],a2['rates'])]
    worst=min(worst,*(p[1] for p in pairs))
    phases={name:math.exp(statistics.mean(math.log(b[pp][i]/a[pp][i]) for pp in a)) for i,name in enumerate(('prefill','decode'))}
    return dict(score=score,worst=worst,paired_scores=[p[0] for p in pairs],phase_speedups=phases,baseline=a,candidate=b,
                accepted=score>=1.03 and min(p[0] for p in pairs)>=1.03 and worst>=.95)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--bin',type=Path,default=r6.DEFAULT_BIN);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tile',type=int,choices=(0,32,64),required=True);p.add_argument('--types',choices=('q5','q6','both'),default='both')
    p.add_argument('--base-iq-tile',type=int,choices=(0,32,64),default=64);p.add_argument('--iq-tile',type=int,choices=(0,32,64),default=64)
    p.add_argument('--mode',choices=('bench','quality','natural','profile','kld'),default='bench')
    p.add_argument('--ctx',type=int,default=8192);p.add_argument('--pp',default='128,512,2048,4096');p.add_argument('--tg',type=int,default=32);p.add_argument('--reps',type=int,default=3)
    p.add_argument('--threads',type=int,choices=(4,6,8,16),default=6);p.add_argument('--chunk',type=int,choices=(128,256,512,1024),default=512)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--min-ram-mb',type=int,default=1024)
    p.add_argument('--corpus',type=Path);p.add_argument('--ctx-len',type=int,default=512);p.add_argument('--chunks',type=int,default=8)
    a=p.parse_args()
    s=r6.settings_ok(dict(ctx=a.ctx,pp=list(map(int,a.pp.split(','))),tg=a.tg,reps=a.reps,timeout=a.timeout,min_ram_mb=a.min_ram_mb,kv='q8'))
    k=r6.knobs_ok({**r6.BASE,'threads':a.threads,'chunk':a.chunk})
    r6.require(a.tile or a.base_iq_tile!=a.iq_tile,'identical control/candidate')
    r6.require(a.mode!='kld' or (a.corpus is not None and 128<=a.ctx_len<=a.ctx and a.chunks>0),'KLD requires valid corpus/coverage')
    session=r6.Session(a.model,a.bin,s,a.out); reports={}
    try:
        ident=r6.identity(a.model,a.bin); holdout=r6.file_id(session.holdout); corpus=r6.file_id(a.corpus) if a.corpus else None
        r6.atomic_json(session.out/'identity.json',dict(environment=ident,holdout=holdout,corpus=corpus))
        for task in (('corpus',) if a.mode=='kld' else ('prose','holdout')+ (('corpus',) if a.corpus else ())):
            runs=[]
            for i,enabled in enumerate((0,1,1,0) if a.mode=='bench' else (0,1),1):
                tile=a.tile if enabled else 0; iq=a.iq_tile if enabled else a.base_iq_tile
                stem=f'{task}-{i}-tile{tile}-iq{iq}'; log=session.out/(stem+'.log'); result=session.out/(stem+'.json'); dump=session.out/(stem+'.bin')
                env=r6.child_env(k); flags=dict(STRATA_PS5_R11_IQ32='1',STRATA_PS5_R12_IQ_INT_TILE=str(iq),STRATA_PS5_R13_LUT='off',STRATA_PS5_R14_DENSE_TILE=str(tile),STRATA_PS5_R14_DENSE_TYPE=a.types)
                env.update(flags)
                if a.mode=='profile': env.update(STRATA_VK_PROF='1',STRATA_VK_STATS='1',STRATA_PS5_R11_PROFILE='1',STRATA_PS5_R13_MEM_TRACE='1',STRATA_PS5_R13_QUANT_TRACE='1',STRATA_R7_TOKEN_TIMING='1')
                if a.mode=='kld':
                    cmd=[str(session.binary),'perplexity',*r6.engine_args(a.model,k,s),'--file',str(a.corpus.resolve()),'--ctx-len',str(a.ctx_len),'--chunks',str(a.chunks),'--kld-base',str(dump)]
                else:
                    cmd=[str(session.binary),'bench',*r6.engine_args(a.model,k,s),'--pp',a.pp,'--tg',str(a.tg),'--reps',str(a.reps),'--json-out',str(result)]
                    if task=='holdout':cmd+=['--prompt-file',str(session.holdout)]
                    if task=='corpus':cmd+=['--prompt-file',str(a.corpus.resolve())]
                    if a.mode!='natural':cmd+=['--fixed-decode']
                    if a.mode=='quality':cmd+=['--fixed-logits',str(dump)]
                r6.atomic_json(session.out/(stem+'-request.json'),dict(cmd=cmd,knobs=k,mode=a.mode,flags=flags))
                print(stem,flush=True)
                def inputs_ok():
                    r6.require(r6.file_id(session.holdout)==holdout and (a.corpus is None or r6.file_id(a.corpus)==corpus),'corpus changed')
                inputs_ok(); mem=r6.run_child(cmd,env,log,a.timeout,a.min_ram_mb);inputs_ok()
                text=log.read_text(errors='replace'); evidence=r6.evidence(text,k);check_marks(text,tile,a.types,iq)
                run=dict(memory=mem,evidence=evidence)
                if a.mode=='kld':
                    quality.check_dump(dump,text,a.ctx_len,a.chunks);run['sha256']=r6.digest(dump)
                else:
                    rec=json.loads(result.read_text())
                    if a.mode=='natural':
                        r6.require('r14_decode' not in rec,'natural control used fixed mode')
                        rates,sig=r6.checked_rates(rec,k,s,a.model,allow_eos=True)
                        run.update(rates=rates,signatures=sig)
                    else:
                        rates,sig,nv,rows=fixed_record(rec,k,s,a.model,'fixed_logits' if a.mode=='quality' else 'fixed')
                        run.update(rates=rates,signatures=sig,vocab=nv)
                        if a.mode=='quality':run['sha256']=check_raw(dump,sig,nv,rows)
                    match(sig,sig,a.mode!='natural')
                    if a.mode=='profile':run['diagnostics']=profile_report(text)
                r6.atomic_json(session.out/(stem+'-checked.json'),run);runs.append(run)
            if a.mode=='bench':reports[task]=compare(runs)
            else:
                if a.mode!='kld':
                    match(runs[0]['signatures'],runs[1]['signatures'],a.mode!='natural')
                    if a.mode!='natural':r6.require(runs[0]['vocab']==runs[1]['vocab'],'vocabulary changed')
                if a.mode in ('quality','kld'):r6.require(runs[0]['sha256']==runs[1]['sha256'],'full-vocabulary identity failed')
                reports[task]=dict(accepted=False,validated=True,mode=a.mode,runs=runs)
        r6.require(r6.identity(a.model,a.bin)==ident,'model, binary or hardware identity changed')
        accepted=a.mode=='bench' and all(x['accepted'] for x in reports.values())
        r6.atomic_json(session.out/'summary.json',dict(mode=a.mode,accepted=accepted,validated=True,reports=reports,production_promotion=False))
        return 0 if accepted or a.mode!='bench' else 1
    except (r6.Reject,ValueError,KeyError,TypeError,OSError,IndexError) as exc:
        r6.atomic_json(session.out/'summary.json',dict(mode=a.mode,accepted=False,validated=False,error=str(exc),reports=reports));print(f'REJECT: {exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
