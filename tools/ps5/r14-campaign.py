#!/usr/bin/env python3
"""R14: controlled dense integer tiles and fixed-work long-context validation.
Never promotes a profile. Natural generation and full-logit readback are not scored.
"""
import argparse
from array import array
import importlib.util
import hashlib
import heapq
import subprocess
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
DENSE = {'r14_dense_q5', 'r14_dense_q6', 'gemmi_dp_ps5_32', 'gemmi_dp_ps5_64', 'r14_base_q5', 'r14_base_q6', 'r14_f16_q5', 'r14_f16_q6'}


def dense_marks(tile, types):
    r6.require(tile in (0, 32, 64) and types in ('q5', 'q6', 'both'), 'invalid R14 selector')
    selected = {t for t in ('q5','q6') if tile and types in (t,'both')}
    return ({f'gemmi_dp_ps5_{tile}'} if tile else set()) | {f'r14_{"dense" if t in selected else "base"}_{t}' for t in ('q5','q6')}


def check_marks(log, tile, types, iq=None):
    want = dense_marks(tile, types)
    if iq is not None:
        want |= r13.expected(iq, 'off')
    got = r6.dispatch_marks(log) - {'gtt_nonlocal'}
    r6.require(got == want, f'dispatch identity: expected {sorted(want)}, got {sorted(got)}')


def fixed_record(record, k, s, model, mode):
    natural = mode == 'natural'
    r6.require(record.get('r14_decode') == (None if natural else mode), 'wrong decode mode')
    nv = record.get('r14_vocab')
    r6.require(type(nv) is int and 5 <= nv <= 2**32, 'invalid vocabulary')
    r6.require(record.get('kv_k') == s['kv'] == record.get('kv_v'), 'mixed or missing KV contract')
    r6.require(record.get('drafts') == {'made':0,'accepted':0}, 'unexpected speculative work')
    rows = len(s['pp'])*s['reps']*(s['tg']+1) if mode == 'fixed_logits' else 0
    if natural:
        r6.require('r14_logit_rows' not in record, 'natural mode has raw logit metadata')
    else:
        r6.require(type(record.get('r14_logit_rows')) is int and record['r14_logit_rows'] == rows, 'wrong logit row count')
    rates, sig = r6.checked_rates(record, k, s, model, allow_eos=natural, decode_mode=None if natural else mode)
    def token_list(ids, size):
        return isinstance(ids,list) and len(ids)==size and all(type(x) is int and 0<=x<nv for x in ids)
    for entry in record['r6']['rows']:
        pp=entry['pp']; inputs=entry.get('input_ids')
        r6.require(token_list(inputs,pp),'invalid prefill inputs')
        data=array('I',inputs)
        r6.require(data.itemsize==4,'unsupported token representation')
        if sys.byteorder!='little':data.byteswap()
        prompt_hash=hashlib.sha256(data.tobytes()).hexdigest()
        forced=entry.get('continuation_ids')
        if not natural: r6.require(token_list(forced,s['tg']), 'missing or invalid prescribed continuation')
        else: r6.require(forced is None,'natural mode contains forced continuation')
        for sample in sig[pp]:
            sample['input_sha256']=prompt_hash
            top=sample['top5']; tokens=sample['tokens']
            r6.require(token_list(tokens,len(tokens)) and all(0<=x[0]<nv for x in top), 'token outside vocabulary')
            r6.require(math.fsum(math.exp(x[1]) for x in top)<=1.000001,'impossible top5 probability mass')
            if natural:
                r6.require('predictions' not in sample, 'natural mode contains forced predictions')
                r6.require(not tokens or tokens[0]==top[0][0], 'natural first token disagrees with prefill greedy')
            else:
                r6.require(tokens==forced,'executed tokens differ from prescribed continuation')
                r6.require(token_list(sample.get('predictions'),s['tg']), 'missing or invalid forced-step predictions')
    return rates, sig, nv, rows


def match(ref, cur, fixed=True):
    r6.match_signatures(ref, cur)
    for pp in ref:
        for a in ref[pp]:
            for b in cur[pp]:
                r6.require(a['input_sha256']==b['input_sha256'], f'prefill token sequence divergence at pp={pp}')
                if fixed: r6.require(a['predictions'] == b['predictions'], f'forced-step argmax divergence at pp={pp}')


def check_raw(path, sig, nv, rows):
    r6.require(path.stat().st_size == nv*rows*4, 'truncated or oversized FP32 dump')
    count = 0
    with path.open('rb') as f:
        for samples in sig.values():
            for sample in samples:
                for step, token in enumerate([sample['top5'][0][0], *sample['predictions']]):
                    data = array('f'); data.frombytes(f.read(nv*4))
                    if sys.byteorder != 'little': data.byteswap()
                    r6.require(len(data)==nv and all(map(math.isfinite,data)), 'invalid raw logits')
                    maximum=max(data)
                    # CPU quality mode uses Rust total_cmp: +0 wins over -0,
                    # then the lowest index for exactly equal representations.
                    def order(i): return (-data[i], -math.copysign(1.,data[i]), i)
                    if step == 0:
                        ids=heapq.nsmallest(5,range(nv),key=order)
                        r6.require(ids==[x[0] for x in sample['top5']], 'FP32 dump disagrees with top5 order')
                        lse=maximum+math.log(math.fsum(math.exp(x-maximum) for x in data))
                        r6.require(all(abs(data[i]-lse-lp)<=1e-8 for i,lp in sample['top5']), 'FP32 dump disagrees with top5 logprob')
                    else:
                        winner=data.index(maximum)
                        if maximum==0 and math.copysign(1.,data[winner])<0:
                            winner=next((i for i,x in enumerate(data) if x==0 and math.copysign(1.,x)>0),winner)
                        r6.require(token==winner, 'FP32 dump disagrees with prediction')
                    count += 1
        r6.require(count == rows and not f.read(1), 'FP32 dump coverage mismatch')
    return r6.digest(path)


def stat_identity(paths):
    # Full hashes at campaign boundaries, cheap inode/ctime/mtime checks per arm.
    # A restored byte-identical file still changes ctime on ordinary Linux filesystems.
    result=[]
    for path in paths:
        p=Path(path).resolve(strict=True); s=p.stat()
        r6.require(p.is_file(),f'not a regular file: {p}')
        result.append((str(p),s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))
    return result


def test_lines(text):
    # libtest --nocapture may prefix the first stderr line with the test name.
    return [re.sub(r'^test (?:shared_bound|fingerprint) \.\.\. (?=R14 )','',line) for line in text.splitlines()]


def check_test_log(text, marker=None):
    summaries=[l for l in text.splitlines() if l.startswith('test result:')]
    r6.require(len(summaries)==1 and re.match(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;',summaries[0]),'missing or contradictory Cargo test result')
    if marker: r6.require(test_lines(text).count(marker)==1,'missing or duplicate parity marker')


def check_lds(text):
    check_test_log(text)
    seen=set()
    for line in test_lines(text):
        if not line.startswith('R14 LDS '): continue
        m=re.fullmatch(r'R14 LDS type=(13|14) base=(\d+) tile32=(\d+) tile64=(\d+)',line)
        r6.require(m is not None,'malformed LDS marker')
        ty,a,b,c=map(int,m.groups())
        r6.require(ty not in seen and 0<b<c<a<=65536,'invalid/duplicate LDS marker')
        seen.add(ty)
    r6.require(seen=={13,14},'missing LDS markers')


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


def load_bench_reference(root, ident, holdout, corpus):
    summary = json.loads((root/'summary.json').read_text())
    r6.require(summary.get('mode')=='bench' and summary.get('validated') is True,
               'cross-mode reference must be a validated benchmark (ROI rejection is allowed)')
    identity = json.loads((root/'identity.json').read_text())
    # Each campaign creates its own copy of the same holdout, so its path differs.
    content = lambda f: (f['size'], f['sha256'])
    r6.require(identity['environment']==ident and content(identity['holdout'])==content(holdout)
               and identity['corpus']==corpus, 'cross-mode reference identity differs')
    tasks = {'prose','holdout'} | ({'corpus'} if corpus is not None else set())
    r6.require(set(summary['reports'])==tasks, 'cross-mode reference workloads differ')


def cross_check_greedy(root, task, enabled, tile, iq, types, k, s, model, flags, sig, nv):
    # Validate the original scored-arm records, not only the generated summary.
    # A/B/B/A means quality's control is matched with 1 and 4, candidate with 2 and 3.
    hashes={}
    for arm in ((2,3) if enabled else (1,4)):
        stem=f'{task}-{arm}-tile{tile}-iq{iq}'
        request_path=root/(stem+'-request.json'); log=root/(stem+'.log'); record=root/(stem+'.json')
        stamp=stat_identity([request_path,log,record])
        request=json.loads(request_path.read_text())
        r6.require(request.get('mode')=='bench' and request.get('flags')==flags and request.get('knobs')==k,
                   'cross-mode reference dispatch/settings differ')
        text=log.read_text(errors='replace');r6.evidence(text,k);check_marks(text,tile,types,iq)
        _,greedy,vocab,_=fixed_record(json.loads(record.read_text()),k,s,model,'fixed')
        r6.require(vocab==nv, 'cross-mode vocabulary differs')
        match(greedy,sig,True)
        hashes[stem]=dict(json=r6.digest(record),log=r6.digest(log),request=r6.digest(request_path))
        r6.require(stat_identity([request_path,log,record])==stamp,'cross-mode reference changed during verification')
    return hashes


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--bin',type=Path,default=r6.DEFAULT_BIN);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tile',type=int,choices=(0,32,64),required=True);p.add_argument('--types',choices=('q5','q6','both'),default='both')
    p.add_argument('--base-iq-tile',type=int,choices=(0,32,64),default=64);p.add_argument('--iq-tile',type=int,choices=(0,32,64),default=64)
    p.add_argument('--mode',choices=('bench','quality','natural','profile','kld'),default='bench')
    p.add_argument('--ctx',type=int,default=8192);p.add_argument('--pp',default='128,512,2048,4096');p.add_argument('--tg',type=int,default=32);p.add_argument('--reps',type=int,default=3)
    p.add_argument('--threads',type=int,choices=(4,6,8,16),default=6);p.add_argument('--chunk',type=int,choices=(128,256,512,1024),default=512)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--min-ram-mb',type=int,default=1024)
    p.add_argument('--bench-reference',type=Path,help='cross-check quality against matching validated fixed-decode A/B/B/A evidence')
    p.add_argument('--corpus',type=Path);p.add_argument('--ctx-len',type=int,default=512);p.add_argument('--chunks',type=int,default=8)
    a=p.parse_args()
    s=r6.settings_ok(dict(ctx=a.ctx,pp=list(map(int,a.pp.split(','))),tg=a.tg,reps=a.reps,timeout=a.timeout,min_ram_mb=a.min_ram_mb,kv='q8'))
    k=r6.knobs_ok({**r6.BASE,'threads':a.threads,'chunk':a.chunk})
    r6.require(a.tile or a.base_iq_tile!=a.iq_tile,'identical control/candidate')
    r6.require(a.bench_reference is None or a.mode=='quality','--bench-reference is only valid in quality mode')
    r6.require(a.mode!='kld' or (a.corpus is not None and 128<=a.ctx_len<=a.ctx and a.chunks>0),'KLD requires valid corpus/coverage')
    session=r6.Session(a.model,a.bin,s,a.out); reports={}
    try:
        paths=[*r6.model_files(a.model),session.binary]
        stamp=stat_identity(paths)
        ident=r6.identity(a.model,a.bin); r6.require(stat_identity(paths)==stamp,'model/binary changed during initial identity'); holdout=r6.file_id(session.holdout); corpus=r6.file_id(a.corpus) if a.corpus else None
        r6.atomic_json(session.out/'identity.json',dict(environment=ident,holdout=holdout,corpus=corpus))
        if a.bench_reference is not None:load_bench_reference(a.bench_reference,ident,holdout,corpus)
        for task in (('corpus',) if a.mode=='kld' else ('prose','holdout')+ (('corpus',) if a.corpus else ())):
            runs=[]
            for i,enabled in enumerate((0,1,1,0) if a.mode=='bench' else (0,1),1):
                tile=a.tile if enabled else 0; iq=a.iq_tile if enabled else a.base_iq_tile
                stem=f'{task}-{i}-tile{tile}-iq{iq}'; log=session.out/(stem+'.log'); result=session.out/(stem+'.json'); dump=session.out/(stem+'.bin')
                env=r6.child_env(k); flags=dict(STRATA_PS5_R11_IQ32='1',STRATA_PS5_R12_IQ_INT_TILE=str(iq),STRATA_PS5_R13_LUT='off',STRATA_PS5_R14_DENSE_TILE=str(tile),STRATA_PS5_R14_DENSE_TYPE=a.types,STRATA_PS5_R14_TRACE='1')
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
                    r6.require(stat_identity(paths)==stamp,'model/binary changed during campaign')
                    r6.require(r6.file_id(session.holdout)==holdout and (a.corpus is None or r6.file_id(a.corpus)==corpus),'corpus changed')
                inputs_ok(); mem=r6.run_child(cmd,env,log,a.timeout,a.min_ram_mb);inputs_ok()
                text=log.read_text(errors='replace'); evidence=r6.evidence(text,k);check_marks(text,tile,a.types,iq)
                run=dict(memory=mem,evidence=evidence)
                if a.mode=='kld':
                    quality.check_dump(dump,text,a.ctx_len,a.chunks);run['sha256']=r6.digest(dump)
                else:
                    rec=json.loads(result.read_text())
                    rates,sig,nv,rows=fixed_record(rec,k,s,a.model,{'natural':'natural','quality':'fixed_logits'}.get(a.mode,'fixed'))
                    run.update(rates=rates,signatures=sig,vocab=nv)
                    if a.mode=='quality':
                        run['sha256']=check_raw(dump,sig,nv,rows)
                        if a.bench_reference is not None:
                            run['greedy_reference']=cross_check_greedy(a.bench_reference,task,enabled,tile,iq,a.types,k,s,a.model,flags,sig,nv)
                    match(sig,sig,a.mode!='natural')
                    if a.mode=='profile':run['diagnostics']=profile_report(text)
                r6.atomic_json(session.out/(stem+'-checked.json'),run);runs.append(run)
            if a.mode=='bench':reports[task]=compare(runs)
            else:
                if a.mode!='kld':
                    match(runs[0]['signatures'],runs[1]['signatures'],a.mode!='natural')
                    r6.require(runs[0]['vocab']==runs[1]['vocab'],'vocabulary changed')
                if a.mode in ('quality','kld'):r6.require(runs[0]['sha256']==runs[1]['sha256'],'full-vocabulary identity failed')
                reports[task]=dict(accepted=False,validated=True,mode=a.mode,runs=runs)
        r6.require(r6.identity(a.model,a.bin)==ident and stat_identity(paths)==stamp,'model, binary or hardware identity changed')
        accepted=a.mode=='bench' and all(x['accepted'] for x in reports.values())
        r6.atomic_json(session.out/'summary.json',dict(mode=a.mode,accepted=accepted,validated=True,reports=reports,production_promotion=False,greedy_cross_validated=a.mode=='quality' and a.bench_reference is not None))
        return 0 if accepted or a.mode!='bench' else 1
    except (r6.Reject,ValueError,KeyError,TypeError,OSError,IndexError,OverflowError,subprocess.SubprocessError) as exc:
        r6.atomic_json(session.out/'summary.json',dict(mode=a.mode,accepted=False,validated=False,error=str(exc),reports=reports));print(f'REJECT: {exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
