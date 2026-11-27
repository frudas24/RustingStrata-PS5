#!/usr/bin/env python3
"""Independent full-vocabulary quality gate for a single R11 GPU candidate.
Strict identity of encoded KLD dumps; not a proof of unquantized logits identity.
"""
import argparse, importlib.util, json, pathlib, sys, struct, re, math
spec=importlib.util.spec_from_file_location('r6',pathlib.Path(__file__).with_name('autotune-r6.py'));r6=importlib.util.module_from_spec(spec);spec.loader.exec_module(r6)
def check_dump(dump, log_text, ctx_len, chunks):
    with dump.open('rb') as f:
        r6.require(f.read(8)==b'_logits_', 'invalid KLD dump')
        header=f.read(12)
        r6.require(len(header)==12, 'truncated KLD header')
        n,nv,nc=struct.unpack('<III',header)
        r6.require(n==ctx_len and n>=4 and nv>0 and nc==chunks and nc>0, 'incomplete corpus coverage')
        scored=(n-1-n//2)*nc
        row_bytes=2*(4+nv+(nv&1))
        r6.require(dump.stat().st_size==20+4*n*nc+row_bytes*scored, 'truncated or oversized KLD dump')
        for _ in range(nc):
            tokens=f.read(4*n)
            r6.require(all(t<nv for (t,) in struct.iter_unpack('<I',tokens)), 'KLD token outside vocabulary')
        for _ in range(scored):
            scale,offset=struct.unpack('<ff',f.read(8))
            r6.require(math.isfinite(scale) and scale>=0 and math.isfinite(offset) and offset<=0, 'invalid KLD row encoding')
            f.seek(row_bytes-8,1)
    matches=re.findall(r'^PPL = (\S+) over (\d+) chunks of (\d+) tokens \((\d+) scored\)$',log_text,re.M)
    r6.require(len(matches)==1,'missing or ambiguous PPL summary')
    ppl,nc_log,n_log,scored_log=matches[0]
    r6.require(math.isfinite(float(ppl)) and float(ppl)>0 and (int(nc_log),int(n_log),int(scored_log))==(nc,n,scored),'invalid PPL or coverage')
    return n,nv,nc

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--bin',type=pathlib.Path,default=r6.DEFAULT_BIN)
    p.add_argument('--corpus',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path)
    p.add_argument('--candidate',choices=('iq32','gemv1','gemv2','gemv4','gemv8','down1','down2','down4','down8','iqint32','iqint64'),required=True)
    p.add_argument('--ctx',type=int,default=8192);p.add_argument('--ctx-len',type=int,default=512);p.add_argument('--chunks',type=int,default=4)
    p.add_argument('--threads',type=int,choices=(4,6,8,16),default=6);p.add_argument('--chunk',type=int,choices=(128,256,512,1024),default=512)
    p.add_argument('--kv',choices=('f16','q8','q4','q2'),default='q8');p.add_argument('--timeout',type=int,default=5400)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    int_tile=a.candidate in ('iqint32','iqint64')
    if int_tile:key,val,mark='STRATA_PS5_R12_IQ_INT_TILE',a.candidate[-2:],'gemmi_dp_ps5_iq'+a.candidate[-2:]
    elif a.candidate=='iq32':key,val,mark='STRATA_PS5_R11_IQ32','1','gemm_f_dp_ps5_iq32'
    elif a.candidate.startswith('gemv'):key,val,mark='STRATA_PS5_R11_GEMV_RPW',a.candidate[-1],'r11_gemv_rpw'
    else:key,val,mark='STRATA_PS5_R11_DOWN_RPW',a.candidate[-1],'r11_down_rpw'
    try:
        r6.require(128<=a.ctx_len<=a.ctx and a.chunks>0,'invalid corpus settings')
        s=r6.settings_ok(dict(ctx=a.ctx,pp=[32],tg=1,reps=2,timeout=a.timeout,min_ram_mb=1024,kv=a.kv))
        k=r6.knobs_ok({**r6.BASE,'threads':a.threads,'chunk':a.chunk})
        ident=r6.identity(a.model,a.bin);corpus=r6.file_id(a.corpus)
        hashes=[]
        for flag in [0,1]:
            env=r6.child_env(k);env[key]=val if flag else '0'
            if int_tile:env['STRATA_PS5_R11_IQ32']='1'
            dump=a.out/f'{flag}.bin';log=a.out/f'{flag}.log'
            cmd=[str(a.bin.resolve()),'perplexity',*r6.engine_args(a.model,k,s),'--file',str(a.corpus.resolve()),'--ctx-len',str(a.ctx_len),'--chunks',str(a.chunks),'--kld-base',str(dump)]
            r6.atomic_json(a.out/f'{flag}-request.json',dict(cmd=cmd,candidate=a.candidate,enabled=bool(flag),base_iq32=int_tile,flags={key:env[key], **({'STRATA_PS5_R11_IQ32':'1'} if int_tile else {})}))
            r6.run_child(cmd,env,log,a.timeout,1024)
            evidence=r6.evidence(log.read_text(errors='replace'),k)
            r6.require((mark in evidence['marks'])==bool(flag),'missing dispatch or leaked into control')
            if int_tile:
                r6.require('gemm_f_dp_ps5_iq32' in evidence['marks'], 'validated IQ32 baseline absent')
                r6.require(not (set(evidence['marks']) & ({'gemmi_dp_ps5_iq32','gemmi_dp_ps5_iq64'} - {mark})), 'other R12 tile leaked')
            check_dump(dump,log.read_text(errors='replace'),a.ctx_len,a.chunks)
            hashes.append(r6.digest(dump))
        r6.require(ident==r6.identity(a.model,a.bin) and corpus==r6.file_id(a.corpus),'identity changed')
        same=hashes[0]==hashes[1];r6.atomic_json(a.out/'summary.json',dict(encoded_identity=same,hashes=hashes,identity=ident,corpus=corpus,production_promotion=False))
        print('PASS encoded full-vocabulary identity' if same else 'REJECT: inspect tools/kld_compare.py on the two dumps');return 0 if same else 1
    except (r6.Reject, OSError,ValueError,KeyError) as e:
        r6.atomic_json(a.out/'summary.json',dict(encoded_identity=False,error=str(e),production_promotion=False));print(e,file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
