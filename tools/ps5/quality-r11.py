#!/usr/bin/env python3
"""Independent full-vocabulary quality gate for a single R11 GPU candidate.
Strict identity of encoded KLD dumps; not a proof of unquantized logits identity.
"""
import argparse, importlib.util, json, pathlib, sys
spec=importlib.util.spec_from_file_location('r6',pathlib.Path(__file__).with_name('autotune-r6.py'));r6=importlib.util.module_from_spec(spec);spec.loader.exec_module(r6)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--bin',type=pathlib.Path,default=r6.DEFAULT_BIN)
    p.add_argument('--corpus',required=True,type=pathlib.Path);p.add_argument('--out',required=True,type=pathlib.Path)
    p.add_argument('--candidate',choices=('iq32','gemv1','gemv2','gemv4','gemv8','down1','down2','down4','down8'),required=True)
    p.add_argument('--ctx',type=int,default=8192);p.add_argument('--ctx-len',type=int,default=512);p.add_argument('--chunks',type=int,default=4)
    p.add_argument('--threads',type=int,choices=(4,6,8,16),default=6);p.add_argument('--chunk',type=int,choices=(128,256,512,1024),default=512)
    p.add_argument('--kv',choices=('f16','q8','q4','q2'),default='q8');p.add_argument('--timeout',type=int,default=5400)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    if a.candidate=='iq32':key,val,mark='STRATA_PS5_R11_IQ32','1','gemm_f_dp_ps5_iq32'
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
            dump=a.out/f'{flag}.bin';log=a.out/f'{flag}.log'
            cmd=[str(a.bin.resolve()),'perplexity',*r6.engine_args(a.model,k,s),'--file',str(a.corpus.resolve()),'--ctx-len',str(a.ctx_len),'--chunks',str(a.chunks),'--kld-base',str(dump)]
            r6.atomic_json(a.out/f'{flag}-request.json',dict(cmd=cmd,candidate=a.candidate,enabled=bool(flag)))
            r6.run_child(cmd,env,log,a.timeout,1024)
            evidence=r6.evidence(log.read_text(errors='replace'),k)
            r6.require((mark in evidence['marks'])==bool(flag),'missing dispatch or leaked into control')
            import struct
            with dump.open('rb') as f:
                r6.require(f.read(8)==b'_logits_','invalid KLD dump');n,nv,nc=struct.unpack('<III',f.read(12))
            r6.require(n==a.ctx_len and nv>0 and nc==a.chunks,'incomplete corpus coverage')
            expected=20+4*n*nc+2*(4+nv+(nv&1))*(n-1-n//2)*nc
            r6.require(dump.stat().st_size==expected,'truncated or oversized KLD dump')
            import re, math
            ppl=re.search(r'PPL = ([0-9.]+) over',log.read_text(errors='replace'))
            r6.require(ppl is not None and math.isfinite(float(ppl[1])) and float(ppl[1])>0,'invalid PPL')
            hashes.append(r6.digest(dump))
        r6.require(ident==r6.identity(a.model,a.bin) and corpus==r6.file_id(a.corpus),'identity changed')
        same=hashes[0]==hashes[1];r6.atomic_json(a.out/'summary.json',dict(encoded_identity=same,hashes=hashes,identity=ident,corpus=corpus,production_promotion=False))
        print('PASS encoded full-vocabulary identity' if same else 'REJECT: inspect tools/kld_compare.py on the two dumps');return 0 if same else 1
    except (r6.Reject, OSError,ValueError,KeyError) as e:
        r6.atomic_json(a.out/'summary.json',dict(encoded_identity=False,error=str(e),production_promotion=False));print(e,file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
