#!/usr/bin/env python3
"""PS5 R6: bounded empirical search; no dependencies beyond Python 3.10+ and Vulkan tools."""
import argparse
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / 'target/ps5/x86_64-unknown-linux-gnu/release/rustingstrata'
SCHEMA = 'strata.ps5.r6.v1'
BASE = dict(gemv=256, small=0, attn=0, threads=6, chunk=512, gtt=1, direct=1, zero=0, sched='off')
SPACE = dict(gemv=(64,128,256), small=(0,1), attn=(0,256,512,1024), threads=(4,6,8,16), chunk=(64,128,256,512,1024), gtt=(0,1), direct=(0,1), zero=(0,1), sched=('off','half','auto'))
ENV_KEYS = dict(gemv='STRATA_PS5_GEMV_WG', small='STRATA_PS5_SMALL_GEMM', attn='STRATA_PS5_ATTN_WG', gtt='STRATA_PS5_GTT', direct='STRATA_PS5_DIRECT', zero='STRATA_PS5_ZERO_COPY', sched='STRATA_PS5_SCHED')
HOLDOUT = '''A software engineer must review this implementation before deploying it.\n```python\ndef merge_sorted(a, b):\n    i = j = 0\n    result = []\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i]); i += 1\n        else:\n            result.append(b[j]); j += 1\n    return result + a[i:] + b[j:]\n```\nExplain invariants, complexity, empty inputs and repeated values. A test should include negative numbers, sorted lists, and a stable reference.\n'''

class Reject(RuntimeError): pass

def require(condition, reason):
    if not condition: raise Reject(reason)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()

def file_id(path):
    p = Path(path).resolve(strict=True)
    a = p.stat()
    require(p.is_file(), f'not a regular file: {p}')
    result = dict(path=str(p), size=a.st_size, sha256=digest(p))
    b = p.stat()
    require((a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns), f'file changed while hashing: {p}')
    return result

def model_files(model):
    model = Path(model).absolute()
    m = re.fullmatch(r'(.*)-(\d{5})-of-(\d{5})\.gguf', model.name)
    if not m: return [model]
    stem, part, count = m.groups(); count = int(count)
    require(int(part)==1 and 1 <= count <= 1024, 'open first GGUF shard; unsupported shard count')
    return [model.with_name(f'{stem}-{i:05}-of-{count:05}.gguf') for i in range(1,count+1)]

def clean_env():
    # No ambient engine overrides. Vulkan/loader environment is retained and fingerprinted.
    return {k:v for k,v in os.environ.items() if not k.startswith(('STRATA_', 'PS5_')) and k not in ('RAYON_NUM_THREADS','LLVM_PROFILE_FILE')}

def meminfo():
    return {k.rstrip(':'):int(v) for k,v,*_ in (line.split() for line in Path('/proc/meminfo').read_text().splitlines())}

def identity(model, binary):
    env = clean_env()
    vk = subprocess.run(['vulkaninfo','--summary'],env=env,capture_output=True,text=True,timeout=60,check=True).stdout
    require('deviceName' in vk and 'driverVersion' in vk and 'vendorID' in vk, 'vulkaninfo summary missing device/driver identity')
    # Strip whitespace only; preserve the enumerated order (device index 0 matters).
    vk = '\n'.join(line.strip() for line in vk.splitlines() if line.strip())
    cpu = sorted(set(line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.split(':')[0].strip() in ('vendor_id','cpu family','model','model name','stepping','microcode','flags')))
    policies=sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor'))
    governors={str(p):p.read_text().strip() for p in policies}
    return dict(model=[file_id(p) for p in model_files(model)], binary=file_id(binary),
        vulkan=vk, kernel=os.uname().release, machine=os.uname().machine, cpu=cpu,
        affinity=sorted(os.sched_getaffinity(0)), governors=governors, ram_kib=meminfo()['MemTotal'],
        cmdline=Path('/proc/cmdline').read_text().strip(),
        loader_env={k:v for k,v in env.items() if k.startswith(('VK_','AMD_','RADV_','MESA_','LD_','DRI_')) or k in ('XDG_DATA_HOME','XDG_DATA_DIRS','XDG_CONFIG_HOME','XDG_CONFIG_DIRS','PATH','HOME')})

def knobs_ok(k):
    require(isinstance(k,dict) and set(k)==set(SPACE), 'invalid profile knobs')
    for key, val in k.items():
        require(type(val) is type(BASE[key]) and val in SPACE[key], f'invalid knob {key}: {val!r}')
    require(not k['zero'] or k['gtt'], 'zero-copy requires GTT')
    return k

def settings_ok(s):
    require(isinstance(s,dict) and set(s)=={'ctx','kv','pp','tg','reps','timeout','min_ram_mb'}, 'invalid settings')
    require(all(type(s[k]) is int for k in ('ctx','tg','reps','timeout','min_ram_mb')), 'invalid numeric settings')
    pp=s['pp']; require(isinstance(pp,list) and pp and all(type(x) is int and x>0 for x in pp) and len(set(pp))==len(pp), 'invalid pp')
    require(s['kv'] in ('f16','q8','q4','q2') and 1<=s['ctx']<=262144 and 1<=s['tg']<=1024 and max(pp)+s['tg']<=s['ctx'], 'invalid context/workload')
    require(2<=s['reps']<=16 and 1<=s['timeout']<=86400 and 128<=s['min_ram_mb']<=1536, 'invalid repetitions/timeout/RAM guard')
    return s

def child_env(k, trace=True):
    knobs_ok(k)
    env=clean_env()
    env.update(STRATA_PS5='1',STRATA_PS5_HOST_RESERVE_MB='1536',STRATA_VK_ATTN_BUDGET='16777216',
        STRATA_VK_NO_COOP='1',STRATA_FIXED_EXPERTS='1',STRATA_PREFILL_PROTECT='0',STRATA_PS5_ZERO_COPY_MB='512',
        STRATA_PS5_SCHED_MAX_ROWS='4',STRATA_PS5_SCHED_SCRATCH_MB='64',RAYON_NUM_THREADS=str(k['threads']))
    env.update({name:str(k[key]) for key,name in ENV_KEYS.items()})
    if trace: env.update(STRATA_PS5_R6='1', STRATA_PS5_SCHED_LOG='0')
    return env

def engine_args(model,k,s):
    return ['--model',str(Path(model).absolute()),'--gpu','--backend','vulkan','--device','0','--ctx',str(s['ctx']),
        '--kv',s['kv'],'--threads',str(k['threads']),'--chunk',str(k['chunk']),'--short','32','--mtp','0',
        '--fixed-experts','--vram-margin','768','--ram-margin','1536','--cache-ram','256']

def run_child(cmd,env,log,timeout,min_ram):
    start=time.monotonic(); low=meminfo()['MemAvailable']; peak=0
    require(low>=min_ram*1024, 'available RAM below guard before launch')
    with open(log,'xb') as f:
        p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
        try:
            while p.poll() is None:
                if time.monotonic()-start>timeout: raise Reject('timeout')
                low=min(low,meminfo()['MemAvailable'])
                if low<min_ram*1024: raise Reject(f'available RAM fell below {min_ram} MiB')
                try:
                    for line in Path(f'/proc/{p.pid}/status').read_text().splitlines():
                        if line.startswith('VmHWM:'): peak=max(peak,int(line.split()[1]))
                except (FileNotFoundError,ProcessLookupError): pass
                time.sleep(.2)
            require(p.returncode==0, f'child exited {p.returncode}; see {log}')
        finally:
            if p.poll() is None:
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try: os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    p.wait()
    return dict(elapsed_s=time.monotonic()-start,min_available_kib=low,sampled_peak_rss_kib=peak)

def checked_rates(record, k, s, model):
    require(record.get('schema')=='strata.bench.v1' and record.get('backend')=='vulkan' and record.get('device')==[0], 'not the requested Vulkan backend')
    require(Path(record['model']['path']).resolve()==Path(model).resolve(), 'wrong model in benchmark')
    require(all(record.get(x)==s[x] for x in ('ctx','kv','tg','reps')) and record.get('chunk')==k['chunk'] and record.get('mtp')==0, 'benchmark changed fixed settings')
    fp=record['r6']; require(fp['threads']==k['threads'] and fp['short']==32,'wrong threads/short threshold')
    rows=record['results']; fps=fp['rows']; require([r['pp'] for r in rows]==s['pp'] and [r['pp'] for r in fps]==s['pp'],'missing/duplicate workload')
    rates={}; signatures={}
    for row, f in zip(rows,fps):
        pp=row['pp']; rates[pp]=[]
        for metric in ('prefill_tok_s','decode_tok_s'):
            stat=row[metric]; a,b,c=(stat[x] for x in ('min','mean','max'))
            require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in (a,b,c)) and a<=b<=c,'bad rate')
            rates[pp].append(b)
        samples=f['samples']; require(len(samples)==s['reps'],'missing fingerprint repetitions')
        for sample in samples:
            t=sample['tokens']; top=sample['top5']; require(len(t)==s['tg'] and all(type(i) is int and i>=0 for i in t) and sample['stopped'] is False,'early EOS or invalid decode length')
            require(len(top)==5 and all(len(x)==2 and type(x[0]) is int and x[0]>=0 and type(x[1]) in (float,int) and math.isfinite(x[1]) and x[1]<=1e-6 for x in top) and len({x[0] for x in top})==5 and all(top[i][1]>=top[i+1][1] for i in range(4)),'invalid top5')
        signatures[pp]=samples
    return rates,signatures

def match_signatures(ref,candidate,tol=.05):
    require(ref.keys()==candidate.keys(),'fingerprint workload differs')
    for pp in ref:
        # Every candidate replicate matches every baseline replicate; reject unstable baselines too.
        for a in ref[pp]:
            for b in candidate[pp]:
                require(a['tokens']==b['tokens'] and a['stopped']==b['stopped'],f'greedy token divergence at pp={pp}')
                require([x[0] for x in a['top5']]==[x[0] for x in b['top5']],f'top5 order differs at pp={pp}')
                require(max(abs(x[1]-y[1]) for x,y in zip(a['top5'],b['top5']))<=tol,f'top5 logprob drift at pp={pp}')

def evidence(log,k):
    marks=set(re.findall(r'^PS5 R6 exercised: (\w+)\s*$',log,re.M))
    require('PS5 profile: GTT ' in log,'PS5 AMD profile not active')
    modes=re.findall(r'PS5 profile: GTT (true|false), mapped upload (true|false)',log)
    require(modes and all(d==str(bool(k['direct'])).lower() and (k['gtt'] or g=='false') for g,d in modes),'GTT/direct settings not respected')
    if k['gemv']!=256: require(any(m.endswith('_'+str(k['gemv'])) and 'gemv' in m for m in marks),'candidate GEMV not dispatched')
    if k['small']: require('gemmi_dp_ps5_32' in marks,'candidate small GEMM not dispatched')
    if k['attn']: require('attn_prefill_ps5_'+str(k['attn']) in marks,'candidate attention not dispatched')
    if k['zero']: require(any(int(x)>0 for x in re.findall(r'PS5 R3: imported_bytes=(\d+)',log)),'zero-copy fell back')
    if k['sched']!='off': require('hybrid_prefill' in marks,'no simultaneous CPU/GPU rows recorded')
    if not k['gtt']: require('gtt_nonlocal' not in marks,'GTT disabled but nonlocal allocation observed')
    uploads=re.findall(r'PS5 fresh weights: mapped (\d+) bytes, staged (\d+) bytes',log)
    require(uploads,'missing upload report')
    mapped=sum(int(a) for a,b in uploads); staged=sum(int(b) for a,b in uploads)
    if not k['direct']: require(mapped==0,'mapped upload active despite being disabled')
    return dict(marks=sorted(marks),mapped=mapped,staged=staged)

def useful_change(old,new,a,b):
    if old['gtt']!=new['gtt']: require('gtt_nonlocal' in (a if old['gtt'] else b)['marks'],'GTT comparison never exercised nonlocal heap')
    if old['direct']!=new['direct']: require((a if old['direct'] else b)['mapped']>0,'mapped upload comparison not exercised')

def metrics(candidate,base):
    ratios=[c/b for pp in base for c,b in zip(candidate[pp],base[pp])]
    return math.exp(statistics.mean(map(math.log,ratios))),min(ratios)

def average(runs):
    return {pp:[statistics.mean(x['rates'][pp][i] for x in runs) for i in (0,1)] for pp in runs[0]['rates']}

def stable(a,b,limit=.15):
    ratios=[min(x,y)/max(x,y) for pp in a for x,y in zip(a[pp],b[pp])]
    require(min(ratios)>=1-limit,'baseline/replicate drift exceeds 15%; rerun under stable load')

def atomic_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(data,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try: os.fsync(d)
        finally: os.close(d)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

class Session:
    def __init__(self,model,binary,s,out):
        self.model=Path(model).absolute(); self.binary=Path(binary).resolve(); self.s=settings_ok(s)
        self.out=Path(out).absolute(); self.out.mkdir(parents=True,exist_ok=False); self.i=0
        self.holdout=self.out/'holdout.txt'; self.holdout.write_text(HOLDOUT)
    def measure(self,k,task='prose',binary=None,profile_pattern=None):
        self.i+=1; stem=f'{self.i:03d}-{task}'; log=self.out/(stem+'.log'); result=self.out/(stem+'.json')
        cmd=[str(binary or self.binary),'bench',*engine_args(self.model,k,self.s),'--pp',','.join(map(str,self.s['pp'])),'--tg',str(self.s['tg']),'--reps',str(self.s['reps']),'--json-out',str(result)]
        if task=='holdout': cmd+=['--prompt-file',str(self.holdout)]
        atomic_json(self.out/(stem+'-request.json'),dict(knobs=k,cmd=cmd))
        print(f'{stem}: {k}',flush=True)
        try:
            env=child_env(k)
            if profile_pattern is not None: env['LLVM_PROFILE_FILE']=str(profile_pattern)
            mem=run_child(cmd,env,log,self.s['timeout'],self.s['min_ram_mb'])
            r,f=checked_rates(json.loads(result.read_text()),k,self.s,self.model)
            match_signatures(f,f)
            e=evidence(log.read_text(errors='replace'),k)
            ans=dict(rates=r,signatures=f,evidence=e,memory=mem,knobs=k)
            atomic_json(self.out/(stem+'-accepted.json'),ans)
            return ans
        except (Reject,KeyError,TypeError,ValueError,OSError,IndexError) as exc:
            atomic_json(self.out/(stem+'-rejected.json'),dict(reason=str(exc)))
            raise Reject(f'{stem}: {exc}') from exc

def evaluate(a,b,old,new):
    match_signatures(a['signatures'],b['signatures']); useful_change(old,new,a['evidence'],b['evidence'])
    return metrics(b['rates'],a['rates'])

def abba(session,base,candidate,task,binary=None):
    a1=session.measure(base,task); b1=session.measure(candidate,task,binary); b2=session.measure(candidate,task,binary); a2=session.measure(base,task)
    stable(a1['rates'],a2['rates']); stable(b1['rates'],b2['rates'])
    for x in (a2,b1,b2): match_signatures(a1['signatures'],x['signatures'])
    for x,y in ((a1,b1),(a2,b2)): useful_change(base,candidate,x['evidence'],y['evidence'])
    score,worst=metrics(average([b1,b2]),average([a1,a2]))
    # Neither paired replicate may regress any phase by more than 5%.
    worst=min(worst,metrics(b1['rates'],a1['rates'])[1],metrics(b2['rates'],a2['rates'])[1])
    return dict(score=score,worst=worst,baseline=average([a1,a2]),candidate=average([b1,b2]))

def search(session,budget,experimental):
    baseline=BASE.copy(); best=baseline.copy(); best_run=session.measure(best); tested=0; history=[]
    # Coordinate descent; each candidate measured against current incumbent and original baseline output.
    baseline_run=best_run
    order=('gemv','small','attn','threads','chunk','gtt','direct')+ (('zero','sched') if experimental else ())
    for key in order:
        original=best[key]
        for value in SPACE[key]:
            if value==original or tested>=budget: continue
            k={**best,key:value}
            if k['zero'] and not k['gtt']: continue
            tested+=1
            try:
                trial=session.measure(k); match_signatures(baseline_run['signatures'],trial['signatures'])
                score,worst=evaluate(best_run,trial,best,k)
                _,base_worst=metrics(trial['rates'],baseline_run['rates'])
                accepted=score>=1.03 and min(worst,base_worst)>=.95
                history.append(dict(knobs=k,score=score,worst=worst,accepted=accepted))
                if accepted: best,best_run=k,trial
            except Reject as exc:
                history.append(dict(knobs=k,error=str(exc))); print(f'rejected: {exc}',file=sys.stderr,flush=True)
        if tested>=budget: break
    return best,history

def load_profile(path):
    p=json.loads(Path(path).read_text()); require(p.get('schema')==SCHEMA and p.get('status') in ('approved','baseline'),'profile not approved (or revoked)')
    knobs_ok(p['knobs']); settings_ok(p['settings']); return p

def cli():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='cmd',required=True)
    def common(p):
        p.add_argument('--model',required=True,type=Path); p.add_argument('--bin',type=Path,default=DEFAULT_BIN); p.add_argument('--profile',required=True,type=Path)
    t=sub.add_parser('tune'); common(t); t.add_argument('--out',required=True,type=Path); t.add_argument('--ctx',type=int,default=4096); t.add_argument('--kv',choices=('f16','q8','q4','q2'),default='q8'); t.add_argument('--pp',default='32,64,512,2048'); t.add_argument('--tg',type=int,default=32); t.add_argument('--reps',type=int,default=3); t.add_argument('--timeout',type=int,default=900); t.add_argument('--min-ram-mb',type=int,default=512); t.add_argument('--budget',type=int,default=24); t.add_argument('--experimental',action='store_true'); t.add_argument('--dry-run',action='store_true')
    v=sub.add_parser('verify'); common(v); v.add_argument('--out',required=True,type=Path)
    c=sub.add_parser('compare-pgo'); common(c); c.add_argument('--out',required=True,type=Path); c.add_argument('--candidate-bin',required=True,type=Path)
    pg=sub.add_parser('train-pgo'); common(pg); pg.add_argument('--out',required=True,type=Path); pg.add_argument('--instrumented-bin',required=True,type=Path); pg.add_argument('--raw-dir',required=True,type=Path)
    r=sub.add_parser('run'); common(r); r.add_argument('mode',choices=('chat','generate')); r.add_argument('--prompt'); r.add_argument('-n',type=int,default=128); r.add_argument('--temp',type=float,default=0.0)
    a=parser.parse_args()
    lock=None
    if a.cmd in ('tune','verify') and not getattr(a,'dry_run',False):
        a.profile.parent.mkdir(parents=True,exist_ok=True)
        lock=open(str(a.profile)+'.lock','a')
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise Reject('profile is locked by another calibration') from exc
    if a.cmd=='tune':
        s=settings_ok(dict(ctx=a.ctx,kv=a.kv,pp=list(map(int,a.pp.split(','))),tg=a.tg,reps=a.reps,timeout=a.timeout,min_ram_mb=a.min_ram_mb)); require(1<=a.budget<=64,'budget outside 1..64')
        if a.dry_run:
            print(json.dumps(dict(baseline=BASE,space=SPACE,settings=s,budget=a.budget,experimental=a.experimental,maximum_child_runs=1+a.budget+12),indent=2)); return
        require(not a.profile.exists(),'profile exists; use a new filename to preserve prior evidence')
        ident=identity(a.model,a.bin); session=Session(a.model,a.bin,s,a.out)
        atomic_json(session.out/'identity.json',ident)
        best,history=search(session,a.budget,a.experimental); atomic_json(session.out/'search.json',history)
        # Always validate baseline on unseen coding text and repeat on prose. No winner on a failed baseline.
        checks={}; status='baseline'; chosen=BASE.copy()
        if best!=BASE:
            try:
                checks={task:abba(session,BASE,best,task) for task in ('prose','holdout')}
                if all(v['score']>=1.03 and v['worst']>=.95 for v in checks.values()): chosen=best; status='approved'
            except Reject as exc: checks={'error':str(exc)}
        if status=='baseline':
            # A fallback is also a measured profile, not an untested assertion.
            for task in ('prose','holdout'):
                ref=session.measure(BASE,task); repeat=session.measure(BASE,task); stable(ref['rates'],repeat['rates']); match_signatures(ref['signatures'],repeat['signatures'])
                checks['baseline_'+task]=dict(rates=average([ref,repeat]))
        require(identity(a.model,a.bin)==ident,'identity changed during calibration; profile not written')
        atomic_json(a.profile,dict(schema=SCHEMA,status=status,identity=ident,knobs=chosen,settings=s,checks=checks,report=str(session.out),created=time.time()))
        print(f'{status}: {chosen}; profile {a.profile}'); return
    p=load_profile(a.profile); ident=identity(a.model,a.bin); require(ident==p['identity'],'stale model/binary/driver/host profile: retune')
    k,s=p['knobs'],p['settings']
    if a.cmd=='run':
        require(a.n>0 and a.n<s['ctx'] and math.isfinite(a.temp) and a.temp>=0,'invalid generation settings')
        require(meminfo()['MemAvailable']>=s['min_ram_mb']*1024,'insufficient available RAM before launch')
        require((a.mode=='generate')==(a.prompt is not None),'generate requires --prompt; chat takes stdin')
        cmd=[str(a.bin.resolve()),a.mode,*engine_args(a.model,k,s),'-n',str(a.n),'--temp',str(a.temp)]
        if a.mode=='generate': cmd+=['--prompt',a.prompt]
        os.execve(cmd[0],cmd,child_env(k,False))
    session=Session(a.model,a.bin,s,a.out)
    atomic_json(session.out/'identity.json',ident)
    if a.cmd=='verify':
        try:
            checks={task:abba(session,BASE,k,task) for task in ('prose','holdout')}
            require(all(x['worst']>=.95 and (p['status']=='baseline' or x['score']>=1.03) for x in checks.values()),'regression in revalidation')
            require(identity(a.model,a.bin)==ident,'identity changed during verification')
        except Reject as exc:
            p['status']='revoked'; p['revoked_reason']=str(exc); p['verification_report']=str(session.out); atomic_json(a.profile,p); raise
        p['last_verified']=time.time(); p['verification']=checks; p['verification_report']=str(session.out); atomic_json(a.profile,p); print('PASS: profile revalidated'); return
    if a.cmd=='train-pgo':
        raw=a.raw_dir.resolve(); raw.mkdir(parents=True,exist_ok=True)
        require(not list(raw.glob('*.profraw')), 'PGO raw directory already contains profiles')
        instrumented=a.instrumented_bin.resolve(strict=True)
        require(digest(instrumented)!=ident['binary']['sha256'], 'instrumented binary is the baseline')
        # Only training prose; the coding holdout is reserved for comparison.
        session.measure(k,'prose',instrumented,raw/'strata-%m-%p.profraw')
        require(any(x.stat().st_size>0 for x in raw.glob('*.profraw')), 'no nonempty LLVM profiles produced')
        print(f'PGO training profiles: {raw}'); return
    # PGO binaries get their own identity, never inherit the original profile automatically.
    new=identity(a.model,a.candidate_bin); require(new['binary']['sha256']!=ident['binary']['sha256'],'same binary')
    require({x:y for x,y in new.items() if x!='binary'}=={x:y for x,y in ident.items() if x!='binary'},'PGO comparison identity mismatch')
    checks={task:abba(session,k,k,task,a.candidate_bin.resolve()) for task in ('prose','holdout')}
    require(identity(a.model,a.bin)==ident and identity(a.model,a.candidate_bin)==new,'identity changed during PGO test')
    passed=all(x['score']>=1.03 and x['worst']>=.95 for x in checks.values()); atomic_json(session.out/'pgo-comparison.json',dict(passed=passed,checks=checks,baseline=ident['binary'],candidate=new['binary']))
    require(passed,'PGO did not pass the gain/regression gates'); print('PASS: PGO candidate; retune using --bin to create its own profile')

if __name__=='__main__':
    try: cli()
    except (Reject,OSError,ValueError,KeyError,TypeError,IndexError,subprocess.SubprocessError) as exc:
        print(f'R6: {exc}',file=sys.stderr); sys.exit(1)
