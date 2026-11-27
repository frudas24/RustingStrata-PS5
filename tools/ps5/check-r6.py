#!/usr/bin/env python3
"""R6 policy/CLI tests using a simulated engine, not GPU or Rust execution."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r6',HERE/'autotune-r6.py'); r6=importlib.util.module_from_spec(spec); spec.loader.exec_module(r6)
SET=dict(ctx=4096,kv='q8',pp=[32,64,512,2048],tg=2,reps=2,timeout=5,min_ram_mb=128)
FAKE='''#!/usr/bin/env python3
import json,os,sys
args=sys.argv[1:]
def v(k): return args[args.index(k)+1]
if args[0]!='bench': print('FAKE RUN',args[0],os.environ['STRATA_PS5_GEMV_WG']); sys.exit(0)
wg=int(os.environ['STRATA_PS5_GEMV_WG']); f={256:100.,64:110.,128:120.}[wg]
if os.environ.get('MOCK_REGRESSION') and wg!=256: f=80.
if 'candidate' in sys.argv[0]: f*=1.12
pp=list(map(int,v('--pp').split(','))); reps=int(v('--reps')); tg=int(v('--tg'))
print('PS5 profile: GTT true, mapped upload true, host reserve 1536 MiB (driver budget true)')
if wg!=256: print('PS5 R6 exercised: gemv_q_ps5_'+str(wg))
print('PS5 fresh weights: mapped 64 bytes, staged 128 bytes')
stat=dict(min=f,mean=f,max=f)
samples=[dict(top5=[[i,-float(i+1)] for i in range(5)],tokens=[7]*tg,stopped=False) for _ in range(reps)]
r=dict(schema='strata.bench.v1',backend='vulkan',device=[0],model=dict(path=v('--model')),ctx=int(v('--ctx')),kv=v('--kv'),chunk=int(v('--chunk')),mtp=0,tg=tg,reps=reps,results=[dict(pp=n,prefill_tok_s=stat,decode_tok_s=stat) for n in pp],r6=dict(threads=int(v('--threads')),short=32,rows=[dict(pp=n,samples=samples) for n in pp]))
open(v('--json-out'),'w').write(json.dumps(r))
'''
class Tests(unittest.TestCase):
    def setUp(self):
        self.sample=dict(top5=[[i,-float(i+1)] for i in range(5)],tokens=[1,2],stopped=False)
        self.sig={p:[copy.deepcopy(self.sample),copy.deepcopy(self.sample)] for p in SET['pp']}
        self.rec=dict(schema='strata.bench.v1',backend='vulkan',device=[0],model=dict(path='/tmp/model.gguf'),ctx=SET['ctx'],kv=SET['kv'],chunk=512,mtp=0,tg=SET['tg'],reps=SET['reps'],results=[dict(pp=p,prefill_tok_s=dict(min=10,mean=11,max=12),decode_tok_s=dict(min=20,mean=21,max=22)) for p in SET['pp']],r6=dict(threads=6,short=32,rows=[dict(pp=p,samples=copy.deepcopy(self.sig[p])) for p in SET['pp']]))
        self.log='PS5 profile: GTT true, mapped upload true, host reserve 1536 MiB\nPS5 fresh weights: mapped 100 bytes, staged 100 bytes\n'
    def test_record(self):
        rates,sig=r6.checked_rates(self.rec,r6.BASE,SET,'/tmp/model.gguf'); self.assertEqual(rates[32],[11,21]); r6.match_signatures(sig,self.sig)
    def test_metadata_and_fingerprint_rejections(self):
        edits=[lambda d:d.update(backend='cpu'),lambda d:d.update(ctx=1),lambda d:d.update(chunk=99),lambda d:d.update(mtp=1),lambda d:d.update(device=[1]),lambda d:d['r6'].update(threads=8),lambda d:d['results'].pop(),lambda d:d['results'][0]['decode_tok_s'].update(mean=float('nan')),lambda d:d['results'][0]['decode_tok_s'].update(mean=0),lambda d:d['r6']['rows'][0]['samples'].pop(),lambda d:d['r6']['rows'][0]['samples'][0].update(tokens=[0]),lambda d:d['r6']['rows'][0]['samples'][0].update(stopped=True),lambda d:d['r6']['rows'][0]['samples'][0]['top5'][0].__setitem__(1,float('inf'))]
        for edit in edits:
            with self.subTest(edit=edit):
                d=copy.deepcopy(self.rec); edit(d)
                with self.assertRaises(r6.Reject): r6.checked_rates(d,r6.BASE,SET,'/tmp/model.gguf')
    def test_numerics(self):
        for edit in [lambda s:s[32][0]['tokens'].__setitem__(0,8),lambda s:s[32][0]['top5'][0].__setitem__(0,8),lambda s:s[32][0]['top5'][0].__setitem__(1,-1.1)]:
            d=copy.deepcopy(self.sig); edit(d)
            with self.assertRaises(r6.Reject): r6.match_signatures(self.sig,d)
        d=copy.deepcopy(self.sig); d[32][0]['top5'][0][1]-=.01; r6.match_signatures(self.sig,d)
    def test_dispatch_noop(self):
        r6.evidence(self.log,r6.BASE)
        for k,v in [('gemv',64),('small',1),('attn',512),('zero',1),('sched','auto')]:
            with self.subTest(k=k):
                with self.assertRaises(r6.Reject):r6.evidence(self.log,{**r6.BASE,k:v})
        r6.evidence(self.log+'PS5 R6 exercised: gemv_q_ps5_64\n',{**r6.BASE,'gemv':64})
        r6.evidence(self.log+'PS5 R6 exercised: hybrid_prefill\n',{**r6.BASE,'sched':'auto'})
        with self.assertRaises(r6.Reject):r6.useful_change(r6.BASE,{**r6.BASE,'gtt':0},{'marks':[]},{'marks':[]})
        with self.assertRaises(r6.Reject):r6.useful_change(r6.BASE,{**r6.BASE,'direct':0},{'mapped':0},{'mapped':0})
    def test_metrics_guard(self):
        a={1:[100.,100.],2:[100.,100.]}; b={1:[300.,90.],2:[200.,200.]}; score,worst=r6.metrics(b,a); self.assertGreater(score,1.03); self.assertLess(worst,.95)
        with self.assertRaises(r6.Reject):r6.stable(a,b)
    def test_settings_and_env(self):
        with self.assertRaises(r6.Reject):r6.settings_ok({**SET,'pp':[4096]})
        with self.assertRaises(r6.Reject):r6.knobs_ok({**r6.BASE,'threads':True})
        with self.assertRaises(r6.Reject):r6.knobs_ok({**r6.BASE,'gtt':0,'zero':1})
        old=os.environ.get('STRATA_EXPERT_MB');os.environ['STRATA_EXPERT_MB']='999'
        try:self.assertNotIn('STRATA_EXPERT_MB',r6.child_env(r6.BASE))
        finally:
            if old is None:os.environ.pop('STRATA_EXPERT_MB')
            else:os.environ['STRATA_EXPERT_MB']=old
    def test_shards_and_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); self.assertEqual([x.name for x in r6.model_files(root/'a-00001-of-00002.gguf')],['a-00001-of-00002.gguf','a-00002-of-00002.gguf'])
            with self.assertRaises(r6.Reject):r6.model_files(root/'a-00002-of-00002.gguf')
            p=root/'profile.json';r6.atomic_json(p,{'x':1});r6.atomic_json(p,{'x':2});self.assertEqual(json.loads(p.read_text()),{'x':2});self.assertEqual(list(root.iterdir()),[p])
    def test_timeout_and_child_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with self.assertRaisesRegex(r6.Reject,'timeout'):r6.run_child([sys.executable,'-c','import time;time.sleep(5)'],os.environ.copy(),root/'timeout.log',.1,0)
            with self.assertRaisesRegex(r6.Reject,'exited 7'):r6.run_child([sys.executable,'-c','raise SystemExit(7)'],os.environ.copy(),root/'fail.log',2,0)
    def test_cli_mock_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=root/'model.gguf';model.write_bytes(b'synthetic fixture only'); binary=root/'mock-engine';binary.write_text(FAKE);binary.chmod(0o755)
            vk=root/'vulkaninfo';vk.write_text('#!/bin/sh\nprintf "deviceName = Mock AMD\\ndriverVersion = 0.0\\nvendorID = 0x1002\\n"\n');vk.chmod(0o755)
            profile=root/'profile.json';env=os.environ.copy();env['PATH']=str(root)+':'+env['PATH'];env.pop('MOCK_REGRESSION',None)
            common=['--model',str(model),'--bin',str(binary),'--profile',str(profile)]
            def call(mode,extra,ok=True,env_=env):
                p=subprocess.run([sys.executable,str(HERE/'autotune-r6.py'),mode,*common,*extra],env=env_,capture_output=True,text=True,timeout=45)
                self.assertEqual(p.returncode==0,ok,p.stdout+'\n'+p.stderr);return p
            call('tune',['--budget','2','--tg','2','--reps','2','--min-ram-mb','128','--out',str(root/'search')]);p=json.loads(profile.read_text());self.assertEqual(p['status'],'approved');self.assertEqual(p['knobs']['gemv'],128)
            self.assertIn('FAKE RUN chat 128',call('run',['chat']).stdout)
            call('verify',['--out',str(root/'verify')]);self.assertIn('last_verified',json.loads(profile.read_text()))
            optimized=root/'candidate-engine';optimized.write_text(FAKE+'\n# separate binary\n');optimized.chmod(0o755)
            call('compare-pgo',['--candidate-bin',str(optimized),'--out',str(root/'pgo')]);self.assertTrue(json.loads((root/'pgo/pgo-comparison.json').read_text())['passed'])
            model.write_bytes(b'changed fixture, new hash');self.assertIn('stale',call('run',['chat'],False).stderr);model.write_bytes(b'synthetic fixture only')
            call('verify',['--out',str(root/'regression')],False,{**env,'MOCK_REGRESSION':'1'});self.assertEqual(json.loads(profile.read_text())['status'],'revoked');call('run',['chat'],False)
    def test_random_metrics(self):
        import random
        rng=random.Random(20261126)
        for _ in range(5000):
            a={i:[rng.uniform(.1,100),rng.uniform(.1,100)] for i in range(4)};q=rng.uniform(.1,3);b={i:[x*q for x in a[i]] for i in a}
            score,worst=r6.metrics(b,a);self.assertTrue(math.isclose(score,q));self.assertTrue(math.isclose(worst,q))
if __name__=='__main__': unittest.main(verbosity=2)
