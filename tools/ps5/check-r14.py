#!/usr/bin/env python3
"""Independent dense geometry and R14 gate tests; never a physical shader test."""
import copy
import importlib.util
import itertools
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import sys
import json

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r14',HERE/'r14-campaign.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r6=c.r6
S=dict(ctx=65536,kv='q8',pp=[65024],tg=2,reps=2,timeout=10,min_ram_mb=128)
MODEL='/tmp/model.gguf'


def record():
    sample=dict(top5=[[i,-1.-i] for i in range(5)],tokens=[1,2],predictions=[0,0],stopped=False)
    stat=dict(min=100,mean=100,max=100)
    return dict(schema='strata.bench.v1',backend='vulkan',device=[0],model=dict(path=MODEL),ctx=S['ctx'],kv=S['kv'],chunk=512,mtp=0,tg=S['tg'],reps=S['reps'],r14_decode='fixed',r14_vocab=5,r14_logit_rows=0,results=[dict(pp=65024,prefill_tok_s=stat.copy(),decode_tok_s=stat.copy())],r6=dict(threads=6,short=32,rows=[dict(pp=65024,input_ids=[1]*65024,samples=[copy.deepcopy(sample),copy.deepcopy(sample)])]))


def text(marks):return ''.join(f'PS5 R6 exercised: {m}\n' for m in sorted(marks))


class Geometry(unittest.TestCase):
    def test_dense_lanes_and_edges(self):
        cases=0
        for tile in (32,64,128):
            coords=[]
            for w,lane in itertools.product(range(8),range(32)):
                bm={32:16*(w&1),64:16*(w&3),128:32*(w>>1)}[tile]
                bn={32:16*(w>>1),64:32*(w>>2),128:32*(w&1)}[tile]
                for mt,nt,i in itertools.product(range(2 if tile==128 else 1),range(1 if tile==32 else 2),range(8)):
                    coords.append((bm+16*mt+(lane>>2)+8*(i>>2),bn+16*nt+2*(lane&3)+(i&1)+8*((i>>1)&1)))
            self.assertEqual(len(coords),tile*64)
            self.assertEqual(set(coords),set(itertools.product(range(tile),range(64))))
            # Every tail rectangle independently matches the intersection of the
            # workgroup domain with active rows/positions. Larger grids partition it.
            for n,r in itertools.product(range(1,tile+1),range(1,65)):
                actual=[(x,y) for x,y in coords if x<n and y<r]
                self.assertEqual(len(actual),n*r);cases+=1
            for n,r in itertools.product([1,31,32,33,63,64,65,127,128,129,255,512,65024],[19,63,64,65,512,2048,4096,8192]):
                xs=[min(tile,n-i) for i in range(0,n,tile)];ys=[min(64,r-i) for i in range(0,r,64)]
                self.assertEqual(sum(xs)*sum(ys),n*r);cases+=1
        print('R14 dense geometry checks:',cases)

    def test_shader_reuse_and_contract(self):
        root=HERE.parents[1];shader=(root/'src/vulkan/shaders/gemmi.glsl').read_text();build=(root/'build.rs').read_text()
        self.assertIn('("gemmi_dp_ps5_64", "gemmi_dp.comp", &["-DSM=2"]',build)
        self.assertIn('("gemmi_dp_ps5_32", "gemmi_dp.comp", &["-DSM=1", "-DSTRATA_PS5_SMALL=1"]',build)
        self.assertIn('kz = gl_WorkGroupID.z',shader)
        # R14 sets z=1, hence k0=kz=0, k1=nk. All types retain the same K sequence.
        for nk in [4,8,32,64]:self.assertEqual(list(range(0,min(nk,nk))),list(range(nk)))
        host=(root/'src/vulkan/mod.rs').read_text()
        self.assertIn('[grid[0], nd.div_ceil(r14_tile), 1]',host)
        self.assertIn('if f {',host)


class Gates(unittest.TestCase):
    def test_marks(self):
        rejected=0
        for tile,types,iq in itertools.product((0,32,64),('q5','q6','both'),(None,0,32,64)):
            want=c.dense_marks(tile,types)|(c.r13.expected(iq,'off') if iq is not None else set())
            c.check_marks(text(want),tile,types,iq)
            for m in want:
                with self.assertRaises(r6.Reject):c.check_marks(text(want-{m}),tile,types,iq)
                rejected+=1
            for m in (c.DENSE|c.r13.KNOWN|{'unrelated'})-want:
                with self.assertRaises(r6.Reject):c.check_marks(text(want|{m}),tile,types,iq)
                rejected+=1
        print('R14 marker mutations rejected:',rejected)

    def test_fixed_valid(self):
        rates,sig,nv,rows=c.fixed_record(record(),r6.BASE,S,MODEL,'fixed');self.assertEqual((nv,rows),(5,0));c.match(sig,sig)

    def test_fixed_rejects_invalid(self):
        for edit in [lambda r:r.pop('r14_decode'),lambda r:r.update(r14_logit_rows=6),lambda r:r.update(r14_vocab=4),lambda r:r['r6']['rows'][0]['samples'][0].update(predictions=[0]),lambda r:r['r6']['rows'][0]['samples'][0].update(predictions=[0,5]),lambda r:r['r6']['rows'][0]['samples'][0].update(tokens=[1],stopped=True),lambda r:r['r6']['rows'][0]['samples'][0].update(tokens=[1,5]),lambda r:r['r6']['rows'][0]['samples'][0].update(predictions=[False,0]),lambda r:r['r6']['rows'][0]['input_ids'].pop()]:
            r=record();edit(r)
            with self.assertRaises(r6.Reject):c.fixed_record(r,r6.BASE,S,MODEL,'fixed')

    def test_predictions_must_match(self):
        _,a,_,_=c.fixed_record(record(),r6.BASE,S,MODEL,'fixed');b=copy.deepcopy(a);b[65024][1]['predictions'][1]=3
        with self.assertRaisesRegex(r6.Reject,'argmax'):c.match(a,b)
        b=copy.deepcopy(a);b[65024][1]['input_sha256']='wrong'
        with self.assertRaisesRegex(r6.Reject,'prefill'):c.match(a,b)

    def test_eos_only_natural(self):
        r=record();r.pop('r14_decode');sample=r['r6']['rows'][0]['samples'][0];sample.update(tokens=[1],stopped=True)
        with self.assertRaises(r6.Reject):r6.checked_rates(r,r6.BASE,S,MODEL)
        r6.checked_rates(r,r6.BASE,S,MODEL,allow_eos=True)
        sample.update(stopped=False)
        with self.assertRaises(r6.Reject):r6.checked_rates(r,r6.BASE,S,MODEL,allow_eos=True)
        sample.update(tokens=[1,2,3],stopped=True)
        with self.assertRaises(r6.Reject):r6.checked_rates(r,r6.BASE,S,MODEL,allow_eos=True)

    def test_raw_gate(self):
        r=record();r.update(r14_decode='fixed_logits',r14_logit_rows=6)
        _,sig,nv,rows=c.fixed_record(r,r6.BASE,S,MODEL,'fixed_logits')
        good=struct.pack('<5f',4.,3.,2.,1.,0.)*rows
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'logits';p.write_bytes(good);self.assertEqual(c.check_raw(p,sig,nv,rows),r6.digest(p))
            for bad in (good[:-4],good+b'1234',struct.pack('<f',float('nan'))+good[4:],struct.pack('<f',-1.)+good[4:]):
                p.write_bytes(bad)
                with self.assertRaises(r6.Reject):c.check_raw(p,sig,nv,rows)

    def test_profile_boundaries(self):
        log="\n".join(["r13_mem_alloc,staging,requested=64,allocated=64,type=5,heap=0,flags=0xe,host_cached=true",
            "r13_mem_copy,staging_write,bytes=64,ns=100",'r7_load {"load_ns":200}',"r13_mem_copy,staging_write,bytes=8,ns=10",
            "r13_q8,hit,rows=32,cols=256,stride=256,type=Q5_K","prof gemmi_dp:Q5_K  1.00 ms 100.0% 2 calls 500.00 us"])
        report=c.profile_report(log);self.assertEqual(report['host_copies_before_load_marker']['staging_write']['bytes'],64)
        self.assertEqual(report['host_copies_after_load_marker']['staging_write']['bytes'],8)
        self.assertEqual(report['gpu_intervals']['gemmi_dp:Q5_K']['calls'],2)
        with self.assertRaises(r6.Reject):c.profile_report(log+'\n'+log)

    def test_paired_speed_gate(self):
        rates,sig,nv,_=c.fixed_record(record(),r6.BASE,S,MODEL,'fixed')
        def run(pp,tg):return dict(rates={65024:[pp,tg]},signatures=sig,vocab=nv)
        self.assertTrue(c.compare([run(100,100),run(115,100),run(115,100),run(100,100)])['accepted'])
        self.assertFalse(c.compare([run(100,100),run(115,100),run(101,100),run(100,100)])['accepted'])
        self.assertFalse(c.compare([run(100,100),run(130,94),run(130,94),run(100,100)])['accepted'])

    def test_mock_campaign(self):
        # Exercise the real runner under a fake engine, including exact flags,
        # fixed-work metadata, all four arms and natural EOS without scoring.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); calls=[]
            def child(cmd,env,log,timeout,min_ram):
                tile=int(env['STRATA_PS5_R14_DENSE_TILE']);iq=int(env['STRATA_PS5_R12_IQ_INT_TILE']);calls.append((tile,iq))
                log.write_text('PS5 profile: GTT true, mapped upload true\nPS5 fresh weights: mapped 1 bytes, staged 1 bytes\n'+text(c.dense_marks(tile,'both')|c.r13.expected(iq,'off')))
                r=record();r['model']['path']=str(root/'model');rate=115 if tile else 100;r['results'][0]['prefill_tok_s']=dict(min=rate,mean=rate,max=rate)
                if '--fixed-decode' not in cmd:
                    for key in ('r14_decode','r14_vocab','r14_logit_rows'):r.pop(key)
                    for sample in r['r6']['rows'][0]['samples']:sample.update(tokens=[1],stopped=True)
                if '--fixed-logits' in cmd:
                    r.update(r14_decode='fixed_logits',r14_logit_rows=6);Path(cmd[cmd.index('--fixed-logits')+1]).write_bytes(struct.pack('<5f',4.,3.,2.,1.,0.)*6)
                Path(cmd[cmd.index('--json-out')+1]).write_text(json.dumps(r));return {}
            with patch.object(r6,'identity',return_value={'mock':True}),patch.object(r6,'run_child',side_effect=child):
                for mode in ('bench','natural','quality'):
                    calls.clear();out=root/mode
                    with patch.object(sys,'argv',['r14','--model',str(root/'model'),'--out',str(out),'--tile','64','--ctx','65536','--pp','65024','--tg','2','--reps','2','--mode',mode]):self.assertEqual(c.main(),0)
                    summary=json.loads((out/'summary.json').read_text());self.assertTrue(summary['validated']);self.assertEqual(summary['accepted'],mode=='bench')
                    self.assertEqual(calls,([(0,64),(64,64),(64,64),(0,64)] if mode=='bench' else [(0,64),(64,64)])*2)


if __name__=='__main__':unittest.main(verbosity=2)
