#!/usr/bin/env python3
"""Final launch-gate tests; mock engine results do not count as physical inference."""
import argparse
import importlib.util
import json
import os
import contextlib
import io
import sys
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('campaign',HERE/'r13-campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r6=c.r6

class Launch(unittest.TestCase):
    def test_mark_parser_rejects_malformed_extra_records(self):
        good=''.join('PS5 R6 exercised: '+m+'\n' for m in c.expected(32,'off'))
        self.assertEqual(set(c.check_marks(good,32,'off')),c.expected(32,'off'))
        for malformed in ('other-marker','other marker','', 'bad!','r13_int_64_iq4xs extra'):
            with self.assertRaises(r6.Reject): c.check_marks(good+'PS5 R6 exercised: '+malformed+'\n',32,'off')
    def test_child_cannot_finish_after_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(r6.Reject,'timeout'):
                r6.run_child([sys.executable,'-c','import time; time.sleep(.15)'],os.environ.copy(),Path(d)/'timeout.log',.05,0)
    def test_batch_gate_rejects_invalid_bytes_even_with_python_optimized(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            for mode in ('valid','cpu','skip','truncated','nan','bad_token','different_vocab'):
                base=p/mode
                for frac in (0,1):
                    vocab=3 if mode=='different_vocab' and frac==1 else 2
                    text=f'R11U1_BATCH layer=0 rows=6 staged=3 cpu={1 if mode=="cpu" else 0}\n'
                    text+=f'R11U1_BATCH_GUARD VOCAB: {vocab}\nR11U1_BATCH_GUARD ARM PASS: frac={frac} 6 finite full-logit rows\n'
                    text+=f'test result: ok. {0 if mode=="skip" else 1} passed; 0 failed; 0 ignored;\n'
                    bits=(struct.pack('<I',vocab if mode=='bad_token' else 1)+struct.pack('<'+'f'*vocab,*([float('nan') if mode=='nan' else 0.]*vocab)))*6
                    if mode=='truncated':bits=bits[:-4]
                    Path(f'{base}.{frac}.log').write_text(text);Path(f'{base}.{frac}.bits').write_bytes(bits)
                # Compile with optimize=2 to remove asserts, exactly as Python -OO.
                code=compile((HERE/'verify-r11u1-batch.py').read_text(),'verify-r11u1-batch.py','exec',optimize=2)
                with patch.object(sys,'argv',['verify-r11u1-batch.py',str(base)]),contextlib.redirect_stdout(io.StringIO()):
                    if mode=='valid':exec(code,{'__name__':'__main__'})
                    else:
                        with self.assertRaises(SystemExit):exec(code,{'__name__':'__main__'})
    def campaign(self,*,quality=False,mutate=False,noop=False):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); corpus=p/'corpus';corpus.write_text('independent fixture')
            a=argparse.Namespace(ctx=8192,pp='128',tg=2,reps=2,timeout=5,min_ram_mb=128,kv='q8',threads=6,chunk=512,
                model=p/'model.gguf',bin=p/'engine',out=p/'out',base_tile=0,tile=0 if noop else 32,lut='off',profile_only=False,
                corpus=corpus if quality else None,ctx_len=128,chunks=1)
            calls=[]
            def child(cmd,env,log,timeout,min_ram):
                tile=int(env['STRATA_PS5_R12_IQ_INT_TILE']);lut=env['STRATA_PS5_R13_LUT'];calls.append((tile,lut))
                text='PS5 profile: GTT true, mapped upload true\nPS5 fresh weights: mapped 1 bytes, staged 1 bytes\n'
                text+=''.join('PS5 R6 exercised: '+m+'\n' for m in c.expected(tile,lut))
                if quality:
                    n,nv,nc=128,2,1;scored=n-1-n//2
                    payload=b'_logits_'+struct.pack('<III',n,nv,nc)+bytes(4*n*nc)
                    payload+=struct.pack('<ffHH',0.,-0.69314718,0,0)*scored
                    Path(cmd[cmd.index('--kld-base')+1]).write_bytes(payload)
                    text+=f'PPL = 2.0 over {nc} chunks of {n} tokens ({scored} scored)\n'
                    if mutate:corpus.write_text('different fixture')
                else:
                    rate=110. if tile else 100.;stat=dict(min=rate,mean=rate,max=rate)
                    sample=dict(tokens=[1,2],top5=[[i,-i-1.] for i in range(5)],stopped=False)
                    record=dict(schema='strata.bench.v1',backend='vulkan',device=[0],model=dict(path=str(a.model)),ctx=a.ctx,kv=a.kv,tg=a.tg,reps=a.reps,chunk=a.chunk,mtp=0,
                        results=[dict(pp=128,prefill_tok_s=stat,decode_tok_s=stat)],r6=dict(threads=6,short=32,rows=[dict(pp=128,samples=[sample]*2)]))
                    Path(cmd[cmd.index('--json-out')+1]).write_text(json.dumps(record))
                    if mutate:(a.out/'holdout.txt').write_text('different fixture')
                log.write_text(text);return {}
            with patch.object(r6,'identity',return_value={'mock':True}),patch.object(r6,'run_child',side_effect=child):
                rc=c.run_campaign(a)
            report=json.loads((a.out/'summary.json').read_text())
            if mutate or noop:
                self.assertEqual(rc,1);self.assertFalse(report['accepted']);self.assertIn('changed' if mutate else 'no experimental change',report['error'])
                if noop:self.assertEqual(calls,[])
            elif quality:
                self.assertEqual(rc,0);self.assertTrue(report['encoded_identity']);self.assertEqual(calls,[(0,'off'),(32,'off')]);self.assertEqual((a.out/'off.bin').read_bytes(),(a.out/'candidate-off.bin').read_bytes())
            else:
                self.assertEqual(rc,0);self.assertTrue(report['accepted']);self.assertEqual(calls,[(0,'off'),(32,'off'),(32,'off'),(0,'off')]*2)
    def test_tile_only_benchmark(self): self.campaign()
    def test_tile_only_quality(self): self.campaign(quality=True)
    def test_noop_rejected(self): self.campaign(noop=True)
    def test_changed_corpus_rejected(self): self.campaign(quality=True,mutate=True)
    def test_changed_holdout_rejected(self): self.campaign(mutate=True)

if __name__=='__main__':unittest.main(verbosity=2)
