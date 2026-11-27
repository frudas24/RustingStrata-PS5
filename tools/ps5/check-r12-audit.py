#!/usr/bin/env python3
"""Host-only negative controls for R12 audit. Fake cargo is never a GPU test."""
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('guard',HERE/'r8-run-guarded.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class Guard(unittest.TestCase):
    def test_empty_proc_fields(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'status'
            p.write_text('VmRSS: 42 kB\nx86_Thread_features:\nName: engine\n')
            self.assertEqual(guard.proc_fields(p,('VmRSS','x86_Thread_features','Missing')),{'VmRSS':'42','x86_Thread_features':'','Missing':''})
            self.assertEqual(guard.proc_fields(p,('VmRSS',)),{'VmRSS':'42'})
    def test_missing_memory_fails_closed(self):
        for v in ('','bad',None,'1023'):
            self.assertFalse(guard.memory_ok(v,1024))
        self.assertTrue(guard.memory_ok('1024',1024))

class ParityScript(unittest.TestCase):
    def campaign(self,mode):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);cargo=p/'cargo'
            cargo.write_text('''#!/bin/bash
set -eu
if [[ " $* " == *" shared_bound "* ]]; then
  echo 'R12 LDS type=18'; echo 'R12 LDS type=23'; exit 0
fi
flag=$STRATA_PS5_R12_IQ_INT_TILE
size=37981440
if [[ $AUDIT_MOCK == size && $flag == 0 ]]; then size=4; fi
truncate -s "$size" "$PS5_R12_GPU_FILE"
echo 'R12 IQ integer GPU parity: 1296 cases'
if [[ $flag != 0 ]]; then echo "PS5 R6 exercised: gemmi_dp_ps5_iq$flag"; fi
tag=$flag; [[ $flag != 0 ]] || tag=base
echo "PS5 R6 exercised: r13_int_${tag}_iq3xxs"
echo "PS5 R6 exercised: r13_int_${tag}_iq4xs"
if [[ $AUDIT_MOCK == leak && $flag == 0 ]]; then echo 'PS5 R6 exercised: gemmi_dp_ps5_iq32'; fi
if [[ $AUDIT_MOCK == wrong && $flag == 32 ]]; then echo 'PS5 R6 exercised: gemmi_dp_ps5_iq64'; fi
''');cargo.chmod(0o755)
            env={**os.environ,'PATH':str(p)+os.pathsep+os.environ['PATH'],'AUDIT_MOCK':mode}
            return subprocess.run(['bash',str(HERE/'test-r12-iq.sh'),str(p/'out')],env=env,capture_output=True,text=True)
    def test_valid_mock(self):
        r=self.campaign('valid');self.assertEqual(r.returncode,0,r.stderr)
    def test_leaked_candidate(self):
        r=self.campaign('leak');self.assertNotEqual(r.returncode,0);self.assertIn('unexpected IQ tile',r.stderr)
    def test_wrong_tile(self):
        r=self.campaign('wrong');self.assertNotEqual(r.returncode,0);self.assertIn('unexpected IQ tile',r.stderr)
    def test_truncated_fingerprint(self):
        r=self.campaign('size');self.assertNotEqual(r.returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
