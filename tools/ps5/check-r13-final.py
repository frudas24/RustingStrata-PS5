#!/usr/bin/env python3
"""Host-only checks: source ordering and the real shell gate with mock Cargo.
The mock never executes Rust, Vulkan or the PS5 physical tests.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class SourceOrder(unittest.TestCase):
    def setUp(self):
        self.source=(ROOT/'src/vulkan/mod.rs').read_text()
    def body(self,start,end):
        a=self.source.index(start); b=self.source.index(end,a+len(start))
        return self.source[a:b]
    def test_drop_commit_after_wait(self):
        s=self.body('    fn graph_drop(', '    fn graph_clear(')
        self.assertLess(s.index('st.graphs.get(&key)'),s.index('self.0.flush(&mut st)?'))
        self.assertLess(s.index('self.0.flush(&mut st)?'),s.index('st.graphs.remove(&key)'))
    def test_readback_guard_before_recording(self):
        s=self.body('    fn dtoh(', '    fn dtod(')
        self.assertLess(s.index('ensure!(st.rec.is_none()'),s.index('self.0.cb(&mut st)?'))
    def test_rejected_nesting_before_mutation(self):
        s=self.body('    fn batch(', '    fn beside_overlaps(')
        self.assertLess(s.index('ensure!(!st.batch'),s.index('st.batch = true'))
        s=self.body('    fn graph(&self', '    fn graph_drop(')
        self.assertLess(s.index('ensure!(st.rec.is_none()'),s.index('self.invalidate_q8_keys()'))
    def test_copy_overlap_guard_before_recording(self):
        s=self.body('    fn dtod(', '    fn memset(')
        self.assertLess(s.index('so.abs_diff(d_o) >= bytes as u64'),s.index('self.0.cb(&mut st)?'))

MOCK='''#!/usr/bin/env python3
import os,sys
name=next(x for x in sys.argv if x in ('q8_graph_cache_lifetime','rejected_graph_operations_are_inert'))
assert os.environ['STRATA_PS5']=='1' and 'STRATA_NO_GRAPH' not in os.environ
marker={'q8_graph_cache_lifetime':'R13 LAUNCH Q8 GRAPH PASS: replay, discard and scratch growth','rejected_graph_operations_are_inert':'R13 FINAL GRAPH PASS: rejected operations inert, device copy disjoint'}[name]
mode=os.environ['MOCK_FINAL'] if name=='rejected_graph_operations_are_inert' else 'valid'
if mode!='missing': print(marker)
if mode=='duplicate': print(marker)
print('test result: ok. '+('0' if mode=='zero' else '1')+' passed; 0 failed; 0 ignored;')
if mode=='extra': print('test result: FAILED. 0 passed; 1 failed; 0 ignored;')
if mode=='exit': sys.exit(1)
'''
class PhysicalGate(unittest.TestCase):
    def run_gate(self,mode):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cargo';p.write_text(MOCK);p.chmod(0o755)
            env={**os.environ,'PATH':d+os.pathsep+os.environ['PATH'],'MOCK_FINAL':mode,'STRATA_NO_GRAPH':'1'}
            return subprocess.run(['bash',str(HERE/'test-r13-final-graph.sh')],env=env,capture_output=True,text=True,timeout=30)
    def test_valid(self):
        r=self.run_gate('valid');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
    def test_reject_invalid(self):
        for mode in ('missing','duplicate','zero','extra','exit'):
            with self.subTest(mode=mode): self.assertNotEqual(self.run_gate(mode).returncode,0)

if __name__=='__main__':unittest.main(verbosity=2)
