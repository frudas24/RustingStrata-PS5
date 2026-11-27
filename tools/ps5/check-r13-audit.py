#!/usr/bin/env python3
"""Host-only adversarial checks. Mock Cargo validates gates, never physical GPU output."""
import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('campaign',HERE/'r13-campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
r6=c.r6


def marks(values):
    return ''.join(f'PS5 R6 exercised: {m}\n' for m in sorted(values))


def lds_lines():
    return ['R13 LDS IQ2 13824 17920']+[f'R13 LDS {t} {k} {n} {n+1024}'
        for t in (18,23) for k,n in [('gemmi_dp_ps5_lut',40960),('gemmi_dp_ps5_iq32_lut',20992),('gemmi_dp_ps5_iq64_lut',27648)]]


class Gates(unittest.TestCase):
    def test_each_format_and_fallback(self):
        cases=0
        for tile,lut,test in itertools.product((0,32,64),('off','iq2','int','all'),('iq2','int','model')):
            want=c.expected(tile,lut,test)
            self.assertEqual(c.check_marks(marks(want|{'gtt_nonlocal'}),tile,lut,test),sorted(want))
            for m in c.KNOWN:
                with self.assertRaises(r6.Reject):c.check_marks(marks(want^{m}),tile,lut,test)
                cases+=1
            for m in ('hybrid_prefill','r11_down_rpw','gemmi_dp_ps5_64','unknown_future_marker'):
                with self.assertRaises(r6.Reject):c.check_marks(marks(want|{m}),tile,lut,test)
                cases+=1
        print(f'R13 adversarial dispatch mutations rejected: {cases}')
    def test_r12_exact_route_markers(self):
        for t in (0,32,64):
            want=c.expected(t,'off')
            r6.require_routed_integer(want|{'gtt_nonlocal'},t)
            for m in c.KNOWN|{'hybrid_prefill'}:
                with self.assertRaises(r6.Reject):r6.require_routed_integer(want^{m},t)
    def test_lds_records(self):
        rows=lds_lines()
        self.assertEqual(c.check_lds('\n'.join(rows)),7)
        invalid=[rows[:-1],rows+[rows[0]],rows[:1]*7,rows[:-1]+['R13 LDS 23 unknown 1024 2048'],
                 ['R13 LDS IQ2 13824 13824']+rows[1:],['R13 LDS IQ2 13824 65537']+rows[1:],
                 ['R13 LDS IQ2 bad 17920']+rows[1:]]
        for v in invalid:
            with self.assertRaises((r6.Reject,ValueError)):c.check_lds('\n'.join(v))
    def test_pairwise_return_gate(self):
        sample=dict(tokens=[1,2],stopped=False,top5=[[i,-i-1.] for i in range(5)])
        def run(rate):return dict(rates={128:[rate,rate]},signatures={128:[sample]})
        a=run(100);b=run(110);weak=run(101)
        self.assertTrue(c.compare([a,b,b,a])['accepted'])
        v=c.compare([a,b,weak,a]);self.assertGreater(v['score'],1.03);self.assertFalse(v['accepted'])
        self.assertFalse(c.compare([a,weak,b,a])['accepted'])
    def test_alignment_counterexample(self):
        # Shader indexes vec4 at floor((row*stride+xc)/4). Odd strides lose the low bits.
        for stride in range(256,264):
            for r in range(1,5):
                off=r*stride; actual=(off//4)*4
                self.assertEqual(actual==off,off%4==0)
            self.assertEqual((stride//4)*4==stride,stride%4==0)
        shader=(ROOT/'src/vulkan/shaders/gemm_f.glsl').read_text()
        self.assertIn('buffer_reference_align = 16',shader)
        host=(ROOT/'src/vulkan/mod.rs').read_text()
        self.assertIn('gemm_f_input_alignment(x, x_stride)?;',host)
        self.assertIn('st.find(x, gemm_matrix_bytes(n_x, x_stride, cols)?)?',host)
    def test_zero_write_poison_wiring(self):
        for f in ('r11_iq.rs','r12_iq.rs'):
            src=(ROOT/'tests'/f).read_text()
            self.assertIn('fill(f32::NAN)',src)
            self.assertIn('let oracle=common::up(&g,&initial);',src)
            self.assertIn('v.is_finite()',src)
    def test_swap_telemetry(self):
        with tempfile.TemporaryDirectory() as d:
            stats=r6.run_child([sys.executable,'-c','pass'],os.environ.copy(),Path(d)/'child.log',5,0)
            self.assertGreaterEqual(stats['system_swap_sampled_peak_kib'],stats['system_swap_start_kib'])
            self.assertGreaterEqual(stats['system_swap_sampled_peak_kib'],stats['system_swap_end_kib'])
            self.assertIn('pswpin',stats['system_swap_pages_delta'])
            self.assertIn('pswpout',stats['system_swap_pages_delta'])


MOCK='''#!/usr/bin/env python3
import os,sys
from pathlib import Path
mode=os.environ['AUDIT_MOCK']
args=sys.argv[1:]
def done():print('test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s')
if 'lds_budget' in args:
    rows=['R13 LDS IQ2 13824 17920']+[f'R13 LDS {t} {k} {n} {n+1024}' for t in (18,23) for k,n in [('gemmi_dp_ps5_lut',40960),('gemmi_dp_ps5_iq32_lut',20992),('gemmi_dp_ps5_iq64_lut',27648)]]
    if mode=='dup_lds':rows[-1]=rows[0]
    print('\\n'.join(rows));done();sys.exit(0)
lut=os.environ['STRATA_PS5_R13_LUT'];tile=int(os.environ.get('STRATA_PS5_R12_IQ_INT_TILE','0'))
if 'r11_iq' in args:
    size=17905536; path=os.environ['PS5_R11_GPU_FILE']
    marks=['gemm_f_dp_ps5_iq32'+('_lut' if lut=='iq2' else '')]
    print('R11 IQ GPU parity: 648 cases')
else:
    size=37981440;path=os.environ['PS5_R12_GPU_FILE']; tag={0:'base',32:'32',64:'64'}[tile]
    kernel={0:'gemmi_dp_ps5',32:'gemmi_dp_ps5_iq32',64:'gemmi_dp_ps5_iq64'}[tile]
    if lut=='int':marks=[kernel+'_lut','r13_lut_iq3xxs','r13_lut_iq4xs']
    else:marks=[f'r13_int_{tag}_iq3xxs',f'r13_int_{tag}_iq4xs']+([kernel] if tile else [])
    if mode=='missing_format' and tile==0 and lut=='off':marks.pop()
    if mode=='mixed_fallback' and tile==0 and lut=='off':marks.append('r13_int_f16_iq4xs')
    print('R12 IQ integer GPU parity: 1296 cases')
if mode=='wrong_arm' and lut=='off':marks.append('gemm_f_dp_ps5_iq32_lut')
if mode=='bad_size' and lut=='off':size=4
with open(path,'wb') as f:f.truncate(size)
for mark in marks:print('PS5 R6 exercised: '+mark)
if mode=='zero_tests':print('test result: ok. 0 passed; 0 failed; 0 ignored;')
else:done()
'''

class MockCampaign(unittest.TestCase):
    def run_mock(self,mode):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d); cargo=p/'cargo';cargo.write_text(MOCK);cargo.chmod(0o755)
            env={**os.environ,'PATH':str(p)+os.pathsep+os.environ['PATH'],'AUDIT_MOCK':mode}
            return subprocess.run(['bash',str(HERE/'test-r13-lut.sh'),str(p/'out')],env=env,capture_output=True,text=True,timeout=30)
    def test_valid_mock(self):
        p=self.run_mock('valid');self.assertEqual(p.returncode,0,p.stdout+p.stderr)
    def reject(self,mode):
        p=self.run_mock(mode);self.assertNotEqual(p.returncode,0,p.stdout+p.stderr)
    def test_duplicate_lds(self): self.reject('dup_lds')
    def test_missing_format(self): self.reject('missing_format')
    def test_mixed_fallback(self): self.reject('mixed_fallback')
    def test_wrong_arm(self): self.reject('wrong_arm')
    def test_bad_size(self): self.reject('bad_size')
    def test_zero_tests(self): self.reject('zero_tests')

if __name__=='__main__':unittest.main(verbosity=2)
