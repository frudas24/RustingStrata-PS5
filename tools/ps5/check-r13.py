#!/usr/bin/env python3
"""Independent source-level LUT arithmetic/coverage and adversarial dispatch gates.
Does not compile GLSL or execute Rust/Vulkan.
"""
import importlib.util
import itertools
import pathlib
import re
import struct
import unittest

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('campaign',HERE/'r13-campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


def rust_array(name):
    text=(ROOT/'src/quant/tables.rs').read_text()
    match=re.search(r'pub static '+name+r':\s*\[\w+;\s*\d+\]\s*=\s*\[(.*?)\];',text,re.S)
    return [int(v.strip(),0) for v in match.group(1).split(',') if v.strip()]


class Lookup(unittest.TestCase):
    def test_cooperative_initialization(self):
        # All table entries have exactly one writer, including hypothetical smaller WGs.
        for length,wg in itertools.product((256,512),(32,64,128,256)):
            writes=[i for lane in range(wg) for i in range(lane,length,wg)]
            self.assertEqual(sorted(writes),list(range(length)))
        src=(ROOT/'src/vulkan/shaders/deq.glsl').read_text()
        self.assertIn('s_iq2xs[i] = iq2xs_grid[i]; barrier();',src)
        self.assertIn('g = IQ2XS_GRID[k & 511];',src)
    def test_grid_data_and_indices(self):
        g2=rust_array('IQ2XS_GRID');g3=rust_array('IQ3XXS_GRID')
        self.assertEqual((len(g2),len(g3)),(512,256))
        # All packed IQ2 index/sign combinations, preserving the original byte representation.
        raw=b''.join(struct.pack('<Q',v) for v in g2)
        for k in range(65536):
            i=k&511;self.assertEqual(struct.unpack_from('<Q',raw,8*i)[0],g2[i])
        raw=b''.join(struct.pack('<I',v) for v in g3)
        for i in range(256): self.assertEqual(struct.unpack_from('<I',raw,4*i)[0],g3[i])
        print('R13 independent IQ grid index cases: 65792')
    def test_iq4_packed_bytes(self):
        # Match the 16 signed IQ4 values from the canonical constants, not float comparisons.
        src=(ROOT/'src/vulkan/shaders/deq.glsl').read_text()
        codes=[int(v,16) for v in re.search(r'uint64_t t = nib < 8u \? (0x[0-9A-F]+)ul : (0x[0-9A-F]+)ul;',src).groups()]
        ref=[-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]
        unpack=lambda n: (codes[n//8]>>(8*(n%8)))&255
        table=[unpack(i&15)|(unpack(i>>4)<<8) for i in range(256)]
        for i in range(16): self.assertEqual(unpack(i),ref[i]&255)
        # Exhaust every 4-nibble combination of the shader's packed lookup.
        for n in range(65536):
            a=[(n>>(4*i))&15 for i in range(4)]
            packed=table[a[0]|a[1]<<4]|table[a[2]|a[3]<<4]<<16
            self.assertEqual(packed,sum((ref[v]&255)<<(8*i) for i,v in enumerate(a)))
        print('R13 independent IQ4 packed cases: 65536')
    def test_shader_wiring(self):
        build=(ROOT/'build.rs').read_text();host=(ROOT/'src/vulkan/mod.rs').read_text();pre=(ROOT/'src/vulkan/preflight.rs').read_text()
        for name in [c.IQ2+'_lut',*(x+'_lut' for x in c.INT.values())]:
            self.assertIn('("'+name+'",',build);self.assertIn('/'+name+'.spv',host)
        self.assertIn('r13_gemm_kernel',pre)
        self.assertIn('self.0.smem_limit >= 64 * 1024',host)
        self.assertIn('STRATA_R13_LUT', (ROOT/'src/vulkan/shaders/gemmi.glsl').read_text())


class Gates(unittest.TestCase):
    def check(self,tile,lut,test,marks):
        return c.check_marks('\n'.join('PS5 R6 exercised: '+x for x in marks),tile,lut,test)
    def test_all_legal_dispatch_sets(self):
        cases=0
        for tile,lut in itertools.product((0,32,64),('off','iq2','int','all')):
            want=c.expected(tile,lut)
            self.assertEqual(self.check(tile,lut,'model',want),sorted(want));cases+=1
            for mark in c.KNOWN:
                with self.assertRaises(c.r6.Reject): self.check(tile,lut,'model',want^{mark})
        print('R13 model arm combinations:',cases,'single-marker corruptions:',cases*len(c.KNOWN))
    def test_distinct_marker_arms(self):
        for tile in (0,32,64):
            for lut in ('off','int'):
                self.check(tile,lut,'int',c.expected(tile,lut,'int'))
        self.check(0,'iq2','iq2',{c.IQ2+'_lut'})
        with self.assertRaises(c.r6.Reject): self.check(0,'iq2','iq2',{c.IQ2})
        with self.assertRaises(c.r6.Reject): self.check(32,'int','int',{c.INT[32]+'_lut','r13_lut_iq3xxs'})
    def test_profile_summary(self):
        text='r13_mem_alloc,host_staging,requested=64,allocated=128,type=2,heap=0,flags=0xe,host_cached=true\nr13_mem_copy,staging_read,bytes=16,ns=10\nr13_q8,hit,rows=1,cols=256,stride=256,type=IQ3_XXS\n'
        report=c.profile_summary(text)
        self.assertEqual(report['host_copies']['staging_read']['bytes'],16)
        self.assertEqual(report['q8']['hit:IQ3_XXS'],1)
        with self.assertRaises(c.r6.Reject):c.profile_summary(text.splitlines()[0])
    def test_profile_flags_do_not_leak(self):
        self.assertEqual(c.flags(64,'off'),dict(STRATA_PS5_R11_IQ32='1',STRATA_PS5_R12_IQ_INT_TILE='64',STRATA_PS5_R13_LUT='off'))
        self.assertNotIn('STRATA_PS5_R13_MEM_TRACE',c.flags(32,'all'))

if __name__=='__main__': unittest.main(verbosity=2)
