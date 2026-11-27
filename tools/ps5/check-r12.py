#!/usr/bin/env python3
"""Independent geometry and diagnostic-gate tests. Not a Vulkan execution."""
import importlib.util
import itertools
import json
import pathlib
import struct
import tempfile
import unittest

HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj

bound=load('compare-r12-boundaries')
base_tests=load('check-r10')

class Geometry(unittest.TestCase):
    def test_routed_subtile_coverage(self):
        cases=0
        for tile in (32,64):
            coords=[]
            for wave,lane in itertools.product(range(8),range(32)):
                bm=16*(wave&1) if tile==32 else 16*(wave&3)
                bn=16*(wave>>1) if tile==32 else 32*(wave>>2)
                for nt,i in itertools.product(range(tile//32),range(8)):
                    coords.append((bm+(lane>>2)+8*(i>>2),bn+16*nt+2*(lane&3)+(i&1)+8*((i>>1)&1)))
            self.assertEqual(len(coords),tile*64)
            self.assertEqual(set(coords),set(itertools.product(range(tile),range(64))))
            for cnt,rows in itertools.product(range(129),[1,19,32,63,64,65,127,128,129]):
                # Independently model z early return and every edge of the original tile ABI.
                got=[(z*tile+r,n0+c) for z in range(128//tile) if z*tile<cnt for n0 in range(0,rows,64)
                     for r,c in coords if z*tile+r<cnt and n0+c<rows]
                self.assertEqual(len(got),cnt*rows)
                self.assertEqual(len(set(got)),cnt*rows)
                cases+=1
        print('R12 routed integer geometries:',cases)
    def test_lds(self):
        # KS=32 for both IQ3_XXS and IQ4_XS. Include all declared arrays.
        def size(x):return 2*4*(x*(20+2+4)+64*(20+4+4))
        self.assertEqual([size(x) for x in (32,64,128)],[20992,27648,40960])
        shader=(ROOT/'src/vulkan/shaders/gemmi.glsl').read_text()
        self.assertIn('#if SM && !defined(STRATA_R12_ROUTED)',shader)
        self.assertIn('base >= cnt) return;',shader)

class Boundaries(base_tests.Gates):
    def setUp(self):
        super().setUp()
        def fp(x):return struct.unpack('<I',struct.pack('<f',x))[0]
        for arm in base_tests.r10.ARMS:
            candidate=arm in bound.CANDIDATES
            # The R10 fixture has independent combine/row logs; make the layer-0
            # output match that combine before adding the full-chunk captures.
            self.edit(arm,lambda s:s.replace('r9_layer_out,0,0,[0.250001, 0.5]','r9_layer_out,0,0,[0.25, 0.5]'))
            with (self.root/(arm+'.log')).open('a') as f:
                for layer in (0,1):
                    for stage in bound.STAGES:
                        v=[1.01 if layer==1 and candidate else 1.0,0.0]
                        if stage=='ffn_norm': v=[1.00001 if layer==1 and candidate else 1.0,-0.0]
                        if stage=='layer_output': v=[.250001 if layer==1 and candidate else .25,.5]
                        f.write(f'r12_boundary,{layer},0,{stage},1,2,{json.dumps([fp(x) for x in v])}\n')
    def test_r12_valid(self):
        q=bound.evaluate(self.root);self.assertEqual(q['controls'],'bitwise identical')
        self.assertEqual(q['arms']['contract'][5]['changed'],1)
    def test_r12_missing_stage(self):
        self.edit('contract',lambda s:'\n'.join(x for x in s.splitlines() if not x.startswith('r12_boundary,1,0,ffn_norm,')))
        with self.assertRaisesRegex(ValueError,'coverage'):bound.evaluate(self.root)
    def test_r12_sham_changed(self):
        self.edit('sham',lambda s:s.replace('r12_boundary,0,0,layer_input,1,2,[1065353216, 0]','r12_boundary,0,0,layer_input,1,2,[1065353217, 0]'))
        with self.assertRaisesRegex(ValueError,'changed raw boundary'):bound.evaluate(self.root)
    def test_r12_nonfinite(self):
        self.edit('contract',lambda s:s.replace('r12_boundary,0,0,layer_input,1,2,[1065353216, 0]','r12_boundary,0,0,layer_input,1,2,[2139095040, 0]'))
        with self.assertRaisesRegex(ValueError,'invalid FP32'):bound.evaluate(self.root)
    def test_r12_changed_replay(self):
        self.edit('replay-contract',lambda s:s.replace('r12_boundary,0,0,layer_output,1,2,[1048576000, 1056964608]','r12_boundary,0,0,layer_output,1,2,[1048576001, 1056964608]'))
        with self.assertRaisesRegex(ValueError,'boundary disagrees|initial full chunk'):bound.evaluate(self.root)

    def test_r12_all_controls_same_but_wrong_stage(self):
        # Previously matching all arms sufficed even when the captured FFN row
        # did not correspond to the independently logged router input.
        for arm in base_tests.r10.ARMS:
            self.edit(arm,lambda s:s.replace('r12_boundary,0,0,ffn_norm,1,2,[1065353216, 2147483648]',
                'r12_boundary,0,0,ffn_norm,1,2,[1065353217, 2147483648]'))
        with self.assertRaisesRegex(ValueError,'boundary disagrees'):bound.evaluate(self.root)
    def test_r12_signed_zero_change(self):
        result=bound.compare({(0,0,'ffn_norm'):(1,1,[0])},{(0,0,'ffn_norm'):(1,1,[2147483648])})[0]
        self.assertEqual((result['changed'],result['changed_tokens'],result['worst_token'],result['max_abs']),(1,[0],0,0.0))

if __name__=='__main__':unittest.main(verbosity=2)
