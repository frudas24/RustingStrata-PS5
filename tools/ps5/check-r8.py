#!/usr/bin/env python3
"""Host-side controller self-test; not a Rust or GPU execution."""
import importlib.util, json, pathlib, tempfile, unittest
spec=importlib.util.spec_from_file_location('r8',pathlib.Path(__file__).with_name('compare-r8.py'))
r8=importlib.util.module_from_spec(spec); spec.loader.exec_module(r8)

class Test(unittest.TestCase):
    def test_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=pathlib.Path(tmp)
            a=[0.,1.,2.,3.,4.,5.,6.,7.]
            b=[0.,1.,2.,3.,4.,5.,7.,6.]
            def write(name,vals,mode,stats):
                (root/(name+'.json')).write_text(json.dumps(dict(mode=mode,stats=stats,logits=[vals]*12)))
            write('gpu',a,'gpu',[1,0,0]); write('probe',a,'gpu',[1,0,0]); write('legacy',b,'half',[1,1,1]); write('contract',a,'half',[1,1,1])
            (root/'legacy.log').write_text(''); (root/'contract.log').write_text('PS5 R8 exercised: layer=0 pairs=1 contracts=[F16, F16, Q8_32]')
            (root/'probe.log').write_text(''.join(f'r7_r4_stage,0,32,0,0,0,{s}_contract_oracle,16,0,0,1\n' for s in ('gate','up','down')))
            result=r8.evaluate(root); self.assertTrue(result['strict_diagnostic_gate']); self.assertTrue(result['hypothesis_improved']); self.assertFalse(result['production_promotion']); self.assertEqual(result['reference_distinct_vectors'],1)
            write('contract',b,'half',[1,1,1]); self.assertFalse(r8.evaluate(root)['strict_diagnostic_gate'])
            write('contract',a,'half',[1,0,0]); self.assertRaises(ValueError,r8.evaluate,root)
            write('contract',a,'half',[1,1,1]); (root/'contract.log').write_text(''); self.assertRaises(ValueError,r8.evaluate,root)
            (root/'contract.log').write_text('PS5 R8 exercised:'); write('probe',b,'gpu',[1,0,0]); self.assertRaises(ValueError,r8.evaluate,root)
            write('probe',a,'gpu',[1,0,0]); (root/'probe.log').write_text(''); self.assertRaises(ValueError,r8.evaluate,root)
    def test_nonfinite(self):
        self.assertRaises(ValueError,r8.metrics,[1.,1.],[1.,2.])
        self.assertRaises(ValueError,r8.metrics,[1.,2.],[1.])

if __name__=='__main__': unittest.main()
