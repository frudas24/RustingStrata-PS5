#!/usr/bin/env python3
"""Independent synthetic acceptance/rejection tests for the R9 offline comparator."""
import importlib.util, json, pathlib, tempfile, unittest
spec=importlib.util.spec_from_file_location('r9',pathlib.Path(__file__).with_name('compare-r9.py'))
r9=importlib.util.module_from_spec(spec); spec.loader.exec_module(r9)
class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=pathlib.Path(self.tmp.name); self.addCleanup(self.tmp.cleanup)
        for arm in ('gpu','shadow','legacy','contract'):
            rows=[[1.,0.,-1.,-.25,.5] for _ in range(12)]
            (self.root/(arm+'.json')).write_text(json.dumps({'mode':'gpu','stats':[960,0,0],'logits':rows}))
            lines=[]
            if arm!='gpu':
                for i in range(6):
                    row={'layer':0,'pos':0,'chunk':32,'token':0,'rank':0,'expert':2+i%2,'slot':3,'ids':[2,3],'hidden':[1.,-0.0],'gpu_down':[.25,.5],'weights':[.5,.5]}
                    lines.append('r9_fixed,'+json.dumps(row))
                    lines.extend(f'r7_r4_stage,0,32,2,3,0,{s},2,0,0,1' for s in r9.STAGES)
                    if arm!='shadow': lines.append(f'r9_write,{arm},0,0,3,2')
                    lines.append('r9_combine,0,0,0,[0.25,0.5]')
            (self.root/(arm+'.log')).write_text('\n'.join(lines))
    def edit(self,arm,fn):
        p=self.root/(arm+'.log');p.write_text(fn(p.read_text()))
    def test_good(self):
        result=r9.evaluate(self.root); self.assertEqual(result['matched_cases'],6); self.assertFalse(result['production_promotion'])
    def test_input_change(self):
        self.edit('contract',lambda s:s.replace('"hidden": [1.0, -0.0]','"hidden": [1.0, 0.0]',1))
        with self.assertRaisesRegex(ValueError,'fixed input'):r9.evaluate(self.root)
    def test_route_change(self):
        self.edit('legacy',lambda s:s.replace('"ids": [2, 3]','"ids": [2, 4]',1))
        with self.assertRaisesRegex(ValueError,'fixed input'):r9.evaluate(self.root)
    def test_missing_write(self):
        self.edit('contract',lambda s:s.replace('r9_write,contract,0,0,3,2','ignored',1))
        with self.assertRaisesRegex(ValueError,'number of writes'):r9.evaluate(self.root)
    def test_missing_stage(self):
        self.edit('shadow',lambda s:s.replace(',r9_gate_full,',',missing,',1))
        with self.assertRaisesRegex(ValueError,'incomplete stages'):r9.evaluate(self.root)
    def test_shadow_changed(self):
        p=self.root/'shadow.json';d=json.loads(p.read_text());d['logits'][0][0]=2;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'shadow changed'):r9.evaluate(self.root)
    def test_nonfinite(self):
        self.edit('shadow',lambda s:s.replace('r9_combine,0,0,0,[0.25,0.5]','r9_combine,0,0,0,[NaN,0.5]',1))
        with self.assertRaisesRegex(ValueError,'invalid combine'):r9.evaluate(self.root)
if __name__=='__main__':unittest.main(verbosity=2)
