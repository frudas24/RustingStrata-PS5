#!/usr/bin/env python3
"""Synthetic falsification tests for offline R10 gates (not physical Vulkan)."""
import copy, importlib.util, json, pathlib, tempfile, unittest
spec=importlib.util.spec_from_file_location('r10',pathlib.Path(__file__).with_name('compare-r10.py'))
r10=importlib.util.module_from_spec(spec);spec.loader.exec_module(r10)
class Gates(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=pathlib.Path(self.tmp.name)
        self.fixed={'layer':0,'pos':0,'chunk':1,'token':0,'rank':0,'expert':1,'slot':0,'ids':[1,2],'hidden':[1.,-0.0],'gpu_down':[.25,.5],'weights':[.5,.5]}
        for arm in ('gpu',*r10.ARMS):
            mode=r10.ARMS.get(arm); control=mode!='contract'
            logits=[[1.,0.,-1.],[.5,-.25,1.]] if control else [[1.1,0.,-1.],[.5,-.25,1.]]
            (self.root/(arm+'.json')).write_text(json.dumps({'mode':'gpu','stats':[10,0,0],'logits':logits,'r10':{'prompt':0,'tokens':[7],'chunk':1,'layers':[0,1]}}))
            if arm=='gpu': continue
            rows=['r9_fixed,'+json.dumps(self.fixed)]
            rows.extend(f'r7_r4_stage,0,1,1,0,0,{s},2,0,0,1' for s in r10.r9.STAGES)
            if mode!='shadow': rows.append(f'r9_write,{mode},0,0,0,2')
            rows.append('r9_combine,0,0,0,[0.25,0.5]')
            for layer in (0,1):
                ref={'layer':layer,'pos':0,'tokens':[7],'top_k':2,'ids':[1,2],'weights':[.5,.5]}
                natural=copy.deepcopy(ref)
                if not control and layer==1: natural.update(ids=[2,3],weights=[.6,.4])
                replay=arm.startswith('replay-'); effective=ref if replay else natural
                rows.append('r10_route,'+json.dumps({'mode':'replay' if replay else 'record','natural':natural,'effective':effective,'raw_cutoff_gaps':[.001],'hidden':[1.,-0.0] if control or layer==0 else [1.00001,-0.0]}))
                rows.append(f'r9_layer_out,{layer},0,'+json.dumps([.25,.5] if control else [.250001,.5]))
            (self.root/(arm+'.log')).write_text('\n'.join(rows)+'\n')
        r10.tape(self.root)
    def edit(self,arm,fn):
        p=self.root/(arm+'.log');p.write_text(fn(p.read_text()))
    def test_valid(self):
        s=r10.evaluate(self.root);self.assertFalse(s['production_promotion']);self.assertEqual(len(s['arms']['contract']['natural_route_changes']),1)
    def test_sham(self):
        self.edit('sham',lambda s:s.replace('r9_layer_out,1,0,[0.25, 0.5]','r9_layer_out,1,0,[0.3, 0.5]'))
        with self.assertRaisesRegex(ValueError,'changed layer'):r10.evaluate(self.root)
    def test_replay_wrong_route(self):
        self.edit('replay-contract',lambda s:s.replace('"effective": {"layer": 1, "pos": 0, "tokens": [7], "top_k": 2, "ids": [1, 2]','"effective": {"layer": 1, "pos": 0, "tokens": [7], "top_k": 2, "ids": [1, 3]'))
        with self.assertRaisesRegex(ValueError,'effective route changed'):r10.evaluate(self.root)
    def test_replay_wrong_weight(self):
        self.edit('replay-contract',lambda s:s.replace('"effective": {"layer": 1, "pos": 0, "tokens": [7], "top_k": 2, "ids": [1, 2], "weights": [0.5, 0.5]', '"effective": {"layer": 1, "pos": 0, "tokens": [7], "top_k": 2, "ids": [1, 2], "weights": [0.6, 0.4]'))
        with self.assertRaisesRegex(ValueError,'effective route changed'):r10.evaluate(self.root)
    def test_changed_target_combine(self):
        self.edit('replay-contract',lambda s:s.replace('r9_combine,0,0,0,[0.25,0.5]','r9_combine,0,0,0,[0.3,0.5]'))
        with self.assertRaisesRegex(ValueError,'intervention differs'):r10.evaluate(self.root)
    def test_changed_target_hidden(self):
        self.edit('contract',lambda s:s.replace('"hidden": [1.0, -0.0]}','"hidden": [1.1, -0.0]}',1))
        with self.assertRaisesRegex(ValueError,'changed pre-router hidden'):r10.evaluate(self.root)
    def test_no_write(self):
        self.edit('sham',lambda s:s.replace('r9_write,sham','ignored'))
        with self.assertRaisesRegex(ValueError,'number of writes'):r10.evaluate(self.root)
    def test_corrupt_tape(self):
        p=self.root/'routes.json';r=json.loads(p.read_text());r[1]['weights'][0]=.6;p.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'tape does not match'):r10.evaluate(self.root)
    def test_missing_layer(self):
        self.edit('replay-contract',lambda s:'\n'.join(x for x in s.splitlines() if not x.startswith('r9_layer_out,1,')))
        with self.assertRaisesRegex(ValueError,'incomplete trace'):r10.evaluate(self.root)
if __name__=='__main__':unittest.main(verbosity=2)
