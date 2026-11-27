#!/usr/bin/env python3
"""Independent Qwen3 BF16 CPU control (not Rust/Vulkan/PS5). Local data only.
Requires numpy, torch, safetensors and GCC; no downloaded model code."""
import argparse, ctypes, hashlib, json, pathlib, platform, subprocess, tempfile, time
import numpy as np
import torch
from safetensors import safe_open


def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=pathlib.Path);p.add_argument('output',type=pathlib.Path);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    if platform.machine() != 'x86_64': raise RuntimeError('requires x86_64')
    flags=pathlib.Path('/proc/cpuinfo').read_text().split('flags',1)[1].splitlines()[0].split(':',1)[1].split()
    if not {'avx2','fma','f16c'}.issubset(flags): raise RuntimeError('requires AVX2/FMA/F16C')
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    c=json.loads((a.model/'config.json').read_text());assert c['model_type']=='qwen3' and c['torch_dtype']=='bfloat16' and c['hidden_act']=='silu' and not c['attention_bias'] and not c['rope_scaling'] and not c['use_sliding_window']
    path=a.model/'model.safetensors';here=pathlib.Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temp, safe_open(str(path),framework='pt',device='cpu') as st:
        lib=pathlib.Path(temp)/'probe.so';subprocess.run(['gcc','-O3','-march=znver2','-ffp-contract=off','-fPIC','-shared',str(here/'venus-real-weights.c'),'-o',str(lib)],check=True)
        dll=ctypes.CDLL(str(lib));P=ctypes.c_void_p
        for fn in ('bf_mv','f16_mv','f16_mm4','bf_scalar'):
            f=getattr(dll,fn);f.argtypes=[P,P,P,ctypes.c_int,ctypes.c_int];f.restype=None
        def call(fn,w,x,y):
            assert w.flags.c_contiguous and x.flags.c_contiguous and y.flags.c_contiguous and w.shape[1]%8==0
            getattr(dll,fn)(w.ctypes.data,x.ctypes.data,y.ctypes.data,*w.shape)
        def get(n):return st.get_tensor(n)
        conversions={'values':0,'changed':0,'nonfinite_bf16':0,'nonfinite_f16':0,'max_abs':0.0}
        for n in st.keys():
            t=get(n);assert t.dtype==torch.bfloat16
            for chunk in t.flatten().split(1<<20):
                f=chunk.float();h=f.half().float();conversions['values']+=f.numel();conversions['changed']+=int((f!=h).sum());conversions['nonfinite_bf16']+=int((~f.isfinite()).sum());conversions['nonfinite_f16']+=int((~h.isfinite()).sum());conversions['max_abs']=max(conversions['max_abs'],float((f-h).abs().max()))
        assert conversions['nonfinite_bf16']==0 and conversions['nonfinite_f16']==0
        print('conversion',conversions,flush=True)
        H=c['num_attention_heads'];K=c['num_key_value_heads'];D=c['head_dim'];L=c['num_hidden_layers'];eps=c['rms_norm_eps'];inv=c['rope_theta']**(-torch.arange(0,D,2,dtype=torch.float32)/D);probes=[]
        def check_probe(name,w,x):
            raw=w[:384].contiguous();xx=x[:4].contiguous().numpy();bf=raw.view(torch.uint16).numpy();hf=raw.float().half().view(torch.uint16).numpy();nr,n=bf.shape
            rd=raw.double().numpy();hd=raw.float().half().double().numpy();xd=xx.astype(np.float64);ref=rd@xd.T;fr=hd@xd.T;asum=np.abs(rd)@np.abs(xd).T;hsum=np.abs(hd)@np.abs(xd).T
            b=np.empty((4,nr),np.float32);f=b.copy();g=b.copy();s=b.copy();jobs={'bf_scalar':(bf,s),'bf_mv':(bf,b),'f16_mv':(hf,f),'f16_mm4':(hf,g)}
            def run(k):
                ww,yy=jobs[k]
                if k=='f16_mm4':call(k,ww,xx,yy)
                else:
                    for i in range(4):call(k,ww,xx[i],yy[i])
            for k in jobs:run(k)
            errors={}
            for k,y,r,ab in [('bf16_gemv',b,ref,asum),('f16_gemv',f,fr,hsum),('f16_gemm',g,fr,hsum)]:
                d=np.abs(y.T-r);assert np.isfinite(y).all() and np.all(d<=2e-6*ab+1e-6),(name,k,float(d.max()));errors[k]=float(d.max())
            times={k:[] for k in jobs}
            for rep in range(8):
                for k in (list(jobs) if rep%2==0 else list(reversed(jobs))):
                    start=time.perf_counter_ns()
                    for _ in range(3):run(k)
                    times[k].append((time.perf_counter_ns()-start)/3)
            probes.append({'tensor':name,'rows':nr,'cols':n,'max_abs':errors,'conversion_max_abs_output':float(np.abs(ref-fr).max()),'times_ns':times})
        def rms(x,w):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w.float()
        def rope(x,co,si):
            u,v=x.chunk(2,-1);return x*co+torch.cat((-v,u),-1)*si
        def linear(name,x,mode,capture):
            w=get(name);xx=x.contiguous();rows,n=w.shape
            if capture and (name.startswith(tuple(f'model.layers.{i}.' for i in [0,L//2,L-1])) or name=='lm_head.weight'):check_probe(name,w,xx)
            if mode=='bf16_avx2':
                raw=w.view(torch.uint16).numpy();nx=xx.numpy();y=np.empty((len(xx),rows),np.float32)
                for i in range(len(xx)):call('bf_mv',raw,nx[i],y[i])
                return torch.from_numpy(y)
            if rows>8192:return torch.cat([xx@part.float().T for part in w.split(4096)],dim=-1)
            return xx@w.float().T
        def forward(ids,mode,capture=False):
            seq=len(ids);x=get('model.embed_tokens.weight')[ids].float();angles=torch.arange(seq).float()[:,None]*inv[None,:];angles=torch.cat((angles,angles),dim=-1);co=angles.cos()[:,None,:];si=angles.sin()[:,None,:];mask=torch.ones(seq,seq,dtype=torch.bool).tril();states=[]
            for i in range(L):
                b=f'model.layers.{i}.';att=b+'self_attn.';mlp=b+'mlp.';z=rms(x,get(b+'input_layernorm.weight'))
                q=linear(att+'q_proj.weight',z,mode,capture).reshape(seq,H,D);k=linear(att+'k_proj.weight',z,mode,capture).reshape(seq,K,D);v=linear(att+'v_proj.weight',z,mode,capture).reshape(seq,K,D)
                q=rope(rms(q,get(att+'q_norm.weight')),co,si);k=rope(rms(k,get(att+'k_norm.weight')),co,si).repeat_interleave(H//K,dim=1);v=v.repeat_interleave(H//K,dim=1)
                scores=(q.transpose(0,1)@k.transpose(0,1).transpose(-1,-2))*(D**-.5);scores=scores.masked_fill(~mask,float('-inf'));z=(scores.softmax(-1)@v.transpose(0,1)).transpose(0,1).reshape(seq,H*D);x=x+linear(att+'o_proj.weight',z,mode,capture)
                z=rms(x,get(b+'post_attention_layernorm.weight'));gate=linear(mlp+'gate_proj.weight',z,mode,capture);up=linear(mlp+'up_proj.weight',z,mode,capture);x=x+linear(mlp+'down_proj.weight',torch.nn.functional.silu(gate)*up,mode,capture);states.append(x.clone())
            return linear('lm_head.weight',rms(x,get('model.norm.weight')),mode,capture),states
        vocab=json.loads((a.model/'vocab.json').read_text());prompts=[[vocab[s] for s in words] for words in [['The','Ġlittle','Ġdog','Ġran'],['def','Ġadd','(','a']]];results=[]
        with torch.inference_mode():
            for j,ids in enumerate(prompts):
                ref,rl=forward(ids,'torch',capture=(j==0));cand,cl=forward(ids,'bf16_avx2');d=(ref-cand).abs();assert torch.isfinite(ref).all() and torch.isfinite(cand).all() and torch.equal(ref.argmax(-1),cand.argmax(-1)) and float(d.max())<0.01
                layer_errors=[float((u-v).abs().max()) for u,v in zip(rl,cl)];gen=[];step_errors=[];full=ids.copy()
                for step in range(3):
                    if step:ref,_=forward(full,'torch');cand,_=forward(full,'bf16_avx2')
                    delta=(ref-cand).abs();assert torch.isfinite(ref).all() and torch.isfinite(cand).all() and torch.equal(ref.argmax(-1),cand.argmax(-1)) and float(delta.max())<0.01
                    step_errors.append(float(delta.max()));ir=int(ref[-1].argmax());ic=int(cand[-1].argmax());assert ir==ic,(j,step,ir,ic);gen.append(ir);full.append(ir)
                results.append({'prompt_ids':ids,'greedy_ids':gen,'max_abs_logits':float(d.max()),'mean_abs_logits':float(d.mean()),'layer_max_abs':layer_errors,'generation_step_max_abs':step_errors});print('causal control',j,results[-1]['max_abs_logits'],gen,flush=True)
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1<<20),b''):digest.update(b)
        out={'scope':'independent CPU Qwen3 reference with C kernels, not Rust or PS5','sha256':digest.hexdigest(),'cpu':next((l.split(':',1)[1].strip() for l in pathlib.Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')),platform.machine()),'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],'torch':torch.__version__,'threads':a.threads,'tensor_count':len(st.keys()),'conversion':conversions,'probes':probes,'inference':results}
        a.output.write_text(json.dumps(out,indent=2)+'\n');print('PASS',len(probes),'real tensor probes;',len(results),'causal controls',flush=True)
if __name__=='__main__':main()
