#!/usr/bin/env python3
"""Independent R11 checks. Not Rust/SPIR-V or physical performance evidence."""
import itertools, pathlib, re, struct, unittest, subprocess, tempfile
ROOT=pathlib.Path(__file__).resolve().parents[2]
class Test(unittest.TestCase):
    def test_tile_geometry(self):
        coords=[(16*(w&1)+(lane>>2)+8*(i>>2),16*(w>>1)+2*(lane&3)+(i&1)+8*((i>>1)&1)) for w in range(8) for lane in range(32) for i in range(8)]
        self.assertEqual(len(coords),2048);self.assertEqual(len(set(coords)),2048)
        cases=0
        for cnt,rows in itertools.product(range(1,129),[1,19,32,63,64,65,127,128,129]):
            out=[(base+r,n0+c) for base in range(0,cnt,32) for n0 in range(0,rows,64) for r,c in coords if base+r<cnt and n0+c<rows]
            self.assertEqual(len(out),cnt*rows);self.assertEqual(len(set(out)),cnt*rows);cases+=1
        print('R11 tile coverage:',cases)
    def test_iq_codes(self):
        src=(ROOT/'src/quant/tables.rs').read_text()
        def table(name):
            s=re.search(r'pub static '+name+r':[^=]+ = \[([^]]+)\]',src).group(1)
            return [int(v.strip(),0) for v in s.split(',') if v.strip()]
        grids=table('IQ3XXS_GRID');signs=table('KSIGNS_IQ2XS');lut=table('KVALUES_IQ4NL')
        f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
        total=0
        for typ,bs in [('IQ3_XXS',98),('IQ4_XS',136)]:
            path=ROOT/'tests/fixtures'/typ
            raw=path.with_suffix('.raw').read_bytes();ref=path.with_suffix('.f32').read_bytes();out=[]
            for p in range(0,len(raw),bs):
                b=raw[p:p+bs];d=struct.unpack_from('<e',b)[0]
                for ib in range(8):
                    if typ=='IQ3_XXS':
                        aux=struct.unpack_from('<I',b,66+4*ib)[0];scale=f32(f32(d*(.5+(aux>>28)))*.5);q=[]
                        for l in range(4):
                            sg=signs[(aux>>(7*l))&127];v=b[2+8*ib+2*l:4+8*ib+2*l];g=grids[v[0]].to_bytes(4,'little')+grids[v[1]].to_bytes(4,'little');q.extend(-u if sg&(1<<j) else u for j,u in enumerate(g))
                    else:
                        sh=struct.unpack_from('<H',b,2)[0];ls=((b[4+ib//2]>>(4*(ib%2)))&15)|(((sh>>(2*ib))&3)<<4);scale=f32(d*(ls-32));v=b[8+16*ib:24+16*ib];q=[lut[x&15] for x in v]+[lut[x>>4] for x in v]
                    out.extend(f32(scale*x) for x in q)
            actual=struct.unpack('<'+'f'*(len(ref)//4),ref)
            self.assertEqual(len(actual),len(out));self.assertLessEqual(max(abs(a-b) for a,b in zip(actual,out)),1e-6);total+=len(out)
        print('R11 IQ fixture values:',total)
    def test_benchmark_gate(self):
        import importlib.util, copy
        spec=importlib.util.spec_from_file_location('bench',ROOT/'tools/ps5/bench-r11.py');bench=importlib.util.module_from_spec(spec);spec.loader.exec_module(bench)
        mark='gemm_f_dp_ps5_iq32'; sample={'tokens':[1,2],'stopped':False,'top5':[[i,-i-1.] for i in range(5)]}
        a={'rates':{128:[100.,100.]},'evidence':{'marks':[]},'signatures':{128:[sample]}}
        b=copy.deepcopy(a);b['rates'][128]=[120.,100.];b['evidence']['marks']=[mark]
        self.assertTrue(bench.compare([a,b,b,a],mark)['accepted'])
        bad=copy.deepcopy(b);bad['evidence']['marks']=[]
        with self.assertRaises(bench.r6.Reject):bench.compare([a,b,bad,a],mark)
        bad=copy.deepcopy(b);bad['rates'][128]=[130.,90.]
        self.assertFalse(bench.compare([a,bad,bad,a],mark)['accepted'])
        bad=copy.deepcopy(b);bad['signatures'][128][0]['tokens']=[1,3]
        with self.assertRaises(bench.r6.Reject):bench.compare([a,bad,bad,a],mark)
    def test_signed_dot(self):
        # Actual independent C AVX2 primitive equivalent to Rust. Extreme mixed signs and random codes.
        c=r'''#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
int main(){uint32_t rng=43;int8_t w[32],x[32];for(int j=0;j<100036;j++){
 int32_t ref=0;for(int i=0;i<32;i++){rng=rng*1664525u+1013904223u;w[i]=j<36?(int8_t[]){-128,-127,-1,0,1,127}[j/6]:(int8_t)(rng>>24);rng=rng*1664525u+1013904223u;x[i]=j<36?(int8_t[]){-128,-127,-1,0,1,127}[j%6]:(int8_t)(rng>>24);ref+=(int)w[i]*x[i];}
 __m256i s=_mm256_setzero_si256();for(int i=0;i<32;i+=16){__m256i a=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w+i))),b=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(x+i)));s=_mm256_add_epi32(s,_mm256_madd_epi16(a,b));}int32_t p[8],z=0;_mm256_storeu_si256((__m256i*)p,s);for(int i=0;i<8;i++)z+=p[i];if(z!=ref)return 1;}puts("100036 signed dot cases PASS");}'''
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d);(p/'test.c').write_text(c);subprocess.run(['cc','-O3','-march=znver2',str(p/'test.c'),'-o',str(p/'test')],check=True);subprocess.run([str(p/'test')],check=True)
if __name__=='__main__':unittest.main(verbosity=2)
