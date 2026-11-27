#!/usr/bin/env python3
"""Check actual batch artifacts; explicit rejection survives python -O."""
import math,pathlib,re,struct,sys
base=sys.argv[1]; arms=[]; vocabs=[]
def require(ok, message):
    if not ok: raise SystemExit('REJECT: '+message)
for frac in (0,1):
    s=pathlib.Path(f'{base}.{frac}.log').read_text()
    require(re.search(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;',s,re.M), 'batch test missing or skipped')
    rows=re.findall(r'R11U1_BATCH layer=\d+ rows=\d+ staged=(\d+) cpu=(\d+)',s)
    require(rows and all(int(staged)>0 and int(cpu)==0 for staged,cpu in rows), 'batch guard not exercised or selected CPU')
    require(f'R11U1_BATCH_GUARD ARM PASS: frac={frac} 6 finite full-logit rows' in s, 'no physical result')
    counts=re.findall(r'^R11U1_BATCH_GUARD VOCAB: (\d+)$',s,re.M)
    require(len(counts)==1 and int(counts[0])>0,'missing/ambiguous batch vocabulary')
    vocab=int(counts[0]);vocabs.append(vocab)
    bits=pathlib.Path(f'{base}.{frac}.bits').read_bytes()
    stride=4*(1+vocab)
    require(len(bits)==6*stride,'batch output truncated or oversized')
    for start in range(0,len(bits),stride):
        require(struct.unpack_from('<I',bits,start)[0]<vocab,'batch token outside vocabulary')
        require(all(math.isfinite(v) for (v,) in struct.iter_unpack('<f',memoryview(bits)[start+4:start+stride])), 'nonfinite batch logits')
    arms.append(bits)
    print(f'arm {frac}: {len(rows)} batch layer decisions; all staged, no CPU batch experts')
require(vocabs[0]==vocabs[1] and arms[0]==arms[1], 'full-logit bits differ between arms')
print('PASS: identical full logits and greedy tokens across isolated frac=0/1 processes')
