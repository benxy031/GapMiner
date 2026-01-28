#!/usr/bin/env python3
import re,sys
if len(sys.argv)<2:
    print('Usage: compare_prods.py <log>')
    sys.exit(2)
log=open(sys.argv[1]).read()
lines=log.splitlines()
prod_un_re=re.compile(r"\[PROD\] unrolled block1 products.*: (.*)")
prod_cl_re=re.compile(r"\[PROD\] classic block1 products.*: (.*)")
sample_re=re.compile(r"\[TRACE\] sample (\d+) idx=(\d+)")

i=0
res=[]
while i<len(lines):
    m=sample_re.search(lines[i])
    if m:
        s=int(m.group(1)); idx=int(m.group(2))
        un=None; cl=None
        j=i+1
        while j<i+12 and j<len(lines):
            mu=prod_un_re.search(lines[j]);
            mc=prod_cl_re.search(lines[j]);
            if mu: un=mu.group(1).strip()
            if mc: cl=mc.group(1).strip()
            if un and cl: break
            j+=1
        if not (un and cl):
            i+=1; continue
        def parse16(s):
            parts=s.split()
            out=[]
            for p in parts:
                if len(p)==16:
                    out.append((p[0:8],p[8:16]))
                elif len(p)==8:
                    out.append(('00000000',p))
                else:
                    p=p.replace('0x','')
                    if len(p)>=16:
                        out.append((p[-16:-8],p[-8:]))
                    else:
                        out.append(('00000000',p[-8:]))
            return out
        vun=parse16(un); vcl=parse16(cl)
        diffs=[]
        for k in range(3):
            if k<len(vun) and k<len(vcl):
                diffs.append((vun[k][0].lower(),vun[k][1].lower(),vcl[k][0].lower(),vcl[k][1].lower(), vun[k]!=vcl[k]))
            else:
                diffs.append(('','','','',True))
        res.append((s,idx,diffs))
        i=j+1
    else:
        i+=1

mismatch_count=0
for s,idx,diffs in res:
    anydiff=False
    for k,(uh,ul,ch,cl,neq) in enumerate(diffs):
        if neq:
            anydiff=True; mismatch_count+=1
    print(f"Sample {s} idx={idx} anydiff={anydiff}")
    for k,(uh,ul,ch,cl,neq) in enumerate(diffs):
        print(f"  prod{k}: unrolled={uh}{ul} classic={ch}{cl} equal={not neq}")
print('\nSummary: parsed %d samples, prod mismatches reported: %d'%(len(res),mismatch_count))
if mismatch_count>0: sys.exit(1)
else: sys.exit(0)
