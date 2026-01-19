#!/usr/bin/env python3
import re
import sys

if len(sys.argv) < 2:
    print("Usage: compare_accs.py <logfile>")
    sys.exit(2)

log = open(sys.argv[1]).read()

sample_re = re.compile(r"\[TRACE\] sample (\d+) idx=(\d+)")
acc_unrolled_re = re.compile(r"\[ACC\] unrolled accBefore \(blk0..3\): (.*)")
acc_classic_re = re.compile(r"\[ACC\] classic accBefore \(blk0..3\): (.*)")

lines = log.splitlines()

i = 0
results = []
while i < len(lines):
    m = sample_re.search(lines[i])
    if m:
        samp = int(m.group(1))
        idx = int(m.group(2))
        # next lines should contain unrolled/classic acc lines
        un = None
        cl = None
        j = i+1
        while j < i+10 and j < len(lines):
            lu = acc_unrolled_re.search(lines[j])
            if lu:
                un = lu.group(1).strip()
            lc = acc_classic_re.search(lines[j])
            if lc:
                cl = lc.group(1).strip()
            if un and cl:
                break
            j += 1
        if not (un and cl):
            i += 1
            continue
        # parse groups: each group printed as HHHHHHHHLLLLLLLL
        def parse_groups(s):
            parts = s.split()
            vals = []
            for p in parts:
                if len(p) == 16:
                    high = p[0:8]
                    low = p[8:16]
                elif len(p) == 8:
                    # maybe separated high/low
                    high = '00000000'
                    low = p
                else:
                    # try hex cleanup
                    p = p.replace('0x','')
                    if len(p) >= 16:
                        high = p[-16:-8]
                        low = p[-8:]
                    else:
                        high = '00000000'
                        low = p[-8:]
                vals.append((high, low))
            return vals
        vun = parse_groups(un)
        vcl = parse_groups(cl)
        # compare block1 low
        if len(vun) >= 2 and len(vcl) >= 2:
            low_un = vun[1][1]
            low_cl = vcl[1][1]
            match = (low_un.lower() == low_cl.lower())
            results.append((samp, idx, low_un, low_cl, match))
        i = j+1
    else:
        i += 1

# print summary
mism = [r for r in results if not r[4]]
for r in results:
    print(f"Sample {r[0]} idx={r[1]} unrolled_blk1_low={r[2]} classic_blk1_low={r[3]} match={r[4]}")
print('\nSummary:')
print(f"  parsed samples: {len(results)}")
print(f"  mismatched block1 low: {len(mism)}")

if len(mism) > 0:
    sys.exit(1)
else:
    sys.exit(0)
