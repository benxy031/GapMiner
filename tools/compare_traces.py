#!/usr/bin/env python3
import re
from pathlib import Path
log = Path('/tmp/gapminer-cuda.log').read_text()
# regex blocks
pattern = re.compile(r"\[TRACE\] sample (\d+) idx=(\d+)\n\[TRACE\] unrolled invValue: ([0-9a-fA-F ]+)\n\[TRACE\] unrolled final op1: ([0-9a-fA-F ]+)\n\[TRACE\] classic invValue: +([0-9a-fA-F ]+)\n\[TRACE\] classic final op1: +([0-9a-fA-F ]+)", re.M)

results = []
for m in pattern.finditer(log):
    s = int(m.group(1))
    idx = int(m.group(2))
    un_inv = m.group(3).strip().split()
    un_final = m.group(4).strip().split()
    cl_inv = m.group(5).strip().split()
    cl_final = m.group(6).strip().split()
    results.append((s, idx, un_inv, cl_inv, un_final, cl_final))

if not results:
    print('No trace blocks found')
    raise SystemExit(1)

# Compare and report
for s, idx, un_inv, cl_inv, un_final, cl_final in results:
    print(f'Sample {s} idx={idx}')
    # compare inv values
    first_inv_diff = None
    for i, (u, c) in enumerate(zip(un_inv, cl_inv)):
        if u.lower() != c.lower():
            first_inv_diff = (i, u, c)
            break
    if first_inv_diff is None:
        print('  invValue: identical')
    else:
        i,u,c = first_inv_diff
        print(f'  invValue mismatch at i={i}: unrolled={u} classic={c}')
    # compare final op1
    first_op_diff = None
    for i, (u, c) in enumerate(zip(un_final, cl_final)):
        if u.lower() != c.lower():
            first_op_diff = (i, u, c)
            break
    if first_op_diff is None:
        print('  final op1: identical')
    else:
        i,u,c = first_op_diff
        print(f'  final op1 mismatch at i={i}: unrolled={u} classic={c}')
    print()

# Summarize counts
inv_mismatch_counts = {}
op_mismatch_counts = {'any':0}
for s, idx, un_inv, cl_inv, un_final, cl_final in results:
    # inv
    matched = True
    for u,c in zip(un_inv, cl_inv):
        if u.lower() != c.lower():
            matched = False
            break
    inv_mismatch_counts['mismatched'] = inv_mismatch_counts.get('mismatched',0) + (0 if matched else 1)
    # op
    matchedop = True
    for u,c in zip(un_final, cl_final):
        if u.lower() != c.lower():
            matchedop = False
            break
    op_mismatch_counts['any'] += 0 if matchedop else 1

print('Summary:')
print(f"  samples parsed: {len(results)}")
print(f"  inv mismatched samples: {inv_mismatch_counts.get('mismatched',0)}")
print(f"  final op1 mismatched samples: {op_mismatch_counts['any']}")
