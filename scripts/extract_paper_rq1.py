"""Extract RQ1 detailed tables from paper text using simple line-by-line parsing.
"""
import json
import os
import re

PAPER_TEXT = '/tmp/paper_text.txt'
OUT_FILE = 'outputs/figure_data/rq1_paper.json'

ATTACKS = ['MEA0','MEA1','MEA2','MEA3','MEA4','MEA5','AdvMEA','CEGA','Realistic',
           'DFEA_I','DFEA_II','DFEA_III']
BUDGETS = [0.05, 0.10, 0.25, 0.50, 1.00]
# Use strict 1-decimal pattern: paper RQ1 tables format values as XX.X or X.X
# (always exactly 1 decimal). Stuck-together bold values look like "63.5±2.081.5±0.7"
# which must split as (63.5, 2.0) and (81.5, 0.7), NOT (63.5, 2.081).
NUM_PATTERN = re.compile(r'(\d+\.\d)±(\d+\.\d)')

with open(PAPER_TEXT) as f:
    lines = f.readlines()

data = {}  # dataset -> metric -> regime -> attack -> {budget: (mean, std)}

# State machine
i = 0
while i < len(lines):
    line = lines[i]
    # Detect main RQ1 table header (Tables 16-36)
    m = re.match(r'\s*Table\s+(\d+):\s+RQ1 detailed for dataset=\s*(\S+?),\s*metric=\s*(Acc|F1|Fidelity)', line)
    if m:
        ds = m.group(2).strip().rstrip(',').replace('OGBN-Arxiv','OGBNArxiv')
        metric = m.group(3)
        # Read until next "Table" or end of section
        i += 1
        cur_regime = None
        while i < len(lines):
            l = lines[i]
            if re.match(r'\s*Table\s+\d+:', l):
                break
            # Detect regime header: (a) Regime=both
            rm = re.match(r'\s*\([a-d]\)\s*Regime=\s*(\w+)', l)
            if rm:
                cur_regime = rm.group(1)
                i += 1
                continue
            # Detect attack row: starts with attack name
            for atk in ATTACKS:
                if l.startswith(atk):
                    # Extract pairs
                    rest = l[len(atk):]
                    pairs = NUM_PATTERN.findall(rest)
                    if len(pairs) >= 5 and cur_regime:
                        data.setdefault(ds, {}).setdefault(metric, {}).setdefault(cur_regime, {})[atk] = {
                            str(b): {'mean': float(pairs[k][0]), 'std': float(pairs[k][1])}
                            for k, b in enumerate(BUDGETS)
                        }
                    break
            i += 1
        continue
    # Detect Tables 41-43 (new 3 datasets, fidelity only)
    m2 = re.match(r'\s*Table\s+(\d+):\s+Detailed RQ1 results on\s*(\S+)\.', line)
    if m2:
        ds = m2.group(2).strip().replace('OGBN-Arxiv','OGBNArxiv')
        metric = 'Fidelity'
        i += 1
        cur_regime = None
        while i < len(lines):
            l = lines[i]
            if re.match(r'\s*Table\s+\d+:', l):
                break
            rm = re.match(r'\s*\([a-d]\)\s*Regime=\s*(\w+)', l)
            if rm:
                cur_regime = rm.group(1)
                i += 1
                continue
            for atk in ATTACKS:
                if l.startswith(atk):
                    rest = l[len(atk):]
                    pairs = NUM_PATTERN.findall(rest)
                    if len(pairs) >= 5 and cur_regime:
                        data.setdefault(ds, {}).setdefault(metric, {}).setdefault(cur_regime, {})[atk] = {
                            str(b): {'mean': float(pairs[k][0]), 'std': float(pairs[k][1])}
                            for k, b in enumerate(BUDGETS)
                        }
                    break
            i += 1
        continue
    i += 1

os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, 'w') as f:
    json.dump(data, f, indent=2)

# Summary
print(f"Extracted RQ1 data for {len(data)} datasets:")
for ds in sorted(data.keys()):
    metrics = list(data[ds].keys())
    regimes = set()
    n_atk = 0
    for met in data[ds]:
        for reg in data[ds][met]:
            regimes.add(reg)
            n_atk = max(n_atk, len(data[ds][met][reg]))
    print(f"  {ds}: metrics={metrics}, regimes={sorted(regimes)}, attacks={n_atk}")
print(f"\nSaved to {OUT_FILE}")
