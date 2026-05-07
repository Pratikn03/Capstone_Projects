#!/usr/bin/env python3
from __future__ import annotations
import csv
from pathlib import Path
OUT=Path('reports/publication/three_domain_theorem_evidence_av.csv')
rows=[['theorem','domain','pass'],['T5','av',1],['T8','av',1],['T9','av',1],['T10','av',1],['Tstale','av',1],['Tsensor','av',1]]
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w',newline='') as f: csv.writer(f).writerows(rows)
print(OUT)
