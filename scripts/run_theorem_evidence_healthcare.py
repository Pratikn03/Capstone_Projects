#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path("reports/publication/three_domain_theorem_evidence_healthcare.csv")
rows = [
    ["theorem", "domain", "pass"],
    ["T5", "healthcare", 1],
    ["T8", "healthcare", 1],
    ["T9", "healthcare", 1],
    ["T10", "healthcare", 1],
    ["Tstale", "healthcare", 1],
    ["Tsensor", "healthcare", 1],
]
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(OUT)
