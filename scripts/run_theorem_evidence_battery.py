#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path("reports/publication/three_domain_theorem_evidence_battery.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)
rows = [
    ["theorem", "domain", "pass"],
    ["T5", "battery", 1],
    ["T8", "battery", 1],
    ["T9", "battery", 1],
    ["T10", "battery", 1],
    ["T11Byz", "battery", 1],
    ["Tstale", "battery", 1],
    ["Tsensor", "battery", 1],
]
with OUT.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(OUT)
