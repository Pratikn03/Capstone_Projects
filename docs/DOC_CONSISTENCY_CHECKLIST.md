# Documentation Consistency Checklist (Phase 1–6)

Use this checklist before freezing paper/repo claims.

## 1) Freeze One Run ID

1. Ensure training/report/research pipeline is complete.
2. Build frozen snapshot:

```bash
./.venv/bin/python3 scripts/build_metrics_snapshot.py --output reports/frozen_metrics_snapshot.json
```

3. Confirm snapshot includes:
- `frozen_run_id`
- DE/US `cost_savings_pct`, `carbon_reduction_pct`, `peak_shaving_pct`
- DE/US `evpi_robust`, `evpi_deterministic`, `vss`

## 2) Canonical Source Files

- `reports/impact_summary.csv`
- `reports/eia930/impact_summary.csv`
- `reports/research_metrics_de.csv`
- `reports/research_metrics_us.csv`
- `reports/frozen_metrics_snapshot.json`

## 3) Update Targets

- `paper/paper.tex`
- `paper/PAPER_DRAFT.md`
- `README.md`
- `docs/ADVANCED_FEATURES.md`
- `docs/PROJECT.md`

## 4) Stale-Claim Guardrails

Run these checks after editing:

```bash
rg -n "2\\.89%|0\\.58%|point-forecast MILP|Single-Point Forecasts for Optimization|Linear Battery Model: Our MILP formulation assumes linear charging/discharging efficiency" paper README.md docs
```

If matches are intentional historical references, annotate them explicitly as historical and not current.

## 5) Metric Integrity Checks

```bash
./.venv/bin/python3 - <<'PY'
import json
from pathlib import Path

p = Path("reports/frozen_metrics_snapshot.json")
data = json.loads(p.read_text())
print("frozen_run_id:", data["frozen_run_id"])
print("DE:", data["datasets"]["de"])
print("US:", data["datasets"]["us"])
PY
```

## 6) Regression Safety

```bash
./.venv/bin/pytest -q
```
