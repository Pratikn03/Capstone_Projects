from __future__ import annotations

import csv
from pathlib import Path


def fmt_pct(val: str | None) -> str:
    if val is None or val == "":
        return "TBD"
    try:
        return f"{float(val):.2f}%"
    except ValueError:
        return "TBD"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = repo_root / "reports" / "impact_summary.csv"
    readme_path = repo_root / "README.md"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run scripts/build_reports.py first.")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise SystemExit("impact_summary.csv is empty.")

    row = rows[0]
    cost = fmt_pct(row.get("cost_savings_pct"))
    carbon = fmt_pct(row.get("carbon_reduction_pct"))
    peak = fmt_pct(row.get("peak_shaving_pct"))

    readme = readme_path.read_text(encoding="utf-8").splitlines()
    updated = []
    in_table = False
    for line in readme:
        if line.strip() == "| Metric | Value |":
            in_table = True
            updated.append(line)
            continue
        if in_table and line.strip().startswith("| Cost savings"):
            updated.append(f"| Cost savings | {cost} |")
            continue
        if in_table and line.strip().startswith("| Carbon reduction"):
            updated.append(f"| Carbon reduction | {carbon} |")
            continue
        if in_table and line.strip().startswith("| Peak shaving"):
            updated.append(f"| Peak shaving | {peak} |")
            in_table = False
            continue
        updated.append(line)

    readme_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    print("README impact table updated.")


if __name__ == "__main__":
    main()
