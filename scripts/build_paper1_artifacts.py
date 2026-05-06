#!/usr/bin/env python3
"""Build Paper 1 battery/DC3S foundation artifacts.

Requires run_cpsbench.py outputs in reports/publication/. If missing, runs CPSBench first.

Outputs:
  reports/paper1/controller_compare.csv
  reports/paper1/per_fault_compare.csv
  reports/paper1/ablation.csv
  reports/paper1/operational_trace.json
  reports/paper1/fig_cost_safety_frontier.png
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _ensure_cpsbench(pub_dir: Path) -> None:
    main_csv = pub_dir / "dc3s_main_table.csv"
    if main_csv.exists() and main_csv.stat().st_size > 0:
        return
    print("Running run_cpsbench.py (required for Paper 1 artifacts)...")
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_cpsbench.py"), "--out-dir", str(pub_dir)],
        cwd=str(REPO_ROOT),
        check=True,
    )


def _build_controller_compare(pub_dir: Path, out_dir: Path) -> None:
    if HAS_PANDAS:
        main = pd.read_csv(pub_dir / "dc3s_main_table.csv")
        agg = (
            main.groupby("controller", as_index=False)
            .agg(
                expected_cost_usd=("expected_cost_usd", "mean"),
                true_soc_violation_rate=("true_soc_violation_rate", "mean"),
                intervention_rate=("intervention_rate", "mean"),
                picp_90=("picp_90", "mean"),
                mean_interval_width=("mean_interval_width", "mean"),
            )
            .round(6)
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(out_dir / "controller_compare.csv", index=False)
    else:
        with open(pub_dir / "dc3s_main_table.csv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        by_ctrl = defaultdict(list)
        for r in rows:
            by_ctrl[r["controller"]].append(r)
        out_rows = []
        for ctrl, sub in by_ctrl.items():
            n = len(sub)
            cost = sum(float(r.get("expected_cost_usd", 0)) for r in sub) / n
            viol = sum(float(r.get("true_soc_violation_rate", 0)) for r in sub) / n
            interv = sum(float(r.get("intervention_rate", 0)) for r in sub) / n
            out_rows.append(
                {
                    "controller": ctrl,
                    "expected_cost_usd": round(cost, 6),
                    "true_soc_violation_rate": round(viol, 6),
                    "intervention_rate": round(interv, 6),
                }
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "controller_compare.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "controller",
                    "expected_cost_usd",
                    "true_soc_violation_rate",
                    "intervention_rate",
                ],
            )
            w.writeheader()
            w.writerows(out_rows)
    print(f"Wrote {out_dir / 'controller_compare.csv'}")


def _build_per_fault_compare(pub_dir: Path, out_dir: Path) -> None:
    fault_breakdown = pub_dir / "dc3s_fault_breakdown.csv"
    if fault_breakdown.exists():
        if HAS_PANDAS:
            df = pd.read_csv(fault_breakdown)
            agg = (
                df.groupby(["scenario", "controller", "fault_type"], as_index=False)
                .agg(
                    true_soc_violation_rate_at_fault=("true_soc_violation_rate_at_fault", "mean"),
                    fault_count=("fault_count", "mean"),
                )
                .round(6)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(out_dir / "per_fault_compare.csv", index=False)
        else:
            with open(fault_breakdown, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            by_key = defaultdict(list)
            for r in rows:
                by_key[(r["scenario"], r["controller"], r["fault_type"])].append(r)
            out_rows = []
            for (sc, ctrl, ft), sub in by_key.items():
                n = len(sub)
                viol = sum(float(r.get("true_soc_violation_rate_at_fault", 0)) for r in sub) / n
                cnt = sum(float(r.get("fault_count", 0)) for r in sub) / n
                out_rows.append(
                    {
                        "scenario": sc,
                        "controller": ctrl,
                        "fault_type": ft,
                        "true_soc_violation_rate_at_fault": round(viol, 6),
                        "fault_count": round(cnt, 2),
                    }
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "per_fault_compare.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "scenario",
                        "controller",
                        "fault_type",
                        "true_soc_violation_rate_at_fault",
                        "fault_count",
                    ],
                )
                w.writeheader()
                w.writerows(out_rows)
    else:
        merged = pd.read_csv(pub_dir / "cpsbench_merged_sweep.csv") if HAS_PANDAS else None
        if merged is not None:
            agg = (
                merged.groupby(["fault_dimension", "severity", "controller"], as_index=False)
                .agg(
                    true_soc_violation_rate=("true_soc_violation_rate", "mean"),
                    expected_cost_usd=("expected_cost_usd", "mean"),
                )
                .round(6)
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            agg.to_csv(out_dir / "per_fault_compare.csv", index=False)
        else:
            with open(pub_dir / "cpsbench_merged_sweep.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            by_key = defaultdict(list)
            for r in rows:
                by_key[(r["fault_dimension"], r["severity"], r["controller"])].append(r)
            out_rows = []
            for (fd, sev, ctrl), sub in by_key.items():
                n = len(sub)
                viol = sum(float(r.get("true_soc_violation_rate", 0)) for r in sub) / n
                cost = sum(float(r.get("expected_cost_usd", 0)) for r in sub) / n
                out_rows.append(
                    {
                        "fault_dimension": fd,
                        "severity": sev,
                        "controller": ctrl,
                        "true_soc_violation_rate": round(viol, 6),
                        "expected_cost_usd": round(cost, 6),
                    }
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "per_fault_compare.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "fault_dimension",
                        "severity",
                        "controller",
                        "true_soc_violation_rate",
                        "expected_cost_usd",
                    ],
                )
                w.writeheader()
                w.writerows(out_rows)
    print(f"Wrote {out_dir / 'per_fault_compare.csv'}")


def _build_ablation(pub_dir: Path, out_dir: Path) -> None:
    ablation_src = pub_dir / "ablation_table.csv"
    if ablation_src.exists():
        if HAS_PANDAS:
            df = pd.read_csv(ablation_src)
            out_dir.mkdir(parents=True, exist_ok=True)
            df.head(50).to_csv(out_dir / "ablation.csv", index=False)
        else:
            with open(ablation_src, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "ablation.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [])
                w.writeheader()
                w.writerows(rows[:50])
    else:
        main = pd.read_csv(pub_dir / "dc3s_main_table.csv")
        df = main[main["scenario"] == "drift_combo"][
            ["controller", "expected_cost_usd", "true_soc_violation_rate", "intervention_rate", "picp_90"]
        ].copy()
        out_dir.mkdir(parents=True, exist_ok=True)
        df.head(50).to_csv(out_dir / "ablation.csv", index=False)
    print(f"Wrote {out_dir / 'ablation.csv'}")


def _build_operational_trace(pub_dir: Path, out_dir: Path) -> None:
    trace_src = pub_dir / "48h_trace_de.csv"
    if not trace_src.exists():
        trace_src = pub_dir / "48h_trace_final_de.csv"
    if not trace_src.exists():
        trace_src = pub_dir / "hil_fault_response_trace.csv"
    if not trace_src.exists():
        trace = {
            "steps": [],
            "note": "No 48h_trace_de.csv or hil_fault_response_trace.csv; run generate_48h_trace.py or run_cpsbench",
        }
    else:
        with open(trace_src, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        trace = {
            "steps": rows[:24],
            "source": str(trace_src.relative_to(REPO_ROOT)),
            "n_steps": len(rows),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "operational_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'operational_trace.json'}")


def _build_fig_cost_safety_frontier(pub_dir: Path, out_dir: Path) -> None:
    with open(pub_dir / "dc3s_main_table.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_ctrl = defaultdict(list)
    for r in rows:
        by_ctrl[r["controller"]].append(r)
    ctrl_data = [
        (
            c,
            sum(float(x.get("expected_cost_usd", 0)) for x in sub) / len(sub),
            sum(float(x.get("true_soc_violation_rate", 0)) for x in sub) / len(sub),
        )
        for c, sub in sorted(by_ctrl.items())
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for controller, cost, viol in ctrl_data:
            ax.scatter(cost, viol, label=controller, alpha=0.85)
        ax.set_xlabel("Expected Cost (USD)")
        ax.set_ylabel("True SOC Violation Rate")
        ax.set_title("Cost-Safety Frontier (Paper 1)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "fig_cost_safety_frontier.png", dpi=150)
        plt.close()
        print(f"Wrote {out_dir / 'fig_cost_safety_frontier.png'}")
    except ImportError:
        _write_svg_frontier(ctrl_data, out_dir)


def _write_svg_frontier(ctrl_data: list[tuple[str, float, float]], out_dir: Path) -> None:
    colors = ["#1a2b3c", "#4c78a8", "#f58518", "#2ca02c", "#e45756", "#9b59b6"]
    w, h = 400, 300
    lines = [
        '<?xml version="1.0"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        '<text x="10" y="20" font-size="12">Cost-Safety Frontier (Paper 1)</text>',
    ]
    costs = [c for _, c, _ in ctrl_data]
    cost_min, cost_max = min(costs), max(costs) or 1
    for i, (ctrl, cost, viol) in enumerate(ctrl_data):
        x = 20 + (cost - cost_min) / max(cost_max - cost_min, 1e-9) * (w - 80)
        y = h - 50 - min(1, max(0, viol)) * (h - 80)
        lines.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4" fill="{colors[i % len(colors)]}"/>')
        lines.append(f'<text x="{min(x + 5, w - 60):.0f}" y="{y:.0f}" font-size="8">{ctrl[:12]}</text>')
    lines.append("</svg>")
    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / "fig_cost_safety_frontier.svg"
    svg_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {svg_path} (matplotlib unavailable)")
    # Try SVG→PNG via rsvg-convert or ImageMagick
    png_path = out_dir / "fig_cost_safety_frontier.png"
    for cmd in [
        ["rsvg-convert", "-f", "png", "-o", str(png_path), str(svg_path)],
        ["convert", str(svg_path), str(png_path)],
    ]:
        try:
            subprocess.run(cmd, check=True, capture_output=True, cwd=str(REPO_ROOT))
            print(f"Wrote {png_path} (via {cmd[0]})")
            break
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Paper 1 battery/DC3S artifacts")
    parser.add_argument("--out", default="reports/paper1", help="Output directory")
    parser.add_argument(
        "--publication-dir",
        default="reports/publication",
        help="Governed publication artifact directory used as the source surface",
    )
    parser.add_argument(
        "--skip-cpsbench", action="store_true", help="Skip CPSBench run; use existing publication outputs"
    )
    args = parser.parse_args()
    out_dir = _resolve_repo_path(args.out)
    pub_dir = _resolve_repo_path(args.publication_dir)
    if not args.skip_cpsbench:
        _ensure_cpsbench(pub_dir)
    _build_controller_compare(pub_dir, out_dir)
    _build_per_fault_compare(pub_dir, out_dir)
    _build_ablation(pub_dir, out_dir)
    _build_operational_trace(pub_dir, out_dir)
    _build_fig_cost_safety_frontier(pub_dir, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
