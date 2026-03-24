#!/usr/bin/env python3
"""Generate software-in-loop validation artifacts for peer ORIUS domains."""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_all_domain_eval import DOMAINS, eval_domain


RUNTIME_DOMAINS = ("av", "industrial", "healthcare", "navigation", "aerospace")
MAX_P95_LATENCY_MS = 50.0


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _tex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _write_summary_tex(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Software-in-loop validation summary for the peer ORIUS domains.}",
        r"\label{tab:domain-sil-summary}",
        r"\begin{tabular}{lrrrrrl}",
        r"\toprule",
        r"Domain & Baseline TSVR (\%) & ORIUS TSVR (\%) & Cert. Rate (\%) & Error Rate (\%) & P95 (ms) & SIL pass \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_tex_escape(row['display'])} & "
            f"{float(row['baseline_tsvr_mean']):.2f} & "
            f"{float(row['orius_tsvr_mean']):.2f} & "
            f"{float(row['certificate_rate_pct']):.2f} & "
            f"{float(row['runtime_error_rate_pct']):.2f} & "
            f"{float(row['p95_latency_ms']):.3f} & "
            f"{'yes' if row['sil_pass'] else 'no'} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["display"]) for row in rows]
    cert_rate = [float(row["certificate_rate_pct"]) for row in rows]
    errors = [float(row["runtime_error_rate_pct"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.8, 4.3))
    x = np.arange(len(rows))
    width = 0.36
    ax.bar(x - width / 2, cert_rate, width, label="Certificate rate", color="#2b6f56", alpha=0.85)
    ax.bar(x + width / 2, errors, width, label="Runtime error rate", color="#b33a3a", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Universal ORIUS Software-in-Loop Validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Universal ORIUS software-in-loop validation")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--out", default="reports/universal_sil_validation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for domain in RUNTIME_DOMAINS:
        cfg = DOMAINS[domain]
        episodes = []
        for seed_idx in range(args.seeds):
            rng = np.random.default_rng(5000 + seed_idx)
            episode = eval_domain(domain, cfg, rng, n_rows=args.rows, capture_trace=True)
            episodes.append(episode)
            trace_rows = episode.get("trace_rows", [])
            if trace_rows:
                _write_csv(traces_dir / f"{domain}_seed{seed_idx}.csv", trace_rows)

        summary_rows.append(
            {
                "domain": domain,
                "display": cfg["display"],
                "data_source": episodes[0]["data_source"],
                "baseline_tsvr_mean": round(_mean([float(ep["violation_rate_before_pct"]) for ep in episodes]), 2),
                "orius_tsvr_mean": round(_mean([float(ep["violation_rate_after_pct"]) for ep in episodes]), 2),
                "repair_rate_pct": round(_mean([float(ep["repair_rate_pct"]) for ep in episodes]), 2),
                "certificate_rate_pct": round(_mean([float(ep["certificate_rate_pct"]) for ep in episodes]), 2),
                "runtime_error_rate_pct": round(_mean([float(ep["runtime_error_rate_pct"]) for ep in episodes]), 2),
                "mean_reliability": round(_mean([float(ep["mean_reliability"]) for ep in episodes]), 4),
                "p95_latency_ms": round(_mean([float(ep["p95_latency_ms"]) for ep in episodes]), 3),
                "sil_pass": all(
                    float(ep["certificate_rate_pct"]) >= 100.0
                    and float(ep["runtime_error_rate_pct"]) <= 0.0
                    and float(ep["p95_latency_ms"]) <= MAX_P95_LATENCY_MS
                    for ep in episodes
                ),
            }
        )

    _write_csv(out_dir / "domain_sil_summary.csv", summary_rows)
    _write_summary_tex(out_dir / "tbl_domain_sil_summary.tex", summary_rows)
    _write_figure(out_dir / "fig_domain_sil_status.png", summary_rows)
    report = {
        "domains": [row["domain"] for row in summary_rows],
        "sil_pass_domains": [row["domain"] for row in summary_rows if row["sil_pass"]],
        "failed_domains": [row["domain"] for row in summary_rows if not row["sil_pass"]],
        "all_passed": all(bool(row["sil_pass"]) for row in summary_rows),
        "summary_csv": str(out_dir / "domain_sil_summary.csv"),
        "summary_tex": str(out_dir / "tbl_domain_sil_summary.tex"),
        "figure": str(out_dir / "fig_domain_sil_status.png"),
        "traces_dir": str(traces_dir),
    }
    (out_dir / "sil_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Universal SIL Validation ===")
    for row in summary_rows:
        print(
            f"  {row['domain']}: sil_pass={row['sil_pass']} "
            f"cert={row['certificate_rate_pct']}% err={row['runtime_error_rate_pct']}% "
            f"tsvr={row['baseline_tsvr_mean']}->{row['orius_tsvr_mean']}"
        )
    print(f"  Report → {out_dir / 'sil_validation_report.json'}")
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
