"""Generate unified all-6-domain training metrics LaTeX table.

Outputs: reports/universal_orius_validation/tbl_all_domain_training_metrics.tex
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "reports/universal_orius_validation/tbl_all_domain_training_metrics.tex"

# Hardcoded from locked artifacts — do not edit without re-running the audit scripts.
# Battery: artifacts/runs/de/r1_gbm_fast_20260311/  (DE locked run, load_mw, h=1)
# Others: reports/{domain}/week2_metrics.json + run_universal_training_audit.py
ROWS = [
    # Domain                              Train   Cal    Test   RMSE     MAE     sMAPE   PICP90  Width
    ("Energy Management (Battery / DE)", 12163, 868, 2537, 253.32, 200.90, "0.0035", "0.924", "1111.05"),
    ("Autonomous Vehicles", 85, 6, 19, 0.0565, 0.0443, "0.0036", "0.947", "0.2663"),
    ("Industrial Process Control", 6680, 477, 1433, 3.6095, 2.7294, "0.0060", "0.896", "11.5782"),
    ("Medical Monitoring (ICU Vitals)", 319, 22, 71, 0.7638, 0.6233, "0.0070", "0.930", "3.1030"),
    ("Aerospace (Flight Envelope)", 3483, 248, 748, 16.32, 12.58, "0.0066", "0.912", "35.61"),
    ("Navigation (Simulation)", None, None, None, None, None, "—", "—", "—"),
]


def _fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:.{decimals}f}"


def main() -> None:
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Training, calibration, and test surfaces for all six ORIUS domains.",
        r"  Battery metrics are from the locked DE ENTSO-E run (load\_mw, $h{=}1$, conformal-adjusted PICP).",
        r"  Navigation uses a closed-loop simulation surface with no trained forecasting model.",
        r"  PICP$_{90}$ target is $\geq 0.90$ for all forecasting-capable domains.}",
        r"\label{tab:all-domain-training-metrics}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Domain & Train & Cal & Test & RMSE & MAE & sMAPE & PICP$_{90}$ & Mean width \\",
        r"\midrule",
    ]

    for row in ROWS:
        domain = row[0]
        train, cal, test = row[1], row[2], row[3]
        rmse, mae, smape, picp, width = row[4], row[5], row[6], row[7], row[8]
        cells = [
            domain,
            _fmt(train),
            _fmt(cal),
            _fmt(test),
            _fmt(rmse, 2) if isinstance(rmse, float) else _fmt(rmse),
            _fmt(mae, 2) if isinstance(mae, float) else _fmt(mae),
            smape,
            picp,
            width,
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
