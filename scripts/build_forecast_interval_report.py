"""Build a conformal interval report from saved calibration/test arrays."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from gridpulse.forecasting.uncertainty.conformal import ConformalConfig, ConformalInterval


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["y_true"], data["y_pred"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/uncertainty.yaml")
    ap.add_argument("--target", default=None)
    ap.add_argument("--cal", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--out-md", default="reports/forecast_intervals.md")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    target = args.target
    if target is None:
        targets_cfg = cfg.get("targets")
        if isinstance(targets_cfg, list) and targets_cfg:
            target = str(targets_cfg[0])
        else:
            target = str(cfg.get("target", "load_mw"))

    cal_template = args.cal or cfg.get("calibration_npz", "artifacts/backtests/calibration.npz")
    test_template = args.test or cfg.get("test_npz", "artifacts/backtests/test.npz")
    cal_path = Path(cal_template.format(target=target)) if "{target}" in str(cal_template) else Path(cal_template)
    test_path = Path(test_template.format(target=target)) if "{target}" in str(test_template) else Path(test_template)
    alpha = args.alpha
    if alpha is None:
        alpha = float(cfg.get("conformal", {}).get("alpha", 0.10))
    if not cal_path.exists() or not test_path.exists():
        raise SystemExit(
            "Missing calibration/test npz. Expected: "
            f"{cal_path} and {test_path}."
        )

    y_true_cal, y_pred_cal = _load_npz(cal_path)
    y_true_test, y_pred_test = _load_npz(test_path)

    ci = ConformalInterval(ConformalConfig(alpha=alpha, horizon_wise=True, rolling=False))
    ci.fit_calibration(y_true_cal, y_pred_cal)

    coverage = ci.coverage(y_true_test, y_pred_test)
    width = ci.mean_width(y_pred_test)

    lines = [
        "# Forecast Interval Report (Conformal Prediction)\n",
        f"- target: {target}\n",
        f"- alpha: {alpha} (target coverage {1 - alpha:.0%})\n",
        f"- Test coverage (PICP): **{coverage:.3f}**\n",
        f"- Mean interval width (MPIW): **{width:.3f}**\n",
        "\n## Notes\n",
        "- Coverage near target indicates calibrated uncertainty.\n",
        "- If coverage is low, increase calibration size or use rolling calibration.\n",
    ]

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
