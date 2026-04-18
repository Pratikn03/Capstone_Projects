"""Grouped coverage validation for reliability-binned Mondrian calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from orius.forecasting.uncertainty.reliability_mondrian import ReliabilityMondrian, ReliabilityMondrianConfig


ROOT = Path(__file__).resolve().parent


def run_validation(seed: int = 42) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    w = rng.uniform(0.2, 1.0, size=2500)
    x = rng.normal(size=2500)
    y = x + rng.normal(scale=0.3, size=2500)
    x_observed = x + rng.normal(scale=0.3, size=2500) / w
    model = ReliabilityMondrian(ReliabilityMondrianConfig(alpha=0.10, n_bins=3, min_bin_size=50))
    model.fit(y_true=y, y_pred=x_observed, reliability=w)

    w_test = rng.uniform(0.2, 1.0, size=1200)
    x_true = rng.normal(size=1200)
    y_true = x_true + rng.normal(scale=0.3, size=1200)
    x_obs = x_true + rng.normal(scale=0.3, size=1200) / w_test
    lower, upper = model.predict_interval(y_pred=x_obs, reliability=w_test)
    return model.group_coverage(y_true=y_true, lower=lower, upper=upper, reliability=w_test)


def main() -> list[dict[str, float | int]]:
    coverage = run_validation()
    lines = ["bin_id,reliability_lower,reliability_upper,count,coverage,avg_width,qhat"]
    for metrics in coverage:
        lines.append(
            ",".join(
                [
                    str(int(metrics["bin_id"])),
                    f"{float(metrics['reliability_lower']):.6f}",
                    f"{float(metrics['reliability_upper']):.6f}",
                    str(int(metrics["n"])),
                    f"{float(metrics['picp']):.6f}",
                    f"{float(metrics['mean_interval_width']):.6f}",
                    f"{float(metrics['qhat']):.6f}",
                ]
            )
        )
    (ROOT / "coverage_validation.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return coverage


if __name__ == "__main__":
    main()
