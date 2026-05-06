#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bootstrap_ci import compute_ci_summary_df, write_ci_outputs

from orius.cpsbench_iot.runner import run_suite


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in _parse_csv_list(raw)]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DC3S CPSBench sensitivity sweeps")
    parser.add_argument("--out-dir", default="reports/publication")
    parser.add_argument("--horizon", type=int, default=168)
    parser.add_argument("--scenarios", default="drift_combo,dropout")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--alpha0", default="0.05,0.10,0.15")
    parser.add_argument("--ph-lambda", dest="ph_lambda", default="3.0,5.0,7.0")
    parser.add_argument("--kappa-drift-penalty", dest="kappa_drift_penalty", default="0.3,0.5,0.7")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _validate_args(
    *,
    horizon: int,
    scenarios: list[str],
    seeds: list[int],
    alpha0_values: list[float],
    ph_lambda_values: list[float],
    kpen_values: list[float],
    n_bootstrap: int,
    confidence: float,
) -> None:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if not alpha0_values:
        raise ValueError("alpha0 must be non-empty")
    if not ph_lambda_values:
        raise ValueError("ph-lambda must be non-empty")
    if not kpen_values:
        raise ValueError("kappa-drift-penalty must be non-empty")
    if n_bootstrap < 1:
        raise ValueError("n-bootstrap must be >= 1")
    if not (0.0 < confidence < 1.0):
        raise ValueError("confidence must be between 0 and 1")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _grid_tag(alpha0: float, ph_lambda: float, kpen: float) -> str:
    return f"alpha0={alpha0:g}__ph_lambda={ph_lambda:g}__kpen={kpen:g}"


def _load_base_dc3s_config() -> dict[str, Any]:
    cfg_path = REPO_ROOT / "configs" / "dc3s.yaml"
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configs/dc3s.yaml must contain a mapping")
    dc3s = payload.get("dc3s")
    if not isinstance(dc3s, dict):
        raise ValueError("configs/dc3s.yaml must contain a dc3s mapping")
    return payload


def run_sensitivity_sweeps(
    *,
    out_dir: str | Path = "reports/publication",
    horizon: int = 168,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    alpha0_values: list[float] | None = None,
    ph_lambda_values: list[float] | None = None,
    kpen_values: list[float] | None = None,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    scenarios = list(scenarios or ["drift_combo", "dropout"])
    seeds = [int(s) for s in (seeds or [11, 22, 33, 44, 55])]
    alpha0_values = list(alpha0_values or [0.05, 0.10, 0.15])
    ph_lambda_values = list(ph_lambda_values or [3.0, 5.0, 7.0])
    kpen_values = list(kpen_values or [0.3, 0.5, 0.7])

    _validate_args(
        horizon=horizon,
        scenarios=scenarios,
        seeds=seeds,
        alpha0_values=alpha0_values,
        ph_lambda_values=ph_lambda_values,
        kpen_values=kpen_values,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )

    out_path = Path(out_dir)
    sweeps_dir = out_path / "sweeps"
    configs_dir = sweeps_dir / "configs"
    runs_dir = sweeps_dir / "runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    seeds_str = ",".join(str(item) for item in seeds)
    prior_env = os.environ.get("ORIUS_DC3S_CONFIG")
    rows: list[pd.DataFrame] = []
    grid_points = 0

    try:
        for alpha0, ph_lambda, kpen in itertools.product(alpha0_values, ph_lambda_values, kpen_values):
            grid_points += 1
            payload = _load_base_dc3s_config()
            payload.setdefault("dc3s", {})
            payload["dc3s"]["alpha0"] = float(alpha0)
            drift_cfg = payload["dc3s"].get("drift")
            if not isinstance(drift_cfg, dict):
                drift_cfg = {}
                payload["dc3s"]["drift"] = drift_cfg
            drift_cfg["ph_lambda"] = float(ph_lambda)
            payload["dc3s"]["kappa_drift_penalty"] = float(kpen)

            tag = _grid_tag(alpha0, ph_lambda, kpen)
            config_path = configs_dir / f"tmp_dc3s_{tag}.yaml"
            config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

            run_dir = runs_dir / tag
            os.environ["ORIUS_DC3S_CONFIG"] = str(config_path)
            try:
                run_suite(
                    scenarios=scenarios,
                    seeds=seeds,
                    out_dir=run_dir,
                    horizon=horizon,
                    include_fault_sweep=False,
                )
            finally:
                if prior_env is None:
                    os.environ.pop("ORIUS_DC3S_CONFIG", None)
                else:
                    os.environ["ORIUS_DC3S_CONFIG"] = prior_env

            run_csv = run_dir / "dc3s_main_table.csv"
            df = pd.read_csv(run_csv)
            df["alpha0"] = float(alpha0)
            df["ph_lambda"] = float(ph_lambda)
            df["kappa_drift_penalty"] = float(kpen)
            df["horizon"] = int(horizon)
            df["seeds"] = seeds_str
            df["run_dir"] = _display_path(run_dir)
            rows.append(df)
    finally:
        if prior_env is None:
            os.environ.pop("ORIUS_DC3S_CONFIG", None)
        else:
            os.environ["ORIUS_DC3S_CONFIG"] = prior_env

    if not rows:
        raise ValueError("No sweep rows were produced")

    grid_df = pd.concat(rows, ignore_index=True)
    grid_df = grid_df.sort_values(
        ["alpha0", "ph_lambda", "kappa_drift_penalty", "scenario", "seed", "controller"],
        kind="stable",
    ).reset_index(drop=True)

    out_path.mkdir(parents=True, exist_ok=True)
    grid_csv = out_path / "sensitivity_grid.csv"
    grid_df.to_csv(grid_csv, index=False, float_format="%.6f")

    summary = compute_ci_summary_df(
        grid_df,
        group_cols=["scenario", "controller", "alpha0", "ph_lambda", "kappa_drift_penalty"],
        metrics=["picp_90", "mean_interval_width", "expected_cost_usd", "intervention_rate"],
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )
    summary_paths = write_ci_outputs(
        summary,
        out_dir=out_path,
        output_stem="sensitivity_summary_ci",
        group_cols=["scenario", "controller", "alpha0", "ph_lambda", "kappa_drift_penalty"],
        title="# DC3S Sensitivity Sweep Bootstrap Confidence Intervals",
        provenance=(
            "Derived from `sensitivity_grid.csv` by grouping on "
            "`(scenario, controller, alpha0, ph_lambda, kappa_drift_penalty)` "
            "and bootstrapping group means across seed rows."
        ),
        include_latex=False,
    )

    return {
        "grid_points": grid_points,
        "rows_grid": int(len(grid_df)),
        "rows_summary": int(len(summary)),
        "out_dir": str(out_path),
        "grid_csv": str(grid_csv),
        "summary_csv": summary_paths["csv"],
        "summary_md": summary_paths["markdown"],
    }


def main() -> None:
    args = _parse_args()
    payload = run_sensitivity_sweeps(
        out_dir=args.out_dir,
        horizon=int(args.horizon),
        scenarios=_parse_csv_list(args.scenarios),
        seeds=_parse_int_list(args.seeds),
        alpha0_values=_parse_float_list(args.alpha0),
        ph_lambda_values=_parse_float_list(args.ph_lambda),
        kpen_values=_parse_float_list(args.kappa_drift_penalty),
        n_bootstrap=int(args.n_bootstrap),
        confidence=float(args.confidence),
        seed=int(args.seed),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
