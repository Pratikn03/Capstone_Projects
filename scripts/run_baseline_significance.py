#!/usr/bin/env python3
"""Run pairwise DM tests, paired-bootstrap CIs, and Holm correction across baselines.

Reads per-(region, target, model, seed) predictions from
``artifacts/runs/{region}/{release_id}/predictions/{model}_{target}_seed{seed}.npz``
(each containing arrays ``y_true`` and ``y_pred``) and emits

    reports/publication/baseline_significance.csv
    reports/publication/baseline_significance.json

The reference model defaults to GBM. For each (region, target, baseline), the
script reports: mean RMSE/MAE delta vs reference, BCa 95% CI on the delta,
Diebold-Mariano statistic and p-value, Cohen's d on per-seed RMSE, and the
Holm-adjusted p-value across the entire (region * target * baseline) family.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from orius.forecasting.stats import (
    cohens_d,
    diebold_mariano,
    holm_bonferroni,
    paired_block_bootstrap,
)
from orius.utils.metrics import mae as _mae
from orius.utils.metrics import rmse as _rmse

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFERENCE = "gbm"
DEFAULT_BASELINES = (
    "lstm",
    "tcn",
    "nbeats",
    "tft",
    "patchtst",
    "prophet",
    "nbeats_darts",
    "ngboost",
    "flaml",
)
DEFAULT_TARGETS = ("load_mw", "wind_mw", "solar_mw")


def _load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return data["y_true"].astype(float), data["y_pred"].astype(float)


def _gather(
    region: str, release_id: str, model: str, target: str, predictions_dir: Path
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    pattern = f"{model}_{target}_seed*.npz"
    for path in sorted(predictions_dir.glob(pattern)):
        seed_str = path.stem.split("_seed")[-1]
        try:
            seed = int(seed_str)
        except ValueError:
            continue
        try:
            yt, yp = _load_predictions(path)
        except (KeyError, OSError):
            continue
        out.append((seed, yt, yp))
    return out


def _per_seed_rmse(records: list[tuple[int, np.ndarray, np.ndarray]]) -> list[float]:
    return [float(_rmse(yt, yp)) for _, yt, yp in records]


def _aligned_predictions(
    ref: list[tuple[int, np.ndarray, np.ndarray]],
    other: list[tuple[int, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_map = {seed: (yt, yp) for seed, yt, yp in ref}
    common = sorted(set(ref_map) & {seed for seed, *_ in other})
    if not common:
        raise ValueError("no overlapping seeds between reference and baseline predictions")
    yts: list[np.ndarray] = []
    p_refs: list[np.ndarray] = []
    p_others: list[np.ndarray] = []
    other_map = {seed: (yt, yp) for seed, yt, yp in other}
    for seed in common:
        yt_r, yp_r = ref_map[seed]
        yt_o, yp_o = other_map[seed]
        n = min(len(yt_r), len(yp_r), len(yt_o), len(yp_o))
        yts.append(yt_r[:n])
        p_refs.append(yp_r[:n])
        p_others.append(yp_o[:n])
    return np.concatenate(yts), np.concatenate(p_refs), np.concatenate(p_others)


def run_significance(
    *,
    region: str,
    release_id: str,
    targets: tuple[str, ...],
    reference: str,
    baselines: tuple[str, ...],
    predictions_dir: Path,
    horizon: int,
    n_resamples: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    p_values: list[float] = []
    indices: list[int] = []

    for target in targets:
        ref_records = _gather(region, release_id, reference, target, predictions_dir)
        if not ref_records:
            continue
        ref_rmses = _per_seed_rmse(ref_records)
        for baseline in baselines:
            other_records = _gather(region, release_id, baseline, target, predictions_dir)
            if not other_records:
                rows.append(
                    {
                        "region": region,
                        "target": target,
                        "reference": reference,
                        "baseline": baseline,
                        "status": "missing",
                    }
                )
                continue
            try:
                yt, p_ref, p_oth = _aligned_predictions(ref_records, other_records)
            except ValueError:
                rows.append(
                    {
                        "region": region,
                        "target": target,
                        "reference": reference,
                        "baseline": baseline,
                        "status": "no_overlap",
                    }
                )
                continue
            dm = diebold_mariano(yt, p_ref, p_oth, horizon=horizon, loss="se")
            ci = paired_block_bootstrap(yt, p_ref, p_oth, metric="rmse", n_resamples=n_resamples, seed=seed)
            mae_ref = float(_mae(yt, p_ref))
            mae_oth = float(_mae(yt, p_oth))
            d = cohens_d(ref_rmses, _per_seed_rmse(other_records))
            row_idx = len(rows)
            rows.append(
                {
                    "region": region,
                    "target": target,
                    "reference": reference,
                    "baseline": baseline,
                    "status": "ok",
                    "n_paired": int(min(len(yt), len(p_ref), len(p_oth))),
                    "rmse_reference": float(_rmse(yt, p_ref)),
                    "rmse_baseline": float(_rmse(yt, p_oth)),
                    "rmse_delta": float(ci.delta),
                    "rmse_ci_low": float(ci.ci_low),
                    "rmse_ci_high": float(ci.ci_high),
                    "mae_reference": mae_ref,
                    "mae_baseline": mae_oth,
                    "dm_statistic": float(dm.statistic),
                    "dm_p_value": float(dm.p_value),
                    "dm_horizon": int(dm.horizon),
                    "cohens_d_rmse": float(d),
                    "n_seeds_reference": len(ref_records),
                    "n_seeds_baseline": len(other_records),
                    "reference_beats_baseline": bool(ci.delta > 0 and ci.ci_low > 0),
                    "baseline_beats_reference": bool(ci.delta < 0 and ci.ci_high < 0),
                }
            )
            p_values.append(float(dm.p_value))
            indices.append(row_idx)

    if p_values:
        adjusted = holm_bonferroni(p_values, alpha=0.05)
        for idx, (_raw, adj, reject) in zip(indices, adjusted, strict=True):
            rows[idx]["dm_p_holm"] = float(adj)
            rows[idx]["dm_significant_holm_05"] = bool(reject)

    return rows


def write_outputs(rows: list[dict[str, object]], out_dir: Path, region: str, release_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "baseline_significance.csv"
    json_path = out_dir / "baseline_significance.json"
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    json_path.write_text(
        json.dumps(
            {"region": region, "release_id": release_id, "rows": rows},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pairwise DM + bootstrap + Holm across baselines")
    p.add_argument("--region", required=True)
    p.add_argument("--release-id", required=True)
    p.add_argument("--predictions-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "publication")
    p.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    p.add_argument("--reference", default=DEFAULT_REFERENCE)
    p.add_argument("--baselines", nargs="+", default=list(DEFAULT_BASELINES))
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    predictions_dir = args.predictions_dir or (
        REPO_ROOT / "artifacts" / "runs" / args.region.lower() / args.release_id / "predictions"
    )
    if not predictions_dir.exists():
        raise FileNotFoundError(f"predictions directory does not exist: {predictions_dir}")
    rows = run_significance(
        region=args.region,
        release_id=args.release_id,
        targets=tuple(args.targets),
        reference=args.reference,
        baselines=tuple(args.baselines),
        predictions_dir=predictions_dir,
        horizon=int(args.horizon),
        n_resamples=int(args.n_resamples),
        seed=int(args.seed),
    )
    if not rows:
        print("no comparable rows generated; check predictions directory")
        return 1
    write_outputs(rows, args.out_dir, args.region, args.release_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
