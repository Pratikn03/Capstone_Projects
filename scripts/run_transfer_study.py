#!/usr/bin/env python3
"""Cross-region transfer study scaffold.

Evaluates how models trained on one BA transfer to another,
producing a transfer matrix showing degradation in MAE/RMSE/PICP
across DE ↔ US and inter-BA (MISO ↔ PJM ↔ ERCOT) transfers.

Usage:
    python scripts/run_transfer_study.py --pairs DE:US_MISO US_MISO:US_PJM
    python scripts/run_transfer_study.py --all-pairs
    python scripts/run_transfer_study.py --all-pairs --models baseline_gbm dl_lstm
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gridpulse.evaluation.metrics import mae, mape, rmse  # noqa: E402

logger = logging.getLogger(__name__)

ALL_REGIONS = ["DE", "US_MISO", "US_PJM", "US_ERCOT"]

MODELS_DIR_MAP = {
    "DE": "artifacts/models",
    "US_MISO": "artifacts/models_eia930",
    "US_PJM": "artifacts/models_eia930_pjm",
    "US_ERCOT": "artifacts/models_eia930_ercot",
}

FEATURES_PATH_MAP = {
    "DE": "data/processed/features.parquet",
    "US_MISO": "data/processed/us_eia930/features.parquet",
    "US_PJM": "data/processed/us_eia930_pjm/features.parquet",
    "US_ERCOT": "data/processed/us_eia930_ercot/features.parquet",
}

TARGETS = ["load_mw", "solar_mw", "wind_mw"]


@dataclass
class TransferResult:
    source_region: str
    target_region: str
    model_name: str
    target_col: str
    mae_native: float
    mae_transfer: float
    rmse_native: float
    rmse_transfer: float
    mape_native: float
    mape_transfer: float
    degradation_mae_pct: float
    degradation_rmse_pct: float


@dataclass
class TransferStudyReport:
    pairs: list[TransferResult] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.pairs])


def _load_test_split(features_path: Path, test_ratio: float = 0.15) -> pd.DataFrame:
    """Load the test partition of a region's feature set."""
    df = pd.read_parquet(features_path)
    n = len(df)
    test_start = int(n * (1.0 - test_ratio))
    return df.iloc[test_start:].copy()


def _load_model(models_dir: Path, model_name: str):
    """Load a trained model artifact. Returns (model, model_type)."""
    import joblib

    gbm_path = models_dir / f"{model_name}.joblib"
    if gbm_path.exists():
        return joblib.load(gbm_path), "gbm"

    pt_path = models_dir / f"{model_name}.pt"
    if pt_path.exists():
        import torch
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=False)
        return checkpoint, "dl"

    raise FileNotFoundError(f"No model artifact found for {model_name} in {models_dir}")


def _predict_gbm(model, X: pd.DataFrame) -> np.ndarray:
    """Predict with a GBM model, handling feature alignment."""
    expected_features = model.feature_name_ if hasattr(model, "feature_name_") else None
    if expected_features is not None:
        missing = set(expected_features) - set(X.columns)
        if missing:
            for col in missing:
                X[col] = 0.0
        X = X[expected_features]
    return model.predict(X)


def evaluate_transfer_pair(
    source_region: str,
    target_region: str,
    model_name: str,
    target_col: str,
) -> TransferResult | None:
    """Evaluate a single source→target transfer for one model and target."""
    source_models = REPO_ROOT / MODELS_DIR_MAP[source_region]
    source_features = REPO_ROOT / FEATURES_PATH_MAP[source_region]
    target_features = REPO_ROOT / FEATURES_PATH_MAP[target_region]

    for p in [source_features, target_features]:
        if not p.exists():
            logger.warning("Features not found: %s", p)
            return None

    try:
        model, model_type = _load_model(source_models, model_name)
    except FileNotFoundError:
        logger.warning("Model %s not found in %s", model_name, source_models)
        return None

    source_test = _load_test_split(source_features)
    target_test = _load_test_split(target_features)

    if target_col not in source_test.columns or target_col not in target_test.columns:
        logger.warning("Target %s missing from one of the datasets", target_col)
        return None

    feature_cols = [c for c in source_test.columns if c not in TARGETS + ["timestamp"]]

    if model_type == "gbm":
        y_native = source_test[target_col].values
        pred_native = _predict_gbm(model, source_test[feature_cols].copy())

        y_transfer = target_test[target_col].values
        target_X = target_test[[c for c in feature_cols if c in target_test.columns]].copy()
        pred_transfer = _predict_gbm(model, target_X)
    else:
        logger.info("DL transfer eval not yet implemented for %s", model_name)
        return None

    mae_n = float(mae(y_native, pred_native))
    mae_t = float(mae(y_transfer, pred_transfer))
    rmse_n = float(rmse(y_native, pred_native))
    rmse_t = float(rmse(y_transfer, pred_transfer))
    mape_n = float(mape(y_native, pred_native))
    mape_t = float(mape(y_transfer, pred_transfer))

    return TransferResult(
        source_region=source_region,
        target_region=target_region,
        model_name=model_name,
        target_col=target_col,
        mae_native=mae_n,
        mae_transfer=mae_t,
        rmse_native=rmse_n,
        rmse_transfer=rmse_t,
        mape_native=mape_n,
        mape_transfer=mape_t,
        degradation_mae_pct=((mae_t - mae_n) / mae_n * 100) if mae_n > 0 else 0.0,
        degradation_rmse_pct=((rmse_t - rmse_n) / rmse_n * 100) if rmse_n > 0 else 0.0,
    )


def run_transfer_study(
    pairs: list[tuple[str, str]],
    model_names: list[str] | None = None,
    target_cols: list[str] | None = None,
) -> TransferStudyReport:
    """Run cross-region transfer evaluation for all specified pairs."""
    if model_names is None:
        model_names = ["baseline_gbm"]
    if target_cols is None:
        target_cols = TARGETS

    report = TransferStudyReport()

    for source, target in pairs:
        for model_name in model_names:
            for target_col in target_cols:
                logger.info("Transfer %s → %s | %s | %s", source, target, model_name, target_col)
                result = evaluate_transfer_pair(source, target, model_name, target_col)
                if result is not None:
                    report.pairs.append(result)

    return report


def _parse_pairs(pair_strs: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for s in pair_strs:
        parts = s.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format '{s}', expected SOURCE:TARGET")
        pairs.append((parts[0], parts[1]))
    return pairs


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Cross-region transfer study")
    parser.add_argument("--pairs", nargs="+", help="Region pairs as SOURCE:TARGET")
    parser.add_argument("--all-pairs", action="store_true", help="Evaluate all region permutations")
    parser.add_argument("--models", nargs="+", default=["baseline_gbm"], help="Model names to evaluate")
    parser.add_argument("--targets", nargs="+", default=TARGETS, help="Target columns")
    parser.add_argument("--out", default="reports/transfer_study.json", help="Output path")
    args = parser.parse_args()

    if args.all_pairs:
        pairs = list(itertools.permutations(ALL_REGIONS, 2))
    elif args.pairs:
        pairs = _parse_pairs(args.pairs)
    else:
        parser.error("Specify --pairs or --all-pairs")
        return 1

    report = run_transfer_study(pairs, args.models, args.targets)

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = report.to_dataframe()
    if not df.empty:
        result = df.to_dict(orient="records")
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %d transfer results to %s", len(result), out_path)

        # Print summary
        print("\n=== Transfer Study Summary ===\n")
        summary = df.groupby(["source_region", "target_region"]).agg(
            mean_degradation_mae=("degradation_mae_pct", "mean"),
            mean_degradation_rmse=("degradation_rmse_pct", "mean"),
        ).round(1)
        print(summary.to_string())
    else:
        logger.warning("No transfer results produced. Check that models and data exist.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
