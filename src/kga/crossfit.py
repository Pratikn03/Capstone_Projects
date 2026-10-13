"""Cell-outcome-disjoint cross-fitting for controlled-grid KGA replays.

This module owns the canonical retrospective controlled-grid estimator.  Each
score fold is kept out of both the benefit-estimator fit and the residual
calibration used to route that fold.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kga.certificate import min_calibration_size, split_conformal_rank_radius
from kga.policy import decide_batch


@dataclass(frozen=True)
class ControlledGridCrossfitResult:
    """Predictions, radii, actions, and serializable cross-fit provenance."""

    prediction: np.ndarray
    radius: np.ndarray
    action: np.ndarray
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the complete result."""

        prediction = [float(value) if math.isfinite(float(value)) else None for value in self.prediction]
        radius = [float(value) if math.isfinite(float(value)) else None for value in self.radius]

        return {
            "prediction": prediction,
            "prediction_status": ["finite" if value is not None else "unavailable" for value in prediction],
            "radius": radius,
            "radius_status": ["finite" if value is not None else "positive_infinity" for value in radius],
            "action": [str(value) for value in self.action],
            "protocol": self.protocol,
        }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _implementation_sha256() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = {
        "crossfit": Path(__file__),
        "certificate": package / "certificate.py",
        "policy": package / "policy.py",
        "numeric_validation": package / "_validation.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def controlled_grid_sample_id(*, track: str, candidate: str, seed: int, condition: str) -> str:
    """Return the stable publication identity used for one grid cell."""

    identity = {"track": str(track), "candidate": str(candidate), "seed": int(seed), "condition": str(condition)}
    payload = (json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    return f"grid-cell-sha256:{hashlib.sha256(payload).hexdigest()}"


def controlled_grid_input_sha256(
    Z: Sequence[Sequence[float]] | np.ndarray,
    B: Sequence[float] | np.ndarray,
    sample_ids: Sequence[str],
) -> dict[str, str]:
    """Hash normalized cross-fit inputs using the canonical representation."""

    return {
        "Z": _sha256_json(np.asarray(Z, dtype=float).tolist()),
        "B": _sha256_json(np.asarray(B, dtype=float).tolist()),
        "sample_ids": _sha256_json([str(value) for value in sample_ids]),
    }


def _stable_order(sample_ids: Sequence[str], *, salt: str, random_state: int) -> list[int]:
    return sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(f"{random_state}|{salt}|{sample_ids[index]}".encode()).hexdigest(),
    )


def _fail_closed(n: int, protocol: dict[str, Any], reason: str) -> ControlledGridCrossfitResult:
    prediction = np.full(n, np.nan, dtype=float)
    radius = np.full(n, np.inf, dtype=float)
    action = np.full(n, "ABSTAIN", dtype=object)
    return ControlledGridCrossfitResult(
        prediction=prediction,
        radius=radius,
        action=action,
        protocol={**protocol, "status": "fail_closed", "reason": reason},
    )


def controlled_grid_crossfit(
    Z: Sequence[Sequence[float]] | np.ndarray,
    B: Sequence[float] | np.ndarray,
    *,
    sample_ids: Sequence[str],
    alpha: float = 0.1,
    n_folds: int = 5,
    n_estimators: int = 250,
    max_depth: int = 2,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    random_state: int = 0,
) -> ControlledGridCrossfitResult:
    """Fit, calibrate, and score disjoint controlled-grid folds.

    Fold membership is a deterministic function of ``sample_ids`` and the
    pinned random state.  For each score fold, its complement is divided into
    disjoint estimator-fit and residual-calibration subsets before one GBRT is
    fitted.  The resulting predictor and exact-rank radius are applied only to
    that score fold.
    """

    try:
        raw_benefit = np.asarray(B, dtype=object)
        n = int(raw_benefit.size) if raw_benefit.ndim == 1 else 0
    except (TypeError, ValueError, OverflowError):
        n = 0

    def finite_float(value: Any) -> float | None:
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return normalized if math.isfinite(normalized) else None

    def integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            return None
        return int(value)

    normalized_alpha = finite_float(alpha)
    normalized_n_folds = integer(n_folds)
    normalized_n_estimators = integer(n_estimators)
    normalized_max_depth = integer(max_depth)
    normalized_learning_rate = finite_float(learning_rate)
    normalized_subsample = finite_float(subsample)
    normalized_random_state = integer(random_state)
    protocol: dict[str, Any] = {
        "schema": "kga-controlled-grid-crossfit-v1",
        "rule": "deterministic cell-outcome-disjoint fit/calibrate/score cross-fit",
        "score_fold_assignment": "SHA-256 order of stable sample_ids, then deterministic round-robin",
        "complement_split": "SHA-256 order of stable sample_ids; calibration first, estimator-fit remainder",
        "requested_n_folds": normalized_n_folds,
        "calibration_fraction_target": 0.30,
        "alpha": normalized_alpha,
        "gbrt": {
            "n_estimators": normalized_n_estimators,
            "max_depth": normalized_max_depth,
            "learning_rate": normalized_learning_rate,
            "subsample": normalized_subsample,
            "random_state": normalized_random_state,
        },
        "implementation_sha256": _implementation_sha256(),
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__},
    }

    try:
        benefit = np.asarray(B, dtype=float)
        n = int(benefit.size) if benefit.ndim == 1 else 0
        ids = [str(value) for value in sample_ids]
        features = np.asarray(Z, dtype=float)
        valid_alpha = normalized_alpha is not None and 0.0 < normalized_alpha < 1.0
        valid_hyperparameters = (
            normalized_n_folds is not None
            and normalized_n_folds >= 2
            and normalized_n_estimators is not None
            and normalized_n_estimators >= 1
            and normalized_max_depth is not None
            and normalized_max_depth >= 1
            and normalized_learning_rate is not None
            and normalized_learning_rate > 0.0
            and normalized_subsample is not None
            and 0.0 < normalized_subsample <= 1.0
            and normalized_random_state is not None
        )
    except (TypeError, ValueError, OverflowError):
        return _fail_closed(n, protocol, "inputs could not be coerced to the canonical numeric schema")

    if len(ids) != len(set(ids)):
        raise ValueError("sample_ids must be unique")

    if len(ids) != n:
        return _fail_closed(n, protocol, "sample_ids length does not match B")
    if (
        features.ndim != 2
        or benefit.ndim != 1
        or features.shape[0] != n
        or features.shape[1] == 0
        or n == 0
        or not np.isfinite(features).all()
        or not np.isfinite(benefit).all()
        or not valid_alpha
        or not valid_hyperparameters
    ):
        return _fail_closed(n, protocol, "malformed controlled-grid inputs")

    assert normalized_alpha is not None
    assert normalized_n_folds is not None
    assert normalized_n_estimators is not None
    assert normalized_max_depth is not None
    assert normalized_learning_rate is not None
    assert normalized_subsample is not None
    assert normalized_random_state is not None
    minimum_calibration = min_calibration_size(normalized_alpha)
    minimum_fit = 2
    maximum_score_fold = n - minimum_calibration - minimum_fit
    if maximum_score_fold < 1:
        return _fail_closed(n, protocol, "three-way split is infeasible")

    folds_required = int(math.ceil(n / maximum_score_fold))
    fold_count = min(n, max(2, normalized_n_folds, folds_required))
    score_order = _stable_order(ids, salt="score-fold", random_state=normalized_random_state)
    score_folds = [score_order[offset::fold_count] for offset in range(fold_count)]
    prediction = np.full(n, np.nan, dtype=float)
    radius = np.full(n, np.inf, dtype=float)
    fold_provenance: list[dict[str, Any]] = []
    all_indices = set(range(n))

    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingRegressor

        protocol.update(
            {
                "input_sha256": controlled_grid_input_sha256(features, benefit, ids),
                "software_versions": {
                    **protocol["software_versions"],
                    "scikit_learn": sklearn.__version__,
                },
                "label_flow_limitation": (
                    "retrospective, opened, dependent, and constructed controlled-grid replay; "
                    "the split removes scored-cell outcome flow but does not establish exchangeability"
                ),
            }
        )

        for fold_index, score_indices in enumerate(score_folds):
            if not score_indices:
                continue
            remaining = sorted(all_indices - set(score_indices))
            local_order = _stable_order(
                [ids[index] for index in remaining],
                salt=f"calibration-fold:{fold_index}",
                random_state=normalized_random_state,
            )
            ordered_remaining = [remaining[index] for index in local_order]
            desired_calibration = max(minimum_calibration, int(math.ceil(0.30 * len(ordered_remaining))))
            calibration_n = min(desired_calibration, len(ordered_remaining) - minimum_fit)
            if calibration_n < minimum_calibration:
                return _fail_closed(n, protocol, "a score fold has no feasible fit/calibration complement")
            calibration_indices = ordered_remaining[:calibration_n]
            fit_indices = ordered_remaining[calibration_n:]
            model = GradientBoostingRegressor(
                n_estimators=normalized_n_estimators,
                max_depth=normalized_max_depth,
                learning_rate=normalized_learning_rate,
                subsample=normalized_subsample,
                random_state=normalized_random_state,
            )
            model.fit(features[fit_indices], benefit[fit_indices])
            calibration_prediction = model.predict(features[calibration_indices])
            fold_radius = split_conformal_rank_radius(
                np.abs(calibration_prediction - benefit[calibration_indices]),
                normalized_alpha,
            )
            prediction[score_indices] = model.predict(features[score_indices])
            radius[score_indices] = fold_radius
            fold_provenance.append(
                {
                    "fold": fold_index,
                    "score_count": len(score_indices),
                    "estimator_fit_count": len(fit_indices),
                    "residual_calibration_count": len(calibration_indices),
                    "score_fraction": len(score_indices) / n,
                    "estimator_fit_fraction": len(fit_indices) / n,
                    "residual_calibration_fraction": len(calibration_indices) / n,
                    "score_ids_sha256": _sha256_json(sorted(ids[index] for index in score_indices)),
                    "estimator_fit_ids_sha256": _sha256_json(sorted(ids[index] for index in fit_indices)),
                    "residual_calibration_ids_sha256": _sha256_json(
                        sorted(ids[index] for index in calibration_indices)
                    ),
                }
            )
    except Exception as exc:  # noqa: BLE001 - canonical integration fails closed
        return _fail_closed(n, protocol, f"cross-fit implementation failure: {type(exc).__name__}")

    if np.isnan(prediction).any() or not np.isfinite(radius).all():
        return _fail_closed(n, protocol, "cross-fit did not produce a finite result for every score cell")

    action = decide_batch(prediction, radius, alpha=normalized_alpha)
    protocol.update(
        {
            "status": "ok",
            "effective_n_folds": fold_count,
            "minimum_estimator_fit_count": minimum_fit,
            "minimum_residual_calibration_count": minimum_calibration,
            "folds": fold_provenance,
        }
    )
    return ControlledGridCrossfitResult(prediction=prediction, radius=radius, action=action, protocol=protocol)


__all__ = [
    "ControlledGridCrossfitResult",
    "controlled_grid_crossfit",
    "controlled_grid_input_sha256",
    "controlled_grid_sample_id",
]
