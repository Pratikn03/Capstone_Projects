"""Frozen benefit-estimator contracts for label-free KGA decisions.

The deployment-time KGA path is label free only after a benefit estimator has
been fitted on development data and its residual radius has been calibrated on
a disjoint labelled split.  This module makes that dependency executable.  A
frozen estimator is bound to both an evidence schema and a protocol digest;
schema or protocol drift fails closed instead of silently reusing a radius.

The bundled :class:`FrozenLinearBenefitEstimator` is a small, auditable
reference artifact.  The paper's benchmark scripts use a gradient-boosting
regressor under the same conceptual contract; callers may implement
:class:`BenefitEstimator` for another frozen model.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from kga._validation import as_float_array

LINEAR_ARTIFACT_SCHEMA = "kga-frozen-linear-benefit/1"


def _is_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _as_finite_vector(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = as_float_array(value).ravel()
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


@runtime_checkable
class BenefitEstimator(Protocol):
    """Protocol implemented by a frozen deployment-time benefit estimator."""

    @property
    def feature_names(self) -> tuple[str, ...]:
        """Ordered names of the estimator's required evidence features."""

    @property
    def evidence_schema_version(self) -> str:
        """Evidence schema identity used when fitting the estimator."""

    @property
    def protocol_sha256(self) -> str:
        """Digest of the deployment protocol bound to the estimator."""

    @property
    def residuals(self) -> np.ndarray:
        """Held-out absolute residuals used for conformal calibration."""

    @property
    def artifact_sha256(self) -> str:
        """Digest of the fitted estimator and calibration payload."""

    def predict(
        self,
        features: Mapping[str, float],
        *,
        evidence_schema_version: str,
        protocol_sha256: str,
    ) -> float:
        """Predict ``Delta`` after validating the frozen deployment contract."""


@dataclass(frozen=True)
class FrozenLinearBenefitEstimator:
    """Immutable linear benefit model with disjoint calibration residuals.

    ``weights`` operate on standardized features
    ``(x - feature_center) / feature_scale``.  The object contains no fitting
    method, and prediction requires the caller to supply the current protocol
    and evidence-schema identities.  This prevents an estimator calibrated for
    one candidate, split, or feature order from being reused accidentally.
    """

    feature_names: tuple[str, ...]
    weights: np.ndarray
    intercept: float
    feature_center: np.ndarray
    feature_scale: np.ndarray
    residuals: np.ndarray
    evidence_schema_version: str
    protocol_sha256: str
    fit_unit: str
    calibration_unit: str

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.feature_names)
        if not names or any(not name for name in names):
            raise ValueError("feature_names must be non-empty strings")
        if len(names) != len(set(names)):
            raise ValueError("feature_names must be unique")
        weights = _as_finite_vector(self.weights, "weights")
        center = _as_finite_vector(self.feature_center, "feature_center")
        scale = _as_finite_vector(self.feature_scale, "feature_scale")
        residuals = _as_finite_vector(self.residuals, "residuals")
        if any(array.size != len(names) for array in (weights, center, scale)):
            raise ValueError("weights, feature_center, and feature_scale must match feature_names")
        if np.any(scale <= 0.0):
            raise ValueError("feature_scale must be strictly positive")
        if np.any(residuals < 0.0):
            raise ValueError("residuals must be non-negative absolute errors")
        if not np.isfinite(float(self.intercept)):
            raise ValueError("intercept must be finite")
        if not self.evidence_schema_version:
            raise ValueError("evidence_schema_version is required")
        if not _is_sha256(self.protocol_sha256):
            raise ValueError("protocol_sha256 must be a 64-character SHA-256 digest")
        if not self.fit_unit or not self.calibration_unit:
            raise ValueError("fit_unit and calibration_unit are required")
        if self.fit_unit == self.calibration_unit:
            raise ValueError("fit_unit and calibration_unit must identify disjoint roles")
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "feature_center", center)
        object.__setattr__(self, "feature_scale", scale)
        object.__setattr__(self, "residuals", residuals)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": LINEAR_ARTIFACT_SCHEMA,
            "feature_names": list(self.feature_names),
            "weights": self.weights.tolist(),
            "intercept": float(self.intercept),
            "feature_center": self.feature_center.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "residuals": self.residuals.tolist(),
            "evidence_schema_version": self.evidence_schema_version,
            "protocol_sha256": self.protocol_sha256,
            "fit_unit": self.fit_unit,
            "calibration_unit": self.calibration_unit,
        }

    @property
    def artifact_sha256(self) -> str:
        """SHA-256 of all model, schema, protocol, and residual fields."""

        return hashlib.sha256(_canonical_json_bytes(self._payload())).hexdigest()

    def predict(
        self,
        features: Mapping[str, float],
        *,
        evidence_schema_version: str,
        protocol_sha256: str,
    ) -> float:
        """Predict benefit after exact schema/protocol validation."""

        if protocol_sha256 != self.protocol_sha256:
            raise ValueError("protocol SHA-256 does not match the frozen benefit estimator")
        if evidence_schema_version != self.evidence_schema_version:
            raise ValueError("evidence schema does not match the frozen benefit estimator")
        supplied = set(features)
        expected = set(self.feature_names)
        missing = sorted(expected - supplied)
        unexpected = sorted(supplied - expected)
        if missing or unexpected:
            raise ValueError(f"evidence feature mismatch: missing={missing}, unexpected={unexpected}")
        x = as_float_array([features[name] for name in self.feature_names])
        if not np.all(np.isfinite(x)):
            raise ValueError("evidence features must be finite")
        standardized = (x - self.feature_center) / self.feature_scale
        return float(self.intercept + standardized @ self.weights)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe artifact including its self-check digest."""

        payload = self._payload()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FrozenLinearBenefitEstimator:
        """Load and verify a serialized estimator artifact."""

        if value.get("schema") != LINEAR_ARTIFACT_SCHEMA:
            raise ValueError(f"unsupported benefit artifact schema: {value.get('schema')!r}")
        estimator = cls(
            feature_names=tuple(value["feature_names"]),
            weights=as_float_array(value["weights"]),
            intercept=float(value["intercept"]),
            feature_center=as_float_array(value["feature_center"]),
            feature_scale=as_float_array(value["feature_scale"]),
            residuals=as_float_array(value["residuals"]),
            evidence_schema_version=str(value["evidence_schema_version"]),
            protocol_sha256=str(value["protocol_sha256"]),
            fit_unit=str(value["fit_unit"]),
            calibration_unit=str(value["calibration_unit"]),
        )
        recorded = value.get("artifact_sha256")
        if recorded is not None and recorded != estimator.artifact_sha256:
            raise ValueError("benefit estimator artifact SHA-256 mismatch")
        return estimator

    @classmethod
    def load_json(cls, path: str | Path) -> FrozenLinearBenefitEstimator:
        """Read a trusted JSON artifact and verify its digest."""

        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("benefit estimator artifact root must be a JSON object")
        return cls.from_dict(value)

    def write_new_json(self, path: str | Path) -> None:
        """Create, but never overwrite, a frozen estimator artifact."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


def fit_frozen_linear_benefit_estimator(
    x_fit: Sequence[Sequence[float]] | np.ndarray,
    y_fit: Sequence[float] | np.ndarray,
    x_calibration: Sequence[Sequence[float]] | np.ndarray,
    y_calibration: Sequence[float] | np.ndarray,
    *,
    feature_names: Sequence[str],
    evidence_schema_version: str,
    protocol_sha256: str,
    fit_unit: str = "estimator_fit",
    calibration_unit: str = "residual_calibration",
    ridge: float = 1e-6,
) -> FrozenLinearBenefitEstimator:
    """Fit a deterministic reference model and calibrate on a disjoint split.

    The function requires separate arrays for model fitting and residual
    calibration.  It does not create cross-fitted or in-pool residuals.
    Callers remain responsible for ensuring the two arrays correspond to
    genuinely disjoint experimental units.
    """

    names = tuple(str(name) for name in feature_names)
    fit_x = as_float_array(x_fit)
    cal_x = as_float_array(x_calibration)
    fit_y = _as_finite_vector(y_fit, "y_fit")
    cal_y = _as_finite_vector(y_calibration, "y_calibration")
    if fit_x.ndim != 2 or cal_x.ndim != 2:
        raise ValueError("x_fit and x_calibration must be 2-D")
    if fit_x.shape[0] != fit_y.size or cal_x.shape[0] != cal_y.size:
        raise ValueError("feature and target row counts must match within each split")
    if fit_x.shape[1] != len(names) or cal_x.shape[1] != len(names):
        raise ValueError("feature matrix widths must match feature_names")
    if fit_x.shape[0] < 2 or cal_x.shape[0] < 1:
        raise ValueError("need at least two fit rows and one calibration row")
    if not np.all(np.isfinite(fit_x)) or not np.all(np.isfinite(cal_x)):
        raise ValueError("feature matrices must contain only finite values")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")

    center = fit_x.mean(axis=0)
    scale = fit_x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x_standardized = (fit_x - center) / scale
    design = np.column_stack([np.ones(fit_x.shape[0]), x_standardized])
    penalty = np.eye(design.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ fit_y
    intercept = float(coefficients[0])
    weights = np.asarray(coefficients[1:], dtype=float)
    cal_prediction = intercept + ((cal_x - center) / scale) @ weights
    residuals = np.abs(cal_prediction - cal_y)
    return FrozenLinearBenefitEstimator(
        feature_names=names,
        weights=weights,
        intercept=intercept,
        feature_center=center,
        feature_scale=scale,
        residuals=residuals,
        evidence_schema_version=evidence_schema_version,
        protocol_sha256=protocol_sha256,
        fit_unit=fit_unit,
        calibration_unit=calibration_unit,
    )
