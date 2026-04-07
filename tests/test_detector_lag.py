"""Regression tests for the runtime detector-lag contract."""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from orius.dc3s.quality import compute_reliability


_CADENCE = 3600.0
_CFG = {"lambda_delay": 0.002, "spike_beta": 0.25, "ooo_gamma": 0.35, "min_w": 0.05}
_FTIT = {"law": "linear", "stale_k": 3, "stale_tol": 1.0e-12}


def _dataset_backed_payload() -> dict[str, float]:
    path = Path("data/processed/features.parquet")
    if not path.exists():
        pytest.skip(f"missing dataset-backed feature surface: {path}")
    row = pd.read_parquet(path).head(1).to_dict(orient="records")[0]
    payload: dict[str, float] = {}
    for key, value in row.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            payload[key] = numeric
        if len(payload) >= 2:
            break
    if len(payload) < 2:
        pytest.skip("dataset-backed payload did not expose at least two numeric telemetry channels")
    return payload


def test_detector_lag_warning_fires_for_dataset_backed_stale_sequence() -> None:
    base = _dataset_backed_payload()
    adaptive_state: dict[str, object] = {}
    previous_event: dict[str, object] | None = None
    last_flags: dict[str, object] | None = None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for hour in range(4):
            event = {"ts_utc": f"2026-01-01T{hour:02d}:00:00+00:00", **base}
            _, flags = compute_reliability(
                event,
                previous_event,
                _CADENCE,
                _CFG,
                adaptive_state=adaptive_state,
                ftit_cfg=_FTIT,
            )
            adaptive_state = {"ftit": {"stale_tracker": flags["stale_tracker"]}}
            previous_event = event
            last_flags = flags

    assert last_flags is not None
    assert last_flags["detector_lag_steps"]["stale_sensor"] == 3
    assert last_flags["stale_max_unchanged_steps"] >= 3
    assert last_flags["detector_lag_warning"] is True
    assert any("Assumption A6 may be violated" in str(w.message) for w in caught)

