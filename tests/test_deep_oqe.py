from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from orius.dc3s.deep_oqe import FEATURE_NAMES, DeepOQEConfig, DeepOQEModel, extract_feature_vector, save_model
from orius.dc3s.quality import compute_reliability

_CADENCE = 3600.0
_BASE_CFG = {"backend": "heuristic", "min_w": 0.05}


def _event(ts: str | None = "2026-01-01T01:00:00+00:00", **kwargs: object) -> dict[str, object]:
    event: dict[str, object] = {
        "load_mw": 50000.0,
        "renewables_mw": 12000.0,
    }
    if ts is not None:
        event["ts_utc"] = ts
    event.update(kwargs)
    return event


def _last(ts: str | None = "2026-01-01T00:00:00+00:00", **kwargs: object) -> dict[str, object]:
    return _event(ts=ts, **kwargs)


def test_extract_feature_vector_matches_runtime_contract() -> None:
    feature = extract_feature_vector(
        event=_event(dropout=True),
        last_event=_last(),
        missing_fraction=0.25,
        delay_seconds=120.0,
        out_of_order=False,
        spike_ratio=0.5,
        ooo_fraction_in_order=0.75,
        stale_tracker={"unchanged_counts": {"load_mw": 2, "renewables_mw": 1}},
        fault_flags={
            "dropout": True,
            "stale_sensor": False,
            "delay_jitter": True,
            "out_of_order": False,
            "spikes": False,
        },
    )
    assert feature.shape == (len(FEATURE_NAMES),)
    assert feature.dtype == np.float32
    assert math.isfinite(float(feature.sum()))


def test_timestamp_missing_penalizes_reliability() -> None:
    w, flags = compute_reliability(_event(ts=None), _last(ts=None), _CADENCE, _BASE_CFG)
    assert flags["timestamp_missing"] is True
    assert flags["delay_seconds"] == pytest.approx(_CADENCE)
    assert flags["out_of_order"] is True
    assert w < 1.0


def test_nan_blackout_counts_as_missing_signal() -> None:
    w, flags = compute_reliability(
        _event(load_mw=float("nan"), renewables_mw=12000.0),
        _last(),
        _CADENCE,
        _BASE_CFG,
    )
    assert flags["missing_fraction"] == pytest.approx(0.5, abs=1e-6)
    assert flags["fault_flags"]["dropout"] is True
    assert w < 1.0


def test_deep_backend_roundtrip_uses_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model_path = tmp_path / "battery_deepoqe_test.pt"
    cfg = DeepOQEConfig(seq_len=4, hidden_size=16, dropout=0.0)
    model = DeepOQEModel(cfg)
    save_model(model_path, model=model, cfg=cfg, metadata={"source": "unit_test"})

    adaptive_state: dict[str, object] = {}
    w, flags = compute_reliability(
        _event(dropout=True, delay_jitter=True),
        _last(),
        _CADENCE,
        {
            "backend": "deep",
            "min_w": 0.05,
            "deep": {"model_path": str(model_path), "seq_len": 4, "strict": True},
        },
        adaptive_state=adaptive_state,
    )
    assert 0.05 <= w <= 1.0
    assert flags["backend_used"] == "deep"
    assert flags["deep_metadata"]["source"] == "unit_test"
    assert flags["heuristic_w_t"] >= 0.05
    assert len(adaptive_state["deep_oqe_history"]) == 1


def test_deep_backend_falls_back_to_heuristic_when_checkpoint_missing(tmp_path: Path) -> None:
    w, flags = compute_reliability(
        _event(dropout=True),
        _last(),
        _CADENCE,
        {
            "backend": "deep",
            "min_w": 0.05,
            "deep": {"model_path": str(tmp_path / "missing.pt"), "seq_len": 4, "strict": False},
        },
        adaptive_state={},
    )
    assert 0.05 <= w <= 1.0
    assert flags["backend_used"] == "heuristic_fallback"
    assert "deep_failure" in flags
