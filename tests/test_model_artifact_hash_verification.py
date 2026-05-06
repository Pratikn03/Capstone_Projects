"""Regression tests for fail-closed model artifact hash checks."""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pytest

from orius.forecasting.predict import load_model_bundle


def _write_pickle(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_model_bundle_verifies_sidecar_sha256(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    digest = _write_pickle(model_path, {"model": "stub", "feature_cols": [], "target": "load_mw"})
    model_path.with_name(f"{model_path.name}.sha256").write_text(
        f"{digest}  {model_path.name}\n", encoding="utf-8"
    )

    bundle = load_model_bundle(model_path)

    assert bundle["target"] == "load_mw"
    assert bundle["_path"] == str(model_path)


def test_load_model_bundle_rejects_hash_mismatch(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    _write_pickle(model_path, {"model": "stub", "feature_cols": [], "target": "load_mw"})
    model_path.with_name(f"{model_path.name}.sha256").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_model_bundle(model_path)


def test_load_model_bundle_requires_hash_in_production(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.pkl"
    _write_pickle(model_path, {"model": "stub", "feature_cols": [], "target": "load_mw"})
    monkeypatch.setenv("ORIUS_ENV", "production")

    with pytest.raises(RuntimeError, match="without sha256 manifest"):
        load_model_bundle(model_path)
