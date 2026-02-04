"""API health and readiness helpers."""
from __future__ import annotations

import os
from pathlib import Path


def readiness_check() -> dict:
    features_path = Path(os.getenv("GRIDPULSE_FEATURES_PATH", "data/processed/features.parquet"))
    models_dir = Path(os.getenv("GRIDPULSE_MODELS_DIR", "artifacts/models"))

    has_features = features_path.exists()
    model_files = []
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pkl"))
    has_models = len(model_files) > 0

    checks = {
        "features_path": str(features_path),
        "features_ready": has_features,
        "models_dir": str(models_dir),
        "models_ready": has_models,
        "model_count": len(model_files),
    }
    status = "ok" if has_features and has_models else "degraded"
    return {"status": status, "checks": checks}

