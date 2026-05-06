"""Run a single in-process DC3S demo step and verify certificate audit retrieval."""

from __future__ import annotations

import json
import math
import sys
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from services.api.main import app
from services.api.routers import dc3s as dc3s_router


def _predict_target(
    *, target: str, horizon: int, features_df: pd.DataFrame, forecast_cfg: dict, required: bool
):
    idx = np.arange(horizon, dtype=float)
    if target == "load_mw":
        y = 52.0 + 4.0 * np.sin((2.0 * math.pi * idx / 24.0) - 0.4)
    elif target == "wind_mw":
        y = 8.0 + 1.8 * np.sin(2.0 * math.pi * (idx + 4.0) / 24.0)
    else:
        y = np.maximum(0.0, 4.0 * np.sin(math.pi * ((idx % 24.0) - 6.0) / 12.0))
    return np.asarray(y, dtype=float), Path(f"demo_{target}.bin")


def main() -> None:
    client = TestClient(app)
    features_df = pd.DataFrame({"price_eur_mwh": [62.0], "carbon_kg_per_mwh": [410.0]})
    with ExitStack() as stack:
        stack.enter_context(patch.object(dc3s_router, "_load_features_df", return_value=features_df))
        stack.enter_context(patch.object(dc3s_router, "_predict_target", side_effect=_predict_target))
        stack.enter_context(
            patch.object(dc3s_router, "_resolve_conformal_q", return_value=np.full(24, 4.0, dtype=float))
        )

        payload = {
            "device_id": "demo-device-001",
            "zone_id": "DE",
            "current_soc_mwh": 1.0,
            "telemetry_event": {
                "ts_utc": datetime.now(UTC).isoformat(),
                "load_mw": 52.0,
                "renewables_mw": 12.0,
            },
            "last_actual_load_mw": 52.0,
            "last_pred_load_mw": 50.0,
            "controller": "deterministic",
            "horizon": 24,
            "include_certificate": True,
        }
        step_resp = client.post("/dc3s/step", json=payload)
        step_resp.raise_for_status()
        step_json = step_resp.json()
        command_id = step_json["command_id"]
        audit_resp = client.get(f"/dc3s/audit/{command_id}")
        audit_resp.raise_for_status()
        audit_json = audit_resp.json()

    summary = {
        "command_id": command_id,
        "certificate_id": step_json.get("certificate_id"),
        "audit_retrieved": True,
        "certificate_hash": audit_json.get("certificate_hash"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
