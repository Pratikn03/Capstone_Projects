from __future__ import annotations

from pathlib import Path

import pandas as pd

from orius.data_pipeline.build_features_av import _build_legacy_features


def test_build_legacy_features_preserves_multiple_vehicle_sequences(tmp_path: Path) -> None:
    csv_path = tmp_path / "av.csv"
    out_dir = tmp_path / "processed"

    rows: list[dict[str, object]] = []
    for vehicle_idx, vehicle_id in enumerate(("veh_a", "veh_b")):
        for step in range(30):
            base = vehicle_idx * 100.0
            rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "step": step,
                    "position_m": base + float(step),
                    "speed_mps": 10.0 + vehicle_idx,
                    "speed_limit_mps": 30.0,
                    "lead_position_m": base + float(step) + 20.0,
                    "rss_safe_gap_m": 8.0,
                    "ts_utc": f"2026-01-01T00:00:{step:02d}.000000Z",
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    features_path = _build_legacy_features(csv_path, out_dir)
    features = pd.read_parquet(features_path)

    assert set(features["vehicle_id"].unique()) == {"veh_a", "veh_b"}
    assert "rss_safe_gap_m" in features.columns
    assert len(features) == 12
