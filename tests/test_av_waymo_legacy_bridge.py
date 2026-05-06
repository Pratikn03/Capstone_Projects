from __future__ import annotations

import numpy as np
import pandas as pd

from scripts import build_av_waymo_legacy_bridge as bridge


def test_bridge_writes_rss_columns_when_lead_gap_cache_exists(tmp_path) -> None:
    full_corpus = tmp_path / "processed_full_corpus"
    processed = tmp_path / "processed"
    splits_dir = full_corpus / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    scenario_id = "scenario_1"
    steps = np.arange(30, dtype=int)
    replay = pd.DataFrame(
        {
            "scenario_id": [scenario_id] * len(steps),
            "step_index": steps,
            "ego_track_id": [7] * len(steps),
            "ego_x_m": steps.astype(float),
            "ego_y_m": np.zeros(len(steps), dtype=float),
            "ego_speed_mps": np.full(len(steps), 12.0, dtype=float),
            "timestamp_us": 1_700_000_000_000_000 + steps * 100_000,
        }
    )
    replay.to_parquet(full_corpus / "replay_windows.parquet", index=False)

    lead_gap = pd.DataFrame(
        {
            "scenario_id": [scenario_id] * len(steps),
            "step_index": steps,
            "lead_present": [True] * len(steps),
            "lead_track_id": [99] * len(steps),
            "lead_rel_x_m": np.full(len(steps), 18.0, dtype=float),
            "lead_rel_y_m": np.zeros(len(steps), dtype=float),
            "lead_speed_mps": np.full(len(steps), 10.0, dtype=float),
        }
    )
    lead_gap.to_parquet(full_corpus / "per_step_lead_gap.parquet", index=False)

    pd.DataFrame({"scenario_id": [scenario_id]}).to_parquet(splits_dir / "train.parquet", index=False)
    for split_name in ("calibration", "val", "test"):
        pd.DataFrame({"scenario_id": pd.Series(dtype=str)}).to_parquet(
            splits_dir / f"{split_name}.parquet",
            index=False,
        )

    replay_loaded = bridge._load_replay_windows(full_corpus)
    trajectories = bridge._translate(replay_loaded)
    trajectories = bridge._augment_with_rss(trajectories, full_corpus / "per_step_lead_gap.parquet")
    features = bridge._add_lag_features(bridge._filter_to_scenarios(trajectories, {scenario_id}))
    bridge._write_outputs(
        features,
        trajectories,
        {"train": {scenario_id}, "calibration": set(), "val": set(), "test": set()},
        processed,
    )

    csv = pd.read_csv(processed / "av_trajectories_orius.csv")
    features_out = pd.read_parquet(processed / "features.parquet")

    assert "rss_safe_gap_m" in csv.columns
    assert "rss_violation_true" in csv.columns
    assert "rss_safe_gap_m" in features_out.columns
    assert csv["rss_safe_gap_m"].notna().all()
    assert features_out["rss_safe_gap_m"].notna().all()
