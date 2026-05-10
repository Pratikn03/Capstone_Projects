import json
from pathlib import Path

import scripts.summarize_split_training as summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_av_run(tmp_path: Path, *, promoted: bool = True, recovery_complete: bool = True) -> Path:
    release_id = "R"
    split_dir = tmp_path / "reports" / "split_training" / release_id
    log_dir = split_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "train_av.log").write_text("started but primary shell stopped before final marker\n", encoding="utf-8")
    if recovery_complete:
        (log_dir / "train_av_release_gbm_completion.log").write_text(
            "===== AV release GBM all complete at 2026-05-10T07:34:54Z =====\n",
            encoding="utf-8",
        )
    registry = tmp_path / "artifacts" / "runs" / "av" / release_id / "registry"
    _write_json(
        registry / "run_manifest.json",
        {
            "accepted": promoted,
            "promoted_at": "2026-05-10T08:05:11+00:00" if promoted else None,
            "expected_targets": ["target_ego_speed_mps__1s"],
        },
    )
    _write_json(registry / "tuning_summary_av.json", {"accepted": promoted, "targets": []})
    if promoted:
        _write_json(registry / "promotion_record.json", {"accepted": True})
    _write_json(tmp_path / "reports" / "runs" / "av" / release_id / "week2_metrics.json", {"targets": {}})
    model_dir = tmp_path / "artifacts" / "runs" / "av" / release_id / "models"
    uncertainty_dir = tmp_path / "artifacts" / "runs" / "av" / release_id / "uncertainty"
    model_dir.mkdir(parents=True, exist_ok=True)
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "gbm_target_ego_speed_mps__1s.joblib").write_text("model", encoding="utf-8")
    (uncertainty_dir / "gbm_target_ego_speed_mps__1s_conformal.json").write_text("{}", encoding="utf-8")
    return split_dir


def test_av_recovery_completion_counts_as_complete(tmp_path: Path, monkeypatch) -> None:
    split_dir = _seed_av_run(tmp_path)
    monkeypatch.setattr(summary, "REPO_ROOT", tmp_path)

    row = summary._domain_summary("R", "AV", split_dir)

    assert row["log_completed"] is False
    assert row["artifact_recovered"] is True
    assert row["status"] == "complete"
    assert row["missing"] == []


def test_av_recovery_requires_promotion_record(tmp_path: Path, monkeypatch) -> None:
    split_dir = _seed_av_run(tmp_path, promoted=False)
    monkeypatch.setattr(summary, "REPO_ROOT", tmp_path)

    row = summary._domain_summary("R", "AV", split_dir)

    assert row["artifact_recovered"] is False
    assert row["status"] == "running_or_pending"
    assert "canonical promotion record" in row["missing"]
