from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_battery_av_pipeline.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_battery_av_pipeline", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline_script = _load_script()


def test_main_writes_combined_summary_and_manifest(tmp_path: Path, monkeypatch) -> None:
    battery_dir = tmp_path / "battery"
    av_dir = tmp_path / "av"
    overall_dir = tmp_path / "overall"
    battery_dir.mkdir(parents=True, exist_ok=True)
    av_dir.mkdir(parents=True, exist_ok=True)

    battery_register = battery_dir / "battery_deep_learning_novelty_register.json"
    battery_register.write_text(json.dumps({"ok": True}), encoding="utf-8")
    av_summary = av_dir / "summary.json"
    av_summary.write_text(json.dumps({"ok": True}), encoding="utf-8")

    monkeypatch.setattr(
        pipeline_script,
        "_parse_args",
        lambda: argparse.Namespace(
            overall_dir=overall_dir,
            skip_battery=False,
            skip_av=False,
            battery_out_dir=battery_dir,
            av_reports_dir=av_dir,
        ),
    )
    monkeypatch.setattr(
        pipeline_script,
        "run_battery_pipeline",
        lambda _args: {
            "register_json": str(battery_register),
            "out_dir": str(battery_dir),
        },
    )
    monkeypatch.setattr(
        pipeline_script,
        "run_av_pipeline",
        lambda _args: {
            "subset_mode": "subset",
            "report": {"summary": str(av_summary)},
        },
    )
    monkeypatch.setattr(
        pipeline_script.closure_script,
        "build_closure",
        lambda **_kwargs: {
            "battery": {"status": "complete"},
            "av": {"status": "complete"},
        },
    )

    assert pipeline_script.main() == 0

    summary_path = overall_dir / "battery_av_pipeline.json"
    manifest_path = overall_dir / "battery_av_manifest.json"
    csv_path = overall_dir / "domain_summary.csv"

    assert summary_path.exists()
    assert manifest_path.exists()
    assert csv_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(payload["domains"]) == {"battery", "av"}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert str(battery_register.resolve()) in manifest["artifacts"]
    assert str(av_summary.resolve()) in manifest["artifacts"]
    assert manifest["input_hashes"] == {"battery": {}, "av": {}}


def test_run_av_pipeline_uses_full_corpus_count(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    uncertainty_dir = tmp_path / "uncertainty"
    reports_dir = tmp_path / "reports"
    processed_dir.mkdir(parents=True, exist_ok=True)
    scenario_index = pd.DataFrame({"scenario_id": ["a", "b", "c"], "shard_id": [0, 0, 1]})
    scenario_index.to_parquet(processed_dir / "scenario_index.parquet", index=False)

    calls: dict[str, object] = {}

    def _fake_subset_manifest(**kwargs):
        calls["subset"] = kwargs
        return {"selected_count": 3}

    monkeypatch.setattr(pipeline_script, "build_subset_manifest", _fake_subset_manifest)
    monkeypatch.setattr(
        pipeline_script,
        "build_replay_surface",
        lambda **kwargs: {"row_count": 273},
    )
    monkeypatch.setattr(
        pipeline_script,
        "build_feature_tables",
        lambda **kwargs: {"anchor_row_count": 3},
    )
    monkeypatch.setattr(
        pipeline_script,
        "train_dry_run_models",
        lambda **kwargs: {"training_summary_csv": str(reports_dir / "training_summary.csv")},
    )
    monkeypatch.setattr(
        pipeline_script,
        "run_runtime_dry_run",
        lambda **kwargs: {"runtime_summary_csv": str(reports_dir / "runtime_summary.csv")},
    )
    monkeypatch.setattr(
        pipeline_script.av_report_script,
        "build_report",
        lambda **kwargs: {"summary": str(reports_dir / "summary.json")},
    )

    args = argparse.Namespace(
        av_raw_dir=raw_dir,
        av_processed_dir=processed_dir,
        av_models_dir=models_dir,
        av_uncertainty_dir=uncertainty_dir,
        av_reports_dir=reports_dir,
        av_subset_size=1000,
        av_full_corpus=True,
        av_max_validation_shards=None,
        av_max_validation_scenarios=None,
        av_max_runtime_scenarios=None,
        av_skip_actor_tracks=True,
        av_skip_validation=True,
        av_skip_training=False,
        av_skip_runtime=False,
        av_skip_report=False,
    )

    report = pipeline_script.run_av_pipeline(args)

    assert report["subset_mode"] == "full_corpus"
    assert report["subset_size"] == 3
    assert calls["subset"]["target_count"] == 3


def test_main_uses_runtime_summary_when_av_report_summary_is_missing(tmp_path: Path, monkeypatch) -> None:
    battery_dir = tmp_path / "battery"
    av_dir = tmp_path / "av"
    overall_dir = tmp_path / "overall"
    battery_dir.mkdir(parents=True, exist_ok=True)
    av_dir.mkdir(parents=True, exist_ok=True)

    battery_register = battery_dir / "battery_deep_learning_novelty_register.json"
    battery_register.write_text(json.dumps({"ok": True}), encoding="utf-8")
    runtime_summary = av_dir / "runtime_summary.csv"
    runtime_summary.write_text("controller,tsvr\norius,0.0\n", encoding="utf-8")

    monkeypatch.setattr(
        pipeline_script,
        "_parse_args",
        lambda: argparse.Namespace(
            overall_dir=overall_dir,
            skip_battery=False,
            skip_av=False,
            battery_out_dir=battery_dir,
            av_reports_dir=av_dir,
        ),
    )
    monkeypatch.setattr(
        pipeline_script,
        "run_battery_pipeline",
        lambda _args: {
            "register_json": str(battery_register),
            "out_dir": str(battery_dir),
        },
    )
    monkeypatch.setattr(
        pipeline_script,
        "run_av_pipeline",
        lambda _args: {
            "subset_mode": "full_corpus",
            "runtime": {"runtime_summary_csv": str(runtime_summary)},
        },
    )
    monkeypatch.setattr(
        pipeline_script.closure_script,
        "build_closure",
        lambda **_kwargs: {
            "battery": {"status": "complete"},
            "av": {"status": "incomplete"},
        },
    )

    assert pipeline_script.main() == 0

    summary_df = pd.read_csv(overall_dir / "domain_summary.csv")
    av_row = summary_df[summary_df["domain"] == "av"].iloc[0]
    assert av_row["key_report"] == str(av_dir / "summary.json")
