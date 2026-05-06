from __future__ import annotations

from pathlib import Path

import orius.av_runtime as av_runtime
import orius.av_runtime.replay as replay_module


def test_av_runtime_facade_preserves_runtime_dry_run_contract(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    def fake_run_runtime_dry_run(**kwargs):
        calls.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(replay_module, "_run_runtime_dry_run", fake_run_runtime_dry_run)

    result = av_runtime.run_runtime_dry_run(
        replay_windows_path=tmp_path / "replay.parquet",
        step_features_path=tmp_path / "steps.parquet",
        models_dir=tmp_path / "models",
        out_dir=tmp_path / "out",
        artifact_prefix="nuplan_av",
        max_scenarios=4,
    )

    assert result == {"status": "ok"}
    assert calls["replay_windows_path"] == tmp_path / "replay.parquet"
    assert calls["step_features_path"] == tmp_path / "steps.parquet"
    assert calls["artifact_prefix"] == "nuplan_av"
    assert calls["max_scenarios"] == 4


def test_av_runtime_exports_source_neutral_adapter_alias() -> None:
    assert av_runtime.AVRuntimeDomainAdapter.__name__ == "WaymoAVDomainAdapter"
    assert callable(av_runtime.load_runtime_bundles)
    assert isinstance(Path("x"), Path)
