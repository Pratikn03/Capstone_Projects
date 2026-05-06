from __future__ import annotations

from pathlib import Path

from orius.utils.manifest import create_run_manifest


def test_run_manifest_snapshots_effective_config_text(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    config_path.write_text("models:\n  dl_lstm:\n    params:\n      epochs: 300\n", encoding="utf-8")

    manifest = create_run_manifest(
        config_path=config_path,
        output_dir=tmp_path / "out",
        run_id="EFFECTIVE",
        config_snapshot_text="models:\n  dl_lstm:\n    params:\n      epochs: 2\n",
        data_manifest_path=None,
    )

    snapshot = Path(manifest["config"]["snapshot"])
    assert manifest["config"]["effective_snapshot"] is True
    assert "epochs: 2" in snapshot.read_text(encoding="utf-8")
    assert "epochs: 2" in manifest["config"]["content"]
    assert "epochs: 300" not in manifest["config"]["content"]
