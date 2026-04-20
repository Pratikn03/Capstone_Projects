from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts import build_features_multi_domain


def test_multi_domain_builder_uses_promoted_healthcare_input_and_canonical_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    av_csv = tmp_path / "data" / "av" / "processed" / "av_trajectories_orius.csv"
    hc_csv = tmp_path / "data" / "healthcare" / "mimic3" / "processed" / "mimic3_healthcare_orius.csv"
    av_csv.parent.mkdir(parents=True, exist_ok=True)
    hc_csv.parent.mkdir(parents=True, exist_ok=True)
    av_csv.write_text("stub\n", encoding="utf-8")
    hc_csv.write_text("stub\n", encoding="utf-8")

    calls: dict[str, tuple[Path, Path]] = {}

    def fake_build_av(csv_path: Path, out_dir: Path) -> None:
        calls["av"] = (csv_path, out_dir)

    def fake_build_healthcare(csv_path: Path, out_dir: Path) -> None:
        calls["healthcare"] = (csv_path, out_dir)

    monkeypatch.setattr(build_features_multi_domain, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "orius.data_pipeline.build_features_av",
        types.SimpleNamespace(build_features=fake_build_av),
    )
    monkeypatch.setitem(
        sys.modules,
        "orius.data_pipeline.build_features_healthcare",
        types.SimpleNamespace(build_promoted_features=fake_build_healthcare),
    )

    assert build_features_multi_domain.main() == 0
    assert calls["av"] == (av_csv, av_csv.parent)
    assert calls["healthcare"] == (hc_csv, tmp_path / "data" / "healthcare" / "processed")
