from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_clean_artifact_release import build_release
from scripts.clean_artifact_release_common import (
    AV_EVIDENCE_DIRS,
    BATTERY_EVIDENCE_DIRS,
    CODE_SCRIPTS,
    CODE_TESTS,
    CONFIG_FILES,
    ENVIRONMENT_FILES,
    HEALTHCARE_EVIDENCE_FILES,
    MANIFEST_FILES,
    MANUSCRIPT_FILES,
    REQUIRED_RELEASE_PATHS,
    TABLE_DIRS,
    TABLE_FILES,
)
from scripts.validate_clean_artifact_release import validate_release


def _write(path: Path, content: str = "ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_release_repo(root: Path) -> Path:
    for rel in ENVIRONMENT_FILES:
        _write(root / rel, "name = 'fixture'\n")

    for rel in CODE_SCRIPTS:
        _write(root / rel, "# fixture script\n")
    for rel in CODE_TESTS:
        _write(root / rel, "# fixture test\n")
    _write(root / "src/orius/__init__.py", "__version__ = 'fixture'\n")

    for rel in CONFIG_FILES:
        _write(root / rel, "fixture: true\n")

    for rel in MANIFEST_FILES:
        if rel.endswith(".csv"):
            _write(root / rel, "domain,status\nBattery,pass\n")
        else:
            _write(root / rel, json.dumps({"status": "pass", "domain": "three-domain"}) + "\n")

    for rel in TABLE_DIRS:
        _write(
            root / rel / "tbl_real_data_validation.tex", "\\begin{tabular}{ll}Battery & pass\\end{tabular}\n"
        )
    for rel in TABLE_FILES:
        _write(root / rel, "domain,metric\nBattery,1\n")

    for rel in BATTERY_EVIDENCE_DIRS:
        if rel.endswith("/battery"):
            _write(root / rel / "runtime_summary.csv", "controller,tsvr\norius,0\n")
            _write(root / rel / "summary.json", json.dumps({"domain": "battery"}) + "\n")
        else:
            _write(root / rel / "hil_summary.json", json.dumps({"domain": "battery", "pass": True}) + "\n")
            _write(root / rel / "hil_step_log.csv", "step,status\n1,pass\n")

    av_runtime = root / AV_EVIDENCE_DIRS[0]
    _write(av_runtime / "runtime_traces.csv", "controller,violation\norius,False\n")
    _write(av_runtime / "dc3s_av_waymo_dryrun.duckdb", "fixture-duckdb\n")
    _write(av_runtime / "runtime_summary.csv", "controller,tsvr\norius,0.0001\n")
    _write(av_runtime / "figures/runtime_metrics.png", "png\n")
    _write(av_runtime / "tables/runtime_summary_table.csv", "controller,tsvr\norius,0.0001\n")

    _write(
        root / AV_EVIDENCE_DIRS[1] / "nuplan_av_ego_speed_mps_1s_bundle.pkl",
        "fixture-model\n",
    )
    _write(
        root / AV_EVIDENCE_DIRS[2] / "nuplan_av_ego_speed_mps_1s_conformal.json",
        json.dumps({"alpha": 0.1}) + "\n",
    )

    for rel in HEALTHCARE_EVIDENCE_FILES:
        if rel.endswith(".csv"):
            _write(root / rel, "domain,status\nHealthcare,pass\n")
        else:
            _write(root / rel, json.dumps({"domain": "healthcare", "pass": True}) + "\n")

    for rel in MANUSCRIPT_FILES:
        _write(root / rel, f"manuscript {rel}\n")

    return root


def test_build_clean_artifact_release_writes_expected_tree_and_manifests(tmp_path: Path) -> None:
    repo = _seed_release_repo(tmp_path / "repo")
    release = build_release(
        repo_root=repo,
        out_root=tmp_path / "out",
        release_id="orius-three-domain-artifact-test",
        include_manuscripts=True,
        copy_mode="copy",
        verify=True,
    )

    for rel in REQUIRED_RELEASE_PATHS:
        assert (release / rel).exists(), rel
    for rel in ("environment", "code", "configs", "manifests", "tables", "evidence", "manuscripts"):
        assert (release / rel).is_dir()

    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "full-derived"
    assert manifest["include_manuscripts"] is True
    assert set(manifest["promoted_domains"]) == {
        "Battery Energy Storage",
        "nuPlan Autonomous Vehicles",
        "Medical and Healthcare Monitoring",
    }
    assert len(manifest["files"]) > 20
    assert "README.md" in (release / "MANIFEST.sha256").read_text(encoding="utf-8")


def test_builder_excludes_raw_downloads_caches_appledouble_and_private_healthcare_rows(
    tmp_path: Path,
) -> None:
    repo = _seed_release_repo(tmp_path / "repo")
    _write(repo / AV_EVIDENCE_DIRS[0] / "nuplan-v1.1.zip", "raw zip should not copy\n")
    _write(repo / AV_EVIDENCE_DIRS[0] / "partial.crdownload", "partial should not copy\n")
    _write(repo / AV_EVIDENCE_DIRS[0] / "._runtime_summary.csv", "appledouble should not copy\n")
    _write(repo / "src/orius/__pycache__/x.pyc", "cache should not copy\n")
    _write(
        repo / "reports/predeployment_external_validation/healthcare_site_splits/train.parquet",
        "private rows\n",
    )

    release = build_release(
        repo_root=repo,
        out_root=tmp_path / "out",
        release_id="orius-three-domain-artifact-test",
        include_manuscripts=True,
        copy_mode="copy",
        verify=True,
    )

    copied_paths = [path.relative_to(release).as_posix() for path in release.rglob("*") if path.is_file()]
    assert not any(path.endswith(".zip") for path in copied_paths)
    assert not any(path.endswith(".crdownload") for path in copied_paths)
    assert not any("/._" in path or path.startswith("._") for path in copied_paths)
    assert not any("__pycache__" in path for path in copied_paths)
    assert not any(
        path.startswith("evidence/healthcare/") and path.endswith(".parquet") for path in copied_paths
    )


def test_builder_includes_symlinked_derived_runtime_files(tmp_path: Path) -> None:
    repo = _seed_release_repo(tmp_path / "repo")
    trace_target = repo / "derived_runtime_traces.csv"
    _write(trace_target, "controller,violation\norius,False\n")
    runtime_trace = repo / AV_EVIDENCE_DIRS[0] / "runtime_traces.csv"
    runtime_trace.unlink()
    runtime_trace.symlink_to(trace_target)

    release = build_release(
        repo_root=repo,
        out_root=tmp_path / "out",
        release_id="orius-three-domain-artifact-test",
        include_manuscripts=True,
        copy_mode="copy",
        verify=True,
    )

    copied_trace = release / "evidence/av_nuplan/runtime/runtime_traces.csv"
    assert copied_trace.exists()
    assert copied_trace.read_text(encoding="utf-8") == trace_target.read_text(encoding="utf-8")


def test_validator_rejects_raw_stale_and_missing_required_artifacts(tmp_path: Path) -> None:
    repo = _seed_release_repo(tmp_path / "repo")
    release = build_release(
        repo_root=repo,
        out_root=tmp_path / "out",
        release_id="orius-three-domain-artifact-test",
        include_manuscripts=True,
        copy_mode="copy",
        verify=True,
    )

    injected_zip = release / "evidence/av_nuplan/runtime/raw.zip"
    _write(injected_zip, "raw archive\n")
    with pytest.raises(ValueError, match="disallowed suffix"):
        validate_release(release)
    injected_zip.unlink()

    stale = release / "evidence/av_nuplan/runtime/stale_claim.md"
    _write(stale, "This stale file claims a six-domain headline.\n")
    with pytest.raises(ValueError, match="Stale public claim"):
        validate_release(release)
    stale.unlink()

    required = release / "manifests/predeployment/nuplan_full_av_gate.json"
    required.unlink()
    with pytest.raises(FileNotFoundError, match="Missing required release file"):
        validate_release(release)
