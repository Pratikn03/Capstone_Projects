from __future__ import annotations

from pathlib import Path

import scripts.build_code_quality_bundle as code_bundle
import scripts.validate_code_quality_bundle as validate_code_bundle


def _write_minimal_repo(root: Path) -> None:
    for dirname in ("src/orius", "scripts", "tests", "configs"):
        (root / dirname).mkdir(parents=True, exist_ok=True)
    (root / "src" / "orius" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "tool.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "tests" / "test_tool.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (root / "configs" / "publish_audit.yaml").write_text("publish_audit: {}\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='orius-test'\n", encoding="utf-8")
    (root / "Makefile").write_text("quality:\n\tpython -m compileall src\n", encoding="utf-8")
    (root / "data").mkdir()
    (root / "data" / "raw.zip").write_bytes(b"raw")
    (root / "reports").mkdir()
    (root / "reports" / "runtime_traces.csv").write_text("subject_id,hadm_id,stay_id\n", encoding="utf-8")
    (root / "scripts" / "._tool.py").write_text("sidecar", encoding="utf-8")


def test_code_quality_bundle_excludes_raw_data_and_validates(tmp_path) -> None:
    _write_minimal_repo(tmp_path)

    manifest = code_bundle.build_bundle(
        tmp_path,
        tmp_path / "out",
        stamp="20260429T000000Z",
        create_tar=False,
    )

    bundle_dir = Path(manifest["bundle_dir"])
    assert (bundle_dir / "src" / "orius" / "__init__.py").is_file()
    assert not (bundle_dir / "data" / "raw.zip").exists()
    assert not (bundle_dir / "reports" / "runtime_traces.csv").exists()
    assert not (bundle_dir / "scripts" / "._tool.py").exists()
    result = validate_code_bundle.validate_bundle(bundle_dir)
    assert result["pass"] is True


def test_code_quality_bundle_validator_rejects_injected_raw_archive(tmp_path) -> None:
    _write_minimal_repo(tmp_path)
    manifest = code_bundle.build_bundle(
        tmp_path,
        tmp_path / "out",
        stamp="20260429T000000Z",
        create_tar=False,
    )
    bundle_dir = Path(manifest["bundle_dir"])
    (bundle_dir / "src" / "bad.zip").write_bytes(b"raw")

    result = validate_code_bundle.validate_bundle(bundle_dir)

    assert result["pass"] is False
    assert any("disallowed heavy/private suffix" in error for error in result["errors"])
