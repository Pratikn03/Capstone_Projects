from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from orius.data_pipeline.external_raw import EXTERNAL_DATASETS, get_strict_external_root

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "setup_external_ssd.py"

DATASET_DIRS = tuple(spec.directory_name for spec in EXTERNAL_DATASETS.values())


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def test_setup_external_ssd_scaffolds_dirs_symlink_and_shell_profile(tmp_path: Path) -> None:
    volume_root = tmp_path / "ORIUS_SSD"
    volume_root.mkdir()
    strict_link = tmp_path / "orius_external_data"
    shell_profile = tmp_path / ".zshrc"

    result = _run(
        [
            sys.executable,
            str(SCRIPT),
            "--volume-root",
            str(volume_root),
            "--strict-link",
            str(strict_link),
            "--append-shell-profile",
            "--shell-profile",
            str(shell_profile),
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    external_root = volume_root / "orius_external_data"
    assert external_root.is_dir()
    for name in DATASET_DIRS:
        assert (external_root / name).is_dir()
    assert strict_link.is_symlink()
    assert strict_link.resolve() == external_root.resolve()

    shell_text = shell_profile.read_text(encoding="utf-8")
    assert "ORIUS_EXTERNAL_DATA_ROOT" in shell_text
    assert str(external_root) in shell_text


def test_setup_external_ssd_verify_only_fails_when_layout_is_incomplete(tmp_path: Path) -> None:
    volume_root = tmp_path / "ORIUS_SSD"
    volume_root.mkdir()
    strict_link = tmp_path / "orius_external_data"

    result = _run(
        [
            sys.executable,
            str(SCRIPT),
            "--volume-root",
            str(volume_root),
            "--strict-link",
            str(strict_link),
            "--verify-only",
            "--skip-symlink",
        ],
        cwd=tmp_path,
    )

    assert result.returncode == 1
    assert "External SSD layout is incomplete" in result.stderr


def test_get_strict_external_root_prefers_strict_env(monkeypatch, tmp_path: Path) -> None:
    strict_root = tmp_path / "strict"
    external_root = tmp_path / "external"
    strict_root.mkdir()
    external_root.mkdir()

    monkeypatch.setenv("ORIUS_STRICT_EXTERNAL_ROOT", str(strict_root))
    monkeypatch.setenv("ORIUS_EXTERNAL_DATA_ROOT", str(external_root))

    assert get_strict_external_root() == strict_root.resolve()


def test_get_strict_external_root_falls_back_to_external_root_env(monkeypatch, tmp_path: Path) -> None:
    external_root = tmp_path / "external"
    external_root.mkdir()

    monkeypatch.delenv("ORIUS_STRICT_EXTERNAL_ROOT", raising=False)
    monkeypatch.setenv("ORIUS_EXTERNAL_DATA_ROOT", str(external_root))

    assert get_strict_external_root() == external_root.resolve()
