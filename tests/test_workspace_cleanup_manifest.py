from __future__ import annotations

import json

import scripts.build_workspace_cleanup_manifest as cleanup_manifest
import scripts.validate_workspace_cleanup_manifest as validate_cleanup_manifest


def test_workspace_cleanup_manifest_lists_only_reviewable_local_artifacts(tmp_path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "._module.py").write_text("sidecar", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "README.md").write_text("cache", encoding="utf-8")
    releases = tmp_path / "artifacts" / "releases"
    old_release = releases / "orius-three-domain-artifact-20260401T000000Z"
    new_release = releases / "orius-three-domain-artifact-20260402T000000Z"
    old_release.mkdir(parents=True)
    new_release.mkdir(parents=True)
    (old_release / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    (new_release / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "download_old_ai_helper.py").write_text("print('old')\n", encoding="utf-8")

    manifest = cleanup_manifest.build_manifest(tmp_path)
    candidates = {item["path"]: item for item in manifest["candidates"]}

    assert candidates["src/._module.py"]["category"] == "appledouble"
    assert candidates["src/._module.py"]["safe_to_delete"] is True
    assert candidates[".pytest_cache"]["category"] == "cache"
    assert candidates[".pytest_cache"]["safe_to_delete"] is True
    assert (
        candidates["artifacts/releases/orius-three-domain-artifact-20260401T000000Z"]["category"]
        == "duplicate_release"
    )
    assert "artifacts/releases/orius-three-domain-artifact-20260402T000000Z" not in candidates
    assert candidates["scripts/download_old_ai_helper.py"]["review_required"] is True
    assert candidates["scripts/download_old_ai_helper.py"]["safe_to_delete"] is False
    assert manifest["deletion_performed"] is False


def test_workspace_cleanup_manifest_validator_rejects_non_sidecar_personal_delete(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "mode": "dry_run",
                "deletion_performed": False,
                "delete_requires_confirmation": True,
                "candidates": [
                    {
                        "path": "Currently Enrolled Letter.pdf",
                        "category": "appledouble",
                        "safe_to_delete": True,
                        "is_dir": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = validate_cleanup_manifest.validate_manifest(manifest_path)

    assert result["pass"] is False
    assert any("not a sidecar" in error for error in result["errors"])
