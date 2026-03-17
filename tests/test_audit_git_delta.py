"""Tests for git delta audit script."""
from __future__ import annotations

from pathlib import Path

import scripts.audit_git_delta as git_delta


def test_build_delta_report_scopes_and_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(git_delta, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "configs" / "publish_audit.yaml"
    cfg.write_text(
        """
publish_audit:
  scope:
    include:
      - src/orius/**
      - scripts/**
""".strip(),
        encoding="utf-8",
    )

    def fake_git(args: list[str]) -> str:
        if args == ["rev-parse", "origin/main"]:
            return "abc123"
        if args == ["rev-parse", "HEAD"]:
            return "def456"
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return "main"
        if args == ["diff", "--numstat", "--find-renames", "origin/main"]:
            return "10\t2\tsrc/orius/a.py\n-\t-\treports/figures/demo.gif"
        if args == ["diff", "--name-status", "--find-renames", "origin/main"]:
            return "M\tsrc/orius/a.py\nM\treports/figures/demo.gif"
        if args == ["status", "--porcelain"]:
            return "?? scripts/new_tool.py"
        return ""

    monkeypatch.setattr(git_delta, "_try_run_git", fake_git)

    payload = git_delta.build_delta_report(baseline_ref="origin/main", config_path=cfg)
    summary = payload["summary"]
    assert summary["total_changed_files"] == 3
    assert summary["tracked_changed_files"] == 2
    assert summary["untracked_files"] == 1
    assert summary["in_scope_files"] == 2
    assert summary["out_of_scope_files"] == 1
    assert summary["added_lines"] == 10
    assert summary["deleted_lines"] == 2

    files = {row["path"]: row for row in payload["files"]}
    assert files["src/orius/a.py"]["in_scope"] is True
    assert files["scripts/new_tool.py"]["status"] == "??"
    assert files["reports/figures/demo.gif"]["in_scope"] is False


def test_audit_git_delta_main_writes_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(git_delta, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "publish_audit.yaml").write_text(
        "publish_audit:\n  scope:\n    include: [src/orius/**]\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        git_delta,
        "build_delta_report",
        lambda **_: {
            "generated_at": "2026-02-24T00:00:00+00:00",
            "baseline_ref": "origin/main",
            "baseline_commit": "abc123",
            "head_commit": "def456",
            "branch": "main",
            "summary": {
                "total_changed_files": 1,
                "tracked_changed_files": 1,
                "untracked_files": 0,
                "in_scope_files": 1,
                "out_of_scope_files": 0,
                "added_lines": 5,
                "deleted_lines": 1,
            },
            "files": [
                {
                    "path": "src/orius/a.py",
                    "status": "M",
                    "added_lines": 5,
                    "deleted_lines": 1,
                    "in_scope": True,
                    "scope_match": "src/orius/**",
                }
            ],
        },
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_git_delta.py",
            "--baseline-ref",
            "origin/main",
            "--out-dir",
            "reports/publish",
            "--scope-config",
            "configs/publish_audit.yaml",
        ],
    )
    git_delta.main()

    assert (tmp_path / "reports" / "publish" / "github_delta_report.json").exists()
    assert (tmp_path / "reports" / "publish" / "github_delta_report.md").exists()
