"""Tests for code health audit script."""

from __future__ import annotations

import scripts.audit_code_health as code_health


def test_code_health_detects_bare_except(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(code_health, "REPO_ROOT", tmp_path)
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    test_file = tmp_path / "src" / "bad.py"
    test_file.write_text(
        """
def f():
    try:
        return 1
    except:
        pass
""".strip(),
        encoding="utf-8",
    )
    cfg = tmp_path / "publish_audit.yaml"
    cfg.write_text(
        """
publish_audit:
  code_health:
    max_function_lines: 50
    max_cyclomatic_estimate: 20
    max_file_lines: 500
    fail_on_bare_except: true
    fail_on_except_pass: true
    fail_on_unused_imports: false
""".strip(),
        encoding="utf-8",
    )

    payload = code_health.run_code_health_audit(
        config_path=cfg,
        include_globs=["src/**/*.py"],
    )
    assert payload["fail"] is True
    vtypes = {v["type"] for v in payload["violations"]}
    assert "bare_except" in vtypes
    assert "except_pass" in vtypes


def test_code_health_skips_appledouble_sidecars(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(code_health, "REPO_ROOT", tmp_path)
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "good.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "._good.py").write_bytes(b"\x00\x05appledouble")
    cfg = tmp_path / "publish_audit.yaml"
    cfg.write_text(
        """
publish_audit:
  code_health:
    max_function_lines: 50
    max_cyclomatic_estimate: 20
    max_file_lines: 500
    fail_on_bare_except: true
    fail_on_except_pass: true
    fail_on_unused_imports: false
""".strip(),
        encoding="utf-8",
    )

    payload = code_health.run_code_health_audit(
        config_path=cfg,
        include_globs=["src/**/*.py"],
    )

    assert payload["fail"] is False
    assert [item["path"] for item in payload["files"]] == [str(tmp_path / "src" / "good.py")]
