"""Tests for strict NA audit script."""

from __future__ import annotations

import pandas as pd

import scripts.audit_na_tables as na_audit


def test_na_audit_fails_on_unexpected_na(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(na_audit, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "publish_audit.yaml").write_text(
        """
publish_audit:
  na_policy:
    strict: true
    default_max_ratio: 0.0
    allowlist: []
""".strip(),
        encoding="utf-8",
    )
    df = pd.DataFrame({"timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"], "x": [1.0, None]})
    df.to_parquet(tmp_path / "data" / "processed" / "sample.parquet", index=False)

    table_df, summary = na_audit.run_na_audit(
        config_path=tmp_path / "configs" / "publish_audit.yaml",
        parquet_glob="data/processed/**/*.parquet",
        csv_glob="reports/**/*.csv",
        duckdb_glob="data/**/*.duckdb",
    )
    assert not table_df.empty
    assert summary["violations"] >= 1
    assert summary["fail"] is True


def test_na_audit_passes_when_allowlisted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(na_audit, "REPO_ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    report_path = tmp_path / "reports" / "metrics.csv"

    (tmp_path / "configs" / "publish_audit.yaml").write_text(
        """
publish_audit:
  na_policy:
    strict: true
    default_max_ratio: 0.0
    allowlist:
      - path_glob: reports/metrics.csv
        columns: [daylight_mape]
        max_ratio: 1.0
""".strip(),
        encoding="utf-8",
    )
    pd.DataFrame({"model": ["gbm", "lstm"], "daylight_mape": [None, None]}).to_csv(report_path, index=False)

    _, summary = na_audit.run_na_audit(
        config_path=tmp_path / "configs" / "publish_audit.yaml",
        parquet_glob="data/processed/**/*.parquet",
        csv_glob="reports/**/*.csv",
        duckdb_glob="data/**/*.duckdb",
    )
    assert summary["violations"] == 0
    assert summary["fail"] is False
