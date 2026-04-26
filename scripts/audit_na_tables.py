"""Strict NA audit for processed data, report tables, and audit DBs."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fnmatch
import json
from pathlib import Path
import sys
from typing import Any

import duckdb
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


@dataclass
class AllowRule:
    path_glob: str
    columns: list[str]
    max_ratio: float


NONCANONICAL_ROW_TOKENS = {
    "not_canonical",
    "pending_artifact",
    "pending_external_review_artifact",
    "not_run",
}


def _load_publish_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("publish_audit", {}) if isinstance(payload, dict) else {}


def _build_allow_rules(cfg: dict[str, Any]) -> list[AllowRule]:
    policy = cfg.get("na_policy", {}) if isinstance(cfg.get("na_policy"), dict) else {}
    rules_raw = policy.get("allowlist", []) if isinstance(policy.get("allowlist"), list) else []
    rules: list[AllowRule] = []
    for item in rules_raw:
        if not isinstance(item, dict):
            continue
        path_glob = str(item.get("path_glob", "")).strip()
        if not path_glob:
            continue
        columns = [str(c).strip() for c in item.get("columns", []) if str(c).strip()]
        if not columns:
            continue
        max_ratio = float(item.get("max_ratio", 1.0))
        rules.append(AllowRule(path_glob=path_glob, columns=columns, max_ratio=max_ratio))
    return rules


def _build_exclude_globs(cfg: dict[str, Any], cli_exclude: list[str] | None) -> list[str]:
    policy = cfg.get("na_policy", {}) if isinstance(cfg.get("na_policy"), dict) else {}
    cfg_globs = [str(x).strip() for x in policy.get("exclude_path_globs", []) if str(x).strip()]
    cli_globs = [str(x).strip() for x in (cli_exclude or []) if str(x).strip()]
    default_globs = ["reports/publish/na_audit.csv"]
    merged: list[str] = []
    for pattern in [*default_globs, *cfg_globs, *cli_globs]:
        if pattern and pattern not in merged:
            merged.append(pattern)
    return merged


def _is_excluded(source_key: str, exclude_globs: list[str]) -> bool:
    return any(fnmatch.fnmatch(source_key, pattern) for pattern in exclude_globs)


def _allowed_max_ratio(source_key: str, column: str, rules: list[AllowRule], default_max_ratio: float) -> float:
    effective = default_max_ratio
    for rule in rules:
        if not fnmatch.fnmatch(source_key, rule.path_glob):
            continue
        if "*" in rule.columns or column in rule.columns:
            effective = max(effective, float(rule.max_ratio))
    return effective


def _scan_dataframe(df: pd.DataFrame, source_key: str, source_type: str, table: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_rows = int(len(df))
    if n_rows == 0:
        return rows
    for col in df.columns:
        na_count = int(df[col].isna().sum())
        na_ratio = float(na_count / n_rows) if n_rows else 0.0
        rows.append(
            {
                "source_type": source_type,
                "source_key": source_key,
                "table": table or "not_applicable",
                "column": str(col),
                "rows": n_rows,
                "na_count": na_count,
                "na_ratio": na_ratio,
            }
        )
    return rows


def _duckdb_noncanonical_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    status_like = [
        column
        for column in df.columns
        if any(marker in str(column).lower() for marker in ("status", "surface", "closure", "tier", "canonical"))
    ]
    for column in status_like:
        values = df[column].astype(str).str.strip().str.lower()
        mask |= values.isin(NONCANONICAL_ROW_TOKENS)
    return mask


def _duckdb_allowed_null_mask(df: pd.DataFrame, column: str) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    col = column.lower()
    if col in {"expires_at", "timeout_at"} and "status" in df.columns:
        status = df["status"].astype(str).str.strip().str.lower()
        mask |= status.isin({"acked", "complete", "completed"})
        if "timeout_reason" in df.columns:
            reason = df["timeout_reason"].astype(str).str.strip().str.lower()
            mask |= reason.isin({"no_failure", "not_applicable", "none_required"})
    return mask


def _scan_duckdb_dataframe(df: pd.DataFrame, source_key: str, table: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_rows = int(len(df))
    if n_rows == 0:
        return rows
    noncanonical = _duckdb_noncanonical_mask(df)
    for col in df.columns:
        allowed_null = _duckdb_allowed_null_mask(df, str(col))
        unresolved = df[col].isna() & ~noncanonical & ~allowed_null
        na_count = int(unresolved.sum())
        na_ratio = float(na_count / n_rows) if n_rows else 0.0
        rows.append(
            {
                "source_type": "duckdb",
                "source_key": source_key,
                "table": table,
                "column": str(col),
                "rows": n_rows,
                "na_count": na_count,
                "na_ratio": na_ratio,
            }
        )
    return rows


def _scan_csv(path: Path, source_key: str | None = None) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return _scan_dataframe(df, source_key=source_key or str(path), source_type="csv")


def _scan_parquet(path: Path, source_key: str | None = None) -> list[dict[str, Any]]:
    try:
        df = pd.read_parquet(path)
    except Exception:
        return []
    return _scan_dataframe(df, source_key=source_key or str(path), source_type="parquet")


def _scan_duckdb(path: Path, source_key_prefix: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    try:
        con = duckdb.connect(str(path))
    except Exception:
        return rows
    try:
        tables = [str(r[0]) for r in con.execute("SHOW TABLES").fetchall()]
        for table in tables:
            try:
                df = con.execute(f"SELECT * FROM {table}").df()
            except Exception:
                continue
            base = source_key_prefix or str(path)
            source_key = f"{base}::{table}"
            rows.extend(_scan_duckdb_dataframe(df, source_key=source_key, table=table))
    finally:
        con.close()
    return rows


def run_na_audit(
    *,
    config_path: Path,
    parquet_glob: str,
    csv_glob: str,
    duckdb_glob: str,
    exclude_globs: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cfg = _load_publish_cfg(config_path)
    policy = cfg.get("na_policy", {}) if isinstance(cfg.get("na_policy"), dict) else {}
    strict = bool(policy.get("strict", True))
    default_max_ratio = float(policy.get("default_max_ratio", 0.0))
    rules = _build_allow_rules(cfg)
    excludes = _build_exclude_globs(cfg, exclude_globs)

    records: list[dict[str, Any]] = []
    for path in sorted(REPO_ROOT.glob(parquet_glob)):
        if path.is_file():
            if path.name.startswith("._"):
                continue
            rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
            if _is_excluded(rel, excludes):
                continue
            records.extend(_scan_parquet(path, source_key=rel))
    for path in sorted(REPO_ROOT.glob(csv_glob)):
        if path.is_file():
            if path.name.startswith("._"):
                continue
            rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
            if _is_excluded(rel, excludes):
                continue
            records.extend(_scan_csv(path, source_key=rel))
    for path in sorted(REPO_ROOT.glob(duckdb_glob)):
        if path.is_file():
            if path.name.startswith("._"):
                continue
            rel = str(path.relative_to(REPO_ROOT)) if path.is_absolute() else str(path)
            if _is_excluded(rel, excludes):
                continue
            records.extend(_scan_duckdb(path, source_key_prefix=rel))

    if records:
        df = pd.DataFrame.from_records(records)
    else:
        df = pd.DataFrame(
            columns=[
                "source_type",
                "source_key",
                "table",
                "column",
                "rows",
                "na_count",
                "na_ratio",
                "allowed_max_ratio",
                "allowed",
                "status",
            ]
        )

    if not df.empty:
        allowed_max = [
            _allowed_max_ratio(source_key=str(r.source_key), column=str(r.column), rules=rules, default_max_ratio=default_max_ratio)
            for r in df.itertuples(index=False)
        ]
        df["allowed_max_ratio"] = allowed_max
        df["allowed"] = df["na_ratio"] <= df["allowed_max_ratio"]
        df["status"] = df["allowed"].map({True: "ok", False: "violation"})
    else:
        df["allowed_max_ratio"] = []
        df["allowed"] = []
        df["status"] = []

    violations_df = df[df["status"] == "violation"] if not df.empty else df
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strict": strict,
        "default_max_ratio": default_max_ratio,
        "excluded_globs": excludes,
        "rows_scanned": int(df["rows"].sum()) if not df.empty else 0,
        "columns_scanned": int(len(df)),
        "violations": int(len(violations_df)),
        "fail": bool(strict and len(violations_df) > 0),
        "top_violations": (
            violations_df.sort_values("na_ratio", ascending=False)
            .head(50)
            .to_dict(orient="records")
            if not violations_df.empty
            else []
        ),
    }
    return df, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit NA values for publish gating")
    parser.add_argument("--config", default="configs/publish_audit.yaml")
    parser.add_argument("--parquet-glob", default="data/processed/**/*.parquet")
    parser.add_argument("--csv-glob", default="reports/**/*.csv")
    parser.add_argument("--duckdb-glob", default="data/**/*.duckdb")
    parser.add_argument("--exclude-glob", action="append", default=None)
    parser.add_argument("--out-csv", default="reports/publish/na_audit.csv")
    parser.add_argument("--out-json", default="reports/publish/na_summary.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df, summary = run_na_audit(
        config_path=Path(args.config),
        parquet_glob=args.parquet_glob,
        csv_glob=args.csv_glob,
        duckdb_glob=args.duckdb_glob,
        exclude_globs=args.exclude_glob,
    )

    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sort_cols = ["status", "na_ratio", "source_key", "column"]
    if not df.empty:
        df.sort_values(by=sort_cols, ascending=[True, False, True, True]).to_csv(out_csv, index=False)
    else:
        df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if summary.get("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
