#!/usr/bin/env python3
"""Audit table/result artifacts for publishable placeholder hygiene.

The audit is intentionally broader than the paper-only manifest checks: it
looks at report CSV/JSON/TeX surfaces and the local audit DuckDB files.  Numeric
zero values are reported as warnings only because zero is a valid safety result
for ORIUS violation/failure metrics.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_ROOTS = (
    "paper/assets/tables",
    "reports",
    "data/audit",
)
SKIP_PARTS = {
    ".git",
    ".venv",
    "node_modules",
    ".next",
    "__pycache__",
    ".pytest_cache",
}

BLOCKING_TOKENS = {
    "",
    "_",
    "---",
    "--",
    "nan",
    "none",
    "null",
    "n/a",
    "na",
    "tbd",
    "todo",
    "placeholder",
}
SEMANTIC_TOKENS = {
    "not_applicable",
    "not_reported",
    "not_run",
    "not_canonical",
    "verified_existing_artifacts",
    "no_failure",
    "no_failed_obligations",
    "no_intervention",
    "none_required",
    "no_open_blocker",
    "no_open_gap",
    "source_location_not_required",
    "external_review_owner_pending",
    "pending_external_review_artifact",
    "pending_artifact",
    "no_release_id",
    "no_supplemental_hf_artifact_required",
}
NONCANONICAL_ROW_TOKENS = {
    "not_canonical",
    "pending_artifact",
    "pending_external_review_artifact",
    "not_run",
}
MISSING_CONTEXT_MARKERS = (
    "artifact",
    "blocker",
    "closure",
    "command",
    "evidence",
    "fail",
    "finding",
    "location",
    "obligation",
    "owner",
    "provenance",
    "reason",
    "source",
    "status",
)
SYNTHETIC_NUMERIC_COLUMNS = {
    "adaptive_quantile",
    "conditional_coverage_gap",
    "e_t_mwh",
    "expires_at_step",
    "gamma_mw",
    "half_life_steps",
    "soc_tube_lower_mwh",
    "soc_tube_upper_mwh",
    "validity_horizon_h_t",
    "validity_score",
}
SYNTHETIC_TIMESTAMP_COLUMNS = {"expires_at", "timeout_at"}
ZERO_REVIEW_MARKERS = (
    "auc",
    "coverage",
    "cost",
    "error",
    "latency",
    "loss",
    "mae",
    "picp",
    "rate",
    "rmse",
    "score",
    "severity",
    "violation",
    "width",
)


@dataclass(frozen=True)
class Finding:
    severity: str
    blocking: bool
    source_type: str
    path: str
    issue: str
    column: str = ""
    table: str = ""
    row_index: str = ""
    value: str = ""
    detail: str = ""
    recommendation: str = ""

    def as_row(self) -> dict[str, Any]:
        def filled(value: Any, default: str = "not_applicable") -> Any:
            if isinstance(value, str) and value == "":
                return default
            return value

        return {
            "severity": self.severity,
            "blocking": self.blocking,
            "source_type": self.source_type,
            "path": self.path,
            "table": filled(self.table),
            "column": filled(self.column),
            "row_index": filled(self.row_index),
            "issue": self.issue,
            "value": filled(self.value),
            "detail": filled(self.detail),
            "recommendation": filled(self.recommendation, "none_required"),
        }


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _should_skip(path: Path) -> bool:
    return path.name.startswith("._") or any(part in SKIP_PARTS for part in path.parts)


def _iter_files(roots: Iterable[str]) -> Iterable[Path]:
    for root in roots:
        base = REPO_ROOT / root
        if not base.exists():
            continue
        if base.is_file():
            if not _should_skip(base):
                yield base
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _should_skip(path):
                yield path


def _looks_missing(value: Any, column: str = "") -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    text = str(value).strip()
    lower = text.lower()
    if lower in SEMANTIC_TOKENS:
        return False
    if lower == "none":
        return any(marker in column.lower() for marker in MISSING_CONTEXT_MARKERS)
    return lower in BLOCKING_TOKENS


def _semantic_recommendation(column: str) -> str:
    col = column.lower()
    if "failed_obligation" in col:
        return "Use no_failed_obligations for passing rows."
    if "failure" in col or "reason" in col:
        return "Use no_failure or a concrete failure reason."
    if "train_command" in col:
        return "Use verified_existing_artifacts or the exact training command."
    if "owner" in col:
        return "Use external_review_owner_pending or a concrete owner."
    if "closure" in col:
        return "Use pending_external_review_artifact or a closure artifact path."
    if "source" in col or "artifact" in col:
        return "Use a concrete artifact path or not_canonical."
    return "Replace with an explicit semantic value such as not_applicable."


def _scan_csv(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    try:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "csv",
                source,
                "csv_read_error",
                detail=str(exc),
                recommendation="Repair CSV syntax or regenerate the artifact.",
            )
        ]

    if df.empty:
        return [
            Finding(
                "error",
                True,
                "csv",
                source,
                "empty_csv_rows",
                detail="CSV has a header but no semantic data row.",
                recommendation="Add an explicit status row or remove it from active surfaces.",
            )
        ]

    for column in df.columns:
        series = df[column].map(lambda value: str(value).strip())
        missing_mask = series.map(lambda value: _looks_missing(value, column))
        count = int(missing_mask.sum())
        if count:
            sample_index = str(int(missing_mask[missing_mask].index[0]) + 2)
            sample = str(series[missing_mask].iloc[0])
            findings.append(
                Finding(
                    "error",
                    True,
                    "csv",
                    source,
                    "placeholder_or_blank_cell",
                    column=str(column),
                    row_index=sample_index,
                    value=sample,
                    detail=f"{count}/{len(series)} rows contain missing/placeholder values.",
                    recommendation=_semantic_recommendation(str(column)),
                )
            )

        numeric = pd.to_numeric(df[column], errors="coerce")
        numeric_count = int(numeric.notna().sum())
        if numeric_count >= 3 and bool((numeric.fillna(0) == 0).all()):
            col_lower = str(column).lower()
            if any(marker in col_lower for marker in ZERO_REVIEW_MARKERS):
                findings.append(
                    Finding(
                        "warning",
                        False,
                        "csv",
                        source,
                        "all_zero_metric_review",
                        column=str(column),
                        detail=f"{numeric_count} numeric values are all zero.",
                        recommendation="Review for semantic zero vs. placeholder zero; not blocking.",
                    )
                )
    return findings


def _scan_json(path: Path) -> list[Finding]:
    source = rel(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "json",
                source,
                "json_read_error",
                detail=str(exc),
                recommendation="Repair JSON syntax or regenerate the artifact.",
            )
        ]

    findings: list[Finding] = []

    def walk(value: Any, parts: list[str]) -> None:
        key = parts[-1] if parts else ""
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, [*parts, str(child_key)])
        elif isinstance(value, list):
            for index, child_value in enumerate(value):
                walk(child_value, [*parts, str(index)])
        elif _looks_missing(value, key):
            findings.append(
                Finding(
                    "error",
                    True,
                    "json",
                    source,
                    "placeholder_or_null_value",
                    column=".".join(parts),
                    value="null" if value is None else str(value),
                    recommendation=_semantic_recommendation(key),
                )
            )

    walk(payload, [])
    return findings


TEX_PLACEHOLDER_RE = re.compile(
    r"(?i)(\bNaN\b|\bNone\b|(?<![A-Za-z-])null\b|TODO|TBD|PLACEHOLDER|&\s*(---|--|N/A|n/a|_)\s*(?=&|\\\\)|&\s*(?=&|\\\\))"
)


def _scan_tex(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if TEX_PLACEHOLDER_RE.search(line):
            findings.append(
                Finding(
                    "error",
                    True,
                    "tex",
                    source,
                    "rendered_placeholder_cell",
                    row_index=str(line_no),
                    value=line.strip()[:220],
                    recommendation="Render an explicit semantic value in the table cell.",
                )
            )
            break
    return findings


def _scan_duckdb(path: Path) -> list[Finding]:
    findings: list[Finding] = []
    source = rel(path)
    try:
        con = duckdb.connect(str(path), read_only=True)
    except Exception as exc:
        return [
            Finding(
                "error",
                True,
                "duckdb",
                source,
                "duckdb_open_error",
                detail=str(exc),
                recommendation="Repair or regenerate the audit database.",
            )
        ]
    try:
        tables = [str(row[0]) for row in con.execute("SHOW TABLES").fetchall()]
        for table in tables:
            try:
                df = con.execute(f'SELECT * FROM "{table}"').df()
            except Exception as exc:
                findings.append(
                    Finding(
                        "error",
                        True,
                        "duckdb",
                        source,
                        "duckdb_table_read_error",
                        table=table,
                        detail=str(exc),
                    )
                )
                continue
            if df.empty:
                findings.append(
                    Finding(
                        "error",
                        True,
                        "duckdb",
                        source,
                        "empty_table",
                        table=table,
                        recommendation="Regenerate or drop stale empty audit table.",
                    )
                )
                continue
            noncanonical_mask = _duckdb_noncanonical_mask(df)
            for column in df.columns:
                allowed_null_mask = _duckdb_allowed_null_mask(df, str(column))
                unresolved_null_mask = df[column].isna() & ~noncanonical_mask & ~allowed_null_mask
                na_count = int(unresolved_null_mask.sum())
                if na_count:
                    findings.append(
                        Finding(
                            "error",
                            True,
                            "duckdb",
                            source,
                            "null_table_value",
                            table=table,
                            column=str(column),
                            detail=f"{na_count}/{len(df)} canonical rows are null.",
                            recommendation=_semantic_recommendation(str(column)),
                        )
                    )
                if pd.api.types.is_numeric_dtype(df[column]):
                    numeric = pd.to_numeric(df.loc[~noncanonical_mask, column], errors="coerce")
                    if str(column).lower() in SYNTHETIC_NUMERIC_COLUMNS:
                        sentinel_count = int((numeric == -1).sum())
                        if sentinel_count:
                            findings.append(
                                Finding(
                                    "error",
                                    True,
                                    "duckdb",
                                    source,
                                    "synthetic_fixed_result_value",
                                    table=table,
                                    column=str(column),
                                    value="-1",
                                    detail=f"{sentinel_count} canonical rows use a fixed sentinel instead of derived evidence.",
                                    recommendation="Regenerate from runtime/training artifacts or mark the rows noncanonical.",
                                )
                            )
                    if int(numeric.notna().sum()) >= 3 and bool((numeric.fillna(0) == 0).all()):
                        col_lower = str(column).lower()
                        if any(marker in col_lower for marker in ZERO_REVIEW_MARKERS):
                            findings.append(
                                Finding(
                                    "warning",
                                    False,
                                    "duckdb",
                                    source,
                                    "all_zero_metric_review",
                                    table=table,
                                    column=str(column),
                                    detail=f"{int(numeric.notna().sum())} numeric values are all zero.",
                                )
                            )
                elif str(column).lower() in SYNTHETIC_TIMESTAMP_COLUMNS:
                    timestamp_text = df.loc[~noncanonical_mask, column].astype(str)
                    sentinel_count = int(timestamp_text.str.startswith("1970-01-01").sum())
                    if sentinel_count:
                        findings.append(
                            Finding(
                                "error",
                                True,
                                "duckdb",
                                source,
                                "synthetic_fixed_result_value",
                                table=table,
                                column=str(column),
                                value="1970-01-01",
                                detail=f"{sentinel_count} canonical rows use an epoch sentinel instead of a real timestamp.",
                                recommendation="Regenerate the timestamp or leave it null where the event is not applicable.",
                            )
                        )
    finally:
        con.close()
    return findings


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


def run_audit(roots: Iterable[str]) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    scanned = {"csv": 0, "json": 0, "tex": 0, "duckdb": 0}
    for path in _iter_files(roots):
        suffix = path.suffix.lower()
        if suffix == ".csv":
            scanned["csv"] += 1
            findings.extend(_scan_csv(path))
        elif suffix == ".json":
            scanned["json"] += 1
            findings.extend(_scan_json(path))
        elif suffix == ".tex":
            scanned["tex"] += 1
            findings.extend(_scan_tex(path))
        elif suffix == ".duckdb" and path.parent.name == "audit":
            scanned["duckdb"] += 1
            findings.extend(_scan_duckdb(path))

    blocking = [finding for finding in findings if finding.blocking]
    warnings = [finding for finding in findings if not finding.blocking]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "roots": list(roots),
        "scanned": scanned,
        "finding_count": len(findings),
        "blocking_count": len(blocking),
        "warning_count": len(warnings),
        "passes": len(blocking) == 0,
        "top_blocking": [finding.as_row() for finding in blocking[:50]],
        "top_warnings": [finding.as_row() for finding in warnings[:50]],
    }
    return findings, summary


def _write_outputs(findings: list[Finding], summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [finding.as_row() for finding in findings]
    if not rows:
        rows = [
            {
                "severity": "info",
                "blocking": False,
                "source_type": "audit",
                "path": "reports/audit/table_result_integrity.csv",
                "table": "not_applicable",
                "column": "not_applicable",
                "row_index": "not_applicable",
                "issue": "no_findings",
                "value": "not_applicable",
                "detail": "No table/result integrity findings were detected.",
                "recommendation": "none_required",
            }
        ]
    csv_path = out_dir / "table_result_integrity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(Finding("", False, "", "", "").as_row().keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "table_result_integrity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Table/Result Integrity Audit",
        "",
        f"- Generated: `{summary['generated_at_utc']}`",
        f"- Passes: `{summary['passes']}`",
        f"- Blocking findings: `{summary['blocking_count']}`",
        f"- Warning findings: `{summary['warning_count']}`",
        f"- Scanned: `{summary['scanned']}`",
        "",
    ]
    if summary["top_blocking"]:
        md_lines.append("## Top Blocking Findings")
        for finding in summary["top_blocking"][:25]:
            md_lines.append(
                f"- `{finding['path']}` `{finding['column']}`: {finding['issue']} ({finding['detail'] or finding['value']})"
            )
        md_lines.append("")
    if summary["top_warnings"]:
        md_lines.append("## Top Warnings")
        for finding in summary["top_warnings"][:25]:
            md_lines.append(
                f"- `{finding['path']}` `{finding['column']}`: {finding['issue']} ({finding['detail']})"
            )
    (out_dir / "table_result_integrity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit ORIUS table/result artifact integrity.")
    parser.add_argument("--root", action="append", default=None, help="Repo-relative root to scan; may be repeated.")
    parser.add_argument("--out-dir", default="reports/audit")
    parser.add_argument("--no-fail", action="store_true", help="Write report but do not exit nonzero on blocking findings.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roots = args.root or list(DEFAULT_ROOTS)
    findings, summary = run_audit(roots)
    _write_outputs(findings, summary, REPO_ROOT / args.out_dir)
    print(json.dumps(summary, indent=2))
    if summary["blocking_count"] and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
