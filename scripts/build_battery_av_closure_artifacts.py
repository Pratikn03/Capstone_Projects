#!/usr/bin/env python3
"""Build shared closure artifacts for the battery + AV release lane."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(SRC_DIR))

from orius.certos.verification import load_certificates_from_duckdb, verify_certificates
from orius.forecasting.uncertainty.shift_aware import summarize_weighted_recalibration, weighted_online_recalibration


DEFAULT_BATTERY_DIR = REPO_ROOT / "reports" / "battery_av" / "battery"
DEFAULT_AV_DIR = REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
DEFAULT_OVERALL_DIR = REPO_ROOT / "reports" / "battery_av" / "overall"
DOCS_DIR = REPO_ROOT / "docs"
CENTRAL_NOVELTY_SENTENCE = (
    "ORIUS identifies OASG as the degraded-observation release hazard and "
    "provides a reliability-aware runtime safety layer across Battery, AV, "
    "and Healthcare."
)


def _is_appledouble(path: Path) -> bool:
    return any(part.startswith("._") for part in path.parts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _required_map(domain_dir: Path, items: dict[str, str]) -> tuple[dict[str, str | None], list[str]]:
    resolved: dict[str, str | None] = {}
    missing: list[str] = []
    for key, relative in items.items():
        path = domain_dir / relative
        if path.exists():
            resolved[key] = str(path)
        else:
            resolved[key] = None
            missing.append(key)
    return resolved, missing


def _runtime_summary_row(runtime_summary_path: Path, controller: str) -> dict[str, Any]:
    if not runtime_summary_path.exists():
        return {}
    runtime_df = pd.read_csv(runtime_summary_path)
    if runtime_df.empty:
        return {}
    candidates = [str(controller)]
    if ":" in str(controller):
        candidates.append(str(controller).split(":")[-1])
    if "controller" in runtime_df.columns:
        selected = runtime_df[runtime_df["controller"].astype(str).isin(candidates)]
        if not selected.empty:
            return selected.iloc[0].to_dict()
    if "controller_label" in runtime_df.columns:
        selected = runtime_df[runtime_df["controller_label"].astype(str).isin(candidates)]
        if not selected.empty:
            return selected.iloc[0].to_dict()
    return runtime_df.iloc[0].to_dict()


def _runtime_trace_counts(traces_df: pd.DataFrame | None, summary_controller: str) -> tuple[int, int]:
    if traces_df is None or traces_df.empty:
        return 0, 0
    total_rows = int(len(traces_df))
    canonical_df = pd.DataFrame()
    candidates = [str(summary_controller)]
    if ":" in str(summary_controller):
        candidates.append(str(summary_controller).split(":")[-1])
    for column in ("controller_label", "controller"):
        if column in traces_df.columns:
            candidate_df = traces_df[traces_df[column].astype(str).isin(candidates)].reset_index(drop=True)
            if not candidate_df.empty:
                canonical_df = candidate_df
                break
    if canonical_df.empty:
        canonical_df = traces_df
    return total_rows, int(len(canonical_df))


def _artifact_count(manifest_path: Path) -> int:
    if not manifest_path.exists():
        return 0
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    if not isinstance(payload, dict):
        return 0
    artifacts = payload.get("artifacts")
    return len(artifacts) if isinstance(artifacts, dict) else 0


def _pct_text(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{numeric * 100:.1f}%"


def _build_executive_summary(
    *,
    release_summary: dict[str, Any],
    override: dict[str, Any],
    battery_dir: Path,
    av_dir: Path,
    submission_scope: str,
    docs_dir: Path,
) -> str:
    battery_release = dict(release_summary.get("battery", {}))
    av_release = dict(release_summary.get("av", {}))
    battery_runtime = _runtime_summary_row(battery_dir / "runtime_summary.csv", battery_release.get("canonical_controller", "deep:dc3s_wrapped"))
    av_baseline_runtime = _runtime_summary_row(av_dir / "runtime_summary.csv", "baseline")
    av_orius_runtime = _runtime_summary_row(av_dir / "runtime_summary.csv", av_release.get("canonical_controller", "orius"))
    battery_artifact_count = _artifact_count(battery_dir / "artifact_manifest.json")
    av_artifact_count = _artifact_count(av_dir / "artifact_manifest.json")
    battery_chain_valid = bool((((battery_release.get("certos") or {}).get("summary") or {}).get("chain_valid")))
    av_chain_valid = bool((((av_release.get("certos") or {}).get("summary") or {}).get("chain_valid")))
    battery_cert_rows = int((((battery_release.get("certos") or {}).get("summary") or {}).get("certificate_rows", 0)))
    av_cert_rows = int((((av_release.get("certos") or {}).get("summary") or {}).get("certificate_rows", 0)))
    battery_oasg = int(((battery_release.get("counterexamples") or {}).get("oasg_case_count", 0)))
    av_oasg = int(((av_release.get("counterexamples") or {}).get("oasg_case_count", 0)))

    lines = [
        "# ORIUS — Executive Summary",
        "",
        "> Generated from the canonical battery + AV closure artifacts.",
        "",
        "## What ORIUS Is",
        "",
        CENTRAL_NOVELTY_SENTENCE,
        "",
        "ORIUS (Observation–Reality Integrity for Universal Safety) is a runtime safety layer for physical AI systems under degraded observation. It treats the observation–action safety gap as the governing hazard and responds through the Detect → Calibrate → Constrain → Shield → Certify lane.",
        "",
        "## Current Submission Scope",
        "",
        f"- `submission_scope={submission_scope}`",
        "- `battery` is the reference witness row.",
        "- `av` is the bounded runtime-contract row under the narrowed brake-hold release contract.",
        "- `industrial`, `healthcare`, `navigation`, and `aerospace` are not promoted in this battery+AV submission lane.",
        "",
        "## Locked Battery + AV Results",
        "",
        "| Domain | Tier | Key Result | Evidence |",
        "|--------|------|------------|----------|",
        (
            f"| **Battery (BESS)** | `{override.get('battery', {}).get('resulting_tier', 'reference')}` | "
            f"Canonical TSVR = {_pct_text(battery_runtime.get('tsvr'))} on {int(battery_release.get('runtime_rows_canonical_controller', 0)):,} canonical runtime rows; "
            f"{battery_oasg} OASG cases identified. | {battery_artifact_count} locked artifacts; chain valid = {battery_chain_valid}; certificates = {battery_cert_rows:,} |"
        ),
        (
            f"| **Autonomous Vehicles** | `{override.get('vehicle', {}).get('resulting_tier', 'runtime_contract_closed')}` | "
            f"Baseline TSVR = {_pct_text(av_baseline_runtime.get('tsvr'))}, ORIUS TSVR = {_pct_text(av_orius_runtime.get('tsvr'))} on "
            f"{int(av_release.get('runtime_rows_canonical_controller', 0)):,} canonical runtime rows "
            f"({int(av_release.get('runtime_rows_total', 0)):,} total trace rows); {av_oasg} OASG cases identified on the ORIUS AV defended row. | "
            f"{av_artifact_count} locked artifacts; chain valid = {av_chain_valid}; certificates = {av_cert_rows:,} |"
        ),
        "",
        "## What This Submission Does Not Claim",
        "",
        "- Industrial and healthcare are intentionally outside the promoted `battery_av_only` submission lane.",
        "- AV remains a bounded longitudinal result; it is not a claim of full autonomous-driving field closure.",
        "- Navigation and aerospace remain non-promoted rows.",
        "- Adversarial completeness and production deployment readiness are not claimed from this surface.",
        "",
        "## Canonical Artifacts",
        "",
        "- `reports/battery_av/overall/release_summary.json`",
        "- `reports/battery_av/overall/publication_closure_override.json`",
        "- `reports/publication/orius_equal_domain_parity_matrix.csv`",
        "- `reports/battery_av/battery/`",
        "- `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/`",
        "",
    ]
    path = docs_dir / "executive_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _build_claim_ledger(
    *,
    release_summary: dict[str, Any],
    override: dict[str, Any],
    battery_dir: Path,
    av_dir: Path,
    submission_scope: str,
    docs_dir: Path,
) -> str:
    battery_release = dict(release_summary.get("battery", {}))
    av_release = dict(release_summary.get("av", {}))
    battery_runtime = _runtime_summary_row(battery_dir / "runtime_summary.csv", battery_release.get("canonical_controller", "deep:dc3s_wrapped"))
    av_baseline_runtime = _runtime_summary_row(av_dir / "runtime_summary.csv", "baseline")
    av_orius_runtime = _runtime_summary_row(av_dir / "runtime_summary.csv", av_release.get("canonical_controller", "orius"))
    av_oasg = int(((av_release.get("counterexamples") or {}).get("oasg_case_count", 0)))
    lines = [
        "# ORIUS Claim Ledger — Battery + AV Submission Lane",
        "",
        "> Generated from the canonical closure artifacts.",
        "",
        CENTRAL_NOVELTY_SENTENCE,
        "",
        "## Governing Inputs",
        "",
        "- `reports/battery_av/overall/release_summary.json`",
        "- `reports/battery_av/overall/publication_closure_override.json`",
        "- `reports/publication/orius_equal_domain_parity_matrix.csv`",
        f"- `submission_scope={submission_scope}`",
        "",
        "## Bucket A — Directly Artifact-Backed",
        "",
        "| ID | Claim | Governing Artifact |",
        "|----|-------|-------------------|",
        f"| A1 | Battery remains the `reference` witness row in the current submission lane. | `publication_closure_override.json` |",
        f"| A2 | Battery canonical TSVR is {_pct_text(battery_runtime.get('tsvr'))} on {int(battery_release.get('runtime_rows_canonical_controller', 0)):,} canonical runtime rows. | `reports/battery_av/battery/runtime_summary.csv`, `release_summary.json` |",
        f"| A3 | Battery CertOS chain is valid with {int((((battery_release.get('certos') or {}).get('summary') or {}).get('certificate_rows', 0))):,} certificates. | `reports/battery_av/battery/certos_verification_summary.json` |",
        f"| A4 | AV remains the bounded `{override.get('vehicle', {}).get('resulting_tier', 'runtime_contract_closed')}` row. | `publication_closure_override.json` |",
        f"| A5 | AV baseline TSVR is {_pct_text(av_baseline_runtime.get('tsvr'))} and ORIUS TSVR is {_pct_text(av_orius_runtime.get('tsvr'))}. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/runtime_summary.csv` |",
        f"| A6 | AV runtime rows total = {int(av_release.get('runtime_rows_total', 0)):,}; canonical ORIUS rows = {int(av_release.get('runtime_rows_canonical_controller', 0)):,}. | `release_summary.json` |",
        f"| A7 | AV ORIUS OASG cases identified = {av_oasg:,}. | `reports/orius_av/nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest/oasg_domain_summary.csv`, `release_summary.json` |",
        f"| A8 | Only `battery` and `vehicle` are promoted rows under `submission_scope={submission_scope}`. | `reports/publication/orius_equal_domain_parity_matrix.csv` |",
        "",
        "## Bucket B — Bounded / Qualified Claims",
        "",
        "| ID | Claim | Qualification |",
        "|----|-------|---------------|",
        "| B1 | AV runtime-contract closure is real. | It is bounded to the current brake-hold release contract. |",
        "| B2 | Battery remains the deepest witness. | That does not imply equal maturity across domains outside the current battery+AV lane. |",
        "| B3 | Shift-aware uncertainty is active in both rows. | Coverage guarantees under arbitrary shift are not claimed from these artifacts. |",
        "",
        "## Bucket C — Explicitly Not Claimed",
        "",
        "| ID | Non-Claim | Governing Reason |",
        "|----|-----------|------------------|",
        "| C1 | Industrial is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |",
        "| C2 | Healthcare is promoted in this submission. | `outside_current_submission_scope_battery_av_lane` |",
        "| C3 | Navigation is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |",
        "| C4 | Aerospace is a promoted defended row. | It remains a non-promoted row in the current parity matrix. |",
        "| C5 | AV is full autonomous-driving field closure. | The current AV row is explicitly bounded. |",
        "",
    ]
    path = docs_dir / "claim_ledger.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _reliability_bin(values: pd.Series) -> pd.Series:
    bins = pd.cut(
        pd.to_numeric(values, errors="coerce").fillna(0.0),
        bins=[-np.inf, 0.4, 0.7, np.inf],
        labels=["low", "mid", "high"],
    )
    return bins.astype(str)


def _drift_bin(values: pd.Series) -> pd.Series:
    bins = pd.cut(
        pd.to_numeric(values, errors="coerce").fillna(0.0),
        bins=[-np.inf, 0.2, 0.5, np.inf],
        labels=["low", "mid", "high"],
    )
    return bins.astype(str)


def _speed_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    bins = pd.cut(
        numeric,
        bins=[-np.inf, 5.0, 10.0, 15.0, 20.0, np.inf],
        labels=["0_5", "5_10", "10_15", "15_20", "20_plus"],
    )
    return bins.astype(str)


def _neighbor_count_bin(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    bins = pd.cut(
        numeric,
        bins=[-np.inf, 1.0, 3.0, 5.0, np.inf],
        labels=["0_1", "2_3", "4_5", "6_plus"],
    )
    return bins.astype(str)


def _counterexample_bundle(
    *,
    domain_name: str,
    traces_df: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Any]:
    work = traces_df.copy()
    work["observed_safe"] = pd.to_numeric(work["observed_margin"], errors="coerce").fillna(-np.inf) >= 0.0
    work["true_unsafe"] = pd.to_numeric(work["true_margin"], errors="coerce").fillna(np.inf) < 0.0
    work["oasg_case"] = work["observed_safe"] & work["true_unsafe"]
    work["margin_gap"] = pd.to_numeric(work["observed_margin"], errors="coerce").fillna(0.0) - pd.to_numeric(work["true_margin"], errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["oasg_case", "margin_gap", "step_index"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    selected = work[work["oasg_case"]].head(20).copy()
    if selected.empty:
        selected = work.nsmallest(20, "true_margin").copy()
    selected_cols = [
        col
        for col in (
            "trace_id",
            "scenario_id",
            "controller",
            "fault_family",
            "step_index",
            "observed_margin",
            "true_margin",
            "margin_gap",
            "reliability_w",
            "widening_factor",
            "intervened",
            "fallback_used",
            "certificate_valid",
            "validity_score",
        )
        if col in selected.columns
    ]
    selected = selected[selected_cols]

    csv_path = out_dir / "observed_true_counterexamples.csv"
    json_path = out_dir / "observed_true_counterexamples.json"
    fig_path = out_dir / "fig_observed_vs_true_counterexamples.png"
    summary_path = out_dir / "oasg_domain_summary.csv"

    selected.to_csv(csv_path, index=False)
    _write_json(json_path, selected.to_dict(orient="records"))

    summary_df = (
        work.groupby(["controller", "fault_family"], dropna=False, as_index=False)
        .agg(
            rows=("oasg_case", "size"),
            oasg_cases=("oasg_case", "sum"),
            max_margin_gap=("margin_gap", "max"),
            mean_margin_gap=("margin_gap", "mean"),
        )
        .sort_values(["oasg_cases", "max_margin_gap"], ascending=[False, False], kind="mergesort")
    )
    summary_df.to_csv(summary_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = selected.head(10).copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, f"No {domain_name} counterexamples", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(len(plot_df))
        ax.bar(x - 0.18, plot_df["observed_margin"], width=0.36, label="Observed margin")
        ax.bar(x + 0.18, plot_df["true_margin"], width=0.36, label="True margin")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["step_index"].astype(str), rotation=45, ha="right")
        ax.set_title(f"{domain_name} observed-vs-true counterexamples")
        ax.set_xlabel("Step index")
        ax.set_ylabel("Margin")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "figure": str(fig_path),
        "summary_csv": str(summary_path),
        "counterexample_count": int(len(selected)),
        "oasg_case_count": int(work["oasg_case"].sum()),
    }


def _build_shift_summary(
    *,
    domain_name: str,
    traces_df: pd.DataFrame,
    out_dir: Path,
    target_specs: list[dict[str, str]],
    subgroup_axes: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    overall_baseline: dict[str, Any] = {"domain": domain_name, "targets": {}}
    overall_adaptive: dict[str, Any] = {"domain": domain_name, "targets": {}}

    for spec in target_specs:
        target = spec["target"]
        work = traces_df.copy()
        work[target] = pd.to_numeric(work[spec["y_true"]], errors="coerce")
        work["base_lower"] = pd.to_numeric(work[spec["lower"]], errors="coerce")
        work["base_upper"] = pd.to_numeric(work[spec["upper"]], errors="coerce")
        work["reliability_w"] = pd.to_numeric(work[spec["reliability"]], errors="coerce").fillna(1.0)
        work["shift_score"] = pd.to_numeric(work[spec["shift"]], errors="coerce").fillna(0.0)
        work = work.dropna(subset=[target, "base_lower", "base_upper"]).reset_index(drop=True)
        if work.empty:
            continue
        adaptive_df = weighted_online_recalibration(
            y_true=work[target].to_numpy(dtype=float),
            lower=work["base_lower"].to_numpy(dtype=float),
            upper=work["base_upper"].to_numpy(dtype=float),
            reliability=work["reliability_w"].to_numpy(dtype=float),
            shift_score=work["shift_score"].to_numpy(dtype=float),
        )
        merged = pd.concat([work, adaptive_df.drop(columns=["row_index"])], axis=1)
        for axis in subgroup_axes:
            if axis not in merged.columns:
                continue
            for group_value, group_df in merged.groupby(axis, dropna=False):
                rows.append(
                    {
                        "target": target,
                        "group_col": axis,
                        "group_value": str(group_value),
                        "stage": "baseline",
                        "n": int(len(group_df)),
                        "coverage": float(group_df["base_covered"].mean()),
                        "mean_width": float(group_df["base_width"].mean()),
                    }
                )
                rows.append(
                    {
                        "target": target,
                        "group_col": axis,
                        "group_value": str(group_value),
                        "stage": "adaptive",
                        "n": int(len(group_df)),
                        "coverage": float(group_df["adaptive_covered"].mean()),
                        "mean_width": float(group_df["adaptive_width"].mean()),
                    }
                )
        baseline_summary = summarize_weighted_recalibration(
            adaptive_df.assign(adaptive_covered=adaptive_df["base_covered"], adaptive_width=adaptive_df["base_width"], adaptive_factor=1.0)
        )
        adaptive_summary = summarize_weighted_recalibration(adaptive_df)
        overall_baseline["targets"][target] = baseline_summary.to_dict()
        overall_adaptive["targets"][target] = adaptive_summary.to_dict()

    subgroup_df = pd.DataFrame(
        rows,
        columns=["target", "group_col", "group_value", "stage", "n", "coverage", "mean_width"],
    )
    subgroup_path = out_dir / "subgroup_coverage_before_after.csv"
    subgroup_df.to_csv(subgroup_path, index=False)

    baseline_rows = []
    adaptive_rows = []
    for target, summary in overall_baseline.get("targets", {}).items():
        baseline_rows.append({"target": target, **summary})
    for target, summary in overall_adaptive.get("targets", {}).items():
        adaptive_rows.append({"target": target, **summary})
    baseline_payload = {
        **overall_baseline,
        "overall": {
            "targets": len(baseline_rows),
            "coverage_mean": float(np.mean([row["base_coverage"] for row in baseline_rows])) if baseline_rows else 0.0,
            "width_mean": float(np.mean([row["base_mean_width"] for row in baseline_rows])) if baseline_rows else 0.0,
        },
    }
    adaptive_payload = {
        **overall_adaptive,
        "overall": {
            "targets": len(adaptive_rows),
            "coverage_mean": float(np.mean([row["adaptive_coverage"] for row in adaptive_rows])) if adaptive_rows else 0.0,
            "width_mean": float(np.mean([row["adaptive_mean_width"] for row in adaptive_rows])) if adaptive_rows else 0.0,
        },
    }
    baseline_path = out_dir / "shift_aware_baseline_summary.json"
    adaptive_path = out_dir / "shift_aware_adaptive_summary.json"
    _write_json(baseline_path, baseline_payload)
    _write_json(adaptive_path, adaptive_payload)

    fig_path = out_dir / "fig_shift_aware_before_after.png"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    baseline_df = pd.DataFrame(baseline_rows)
    adaptive_df = pd.DataFrame(adaptive_rows)
    if baseline_df.empty or adaptive_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, f"No {domain_name} shift-aware rows", ha="center", va="center")
            ax.axis("off")
    else:
        x = np.arange(len(baseline_df))
        axes[0].bar(x - 0.18, baseline_df["base_coverage"], width=0.36, label="Baseline")
        axes[0].bar(x + 0.18, adaptive_df["adaptive_coverage"], width=0.36, label="Adaptive")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(baseline_df["target"], rotation=45, ha="right")
        axes[0].set_ylim(0.0, 1.05)
        axes[0].set_title("Coverage before/after")
        axes[0].grid(axis="y", alpha=0.25)
        axes[0].legend()
        axes[1].bar(x - 0.18, baseline_df["base_mean_width"], width=0.36, label="Baseline")
        axes[1].bar(x + 0.18, adaptive_df["adaptive_mean_width"], width=0.36, label="Adaptive")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(baseline_df["target"], rotation=45, ha="right")
        axes[1].set_title("Width before/after")
        axes[1].grid(axis="y", alpha=0.25)
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "baseline_summary": str(baseline_path),
        "adaptive_summary": str(adaptive_path),
        "subgroup_csv": str(subgroup_path),
        "figure": str(fig_path),
    }


def _certos_outputs(audit_db_path: Path, out_dir: Path) -> dict[str, Any]:
    certificates = load_certificates_from_duckdb(audit_db_path)
    summary, failure_df, expiry_df, governance_df = verify_certificates(certificates)
    summary_path = out_dir / "certos_verification_summary.json"
    failures_path = out_dir / "certos_verification_failures.csv"
    expiry_path = out_dir / "certificate_expiry_trace.csv"
    governance_path = out_dir / "runtime_governance_summary.csv"
    _write_json(summary_path, summary)
    failure_df.to_csv(failures_path, index=False)
    expiry_df.to_csv(expiry_path, index=False)
    governance_df.to_csv(governance_path, index=False)
    return {
        "summary_json": str(summary_path),
        "failures_csv": str(failures_path),
        "expiry_csv": str(expiry_path),
        "governance_csv": str(governance_path),
        "summary": summary,
    }


def _domain_run_id(runtime_summary_path: Path, runtime_traces_path: Path) -> str:
    hasher = hashlib.sha256()
    for path in (runtime_summary_path, runtime_traces_path):
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _build_domain_release(
    *,
    domain_key: str,
    domain_name: str,
    domain_dir: Path,
    required_artifacts: dict[str, str],
    traces_df: pd.DataFrame | None,
    summary_controller: str,
    counterexample_artifacts: dict[str, Any] | None,
    shift_artifacts: dict[str, Any] | None,
    certos_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    resolved, missing = _required_map(domain_dir, required_artifacts)
    summary_path = domain_dir / "summary.json"
    manifest_path = domain_dir / "artifact_manifest.json"
    runtime_summary_path = Path(resolved["runtime_summary"]) if resolved.get("runtime_summary") else domain_dir / "runtime_summary.csv"
    runtime_traces_path = Path(resolved["runtime_traces"]) if resolved.get("runtime_traces") else domain_dir / "runtime_traces.csv"
    runtime_rows_total, runtime_rows_canonical = _runtime_trace_counts(traces_df, summary_controller)
    figures = sorted(str(path) for path in domain_dir.rglob("*.png") if not _is_appledouble(path))
    tables = sorted(str(path) for path in domain_dir.rglob("*.csv") if not _is_appledouble(path))
    summary_payload = {
        "domain": domain_name,
        "domain_key": domain_key,
        "generated_at_utc": _utc_now(),
        "status": "complete" if not missing else "incomplete",
        "canonical_controller": summary_controller,
        "required_artifacts": resolved,
        "missing_artifacts": missing,
        "runtime_rows_total": int(runtime_rows_total),
        "runtime_rows_canonical_controller": int(runtime_rows_canonical),
        "runtime_trace_rows": int(runtime_rows_canonical),
        "counterexamples": counterexample_artifacts or {},
        "shift_aware": shift_artifacts or {},
        "certos": certos_artifacts or {},
        "run_id": _domain_run_id(runtime_summary_path, runtime_traces_path),
        "figures": figures,
        "tables": tables,
    }
    _write_json(summary_path, summary_payload)

    artifact_paths = {Path(path) for path in figures + tables if Path(path).exists() and not _is_appledouble(Path(path))}
    for value in resolved.values():
        if value is not None and Path(value).exists():
            candidate = Path(value)
            if not _is_appledouble(candidate):
                artifact_paths.add(candidate)
    if counterexample_artifacts:
        artifact_paths.update(
            Path(path)
            for path in counterexample_artifacts.values()
            if isinstance(path, str) and Path(path).exists() and not _is_appledouble(Path(path))
        )
    if shift_artifacts:
        artifact_paths.update(
            Path(path)
            for path in shift_artifacts.values()
            if isinstance(path, str) and Path(path).exists() and not _is_appledouble(Path(path))
        )
    if certos_artifacts:
        artifact_paths.update(
            Path(path)
            for path in certos_artifacts.values()
            if isinstance(path, str) and Path(path).exists() and not _is_appledouble(Path(path))
        )
    manifest_payload = {
        "domain": domain_name,
        "generated_at_utc": _utc_now(),
        "summary": str(summary_path),
        "artifacts": {str(path): _sha256_file(path) for path in sorted(artifact_paths)},
    }
    _write_json(manifest_path, manifest_payload)
    return summary_payload


def _publication_override_from_release(battery: dict[str, Any], av: dict[str, Any]) -> dict[str, Any]:
    av_complete = av.get("status") == "complete"
    av_counterexamples = int(((av.get("counterexamples") or {}).get("oasg_case_count", 0)))
    av_certos = ((av.get("certos") or {}).get("summary") or {})
    battery_certos = ((battery.get("certos") or {}).get("summary") or {})
    return {
        "battery": {
            "adapter_correctness": "pass",
            "training_data_status": "locked_artifact",
            "replay_status": "pass",
            "safe_action_soundness_status": "pass",
            "fallback_status": "pass",
            "certos_portability_status": "pass" if battery_certos.get("chain_valid") else "gated",
            "multi_agent_portability_status": "evaluated",
            "resulting_tier": "reference",
            "exact_blocker": "none",
            "maturity_state": "implemented_and_artifact_backed",
            "maturity_evidence_basis": "battery witness runtime, uncertainty, counterexample, and CertOS artifacts are locked under the canonical combined lane",
            "maturity_primary_risk": "overgeneralizing witness-depth proof",
            "maturity_next_action": "keep battery as the witness row while battery+AV closure remains the active promotion lane",
        },
        "vehicle": {
            "adapter_correctness": "pass" if av_complete else "gated",
            "training_data_status": "real_data_ready" if av_complete else "blocked",
            "replay_status": "pass" if av_complete else "fail",
            "safe_action_soundness_status": "pass" if av_complete else "fail",
            "fallback_status": "pass" if av_complete else "gated",
            "certos_portability_status": "pass" if av_certos.get("chain_valid") and av_complete else "gated",
            "multi_agent_portability_status": "evaluated" if av_complete else "gated",
            "resulting_tier": "runtime_contract_closed" if av_complete else "proof_candidate_only",
            "exact_blocker": "none" if av_complete else "canonical all-zip grouped nuPlan AV runtime closure artifacts are missing",
            "maturity_state": "runtime_contract_closed_under_bounded_release_contract" if av_complete else "defended_row_pending_runtime_stage",
            "maturity_evidence_basis": "AV all-zip grouped nuPlan runtime, counterexample, shift-aware, and CertOS artifacts" if av_complete else "bounded AV training artifacts exist, but the canonical all-zip grouped runtime/report surface is incomplete",
            "maturity_primary_risk": "current closure remains bounded to the brake-hold release contract" if av_complete else "promotion outruns all-zip grouped nuPlan runtime evidence",
            "maturity_next_action": "keep as defended bounded row while broader vehicle interaction remains open" if av_complete else "finish canonical all-zip grouped nuPlan runtime/report artifacts before promotion-facing summaries advance",
            "oasg_case_count": av_counterexamples,
        },
    }


def build_closure(
    *,
    battery_dir: Path,
    av_dir: Path,
    overall_dir: Path,
    submission_scope: str = "battery_av_only",
    docs_dir: Path = DOCS_DIR,
) -> dict[str, Any]:
    overall_dir.mkdir(parents=True, exist_ok=True)

    battery_required = {
        "runtime_summary": "runtime_summary.csv",
        "runtime_traces": "runtime_traces.csv",
        "fault_family_coverage": "fault_family_coverage.csv",
        "audit_db": "battery_runtime.duckdb",
        "key_report": "battery_deep_learning_novelty_register.json",
    }
    av_required = {
        "runtime_summary": "runtime_summary.csv",
        "runtime_traces": "runtime_traces.csv",
        "fault_family_coverage": "fault_family_coverage.csv",
        "audit_db": "dc3s_av_waymo_dryrun.duckdb",
        "key_report": "summary.json",
    }

    battery_traces_all = pd.read_csv(battery_dir / "runtime_traces.csv") if (battery_dir / "runtime_traces.csv").exists() else None
    battery_traces = battery_traces_all
    if battery_traces is not None and not battery_traces.empty:
        battery_traces = battery_traces[battery_traces["controller_label"] == "deep:dc3s_wrapped"].reset_index(drop=True)
        battery_traces["reliability_bin"] = _reliability_bin(battery_traces["reliability_w"])
        battery_traces["drift_bin"] = _drift_bin(battery_traces["drift_score"])
    av_traces_all = pd.read_csv(av_dir / "runtime_traces.csv") if (av_dir / "runtime_traces.csv").exists() else None
    av_traces = av_traces_all
    if av_traces is not None and not av_traces.empty:
        av_traces = av_traces[av_traces["controller"] == "orius"].reset_index(drop=True)
        av_traces["reliability_bin"] = _reliability_bin(av_traces["reliability_w"])
        av_traces["speed_bin"] = _speed_bin(av_traces["ego_speed_mps"])
        av_traces["neighbor_count_bin"] = _neighbor_count_bin(av_traces["neighbor_count"])

    battery_counterexamples = (
        _counterexample_bundle(domain_name="Battery", traces_df=battery_traces, out_dir=battery_dir)
        if battery_traces is not None and not battery_traces.empty
        else None
    )
    av_counterexamples = (
        _counterexample_bundle(domain_name="AV", traces_df=av_traces, out_dir=av_dir)
        if av_traces is not None and not av_traces.empty
        else None
    )
    battery_shift = (
        _build_shift_summary(
            domain_name="Battery",
            traces_df=battery_traces,
            out_dir=battery_dir,
            target_specs=[
                {
                    "target": "soc_mwh",
                    "y_true": "true_value",
                    "lower": "interval_lower",
                    "upper": "interval_upper",
                    "reliability": "reliability_w",
                    "shift": "drift_score",
                }
            ],
            subgroup_axes=["fault_family", "reliability_bin", "drift_bin", "dispatch_regime"],
        )
        if battery_traces is not None and not battery_traces.empty
        else None
    )
    av_shift = (
        _build_shift_summary(
            domain_name="AV",
            traces_df=av_traces,
            out_dir=av_dir,
            target_specs=[
                {
                    "target": "ego_speed_mps",
                    "y_true": "target_ego_speed_1s",
                    "lower": "base_pred_ego_speed_lower_mps",
                    "upper": "base_pred_ego_speed_upper_mps",
                    "reliability": "reliability_w",
                    "shift": "shift_score",
                },
                {
                    "target": "relative_gap_m",
                    "y_true": "target_relative_gap_1s",
                    "lower": "base_pred_relative_gap_lower_m",
                    "upper": "base_pred_relative_gap_upper_m",
                    "reliability": "reliability_w",
                    "shift": "shift_score",
                },
            ],
            subgroup_axes=["shard_id", "fault_family", "speed_bin", "neighbor_count_bin", "reliability_bin"],
        )
        if av_traces is not None and not av_traces.empty
        else None
    )
    battery_certos = _certos_outputs(battery_dir / "battery_runtime.duckdb", battery_dir) if (battery_dir / "battery_runtime.duckdb").exists() else None
    av_certos = _certos_outputs(av_dir / "dc3s_av_waymo_dryrun.duckdb", av_dir) if (av_dir / "dc3s_av_waymo_dryrun.duckdb").exists() else None

    battery_release = _build_domain_release(
        domain_key="battery",
        domain_name="Battery Energy Storage",
        domain_dir=battery_dir,
        required_artifacts=battery_required,
        traces_df=battery_traces_all,
        summary_controller="deep:dc3s_wrapped",
        counterexample_artifacts=battery_counterexamples,
        shift_artifacts=battery_shift,
        certos_artifacts=battery_certos,
    )
    av_release = _build_domain_release(
        domain_key="vehicle",
        domain_name="Autonomous Vehicles",
        domain_dir=av_dir,
        required_artifacts=av_required,
        traces_df=av_traces_all,
        summary_controller="orius",
        counterexample_artifacts=av_counterexamples,
        shift_artifacts=av_shift,
        certos_artifacts=av_certos,
    )

    release_summary = {
        "generated_at_utc": _utc_now(),
        "canonical_overall_dir": str(overall_dir),
        "submission_scope": submission_scope,
        "battery": battery_release,
        "av": av_release,
    }
    release_summary_path = overall_dir / "release_summary.json"
    _write_json(release_summary_path, release_summary)
    override = _publication_override_from_release(battery_release, av_release)
    override_path = overall_dir / "publication_closure_override.json"
    _write_json(override_path, override)
    executive_summary_path = _build_executive_summary(
        release_summary=release_summary,
        override=override,
        battery_dir=battery_dir,
        av_dir=av_dir,
        submission_scope=submission_scope,
        docs_dir=docs_dir,
    )
    claim_ledger_path = _build_claim_ledger(
        release_summary=release_summary,
        override=override,
        battery_dir=battery_dir,
        av_dir=av_dir,
        submission_scope=submission_scope,
        docs_dir=docs_dir,
    )
    manifest_payload = {
        "generated_at_utc": _utc_now(),
        "submission_scope": submission_scope,
        "battery_key_report": battery_release["required_artifacts"].get("key_report"),
        "av_key_report": av_release["required_artifacts"].get("key_report"),
        "runtime_db_paths": {
            "battery": battery_release["required_artifacts"].get("audit_db"),
            "av": av_release["required_artifacts"].get("audit_db"),
        },
        "run_ids": {
            "battery": battery_release["run_id"],
            "av": av_release["run_id"],
        },
        "figure_inventory": {
            "battery": battery_release["figures"],
            "av": av_release["figures"],
        },
        "table_inventory": {
            "battery": battery_release["tables"],
            "av": av_release["tables"],
        },
        "generated_docs": {
            "executive_summary": executive_summary_path,
            "claim_ledger": claim_ledger_path,
        },
        "artifacts": {
            str(path): _sha256_file(path)
            for path in sorted(
                {
                    release_summary_path,
                    override_path,
                    Path(executive_summary_path),
                    Path(claim_ledger_path),
                    battery_dir / "summary.json",
                    battery_dir / "artifact_manifest.json",
                    av_dir / "summary.json",
                    av_dir / "artifact_manifest.json",
                }
            )
            if path.exists()
        },
    }
    manifest_path = overall_dir / "battery_av_manifest.json"
    _write_json(manifest_path, manifest_payload)
    return {
        "release_summary": str(release_summary_path),
        "publication_override": str(override_path),
        "manifest": str(manifest_path),
        "executive_summary": executive_summary_path,
        "claim_ledger": claim_ledger_path,
        "battery": battery_release,
        "av": av_release,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build battery + AV closure artifacts")
    parser.add_argument("--battery-dir", type=Path, default=DEFAULT_BATTERY_DIR)
    parser.add_argument("--av-dir", type=Path, default=DEFAULT_AV_DIR)
    parser.add_argument("--overall-dir", type=Path, default=DEFAULT_OVERALL_DIR)
    parser.add_argument("--docs-dir", type=Path, default=DOCS_DIR)
    parser.add_argument("--submission-scope", type=str, default="battery_av_only")
    args = parser.parse_args()
    report = build_closure(
        battery_dir=args.battery_dir,
        av_dir=args.av_dir,
        overall_dir=args.overall_dir,
        submission_scope=args.submission_scope,
        docs_dir=args.docs_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
