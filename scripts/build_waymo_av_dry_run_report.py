#!/usr/bin/env python3
"""Build tables, figures, and a manifest for a Waymo AV dry run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridpulse")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROCESSED = REPO_ROOT / "data" / "orius_av" / "av" / "processed_full_1k"
DEFAULT_REPORTS = REPO_ROOT / "reports" / "orius_av" / "full_dry_run"
DEFAULT_MODELS = REPO_ROOT / "artifacts" / "models_orius_av_full_1k"
DEFAULT_UNCERTAINTY = REPO_ROOT / "artifacts" / "uncertainty" / "orius_av_full_1k"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _make_runtime_figure(runtime_df: pd.DataFrame, out_path: Path) -> str:
    metric_cols = ["tsvr", "oasg", "cva", "gdq", "intervention_rate", "audit_completeness"]
    plot_df = runtime_df.set_index("controller")[metric_cols].transpose()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("Waymo AV Dry Run Runtime Metrics")
    ax.set_ylabel("Metric value")
    ax.set_xlabel("Metric")
    ax.set_ylim(0.0, max(1.05, float(plot_df.to_numpy().max()) * 1.1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Controller")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _make_training_width_figure(training_df: pd.DataFrame, out_path: Path) -> str:
    plot_df = training_df.copy()
    plot_df["task"] = plot_df["target"] + "@" + plot_df["horizon_label"] + ":" + plot_df["split"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plot_df))
    ax.bar(x, plot_df["base_mean_width"], label="Base width", alpha=0.75)
    ax.bar(x, plot_df["widened_mean_width"], label="Widened width", alpha=0.55)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["task"], rotation=45, ha="right")
    ax.set_title("Conformal Interval Widths")
    ax.set_ylabel("Mean interval width")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _make_fault_coverage_figure(coverage_df: pd.DataFrame, out_path: Path) -> str:
    if coverage_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No fault-family coverage rows", ha="center", va="center")
        ax.axis("off")
    else:
        plot_df = coverage_df.copy()
        if "controller" in plot_df.columns:
            plot_df["series"] = plot_df["controller"].astype(str) + ":" + plot_df["target"].astype(str)
            legend_title = "Controller:target"
            pivot = plot_df.pivot(index="fault_family", columns="series", values="coverage").fillna(0.0)
        else:
            legend_title = "Target"
            pivot = plot_df.pivot(index="fault_family", columns="target", values="coverage").fillna(0.0)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        pivot.plot(kind="bar", ax=ax)
        ax.set_title("Fault-Family Coverage")
        ax.set_ylabel("Coverage")
        ax.set_xlabel("Fault family")
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title=legend_title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _make_shift_aware_figure(shift_df: pd.DataFrame, out_path: Path) -> str:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    if shift_df.empty:
        ax.text(0.5, 0.5, "No shift-aware runtime summary", ha="center", va="center")
        ax.axis("off")
    else:
        x = range(len(shift_df))
        ax.bar(x, shift_df["final_validity_score"], label="Final validity score", alpha=0.8)
        ax.plot(x, shift_df["shift_alert_rate"], marker="o", label="Shift alert rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels(shift_df["target"], rotation=0)
        ax.set_ylim(0.0, 1.05)
        ax.set_title("Shift-Aware Runtime Summary")
        ax.set_ylabel("Score / rate")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def _split_counts(anchor_features_path: Path) -> pd.DataFrame:
    anchor_df = pd.read_parquet(anchor_features_path)
    counts = anchor_df.groupby("split").size().rename("scenario_count").reset_index()
    return counts.sort_values("split").reset_index(drop=True)


def _summary_payload(
    *,
    runtime_df: pd.DataFrame,
    training_df: pd.DataFrame,
    split_counts_df: pd.DataFrame,
    runtime_trace_count: int,
    raw_file_hashes: dict[str, str],
) -> dict[str, Any]:
    best_runtime = runtime_df.set_index("controller").to_dict(orient="index")
    return {
        "runtime": best_runtime,
        "training_rows": int(len(training_df)),
        "runtime_trace_rows": int(runtime_trace_count),
        "split_counts": {
            str(row["split"]): int(row["scenario_count"])
            for row in split_counts_df.to_dict(orient="records")
        },
        "mean_widening_factor": float(training_df["mean_widening_factor"].mean()),
        "max_widened_coverage": float(training_df["widened_coverage"].max()),
        "raw_file_count": int(len(raw_file_hashes)),
        "raw_file_hashes": dict(raw_file_hashes),
    }


def build_report(
    *,
    processed_dir: Path,
    reports_dir: Path,
    models_dir: Path,
    uncertainty_dir: Path,
) -> dict[str, Any]:
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    training_summary_path = reports_dir / "training_summary.csv"
    subgroup_coverage_path = reports_dir / "subgroup_coverage.csv"
    runtime_summary_path = reports_dir / "runtime_summary.csv"
    runtime_traces_path = reports_dir / "runtime_traces.csv"
    fault_coverage_path = reports_dir / "fault_family_coverage.csv"
    shift_runtime_summary_path = reports_dir / "shift_aware" / "shift_aware_runtime_summary.csv"
    anchor_features_path = processed_dir / "anchor_features.parquet"
    subset_manifest_path = processed_dir / "dry_run_subset_manifest.json"

    training_df = _load_csv(training_summary_path)
    subgroup_df = _load_csv(subgroup_coverage_path)
    runtime_df = _load_csv(runtime_summary_path)
    fault_df = _load_csv(fault_coverage_path)
    shift_df = pd.read_csv(shift_runtime_summary_path) if shift_runtime_summary_path.exists() else pd.DataFrame()
    split_counts_df = _split_counts(anchor_features_path)
    runtime_trace_count = 0
    if runtime_traces_path.exists():
        runtime_trace_count = int(sum(1 for _ in runtime_traces_path.open("r", encoding="utf-8")) - 1)
    subset_manifest = json.loads(subset_manifest_path.read_text(encoding="utf-8")) if subset_manifest_path.exists() else {}
    raw_file_hashes = dict(subset_manifest.get("raw_file_hashes", {})) if isinstance(subset_manifest, dict) else {}

    table_artifacts = {
        "training_summary": _write_table(training_df, tables_dir / "training_summary_table.csv"),
        "runtime_summary": _write_table(runtime_df, tables_dir / "runtime_summary_table.csv"),
        "fault_family_coverage": _write_table(fault_df, tables_dir / "fault_family_coverage_table.csv"),
        "subgroup_coverage": _write_table(subgroup_df, tables_dir / "subgroup_coverage_table.csv"),
        "split_counts": _write_table(split_counts_df, tables_dir / "split_counts_table.csv"),
    }

    figure_artifacts = {
        "runtime_metrics": _make_runtime_figure(runtime_df, figures_dir / "runtime_metrics.png"),
        "training_widths": _make_training_width_figure(training_df, figures_dir / "training_interval_widths.png"),
        "fault_family_coverage": _make_fault_coverage_figure(fault_df, figures_dir / "fault_family_coverage.png"),
    }
    if shift_runtime_summary_path.exists():
        table_artifacts["shift_aware_runtime"] = _write_table(shift_df, tables_dir / "shift_aware_runtime_table.csv")
        figure_artifacts["shift_aware_runtime"] = _make_shift_aware_figure(shift_df, figures_dir / "shift_aware_runtime.png")

    summary = _summary_payload(
        runtime_df=runtime_df,
        training_df=training_df,
        split_counts_df=split_counts_df,
        runtime_trace_count=runtime_trace_count,
        raw_file_hashes=raw_file_hashes,
    )
    summary_path = reports_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest_paths = [
        training_summary_path,
        subgroup_coverage_path,
        runtime_summary_path,
        runtime_traces_path,
        fault_coverage_path,
        anchor_features_path,
        subset_manifest_path,
        *[Path(path) for path in table_artifacts.values()],
        *[Path(path) for path in figure_artifacts.values()],
    ]
    manifest_paths.extend(sorted(models_dir.glob("*.pkl")))
    manifest_paths.extend(sorted(uncertainty_dir.glob("*.json")))

    manifest = {
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir),
        "models_dir": str(models_dir),
        "uncertainty_dir": str(uncertainty_dir),
        "tables": table_artifacts,
        "figures": figure_artifacts,
        "summary": str(summary_path),
        "raw_file_hashes": raw_file_hashes,
        "artifacts": {
            str(path): _sha256_file(path)
            for path in manifest_paths
            if path.exists() and path.is_file()
        },
    }
    manifest_path = reports_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["artifact_manifest"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build figures/tables for a Waymo AV dry run")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--uncertainty-dir", type=Path, default=DEFAULT_UNCERTAINTY)
    args = parser.parse_args()

    manifest = build_report(
        processed_dir=args.processed_dir,
        reports_dir=args.reports_dir,
        models_dir=args.models_dir,
        uncertainty_dir=args.uncertainty_dir,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
