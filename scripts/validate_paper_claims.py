#!/usr/bin/env python3
"""Validate manuscript claims against paper/metrics_manifest.json.

This checker is intentionally non-mutating. It validates:
1. Required metric/run-id patterns exist in markdown and LaTeX.
2. Banned legacy patterns are absent.
3. Placeholder tokens are absent.
4. LaTeX \\input references resolve.
5. Run IDs used in manuscript match allowed dataset-scoped run IDs.
6. claim_matrix status values are valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ALLOWED_CLAIM_STATUSES = {
    "Verified",
    "Historical",
    "Inactive",
    "Conflicting",
    "Unsupported",
    "Needs Citation",
}
ACTIVE_CLAIM_STATUSES = {"Verified", "Conflicting", "Unsupported", "Needs Citation"}
RUN_ID_RE = re.compile(r"\b20\d{6}_\d{6}\b")

# Verified claims with canonical_value must match metrics_manifest (the lock).
# When claim_matrix is intentionally changed, validator fails.
CLAIM_TO_MANIFEST_PATH: dict[str, str] = {
    "C001": "claim_family.schema_version",
    "C002": "metric_policy.master_manuscript",
    "C003": "universal_claims.dataset_profiles.de_rows",
    "C004": "universal_claims.dataset_profiles.us_rows",
    "C005": "universal_claims.dataset_profiles.de_feature_count",
    "C006": "universal_claims.dataset_profiles.us_feature_count",
    "C007": "universal_claims.benchmark_summary.nominal_dc3s_ftit_tsvr",
    "C008": "universal_claims.benchmark_summary.full_dc3s_step_p95_ms",
    "C009": "universal_claims.artifact_provenance.release_manifest_frozen_at_utc",
}

CANONICAL_ORIUS_FRAMING = (
    "ORIUS provides a reliability-aware runtime safety layer for physical AI under "
    "degraded observation, enforcing certificate-backed action release through "
    "uncertainty coverage, repair, and fallback."
)

CANONICAL_FRAMING_SURFACES = (
    "README.md",
    "docs/claim_ledger.md",
    "docs/executive_summary.md",
    "reports/publication/README.md",
    "reports/publication/top_venue_research_package.md",
    "paper/review/orius_review_dossier.tex",
    "paper/monograph/ch01_introduction_and_thesis_claims.tex",
    "paper/monograph/ch02_related_work_and_novelty_gap.tex",
)

FORBIDDEN_CANONICAL_FRAMING_DRIFT = (
    "ORIUS identifies OASG as the degraded-observation release hazard and provides a reliability-aware runtime safety layer across Battery, AV, and Healthcare.",
    "ORIUS identifies degraded observation as a physical-AI release hazard and provides a reliability-aware runtime safety layer that enforces certificate-backed action release across three bounded domains.",
    "ORIUS is a universal runtime safety layer for Physical AI under degraded observation",
    "ORIUS is a universal runtime safety layer for physical AI under degraded observation",
)


@dataclass
class Finding:
    severity: str  # ERROR or WARN
    check: str
    file: str
    detail: str


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing required JSON file: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SystemExit(f"Missing required text file: {path}") from None


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _check_canonical_project_framing(findings: list[Finding], repo_root: Path) -> None:
    expected = _normalize_space(CANONICAL_ORIUS_FRAMING)
    for rel_path in CANONICAL_FRAMING_SURFACES:
        path = repo_root / rel_path
        if not path.exists():
            findings.append(Finding("ERROR", "canonical_framing", rel_path, "Missing framing surface"))
            continue
        text = _normalize_space(path.read_text(encoding="utf-8"))
        if expected not in text:
            findings.append(
                Finding("ERROR", "canonical_framing", rel_path, "Missing canonical ORIUS framing")
            )
        for legacy in FORBIDDEN_CANONICAL_FRAMING_DRIFT:
            if _normalize_space(legacy) in text:
                findings.append(
                    Finding("ERROR", "canonical_framing", rel_path, f"Legacy framing remains: {legacy}")
                )


def _check_patterns(
    findings: list[Finding],
    pattern_specs: Iterable[dict],
    markdown_text: str,
    tex_text: str,
    markdown_path: Path,
    tex_path: Path,
    required: bool,
) -> None:
    for spec in pattern_specs:
        pid = spec.get("id", "unnamed")
        regex = spec.get("regex")
        files = spec.get("files", ["markdown", "tex"]) if required else ["markdown", "tex"]
        if not regex:
            findings.append(
                Finding("ERROR", "manifest_pattern", "manifest", f"Pattern '{pid}' missing regex")
            )
            continue

        try:
            compiled = re.compile(regex)
        except re.error as exc:
            findings.append(
                Finding("ERROR", "manifest_pattern", "manifest", f"Invalid regex for '{pid}': {exc}")
            )
            continue

        targets = []
        if "markdown" in files:
            targets.append((markdown_path, markdown_text))
        if "tex" in files:
            targets.append((tex_path, tex_text))

        for path, text in targets:
            matched = compiled.search(text) is not None
            if required and not matched:
                findings.append(
                    Finding(
                        "ERROR", "required_pattern", str(path), f"Missing required pattern '{pid}' /{regex}/"
                    )
                )
            if not required and matched:
                reason = spec.get("reason", "Banned pattern detected")
                findings.append(
                    Finding(
                        "ERROR", "banned_pattern", str(path), f"Pattern '{pid}' matched /{regex}/ ({reason})"
                    )
                )


def _check_placeholders(
    findings: list[Finding],
    placeholder_patterns: Iterable[str],
    markdown_text: str,
    tex_text: str,
    markdown_path: Path,
    tex_path: Path,
) -> None:
    for patt in placeholder_patterns:
        try:
            compiled = re.compile(patt)
        except re.error as exc:
            findings.append(
                Finding(
                    "ERROR", "placeholder_pattern", "manifest", f"Invalid placeholder regex '{patt}': {exc}"
                )
            )
            continue
        for path, text in ((markdown_path, markdown_text), (tex_path, tex_text)):
            if compiled.search(text):
                findings.append(
                    Finding("ERROR", "placeholder", str(path), f"Placeholder pattern found: /{patt}/")
                )


def _check_run_ids(
    findings: list[Finding],
    expected_run_ids: set[str],
    markdown_text: str,
    tex_text: str,
    markdown_path: Path,
    tex_path: Path,
) -> None:
    for path, text in ((markdown_path, markdown_text), (tex_path, tex_text)):
        # LaTeX escapes underscores as "\\_", so normalize before run-id extraction.
        normalized = text.replace("\\_", "_")
        found = set(RUN_ID_RE.findall(normalized))
        unknown = sorted(found - expected_run_ids)
        missing = sorted(expected_run_ids - found)
        if unknown:
            findings.append(
                Finding("ERROR", "run_id_scope", str(path), f"Unknown run IDs present: {', '.join(unknown)}")
            )
        if missing:
            findings.append(
                Finding(
                    "WARN", "run_id_scope", str(path), f"Expected run IDs not found: {', '.join(missing)}"
                )
            )


def _check_tex_inputs(findings: list[Finding], tex_text: str, tex_path: Path, repo_root: Path) -> None:
    inputs = re.findall(r"\\input\{([^}]+)\}", tex_text)
    tex_dir = tex_path.parent
    for ref in inputs:
        candidates = [
            (tex_dir / ref).resolve(),
            (repo_root / ref).resolve(),
        ]
        if not any(path.exists() for path in candidates):
            findings.append(Finding("ERROR", "latex_input", str(tex_path), f"Missing \\input target: {ref}"))


def _check_title_alignment(
    findings: list[Finding], markdown_text: str, tex_text: str, markdown_path: Path, tex_path: Path
) -> None:
    md_title_match = re.search(r"^#\s+(.+)$", markdown_text, flags=re.MULTILINE)
    tex_title_match = re.search(r"\\title\{(.+?)\}", tex_text, flags=re.DOTALL)

    if not md_title_match:
        findings.append(
            Finding("WARN", "title_alignment", str(markdown_path), "Could not find markdown title")
        )
        return
    if not tex_title_match:
        findings.append(Finding("WARN", "title_alignment", str(tex_path), "Could not find LaTeX title"))
        return

    md_title = _normalize_space(md_title_match.group(1).replace("\\", " "))
    tex_title = _normalize_space(tex_title_match.group(1).replace("\\", " "))

    md_norm = re.sub(r"[^a-z0-9]+", "", md_title.lower())
    tex_norm = re.sub(r"[^a-z0-9]+", "", tex_title.lower())

    if md_norm != tex_norm:
        findings.append(
            Finding(
                "WARN",
                "title_alignment",
                f"{markdown_path} <-> {tex_path}",
                "Title mismatch between markdown and LaTeX",
            )
        )


def _get_nested(obj: dict, path: str) -> object:
    for key in path.split("."):
        obj = obj.get(key) if isinstance(obj, dict) else None
        if obj is None:
            return None
    return obj


def _normalize_canonical(a: object, b: object) -> tuple[str, str]:
    """Normalize for comparison; handles int, float, and formatted strings."""
    sa = str(a).strip() if a is not None else ""
    sb = str(b).strip() if b is not None else ""
    # Remove extra spaces in comma-separated numbers for consistency
    sa = "".join(sa.split())
    sb = "".join(sb.split())
    return sa, sb


def _check_canonical_value_lock(
    findings: list[Finding],
    manifest: dict,
    claim_matrix_path: Path,
    repo_root: Path,
) -> None:
    """Validate that Verified claims' canonical_value matches metrics_manifest (the lock).
    Proves validator fails when a locked claim is intentionally changed."""
    if not claim_matrix_path.exists():
        return

    with claim_matrix_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    for row in rows:
        cid = (row.get("claim_id") or "").strip()
        status = (row.get("status") or "").strip()
        canonical_value = (row.get("canonical_value") or "").strip()
        if status != "Verified" or not canonical_value or cid not in CLAIM_TO_MANIFEST_PATH:
            continue

        manifest_path = CLAIM_TO_MANIFEST_PATH[cid]
        manifest_val = _get_nested(manifest, manifest_path)
        if manifest_val is None:
            continue

        norm_claim, norm_manifest = _normalize_canonical(canonical_value, manifest_val)
        if norm_claim != norm_manifest:
            findings.append(
                Finding(
                    "ERROR",
                    "canonical_value_lock",
                    str(claim_matrix_path),
                    f"Claim {cid} canonical_value '{canonical_value}' does not match locked manifest value '{manifest_val}' (path: {manifest_path})",
                )
            )


def _check_claim_matrix(findings: list[Finding], claim_matrix_path: Path) -> None:
    if not claim_matrix_path.exists():
        findings.append(Finding("ERROR", "claim_matrix", str(claim_matrix_path), "Missing claim matrix CSV"))
        return

    with claim_matrix_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        findings.append(Finding("ERROR", "claim_matrix", str(claim_matrix_path), "Claim matrix is empty"))
        return

    required_cols = {
        "claim_id",
        "status",
        "category",
        "manuscript_locations",
        "claim_text",
        "source_file",
    }
    missing_cols = required_cols - set(reader.fieldnames or [])
    if missing_cols:
        findings.append(
            Finding(
                "ERROR",
                "claim_matrix",
                str(claim_matrix_path),
                f"Missing columns: {', '.join(sorted(missing_cols))}",
            )
        )

    seen_ids: set[str] = set()
    for row in rows:
        cid = (row.get("claim_id") or "").strip()
        status = (row.get("status") or "").strip()
        manuscript_locations = (row.get("manuscript_locations") or "").strip()
        if not cid:
            findings.append(
                Finding("ERROR", "claim_matrix", str(claim_matrix_path), "Found row without claim_id")
            )
            continue
        if cid in seen_ids:
            findings.append(
                Finding("ERROR", "claim_matrix", str(claim_matrix_path), f"Duplicate claim_id: {cid}")
            )
        seen_ids.add(cid)
        if status not in ALLOWED_CLAIM_STATUSES:
            findings.append(
                Finding(
                    "ERROR", "claim_matrix", str(claim_matrix_path), f"Invalid status '{status}' for {cid}"
                )
            )
            continue

        is_not_present = manuscript_locations.upper() == "NOT PRESENT"
        is_historical = manuscript_locations.startswith("historical_")

        if status == "Verified" and (is_not_present or is_historical):
            findings.append(
                Finding(
                    "ERROR",
                    "claim_matrix",
                    str(claim_matrix_path),
                    f"Verified claim {cid} cannot use inactive manuscript location '{manuscript_locations}'",
                )
            )
        if status == "Historical" and not is_historical:
            findings.append(
                Finding(
                    "ERROR",
                    "claim_matrix",
                    str(claim_matrix_path),
                    f"Historical claim {cid} must use historical_* manuscript_locations",
                )
            )
        if status == "Inactive" and not is_not_present:
            findings.append(
                Finding(
                    "ERROR",
                    "claim_matrix",
                    str(claim_matrix_path),
                    f"Inactive claim {cid} must use manuscript_locations=NOT PRESENT",
                )
            )
        if status in ACTIVE_CLAIM_STATUSES and (is_not_present or is_historical):
            findings.append(
                Finding(
                    "WARN",
                    "claim_matrix",
                    str(claim_matrix_path),
                    f"Active review status '{status}' is attached to inactive claim {cid}; use Historical or Inactive instead",
                )
            )


def _check_impact_alignment_with_manifest(
    findings: list[Finding],
    manifest: dict,
    repo_root: Path,
) -> None:
    canonical = manifest.get("canonical_metrics", {})
    tolerance = 1e-6
    targets = [
        ("de", repo_root / "reports" / "impact_summary.csv"),
        ("us", repo_root / "reports" / "eia930" / "impact_summary.csv"),
    ]
    for region, csv_path in targets:
        if not csv_path.exists():
            findings.append(Finding("ERROR", "impact_alignment", str(csv_path), "Missing impact summary CSV"))
            continue
        locked = canonical.get(region, {}).get("impact", {})
        if not locked:
            findings.append(
                Finding(
                    "ERROR",
                    "impact_alignment",
                    str(csv_path),
                    f"Missing canonical manifest metrics for region '{region}'",
                )
            )
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            findings.append(
                Finding("ERROR", "impact_alignment", str(csv_path), "Impact summary CSV is empty")
            )
            continue
        row = rows[0]
        for field, manifest_key in (
            ("cost_savings_pct", "cost_savings_pct_raw"),
            ("carbon_reduction_pct", "carbon_reduction_pct_raw"),
            ("peak_shaving_pct", "peak_shaving_pct_raw"),
        ):
            try:
                actual = float(row[field])
                expected = float(locked[manifest_key])
            except (KeyError, TypeError, ValueError) as exc:
                findings.append(
                    Finding("ERROR", "impact_alignment", str(csv_path), f"Could not compare {field}: {exc}")
                )
                continue
            if abs(actual - expected) > tolerance:
                findings.append(
                    Finding(
                        "ERROR",
                        "impact_alignment",
                        str(csv_path),
                        f"{field}={actual} diverges from locked manifest value {expected}",
                    )
                )


def _locked_battery_reference_metrics(repo_root: Path) -> dict[str, float]:
    table_path = repo_root / "reports" / "publication" / "dc3s_main_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    with table_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"Locked battery witness table is empty: {table_path}")

    def _collect(controller: str) -> list[float]:
        vals = [
            float(row["violation_rate"])
            for row in rows
            if row.get("scenario") == "nominal" and row.get("controller") == controller
        ]
        if not vals:
            raise ValueError(f"Missing nominal locked battery rows for controller '{controller}'")
        return vals

    deterministic_vals = _collect("deterministic_lp")
    dc3s_vals = _collect("dc3s_ftit")
    return {
        "baseline_tsvr_mean": sum(deterministic_vals) / len(deterministic_vals),
        "orius_tsvr_mean": sum(dc3s_vals) / len(dc3s_vals),
    }


def _check_reference_domain_validation_alignment(
    findings: list[Finding],
    repo_root: Path,
) -> None:
    summary_path = repo_root / "reports" / "universal_orius_validation" / "domain_validation_summary.csv"
    if not summary_path.exists():
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                "Missing universal validation summary",
            )
        )
        return

    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    battery_row = next((row for row in rows if row.get("domain") == "battery"), None)
    if battery_row is None:
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                "Battery reference row missing from validation summary",
            )
        )
        return

    try:
        locked = _locked_battery_reference_metrics(repo_root)
    except (FileNotFoundError, ValueError) as exc:
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                f"Could not load locked battery witness metrics: {exc}",
            )
        )
        return

    try:
        baseline = float(battery_row["baseline_tsvr_mean"])
        orius = float(battery_row["orius_tsvr_mean"])
    except (KeyError, TypeError, ValueError) as exc:
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                f"Battery row is malformed: {exc}",
            )
        )
        return

    if abs(baseline - locked["baseline_tsvr_mean"]) > 1e-4:
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                f"battery baseline_tsvr_mean={baseline} diverges from locked reference {locked['baseline_tsvr_mean']}",
            )
        )
    if abs(orius - locked["orius_tsvr_mean"]) > 1e-9:
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                f"battery orius_tsvr_mean={orius} diverges from locked reference {locked['orius_tsvr_mean']}",
            )
        )
    metric_surface = (battery_row.get("metric_surface") or "").strip()
    if metric_surface and metric_surface != "locked_publication_nominal":
        findings.append(
            Finding(
                "ERROR",
                "reference_validation_alignment",
                str(summary_path),
                f"battery metric_surface='{metric_surface}' should be locked_publication_nominal",
            )
        )


def _check_promoted_validation_surface_alignment(
    findings: list[Finding],
    repo_root: Path,
) -> None:
    benchmark_path = repo_root / "reports" / "publication" / "three_domain_ml_benchmark.csv"
    closure_path = repo_root / "reports" / "publication" / "orius_domain_closure_matrix.csv"
    proxy_runtime_path = repo_root / "reports" / "publication" / "three_domain_proxy_runtime_comparison.csv"
    av_runtime_summary_path = (
        repo_root
        / "reports"
        / "orius_av"
        / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest"
        / "runtime_summary.csv"
    )
    healthcare_runtime_summary_path = repo_root / "reports" / "healthcare" / "runtime_summary.csv"

    required_paths = (
        benchmark_path,
        closure_path,
        proxy_runtime_path,
        av_runtime_summary_path,
        healthcare_runtime_summary_path,
    )
    for path in required_paths:
        if not path.exists():
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(path),
                    "Missing required promoted validation surface",
                )
            )
            return

    with benchmark_path.open("r", encoding="utf-8", newline="") as fh:
        benchmark_rows = {row["domain"]: row for row in csv.DictReader(fh) if row.get("domain")}
    with closure_path.open("r", encoding="utf-8", newline="") as fh:
        closure_rows = {row["domain"]: row for row in csv.DictReader(fh) if row.get("domain")}
    with proxy_runtime_path.open("r", encoding="utf-8", newline="") as fh:
        proxy_runtime_rows = {row["domain"]: row for row in csv.DictReader(fh) if row.get("domain")}
    with av_runtime_summary_path.open("r", encoding="utf-8", newline="") as fh:
        av_runtime_rows = {row["controller"]: row for row in csv.DictReader(fh) if row.get("controller")}
    with healthcare_runtime_summary_path.open("r", encoding="utf-8", newline="") as fh:
        healthcare_runtime_rows = {
            row["controller"]: row for row in csv.DictReader(fh) if row.get("controller")
        }

    promoted_domains = {
        "Autonomous Vehicles": av_runtime_rows,
        "Medical and Healthcare Monitoring": healthcare_runtime_rows,
    }
    for display_name, runtime_rows in promoted_domains.items():
        benchmark_row = benchmark_rows.get(display_name)
        closure_row = closure_rows.get(display_name)
        proxy_runtime_row = proxy_runtime_rows.get(display_name)
        if benchmark_row is None:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(benchmark_path),
                    f"Missing promoted benchmark row for {display_name}",
                )
            )
            continue
        if closure_row is None:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(closure_path),
                    f"Missing closure matrix row for {display_name}",
                )
            )
            continue
        if proxy_runtime_row is None:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(proxy_runtime_path),
                    f"Missing proxy-vs-runtime comparison row for {display_name}",
                )
            )
            continue

        try:
            runtime_baseline = float(runtime_rows["baseline"]["tsvr"])
            runtime_orius = float(runtime_rows["orius"]["tsvr"])
            benchmark_baseline = float(benchmark_row["baseline_tsvr_mean"])
            benchmark_orius = float(benchmark_row["orius_tsvr_mean"])
        except (KeyError, TypeError, ValueError) as exc:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(benchmark_path),
                    f"{display_name} promoted surfaces are malformed: {exc}",
                )
            )
            continue

        if abs(benchmark_baseline - runtime_baseline) > 1e-6:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(benchmark_path),
                    (
                        f"{display_name} benchmark baseline_tsvr_mean={benchmark_baseline} diverges from "
                        f"runtime summary {runtime_baseline}"
                    ),
                )
            )
        if abs(benchmark_orius - runtime_orius) > 1e-6:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(benchmark_path),
                    (
                        f"{display_name} benchmark orius_tsvr_mean={benchmark_orius} diverges from "
                        f"runtime summary {runtime_orius}"
                    ),
                )
            )

        benchmark_metric_surface = (benchmark_row.get("metric_surface") or "").strip()
        if benchmark_metric_surface != "runtime_denominator":
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(benchmark_path),
                    f"{display_name} metric_surface='{benchmark_metric_surface}' should be runtime_denominator",
                )
            )

        expected_closure_baseline = f"{runtime_baseline:.4f}"
        expected_closure_orius = f"{runtime_orius:.4f}"
        if (closure_row.get("baseline_tsvr") or "").strip() != expected_closure_baseline:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(closure_path),
                    (
                        f"{display_name} closure baseline_tsvr='{closure_row.get('baseline_tsvr', '')}' diverges from "
                        f"runtime summary {expected_closure_baseline}"
                    ),
                )
            )
        if (closure_row.get("orius_tsvr") or "").strip() != expected_closure_orius:
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(closure_path),
                    (
                        f"{display_name} closure orius_tsvr='{closure_row.get('orius_tsvr', '')}' diverges from "
                        f"runtime summary {expected_closure_orius}"
                    ),
                )
            )

        current_status = (closure_row.get("current_status") or "").strip()
        has_runtime_denominator = "runtime denominator" in current_status
        has_secondary_proxy_disclosure = "secondary proxy" in current_status
        has_nuplan_disclosure = (
            display_name == "Autonomous Vehicles"
            and "nuplan_allzip_grouped_runtime_replay_surrogate" in current_status
            and "no road deployment" in current_status
            and "full autonomous-driving field closure" in current_status
        )
        if not has_runtime_denominator or not (has_secondary_proxy_disclosure or has_nuplan_disclosure):
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(closure_path),
                    (
                        f"{display_name} current_status must disclose runtime-denominator governance and "
                        "either secondary proxy status or bounded nuPlan claim boundaries"
                    ),
                )
            )
        if (proxy_runtime_row.get("claim_governs_from") or "").strip() != "runtime_denominator":
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(proxy_runtime_path),
                    f"{display_name} proxy/runtime row must declare claim_governs_from=runtime_denominator",
                )
            )
        if (proxy_runtime_row.get("proxy_metric_surface") or "").strip() != "validation_harness":
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(proxy_runtime_path),
                    f"{display_name} proxy/runtime row must keep proxy_metric_surface=validation_harness",
                )
            )
        if (proxy_runtime_row.get("diagnostic_only") or "").strip() != "True":
            findings.append(
                Finding(
                    "ERROR",
                    "promoted_validation_alignment",
                    str(proxy_runtime_path),
                    f"{display_name} proxy/runtime row must declare diagnostic_only=True",
                )
            )


def _check_healthcare_runtime_metric_alignment(
    findings: list[Finding],
    repo_root: Path,
) -> None:
    benchmark_path = repo_root / "reports" / "publication" / "three_domain_ml_benchmark.csv"
    witness_summary_path = repo_root / "reports" / "publication" / "domain_runtime_contract_summary.json"
    runtime_summary_path = repo_root / "reports" / "healthcare" / "runtime_summary.csv"
    runtime_traces_path = repo_root / "reports" / "healthcare" / "runtime_traces.csv"

    required_paths = (benchmark_path, witness_summary_path, runtime_summary_path)
    for path in required_paths:
        if not path.exists():
            findings.append(
                Finding(
                    "ERROR",
                    "healthcare_runtime_alignment",
                    str(path),
                    "Missing required healthcare runtime surface",
                )
            )
            return

    with benchmark_path.open("r", encoding="utf-8", newline="") as fh:
        benchmark_rows = {row["domain"]: row for row in csv.DictReader(fh) if row.get("domain")}
    healthcare_row = benchmark_rows.get("Medical and Healthcare Monitoring")
    if healthcare_row is None:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                "Missing healthcare benchmark row",
            )
        )
        return
    try:
        witness_summary = json.loads(witness_summary_path.read_text(encoding="utf-8"))
        witness_certificate_valid_rate = float(
            witness_summary["domains"]["healthcare"]["certificate_valid_rate"]
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(witness_summary_path),
                f"Healthcare domain runtime witness summary is malformed: {exc}",
            )
        )
        return

    with runtime_summary_path.open("r", encoding="utf-8", newline="") as fh:
        summary_rows = {row["controller"]: row for row in csv.DictReader(fh) if row.get("controller")}
    runtime_row = summary_rows.get("orius")
    if runtime_row is None:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(runtime_summary_path),
                "Missing ORIUS healthcare runtime summary row",
            )
        )
        return

    def _parse_bool(value: str) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes"}

    try:
        runtime_ir = float(runtime_row["intervention_rate"])
        if (
            "fallback_activation_rate" in runtime_row
            and str(runtime_row["fallback_activation_rate"]).strip() != ""
        ):
            runtime_fallback = float(runtime_row["fallback_activation_rate"])
        elif runtime_traces_path.exists():
            with runtime_traces_path.open("r", encoding="utf-8", newline="") as fh:
                trace_rows = [
                    row for row in csv.DictReader(fh) if (row.get("controller") or "").strip() == "orius"
                ]
            runtime_fallback = (
                sum(1 for row in trace_rows if _parse_bool(row.get("fallback_used", ""))) / len(trace_rows)
                if trace_rows
                else float("nan")
            )
        else:
            raise KeyError("fallback_activation_rate")
        benchmark_cva = float(healthcare_row["certificate_valid_release_rate"])
        benchmark_ir = float(healthcare_row["intervention_rate"])
        benchmark_fallback = float(healthcare_row["fallback_activation_rate"])
    except (KeyError, TypeError, ValueError) as exc:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                f"Healthcare runtime surfaces are malformed: {exc}",
            )
        )
        return

    semantics = (healthcare_row.get("certificate_valid_release_rate_semantics") or "").strip()
    expected_semantics = "runtime_witness_certificate_valid_rate_promoted_healthcare_row"
    if semantics != expected_semantics:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                (
                    "Healthcare certificate_valid_release_rate_semantics must be "
                    f"'{expected_semantics}', found '{semantics}'"
                ),
            )
        )

    runtime_source = (healthcare_row.get("runtime_source") or "").strip()
    if runtime_source != "reports/healthcare/runtime_summary.csv":
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                (
                    "Healthcare runtime_source must be "
                    f"'reports/healthcare/runtime_summary.csv', found '{runtime_source}'"
                ),
            )
        )

    if abs(benchmark_cva - witness_certificate_valid_rate) > 1e-6:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                (
                    f"Healthcare certificate_valid_release_rate={benchmark_cva} diverges from "
                    f"domain runtime witness certificate_valid_rate={witness_certificate_valid_rate}"
                ),
            )
        )
    if abs(benchmark_ir - runtime_ir) > 1e-6:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                (
                    f"Healthcare intervention_rate={benchmark_ir} diverges from "
                    f"runtime summary intervention_rate={runtime_ir}"
                ),
            )
        )
    if abs(benchmark_fallback - runtime_fallback) > 1e-6:
        findings.append(
            Finding(
                "ERROR",
                "healthcare_runtime_alignment",
                str(benchmark_path),
                (
                    f"Healthcare fallback_activation_rate={benchmark_fallback} diverges from "
                    f"runtime summary fallback_activation_rate={runtime_fallback}"
                ),
            )
        )


def _has_positive_completion_claim(text: str, positive_patterns: Iterable[str]) -> bool:
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    negative_markers = (
        "not ",
        "not-",
        "does not",
        "do not",
        "no ",
        "without ",
        "future",
        "next-tier",
        "prepared_not_completed",
        "blocked",
        "incomplete",
        "out of scope",
    )
    for sentence in sentences:
        lower = sentence.lower()
        if any(marker in lower for marker in negative_markers):
            continue
        if any(re.search(pattern, lower) for pattern in positive_patterns):
            return True
    return False


def _check_95_validation_claim_boundaries(
    findings: list[Finding],
    repo_root: Path,
    markdown_text: str,
    tex_text: str,
    markdown_path: Path,
    tex_path: Path,
) -> None:
    """Prevent next-tier validation claims from outrunning the 95 manifest."""
    manifest_path = (
        repo_root / "reports" / "predeployment_external_validation" / "orius_95_validation_manifest.json"
    )
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    domains = manifest.get("domains", {}) if isinstance(manifest, dict) else {}
    carla_completed = bool(domains.get("av_carla", {}).get("completed"))
    eicu_staged = domains.get("healthcare", {}).get("eicu_status") == "staged"
    physical_hil_completed = bool(domains.get("battery", {}).get("physical_hil_completed"))

    checks = [
        (
            "carla_95_boundary",
            carla_completed,
            [
                r"\bcarla\b.*\bclosed[- ]loop\b.*\b(completed|complete|validated|passed)\b",
                r"\bcompleted\b.*\bcarla\b",
            ],
            "CARLA closed-loop completion is claimed without manifest completion.",
        ),
        (
            "eicu_95_boundary",
            eicu_staged,
            [
                r"\beicu\b.*\b(held[- ]out|source[- ]holdout|validation)\b.*\b(completed|complete|validated|passed)\b"
            ],
            "eICU held-out validation is claimed without staged eICU evidence.",
        ),
        (
            "physical_hil_95_boundary",
            physical_hil_completed,
            [r"\bphysical\b.*\b(hil|bench)\b.*\b(completed|complete|validated|passed)\b"],
            "Physical HIL/bench validation is claimed during a software-HIL-only run.",
        ),
    ]
    for path, text in ((markdown_path, markdown_text), (tex_path, tex_text)):
        for check, allowed, patterns, detail in checks:
            if not allowed and _has_positive_completion_claim(text, patterns):
                findings.append(Finding("ERROR", check, str(path), detail))


def _print_findings(findings: list[Finding]) -> None:
    if not findings:
        print("[validate_paper_claims] PASS: no findings")
        return

    print("[validate_paper_claims] Findings:")
    for f in findings:
        print(f"- {f.severity}: {f.check}: {f.file}: {f.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manuscript claims and references")
    parser.add_argument(
        "--manifest", default="paper/metrics_manifest.json", help="Path to metrics manifest JSON"
    )
    parser.add_argument(
        "--markdown", default="paper/PAPER_DRAFT.md", help="Path to master markdown manuscript"
    )
    parser.add_argument("--tex", default="orius_book.tex", help="Path to LaTeX manuscript")
    parser.add_argument("--claim-matrix", default="paper/claim_matrix.csv", help="Path to claim matrix CSV")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    markdown_path = Path(args.markdown)
    tex_path = Path(args.tex)
    claim_matrix_path = Path(args.claim_matrix)

    manifest = _load_json(manifest_path)
    markdown_text = _read_text(markdown_path)
    tex_text = _read_text(tex_path)
    repo_root = manifest_path.resolve().parents[1]
    monograph_mode = r"\documentclass[12pt,oneside]{book}" in tex_text

    findings: list[Finding] = []

    validation = manifest.get("validation", {})
    required_patterns = validation.get("required_patterns", [])
    banned_patterns = validation.get("banned_patterns", [])
    placeholder_patterns = validation.get("placeholder_patterns", [])

    if not monograph_mode:
        _check_patterns(
            findings,
            required_patterns,
            markdown_text,
            tex_text,
            markdown_path,
            tex_path,
            required=True,
        )
    _check_patterns(
        findings,
        banned_patterns,
        markdown_text,
        tex_text,
        markdown_path,
        tex_path,
        required=False,
    )
    _check_placeholders(findings, placeholder_patterns, markdown_text, tex_text, markdown_path, tex_path)

    run_ids = set((manifest.get("run_ids") or {}).values())
    if not monograph_mode:
        if not run_ids:
            findings.append(
                Finding("ERROR", "manifest", str(manifest_path), "No run_ids configured in manifest")
            )
        else:
            _check_run_ids(findings, run_ids, markdown_text, tex_text, markdown_path, tex_path)

    _check_tex_inputs(findings, tex_text, tex_path, repo_root)
    _check_title_alignment(findings, markdown_text, tex_text, markdown_path, tex_path)
    _check_claim_matrix(findings, claim_matrix_path)
    _check_canonical_value_lock(findings, manifest, claim_matrix_path, repo_root)
    _check_impact_alignment_with_manifest(findings, manifest, repo_root)
    _check_reference_domain_validation_alignment(findings, repo_root)
    _check_promoted_validation_surface_alignment(findings, repo_root)
    _check_healthcare_runtime_metric_alignment(findings, repo_root)
    _check_canonical_project_framing(findings, repo_root)
    _check_95_validation_claim_boundaries(
        findings, repo_root, markdown_text, tex_text, markdown_path, tex_path
    )

    _print_findings(findings)

    errors = [f for f in findings if f.severity == "ERROR"]
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
