from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = REPO_ROOT / "reports" / "publication"
DOMAIN_DIRS = {
    "Battery Energy Storage": REPO_ROOT / "reports" / "battery_av" / "battery",
    "Autonomous Vehicles": REPO_ROOT / "reports" / "orius_av" / "nuplan_allzip_grouped_runtime_dropout_aligned_m15_fulltest",
    "Medical and Healthcare Monitoring": REPO_ROOT / "reports" / "healthcare",
}
PROMOTED_RUNTIME_MAX_TSVR = 1e-3
PROMOTED_RUNTIME_MIN_PASS_RATE = 1.0 - PROMOTED_RUNTIME_MAX_TSVR
REQUIRED_BASELINES = {
    "nominal_deterministic_controller",
    "fixed_threshold_or_fixed_inflation_runtime",
    "standard_conformal_nonreliability_runtime",
    "no_quality_signal_runtime",
    "no_adaptive_response_runtime",
    "no_temporal_guard_or_no_certificate_refresh_runtime",
    "orius_full_stack",
}
REQUIRED_ABLATIONS = {
    "no_quality_signal",
    "no_reliability_conditioned_widening",
    "no_repair_release_without_repair",
    "no_fallback_or_no_temporal_guard",
    "no_certificate_refresh_stale_certificate_policy",
}
REQUIRED_NEGATIVE_CONTROLS = {
    "actual_reliability",
    "shuffled_reliability_score",
    "delayed_reliability_score",
    "constant_low_reliability_conservative_policy",
    "stronger_predictor_without_runtime_adaptation",
}
FORBIDDEN = {
    "validation_harness",
    "diagnostic_cross_domain_proxy",
    "proxy_current_shared_harness",
    "missing",
    "missing_on_current_cross_domain_lane",
    "future_cross_domain_benchmark_extension",
}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _forbidden(value: str) -> bool:
    lowered = value.strip().lower()
    return lowered in FORBIDDEN or lowered.startswith("future_")


def test_equal_domain_artifact_discipline_gates_pass_for_all_domains() -> None:
    rows = _rows(PUBLICATION / "equal_domain_artifact_discipline.csv")
    assert {row["domain"] for row in rows} == set(DOMAIN_DIRS)
    for row in rows:
        assert row["artifact_discipline_gate"] == "True", row
        assert row["runtime_native_gate"] == "True", row
        assert row["theorem_gate"] == "True", row
        assert row["proof_appendix_gate"] == "True", row
        assert row["baseline_gate"] == "True", row
        assert row["ablation_gate"] == "True", row
        assert row["negative_control_gate"] == "True", row
        assert row["utility_gate"] == "True", row
        assert row["reproducibility_gate"] == "True", row
        assert row["blockers"] == ""


def test_each_domain_has_runtime_native_baselines_ablations_and_negative_controls() -> None:
    for domain, domain_dir in DOMAIN_DIRS.items():
        comparator_rows = _rows(domain_dir / "runtime_comparator_summary.csv")
        ablation_rows = _rows(domain_dir / "runtime_ablation_summary.csv")
        negative_rows = _rows(domain_dir / "runtime_negative_controls.csv")
        trace_rows = _rows(domain_dir / "runtime_comparator_traces.csv")

        assert REQUIRED_BASELINES <= {row["baseline_family"] for row in comparator_rows}, domain
        assert REQUIRED_ABLATIONS == {row["ablation_name"] for row in ablation_rows}, domain
        assert REQUIRED_NEGATIVE_CONTROLS == {row["control_name"] for row in negative_rows}, domain
        assert trace_rows, domain

        for row in comparator_rows:
            assert row["metric_surface"] == "runtime_denominator"
            assert not _forbidden(row["metric_surface"])
            assert not _forbidden(row["evidence_status"])
            assert row["runtime_source"]
            assert row["n_steps"] != ""
        if domain in {"Autonomous Vehicles", "Medical and Healthcare Monitoring"}:
            independent_rows = [
                row
                for row in comparator_rows
                if row["baseline_family"] not in {"orius_full_stack", "degenerate_fallback_runtime"}
            ]
            assert all(row["independent_baseline"] == "True" for row in independent_rows), domain
            controllers = [row["controller"] for row in independent_rows]
            assert len(controllers) == len(set(controllers)), domain
        for row in ablation_rows:
            assert row["metric_surface"] == "runtime_denominator"
            assert not _forbidden(row["metric_surface"])
            assert not _forbidden(row["evidence_status"])
        for row in negative_rows:
            assert row["surface"] == "runtime_denominator"
            assert not _forbidden(row["surface"])
            assert not _forbidden(row["status"])


def test_each_domain_orius_row_is_non_degenerate_and_witness_closed() -> None:
    for domain, domain_dir in DOMAIN_DIRS.items():
        rows = {
            row["baseline_family"]: row
            for row in _rows(domain_dir / "runtime_comparator_summary.csv")
        }
        orius = rows["orius_full_stack"]
        degenerate = rows["degenerate_fallback_runtime"]
        assert _safe_float(orius["tsvr"]) <= PROMOTED_RUNTIME_MAX_TSVR, domain
        assert _safe_float(orius["certificate_valid_rate"]) >= PROMOTED_RUNTIME_MIN_PASS_RATE, domain
        assert _safe_float(orius["runtime_witness_pass_rate"]) >= PROMOTED_RUNTIME_MIN_PASS_RATE, domain
        assert _safe_float(orius["useful_work_total"]) > _safe_float(degenerate["useful_work_total"]), domain
        if domain == "Autonomous Vehicles":
            assert _safe_float(orius["fallback_activation_rate"]) <= 0.50, domain
        if domain == "Medical and Healthcare Monitoring":
            assert _safe_float(orius["fallback_activation_rate"]) <= 0.50, domain


def test_equal_domain_reproducibility_manifest_is_complete() -> None:
    payload = json.loads((PUBLICATION / "equal_domain_reproducibility_manifest.json").read_text(encoding="utf-8"))
    assert ".venv/bin/python scripts/build_equal_domain_artifact_discipline.py" in payload["commands"]
    assert ".venv/bin/python scripts/validate_equal_domain_artifact_discipline.py" in payload["commands"]
    missing = [row["path"] for row in payload["artifacts"] if not row["exists"]]
    assert missing == []


def test_equal_domain_validator_passes() -> None:
    result = subprocess.run(
        [str(REPO_ROOT / ".venv" / "bin" / "python"), "scripts/validate_equal_domain_artifact_discipline.py"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
