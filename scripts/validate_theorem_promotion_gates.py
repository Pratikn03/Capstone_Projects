#!/usr/bin/env python3
"""Validate the T9/T10 theorem-promotion gate package."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts import build_theorem_promotion_gates as builder
except ModuleNotFoundError:  # pragma: no cover - script execution from scripts/
    import build_theorem_promotion_gates as builder  # type: ignore[no-redef]

REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_DIR = REPO_ROOT / "reports" / "publication"
CANDIDATE_THEOREMS = ("T9", "T10")
DOMAINS = ("battery", "av", "healthcare")
REQUIRED_GATES = {
    "formal_theorem_statement",
    "explicit_assumptions",
    "mathematical_proof",
    "mechanized_proof",
    "research_package",
    "code_anchor",
    "tests",
    "artifact_evidence:battery",
    "artifact_evidence:av",
    "artifact_evidence:healthcare",
    "domain_applicability_matrix",
    "universal_constants_and_assumptions",
}
PROMOTION_STATUSES = {"promote_requested", "flagship_candidate", "promoted"}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _is_true(value: str | bool | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _expected_package(
    repo_root: Path, publication_dir: Path
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="orius-theorem-promotion-") as tmp:
        expected_publication_dir = Path(tmp) / "publication"
        evidence_dir = publication_dir / "theorem_promotion_evidence"
        if evidence_dir.exists():
            shutil.copytree(evidence_dir, expected_publication_dir / "theorem_promotion_evidence")
        for name in (
            "t9_t10_mechanized_proof_status.json",
            "t9_t10_research_scorecard.json",
        ):
            source = publication_dir / name
            if source.exists():
                expected_publication_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, expected_publication_dir / name)
        builder.build_promotion_package(repo_root=repo_root, publication_dir=expected_publication_dir)
        return (
            _read_csv(expected_publication_dir / "theorem_promotion_gates.csv"),
            _read_csv(expected_publication_dir / "theorem_promotion_domain_matrix.csv"),
            _read_json(expected_publication_dir / "theorem_promotion_scorecard.json"),
        )


def _index_rows(
    rows: list[dict[str, str]], key_fields: tuple[str, ...]
) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(field, "") for field in key_fields): row for row in rows}


def _diff_fields(actual: dict[str, str], expected: dict[str, str]) -> list[str]:
    return [field for field, expected_value in expected.items() if actual.get(field, "") != expected_value]


def _path_candidates(reference: str, repo_root: Path, publication_dir: Path) -> list[Path]:
    item = reference.strip()
    if not item or item.startswith(("http://", "https://")):
        return []
    file_suffixes = (".csv", ".json", ".py", ".tex", ".md", ".yaml", ".yml", ".parquet", ".txt")
    if "/" not in item and not item.endswith(file_suffixes):
        return []
    if ":" in item and item.rsplit(":", 1)[1].isdigit():
        item = item.rsplit(":", 1)[0]
    raw = Path(item)
    if raw.is_absolute():
        return [raw]
    candidates = [repo_root / raw]
    publication_prefix = "reports/publication/"
    if item.startswith(publication_prefix):
        candidates.append(publication_dir / item.removeprefix(publication_prefix))
    return candidates


def _verify_evidence_paths(
    gate_rows: list[dict[str, str]],
    *,
    repo_root: Path,
    publication_dir: Path,
) -> list[str]:
    findings: list[str] = []
    for row in gate_rows:
        if not _is_true(row.get("gate_pass")):
            continue
        for reference in row.get("evidence", "").split("|"):
            reference = reference.strip()
            if not reference or reference.startswith("three-domain constants"):
                continue
            candidates = _path_candidates(reference.split(";", 1)[0], repo_root, publication_dir)
            if candidates and not any(candidate.exists() for candidate in candidates):
                findings.append(
                    f"{row.get('theorem_id')}: passing gate {row.get('gate')} references missing evidence path {reference}"
                )
    return findings


def _compare_against_recomputed_package(
    actual_gate_rows: list[dict[str, str]],
    actual_domain_rows: list[dict[str, str]],
    actual_scorecard: dict[str, Any],
    *,
    repo_root: Path,
    publication_dir: Path,
) -> list[str]:
    findings: list[str] = []
    expected_gate_rows, expected_domain_rows, expected_scorecard = _expected_package(
        repo_root, publication_dir
    )
    actual_gates = _index_rows(actual_gate_rows, ("theorem_id", "gate"))
    expected_gates = _index_rows(expected_gate_rows, ("theorem_id", "gate"))
    actual_domains = _index_rows(actual_domain_rows, ("theorem_id", "domain"))
    expected_domains = _index_rows(expected_domain_rows, ("theorem_id", "domain"))

    for key, expected in sorted(expected_gates.items()):
        theorem_id, gate = key
        actual = actual_gates.get(key)
        if not actual:
            findings.append(f"{theorem_id}: missing recomputed gate {gate}")
            continue
        diff_fields = _diff_fields(actual, expected)
        if diff_fields:
            findings.append(
                f"{theorem_id}: gate {gate} does not match recomputed gate fields {sorted(diff_fields)}"
            )

    for key in sorted(set(actual_gates) - set(expected_gates)):
        theorem_id, gate = key
        findings.append(f"{theorem_id}: unexpected gate {gate} outside recomputed package")

    for key, expected in sorted(expected_domains.items()):
        theorem_id, domain = key
        actual = actual_domains.get(key)
        if not actual:
            findings.append(f"{theorem_id}: missing recomputed domain row {domain}")
            continue
        diff_fields = _diff_fields(actual, expected)
        if diff_fields:
            findings.append(
                f"{theorem_id}: domain row {domain} does not match recomputed package fields {sorted(diff_fields)}"
            )

    for key in sorted(set(actual_domains) - set(expected_domains)):
        theorem_id, domain = key
        findings.append(f"{theorem_id}: unexpected domain row {domain} outside recomputed package")

    if actual_scorecard.get("source_sha256") != expected_scorecard.get("source_sha256"):
        findings.append("scorecard source_sha256 does not match current active_theorem_audit.json")

    expected_candidates = expected_scorecard.get("candidates", {})
    actual_candidates = actual_scorecard.get("candidates", {})
    for theorem_id, expected_candidate in expected_candidates.items():
        actual_candidate = actual_candidates.get(theorem_id, {})
        for field in ("current_tier", "candidate_status", "promotion_ready", "blocking_gates"):
            if actual_candidate.get(field) != expected_candidate.get(field):
                findings.append(f"{theorem_id}: scorecard {field} does not match recomputed package")

    return findings


def validate_promotion_package(
    publication_dir: Path = PUBLICATION_DIR,
    repo_root: Path = REPO_ROOT,
    require_promoted: set[str] | None = None,
) -> dict[str, Any]:
    findings: list[str] = []
    blockers: list[str] = []
    require_promoted = require_promoted or set()

    gate_path = publication_dir / "theorem_promotion_gates.csv"
    matrix_path = publication_dir / "theorem_promotion_domain_matrix.csv"
    scorecard_path = publication_dir / "theorem_promotion_scorecard.json"
    try:
        gate_rows = _read_csv(gate_path)
        domain_rows = _read_csv(matrix_path)
        scorecard = _read_json(scorecard_path)
    except FileNotFoundError as exc:
        return {
            "pass": False,
            "promotion_ready": False,
            "findings": [f"missing promotion package artifact: {exc.filename}"],
            "blockers": [],
        }

    findings.extend(
        _compare_against_recomputed_package(
            gate_rows,
            domain_rows,
            scorecard,
            repo_root=repo_root,
            publication_dir=publication_dir,
        )
    )
    findings.extend(_verify_evidence_paths(gate_rows, repo_root=repo_root, publication_dir=publication_dir))

    gates_by_theorem: dict[str, list[dict[str, str]]] = {theorem_id: [] for theorem_id in CANDIDATE_THEOREMS}
    domains_by_theorem: dict[str, list[dict[str, str]]] = {
        theorem_id: [] for theorem_id in CANDIDATE_THEOREMS
    }
    for row in gate_rows:
        if row.get("theorem_id") in gates_by_theorem:
            gates_by_theorem[row["theorem_id"]].append(row)
    for row in domain_rows:
        if row.get("theorem_id") in domains_by_theorem:
            domains_by_theorem[row["theorem_id"]].append(row)

    for theorem_id in CANDIDATE_THEOREMS:
        theorem_gates = gates_by_theorem[theorem_id]
        observed_gates = {row.get("gate", "") for row in theorem_gates}
        missing_gates = sorted(REQUIRED_GATES - observed_gates)
        extra_gates = sorted(observed_gates - REQUIRED_GATES)
        if missing_gates:
            findings.append(f"{theorem_id}: missing promotion gates: {missing_gates}")
        if extra_gates:
            findings.append(f"{theorem_id}: unknown promotion gates: {extra_gates}")

        theorem_domains = domains_by_theorem[theorem_id]
        observed_domains = {row.get("domain", "") for row in theorem_domains}
        missing_domains = sorted(set(DOMAINS) - observed_domains)
        if missing_domains:
            findings.append(f"{theorem_id}: missing domain applicability rows: {missing_domains}")

        failing_gates = [row for row in theorem_gates if not _is_true(row.get("gate_pass"))]
        for row in failing_gates:
            if not row.get("blocker", "").strip():
                findings.append(f"{theorem_id}: failed gate {row.get('gate')} has no blocker")
            else:
                blockers.append(f"{theorem_id}:{row.get('gate')} - {row.get('blocker')}")

        statuses = {row.get("candidate_status", "") for row in theorem_gates}
        if statuses & PROMOTION_STATUSES and failing_gates:
            findings.append(
                f"{theorem_id}: candidate_status promote_requested/flagship/promoted is blocked by "
                f"{len(failing_gates)} failing gate(s)"
            )

        ready = not failing_gates and not missing_gates and not missing_domains
        if theorem_id in require_promoted and not ready:
            findings.append(f"{theorem_id}: not promotion-ready under --require-promoted")

        candidate = scorecard.get("candidates", {}).get(theorem_id)
        if not candidate:
            findings.append(f"{theorem_id}: missing scorecard candidate row")
        elif bool(candidate.get("promotion_ready")) != ready:
            findings.append(f"{theorem_id}: scorecard promotion_ready does not match gate rows")
        elif not ready and not candidate.get("blocking_gates"):
            findings.append(f"{theorem_id}: scorecard must list blocking gates while not ready")

    promotion_ready = all(
        not [row for row in gates_by_theorem[theorem_id] if not _is_true(row.get("gate_pass"))]
        for theorem_id in CANDIDATE_THEOREMS
    )
    if bool(scorecard.get("promotion_ready")) != promotion_ready:
        findings.append("scorecard promotion_ready does not match candidate gate state")

    return {
        "pass": not findings,
        "promotion_ready": promotion_ready,
        "findings": findings,
        "blockers": sorted(set(blockers)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-dir", type=Path, default=PUBLICATION_DIR)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--require-promoted",
        action="append",
        default=[],
        choices=list(CANDIDATE_THEOREMS),
        help="Require a theorem to be promotion-ready. Can be repeated.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full validation result as JSON.")
    args = parser.parse_args()

    result = validate_promotion_package(
        publication_dir=args.publication_dir.resolve(),
        repo_root=args.repo_root.resolve(),
        require_promoted=set(args.require_promoted),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "PASS" if result["pass"] else "FAIL"
        print(f"[validate_theorem_promotion_gates] {status}")
        for finding in result["findings"]:
            print(f"- {finding}")
        if result["blockers"]:
            print("[validate_theorem_promotion_gates] tracked blockers:")
            for blocker in result["blockers"]:
                print(f"- {blocker}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
