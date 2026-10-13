"""Fail-closed protocol and artifact contracts for confirmatory KGA studies.

This module deliberately contains no training code.  It validates the boundary
between a sealed experimental design, label-free live decisions, and the later
offline label join used for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kga.policy import Decision

SCHEMA_VERSION = 1
PROTOCOL_STATUSES = {"DRAFT_UNSEALED", "SEALED", "EXECUTED"}
SPLIT_ROLES = (
    "source_train",
    "development",
    "estimator_fit",
    "residual_calibration",
    "test",
    "replication",
)
DECISION_FIELDS = (
    "run_id",
    "protocol_id",
    "protocol_sha256",
    "git_sha",
    "dataset_version",
    "split_role",
    "unit_id",
    "environment_id",
    "model_seed",
    "checkpoint_sha256",
    "adapter",
    "adapter_config_sha256",
    "estimator_config_sha256",
    "estimator_artifact_sha256",
    "calibration_pool_sha256",
    "alpha",
    "evidence_schema_version",
    "evidence_sha256",
    "delta_hat",
    "epsilon",
    "action",
    "decision_timestamp_utc",
)
OFFLINE_FIELDS = (
    "run_id",
    "protocol_id",
    "unit_id",
    "delta",
    "risk_freeze",
    "risk_adapt",
    "oracle_action",
    "regret",
    "false_adapt",
    "balanced_accuracy",
    "macro_f1",
    "evaluation_timestamp_utc",
)
LABEL_BEARING_FIELDS = {
    "delta",
    "risk_freeze",
    "risk_adapt",
    "oracle_action",
    "regret",
    "false_adapt",
    "balanced_accuracy",
    "macro_f1",
    "label",
    "labels",
    "target_label",
    "target_labels",
    "y",
    "y_true",
}


class ContractError(ValueError):
    """Raised when a protocol or artifact violates the closure contract."""


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_fields(record: Mapping[str, Any], fields: Iterable[str], *, context: str) -> list[str]:
    return [f"{context}: missing required field {name!r}" for name in fields if name not in record]


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_git_sha(value: Any) -> bool:
    if not isinstance(value, str) or not 7 <= len(value) <= 40:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.endswith("Z"):
        return None
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        return None


def canonical_protocol_bytes(document: Mapping[str, Any]) -> bytes:
    """Return deterministic bytes for hashing a parsed protocol document."""

    return json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def protocol_sha256(document: Mapping[str, Any]) -> str:
    """Hash protocol meaning rather than YAML formatting."""

    return hashlib.sha256(canonical_protocol_bytes(document)).hexdigest()


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load a YAML protocol and require a mapping at the document root."""

    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ContractError(f"cannot read protocol {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContractError(f"protocol {source} must contain a mapping at its root")
    return document


def validate_protocol(document: Mapping[str, Any], *, require_sealed: bool = False) -> list[str]:
    """Return every structural or confirmatory-readiness violation."""

    errors = _require_fields(
        document,
        (
            "schema_version",
            "protocol_id",
            "status",
            "alpha",
            "model_seeds",
            "decision_rule",
            "inference",
            "primary_natural_track",
            "replication_tracks",
            "launcher_compatibility",
            "execution",
        ),
        context="protocol",
    )
    if errors:
        return errors

    if document["schema_version"] != SCHEMA_VERSION:
        errors.append(f"protocol: schema_version must be {SCHEMA_VERSION}, got {document['schema_version']!r}")
    if not isinstance(document["protocol_id"], str) or not document["protocol_id"].strip():
        errors.append("protocol: protocol_id must be a non-empty string")
    status = document["status"]
    if status not in PROTOCOL_STATUSES:
        errors.append(f"protocol: status must be one of {sorted(PROTOCOL_STATUSES)}, got {status!r}")
    alpha = document["alpha"]
    if not _finite_number(alpha) or not 0.0 < float(alpha) < 1.0:
        errors.append(f"protocol: alpha must be finite and in (0, 1), got {alpha!r}")

    seeds = document["model_seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(s, int) or s < 0 for s in seeds):
        errors.append("protocol: model_seeds must be a non-empty list of non-negative integers")
    elif len(seeds) != len(set(seeds)):
        errors.append("protocol: model_seeds contains duplicates")

    rule = document["decision_rule"]
    expected_rule = {
        "adapt": "delta_hat - epsilon > 0",
        "freeze": "delta_hat + epsilon < 0",
        "otherwise": "abstain",
    }
    if rule != expected_rule:
        errors.append(f"protocol: decision_rule must equal the canonical strict rule {expected_rule!r}")

    inference = document["inference"]
    if not isinstance(inference, Mapping):
        errors.append("protocol: inference must be a mapping")
    else:
        if inference.get("comparison_family") != ["kga_vs_always_adapt", "kga_vs_always_freeze"]:
            errors.append("protocol: inference comparison_family must contain the two fixed-policy comparisons")
        if inference.get("multiplicity") != "holm":
            errors.append("protocol: inference multiplicity must be 'holm'")
        if inference.get("confidence_level") != 0.95:
            errors.append("protocol: inference confidence_level must be 0.95")
        unit = inference.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            errors.append("protocol: inference unit must be a non-empty predeclared string")

    primary = document["primary_natural_track"]
    if not isinstance(primary, Mapping):
        errors.append("protocol: primary_natural_track must be a mapping")
    else:
        splits = primary.get("splits")
        if not isinstance(splits, Mapping):
            errors.append("protocol: primary_natural_track.splits must be a mapping")
        else:
            unknown = set(splits) - set(SPLIT_ROLES)
            if unknown:
                errors.append(f"protocol: unknown split roles: {sorted(unknown)}")
            seen: dict[str, str] = {}
            for role in SPLIT_ROLES:
                ids = splits.get(role, [])
                if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
                    errors.append(f"protocol: split {role!r} must be a list of non-empty unit IDs")
                    continue
                for unit_id in ids:
                    previous = seen.get(unit_id)
                    if previous is not None:
                        errors.append(f"protocol: unit {unit_id!r} overlaps split roles {previous!r} and {role!r}")
                    else:
                        seen[unit_id] = role

        if require_sealed:
            if not isinstance(primary.get("dataset"), str) or not primary.get("dataset", "").strip():
                errors.append("sealed protocol: primary natural dataset is not selected")
            if primary.get("provenance_status") != "UNOPENED_VERIFIED":
                errors.append("sealed protocol: natural test provenance must be UNOPENED_VERIFIED")
            if not primary.get("splits", {}).get("test"):
                errors.append("sealed protocol: primary natural test split is empty")
            for role in ("estimator_fit", "residual_calibration"):
                if not primary.get("splits", {}).get(role):
                    errors.append(f"sealed protocol: primary natural {role} split is empty")

    replications = document["replication_tracks"]
    if not isinstance(replications, list) or any(not isinstance(x, Mapping) for x in replications):
        errors.append("protocol: replication_tracks must be a list of mappings")

    compatibility = document["launcher_compatibility"]
    if not isinstance(compatibility, Mapping):
        errors.append("protocol: launcher_compatibility must be a mapping")
    elif require_sealed:
        unverified = sorted(name for name, state in compatibility.items() if state != "VERIFIED")
        if unverified:
            errors.append(f"sealed protocol: launchers are not VERIFIED: {unverified}")

    execution = document["execution"]
    if not isinstance(execution, Mapping):
        errors.append("protocol: execution must be a mapping")
    else:
        for stage in ("train", "evaluate"):
            commands = execution.get(stage)
            if not isinstance(commands, list):
                errors.append(f"protocol: execution.{stage} must be a list")
                continue
            for index, command in enumerate(commands):
                context = f"protocol: execution.{stage}[{index}]"
                if not isinstance(command, Mapping):
                    errors.append(f"{context} must be a mapping")
                    continue
                if not isinstance(command.get("name"), str) or not command.get("name", "").strip():
                    errors.append(f"{context}.name must be a non-empty string")
                argv = command.get("argv")
                if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv):
                    errors.append(f"{context}.argv must be a non-empty list of strings")
        if require_sealed:
            for stage in ("train", "evaluate"):
                if not execution.get(stage):
                    errors.append(f"sealed protocol: execution.{stage} has no commands")

    if require_sealed and status != "SEALED":
        errors.append(f"sealed protocol required, current status is {status!r}")
    return errors


def validate_decision_record(record: Mapping[str, Any]) -> list[str]:
    """Validate one label-free decision record and its canonical action."""

    errors = _require_fields(record, DECISION_FIELDS, context="decision")
    forbidden = sorted(LABEL_BEARING_FIELDS.intersection(record))
    if forbidden:
        errors.append(f"decision: label-bearing fields are forbidden: {forbidden}")
    if errors:
        return errors

    if record["split_role"] not in {"test", "replication"}:
        errors.append("decision: split_role must be 'test' or 'replication'")
    for field in (
        "run_id",
        "protocol_id",
        "dataset_version",
        "unit_id",
        "environment_id",
        "adapter",
        "evidence_schema_version",
    ):
        if not _nonempty_string(record[field]):
            errors.append(f"decision: {field} must be a non-empty string")
    if not _is_git_sha(record["git_sha"]):
        errors.append("decision: git_sha must be a 7-40 character hexadecimal Git object ID")
    if not isinstance(record["model_seed"], int) or isinstance(record["model_seed"], bool) or record["model_seed"] < 0:
        errors.append("decision: model_seed must be a non-negative integer")
    if _parse_utc(record["decision_timestamp_utc"]) is None:
        errors.append("decision: decision_timestamp_utc must be an ISO-8601 UTC timestamp ending in Z")
    if not _is_sha256(record["protocol_sha256"]):
        errors.append("decision: protocol_sha256 is not a SHA-256 digest")
    for field in (
        "checkpoint_sha256",
        "adapter_config_sha256",
        "estimator_config_sha256",
        "estimator_artifact_sha256",
        "calibration_pool_sha256",
        "evidence_sha256",
    ):
        if not _is_sha256(record[field]):
            errors.append(f"decision: {field} is not a SHA-256 digest")
    if not _finite_number(record["delta_hat"]):
        errors.append("decision: delta_hat must be finite")
    epsilon = record["epsilon"]
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or math.isnan(float(epsilon)):
        errors.append("decision: epsilon must be a non-negative number or +inf")
    elif float(epsilon) < 0.0:
        errors.append("decision: epsilon must be non-negative")
    if not _finite_number(record["alpha"]) or not 0.0 < float(record["alpha"]) < 1.0:
        errors.append("decision: alpha must be finite and in (0, 1)")
    if errors:
        return errors

    delta_hat = float(record["delta_hat"])
    eps = float(epsilon)
    if math.isinf(eps):
        expected = Decision.ABSTAIN.value
    elif delta_hat - eps > 0.0:
        expected = Decision.ADAPT.value
    elif delta_hat + eps < 0.0:
        expected = Decision.FREEZE.value
    else:
        expected = Decision.ABSTAIN.value
    if record["action"] != expected:
        errors.append(f"decision: action {record['action']!r} disagrees with canonical action {expected!r}")
    return errors


def validate_offline_record(record: Mapping[str, Any]) -> list[str]:
    """Validate an offline record after target labels are revealed."""

    errors = _require_fields(record, OFFLINE_FIELDS, context="offline")
    if errors:
        return errors
    for field in (
        "delta",
        "risk_freeze",
        "risk_adapt",
        "regret",
        "balanced_accuracy",
        "macro_f1",
    ):
        if not _finite_number(record[field]):
            errors.append(f"offline: {field} must be finite")
    for field in ("run_id", "protocol_id", "unit_id"):
        if not _nonempty_string(record[field]):
            errors.append(f"offline: {field} must be a non-empty string")
    if record["oracle_action"] not in {Decision.ADAPT.value, Decision.FREEZE.value}:
        errors.append("offline: oracle_action must be ADAPT or FREEZE")
    if not isinstance(record["false_adapt"], bool):
        errors.append("offline: false_adapt must be boolean")
    if _parse_utc(record["evaluation_timestamp_utc"]) is None:
        errors.append("offline: evaluation_timestamp_utc must be an ISO-8601 UTC timestamp ending in Z")
    if errors:
        return errors

    delta = float(record["delta"])
    risk_freeze = float(record["risk_freeze"])
    risk_adapt = float(record["risk_adapt"])
    if not math.isclose(delta, risk_freeze - risk_adapt, rel_tol=0.0, abs_tol=1e-10):
        errors.append("offline: delta must equal risk_freeze - risk_adapt")
    expected_oracle = Decision.ADAPT.value if delta > 0.0 else Decision.FREEZE.value
    if record["oracle_action"] != expected_oracle:
        errors.append(
            f"offline: oracle_action {record['oracle_action']!r} disagrees with delta; expected {expected_oracle!r}"
        )
    if float(record["regret"]) < -1e-12:
        errors.append("offline: regret must be non-negative")
    for field in ("balanced_accuracy", "macro_f1"):
        value = float(record[field])
        if not 0.0 <= value <= 1.0:
            errors.append(f"offline: {field} must be in [0, 1]")
    return errors


def validate_joined_records(
    decisions: Iterable[Mapping[str, Any]],
    offline: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Validate the one-to-one offline label join and recompute derived fields.

    The join key is ``(run_id, protocol_id, unit_id)``. ``ABSTAIN`` means the
    candidate update is not committed, so its deployed risk is the frozen risk.
    """

    errors: list[str] = []
    decision_index: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    offline_index: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}

    for index, row in enumerate(decisions):
        row_errors = validate_decision_record(row)
        errors.extend(f"decision[{index}]: {error}" for error in row_errors)
        key = (row.get("run_id"), row.get("protocol_id"), row.get("unit_id"))
        if key in decision_index:
            errors.append(f"decision: duplicate join key {key!r}")
        decision_index[key] = row
    for index, row in enumerate(offline):
        row_errors = validate_offline_record(row)
        errors.extend(f"offline[{index}]: {error}" for error in row_errors)
        key = (row.get("run_id"), row.get("protocol_id"), row.get("unit_id"))
        if key in offline_index:
            errors.append(f"offline: duplicate join key {key!r}")
        offline_index[key] = row

    missing_offline = sorted(set(decision_index) - set(offline_index), key=repr)
    missing_decisions = sorted(set(offline_index) - set(decision_index), key=repr)
    if missing_offline:
        errors.append(f"join: decision rows without offline labels: {missing_offline!r}")
    if missing_decisions:
        errors.append(f"join: offline rows without decisions: {missing_decisions!r}")

    for key in set(decision_index).intersection(offline_index):
        decision_row = decision_index[key]
        offline_row = offline_index[key]
        if validate_decision_record(decision_row) or validate_offline_record(offline_row):
            continue
        action = str(decision_row["action"])
        delta = float(offline_row["delta"])
        risk_freeze = float(offline_row["risk_freeze"])
        risk_adapt = float(offline_row["risk_adapt"])
        deployed_risk = risk_adapt if action == Decision.ADAPT.value else risk_freeze
        expected_regret = deployed_risk - min(risk_freeze, risk_adapt)
        if not math.isclose(float(offline_row["regret"]), expected_regret, rel_tol=0.0, abs_tol=1e-10):
            errors.append(f"join {key!r}: regret must be {expected_regret:.12g} for action {action}")
        expected_false_adapt = action == Decision.ADAPT.value and delta <= 0.0
        if offline_row["false_adapt"] is not expected_false_adapt:
            errors.append(f"join {key!r}: false_adapt must be {expected_false_adapt} for action/delta")
        decision_time = _parse_utc(decision_row["decision_timestamp_utc"])
        evaluation_time = _parse_utc(offline_row["evaluation_timestamp_utc"])
        if decision_time is not None and evaluation_time is not None and evaluation_time < decision_time:
            errors.append(f"join {key!r}: offline evaluation precedes the live decision")
    return errors


def write_new_jsonl(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    validator: Callable[[Mapping[str, Any]], list[str]],
) -> int:
    """Validate and atomically create a JSONL artifact without overwriting evidence."""

    destination = Path(path)
    rows = list(records)
    if not rows:
        raise ContractError("refusing to create an empty evidence artifact")
    for index, row in enumerate(rows):
        errors = validator(row)
        if errors:
            raise ContractError(f"record {index} failed validation:\n  " + "\n  ".join(errors))

    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(destination, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return len(rows)


def assert_valid(errors: Iterable[str]) -> None:
    """Raise one readable exception for a validation result."""

    failures = list(errors)
    if failures:
        raise ContractError("contract validation failed:\n  " + "\n  ".join(failures))
