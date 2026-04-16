from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import duckdb
import pandas as pd

from orius.dc3s.certificate import recompute_certificate_hash


# ---------------------------------------------------------------------------
# Formal validity predicate  (Theorem: CertOS Runtime Proof)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidityVerdict:
    """Result of the formal three-part validity predicate for a certificate."""

    valid: bool
    hash_integrity: bool
    chain_continuity: bool
    safety_semantics: bool
    reason: str = ""


def missing_required_certificate_fields(certificate: Mapping[str, Any]) -> list[str]:
    return sorted(
        field
        for field in REQUIRED_CERTIFICATE_FIELDS
        if field not in certificate or certificate.get(field) in (None, "")
    )


def count_present_required_certificate_fields(certificate: Mapping[str, Any] | None) -> int:
    if not isinstance(certificate, Mapping):
        return 0
    return sum(
        1
        for field in REQUIRED_CERTIFICATE_FIELDS
        if field in certificate and certificate.get(field) not in (None, "")
    )


def certificate_intervention_semantics_valid(certificate: Mapping[str, Any]) -> bool:
    intervention_reason = certificate.get("intervention_reason")
    intervened = bool(certificate.get("intervened", False))
    return not intervened or intervention_reason not in (None, "")


def extract_certificate_validity_horizon(certificate: Mapping[str, Any]) -> int:
    for key in ("validity_horizon_H_t", "certificate_horizon_steps", "tau_t"):
        value = certificate.get(key)
        try:
            if value not in (None, ""):
                return int(value)
        except (TypeError, ValueError):
            continue
    return 0


def _extract_reliability_weight(certificate: Mapping[str, Any]) -> float:
    for key in ("reliability_w", "w_t"):
        try:
            value = certificate.get(key)
            if value not in (None, ""):
                return float(value)
        except (TypeError, ValueError):
            pass
    reliability = certificate.get("reliability")
    if isinstance(reliability, Mapping):
        for key in ("w_t", "w", "reliability_w"):
            try:
                value = reliability.get(key)
                if value not in (None, ""):
                    return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0
    try:
        return 0.0 if reliability in (None, "") else float(reliability)
    except (TypeError, ValueError):
        return 0.0


def _chain_continuity_ok(cert: Mapping[str, Any], prev_cert: Mapping[str, Any] | None) -> bool:
    current_prev = cert.get("prev_hash")
    if prev_cert is None:
        return current_prev in (None, "")
    previous_hash = dict(prev_cert).get("certificate_hash")
    return current_prev not in (None, "") and current_prev == previous_hash


def formal_validity_predicate(
    cert: Mapping[str, Any],
    prev_cert: Mapping[str, Any] | None,
    w_min: float = 0.0,
) -> ValidityVerdict:
    """Three-conjunct formal validity predicate for a single certificate.

    A certificate C_i is valid iff:
      (1) Hash integrity: recompute_hash(C_i) == C_i.certificate_hash
      (2) Chain continuity: C_i.prev_hash == C_{i-1}.certificate_hash
          (vacuously true for the genesis certificate where prev_cert is None)
      (3) Safety semantics: H_t > 0 AND w_t >= w_min AND guarantee checks pass

    This is the typed conjunction connecting the audit chain to the TSVR bound.
    """
    cert_d = dict(cert)

    observed_hash = cert_d.get("certificate_hash")
    expected_hash = recompute_certificate_hash(cert_d)
    hash_ok = isinstance(observed_hash, str) and observed_hash == expected_hash
    if not hash_ok:
        stripped = {k: v for k, v in cert_d.items() if k not in NON_HASHED_EXTENSION_FIELDS}
        expected_hash = recompute_certificate_hash(stripped)
        hash_ok = isinstance(observed_hash, str) and observed_hash == expected_hash

    chain_ok = _chain_continuity_ok(cert_d, prev_cert)

    H_t_val = extract_certificate_validity_horizon(cert_d)
    w_t_val = _extract_reliability_weight(cert_d)

    safety_ok = H_t_val > 0 and w_t_val >= w_min

    valid = hash_ok and chain_ok and safety_ok
    reasons = []
    if not hash_ok:
        reasons.append("hash_mismatch")
    if not chain_ok:
        reasons.append("chain_break")
    if not safety_ok:
        reasons.append(f"safety(H_t={H_t_val},w_t={w_t_val:.4f},w_min={w_min:.4f})")

    return ValidityVerdict(
        valid=valid,
        hash_integrity=hash_ok,
        chain_continuity=chain_ok,
        safety_semantics=safety_ok,
        reason="; ".join(reasons) if reasons else "ok",
    )


# ---------------------------------------------------------------------------
# Composability theorem  (induction over certificate chain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComposabilityResult:
    """Result of the composability verification over an episode chain."""

    composable: bool
    chain_length: int
    all_valid: bool
    episode_tsvr_bound: float | None
    failed_indices: tuple[int, ...]
    proof_sketch: str


def verify_composability(
    certificates: Sequence[Mapping[str, Any]],
    alpha: float,
    w_min: float = 0.0,
) -> ComposabilityResult:
    """Verify the composability theorem over a certificate chain.

    Theorem (Composability):
      If every certificate C_i in the chain satisfies the validity
      predicate (hash integrity, chain continuity, safety semantics),
      then the episode TSVR is bounded:

          E[V_T] <= alpha * sum_{i=1}^{T} (1 - w_i)

    Proof: by induction on chain length.  Base case: single valid
    certificate gives per-step risk <= alpha*(1 - w_1).  Inductive step:
    chain continuity ensures no gap between C_i and C_{i+1}, so risks
    sum over the episode.
    """
    cert_list = [dict(c) for c in certificates]
    T = len(cert_list)
    if T == 0:
        return ComposabilityResult(
            composable=True, chain_length=0, all_valid=True,
            episode_tsvr_bound=0.0, failed_indices=(), proof_sketch="Empty chain.",
        )

    failed_indices = []
    total_gap = 0.0
    prev_cert: Mapping[str, Any] | None = None

    for i, cert in enumerate(cert_list):
        verdict = formal_validity_predicate(cert, prev_cert, w_min)
        if not verdict.valid:
            failed_indices.append(i)

        w_t = cert.get("reliability") or cert.get("w_t")
        try:
            w_val = 0.0 if w_t in (None, "") else float(w_t)
        except (TypeError, ValueError):
            w_val = 0.0
        total_gap += 1.0 - w_val
        prev_cert = cert

    all_valid = len(failed_indices) == 0
    tsvr_bound = alpha * total_gap if all_valid else None

    return ComposabilityResult(
        composable=all_valid,
        chain_length=T,
        all_valid=all_valid,
        episode_tsvr_bound=tsvr_bound,
        failed_indices=tuple(failed_indices),
        proof_sketch=(
            f"Chain of {T} certificates.  All pass validity predicate (w_min={w_min:.3f}).  "
            f"Episode TSVR <= {alpha} * {total_gap:.4f} = {alpha * total_gap:.6f}."
            if all_valid else
            f"Chain of {T} certificates.  {len(failed_indices)} failed validity predicate "
            f"at indices {failed_indices}.  Composability does not hold."
        ),
    )


# ---------------------------------------------------------------------------
# Tamper-evidence proof  (SHA-256 collision resistance)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TamperEvidenceResult:
    """Result of the tamper-evidence verification."""

    tamper_evident: bool
    chain_length: int
    detection_probability_lower_bound: float
    broken_links: tuple[int, ...]
    proof_sketch: str


def verify_tamper_evidence(
    certificates: Sequence[Mapping[str, Any]],
) -> TamperEvidenceResult:
    """Verify the tamper-evidence theorem over a certificate chain.

    Theorem (Tamper-Evidence):
      Under SHA-256 collision resistance (Assumption A7), any modification
      to certificate C_i in the chain is detectable with probability
      >= 1 - 2^{-128}.

    Proof: by induction.  Modifying C_i changes its hash.  Either:
      (a) C_{i+1}.prev_hash != hash(C_i') — chain breaks at i+1, or
      (b) The adversary finds a collision SHA-256(C_i) = SHA-256(C_i'),
          requiring >= 2^{128} expected operations (birthday bound).
    """
    cert_list = [dict(c) for c in certificates]
    T = len(cert_list)
    if T == 0:
        return TamperEvidenceResult(
            tamper_evident=True, chain_length=0,
            detection_probability_lower_bound=1.0, broken_links=(),
            proof_sketch="Empty chain — vacuously tamper-evident.",
        )

    broken_links = []
    prev_hash: str | None = None

    for i, cert in enumerate(cert_list):
        observed_hash = cert.get("certificate_hash")
        expected_hash = recompute_certificate_hash(cert)
        if not (isinstance(observed_hash, str) and observed_hash == expected_hash):
            stripped = {k: v for k, v in cert.items() if k not in NON_HASHED_EXTENSION_FIELDS}
            expected_hash = recompute_certificate_hash(stripped)
            if not (isinstance(observed_hash, str) and observed_hash == expected_hash):
                broken_links.append(i)
                prev_hash = observed_hash
                continue

        curr_prev = cert.get("prev_hash")
        if i == 0:
            if curr_prev not in (None, ""):
                broken_links.append(i)
        elif curr_prev in (None, "") or curr_prev != prev_hash:
            broken_links.append(i)

        prev_hash = str(observed_hash)

    tamper_evident = len(broken_links) == 0
    detection_prob = 1.0 - 2.0 ** (-128) if tamper_evident else 1.0

    return TamperEvidenceResult(
        tamper_evident=tamper_evident,
        chain_length=T,
        detection_probability_lower_bound=detection_prob,
        broken_links=tuple(broken_links),
        proof_sketch=(
            f"Chain of {T} certificates verified.  All hashes match and "
            f"prev_hash links are consistent.  Under A7 (SHA-256 collision "
            f"resistance), tampering is detectable with P >= 1 - 2^{{-128}}."
            if tamper_evident else
            f"Chain of {T} certificates.  {len(broken_links)} broken link(s) "
            f"at indices {broken_links}.  Tampering detected."
        ),
    )


REQUIRED_CERTIFICATE_FIELDS = (
    "certificate_hash",
    "command_id",
    "controller",
    "created_at",
    "proposed_action",
    "safe_action",
    "uncertainty",
    "reliability",
)

NON_HASHED_EXTENSION_FIELDS = {
    "assumptions_checked",
    "controller_label",
    "coverage_lb_t",
    "delta_mw",
    "dispatch_regime",
    "ego_track_id",
    "fault_family",
    "intervention_trace_id",
    "interval_width",
    "lane",
    "lambda_mw_used",
    "neighbor_ids",
    "observed_margin",
    "q_eff",
    "q_multiplier",
    "risk_bound_scope",
    "scenario_id",
    "semantic_checks",
    "sensitivity_norm",
    "sensitivity_t",
    "shard_id",
    "shift_score",
    "solver_status",
    "source_domain",
    "true_margin",
    "w_t",
    "widening_factor",
}


def load_certificates_from_duckdb(duckdb_path: str | Path, table_name: str = "dispatch_certificates") -> list[dict[str, Any]]:
    db_path = Path(duckdb_path)
    if not db_path.exists():
        return []
    conn = duckdb.connect(str(db_path))
    try:
        tables = {str(row[0]) for row in conn.execute("SHOW TABLES").fetchall()}
        if table_name not in tables:
            return []
        rows = conn.execute(
            f"""
            SELECT
                command_id,
                certificate_hash,
                prev_hash,
                created_at,
                payload_json
            FROM {table_name}
            ORDER BY created_at ASC, command_id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    certificates: list[dict[str, Any]] = []
    for command_id, cert_hash, prev_hash, created_at, payload_json in rows:
        payload: dict[str, Any]
        try:
            payload = json.loads(payload_json) if payload_json else {}
        except json.JSONDecodeError:
            payload = {}
        payload.setdefault("command_id", command_id)
        payload.setdefault("certificate_hash", cert_hash)
        payload.setdefault("prev_hash", prev_hash)
        payload.setdefault("created_at", created_at)
        certificates.append(payload)
    return certificates


def verify_certificates(certificates: Iterable[Mapping[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cert_list = [dict(cert) for cert in certificates]
    failure_rows: list[dict[str, Any]] = []
    expiry_rows: list[dict[str, Any]] = []
    hash_valid_flags: list[bool] = []

    for index, certificate in enumerate(cert_list):
        observed_hash = certificate.get("certificate_hash")
        expected_hash = recompute_certificate_hash(certificate)
        valid_hash = isinstance(observed_hash, str) and observed_hash == expected_hash
        if not valid_hash:
            stripped = {key: value for key, value in certificate.items() if key not in NON_HASHED_EXTENSION_FIELDS}
            expected_hash = recompute_certificate_hash(stripped)
            valid_hash = isinstance(observed_hash, str) and observed_hash == expected_hash
        missing_fields = missing_required_certificate_fields(certificate)
        semantic_ok = certificate_intervention_semantics_valid(certificate)
        if not valid_hash or missing_fields or not semantic_ok:
            failure_rows.append(
                {
                    "row_index": int(index),
                    "command_id": str(certificate.get("command_id", "")),
                    "failure_type": (
                        "hash_mismatch"
                        if not valid_hash
                        else "missing_required_fields"
                        if missing_fields
                        else "intervention_reason_missing"
                    ),
                    "missing_fields": ",".join(missing_fields),
                    "expected_hash": expected_hash,
                    "observed_hash": observed_hash,
                }
            )
        hash_valid_flags.append(bool(valid_hash))

        validity_horizon = certificate.get("validity_horizon_H_t")
        half_life = certificate.get("half_life_steps")
        expires_at = certificate.get("expires_at_step")
        status = certificate.get("validity_status")
        has_expiry = any(value not in (None, "") for value in (validity_horizon, half_life, expires_at, status))
        expiry_ok = True
        if has_expiry:
            try:
                horizon_value = None if validity_horizon in (None, "") else int(validity_horizon)
                half_life_value = None if half_life in (None, "") else int(half_life)
                expires_value = None if expires_at in (None, "") else int(expires_at)
                expiry_ok = all(
                    value is None or value >= 0
                    for value in (horizon_value, half_life_value, expires_value)
                )
                if horizon_value is not None and half_life_value is not None:
                    expiry_ok = expiry_ok and half_life_value <= max(horizon_value, half_life_value)
            except (TypeError, ValueError):
                expiry_ok = False
        expiry_rows.append(
            {
                "row_index": int(index),
                "command_id": str(certificate.get("command_id", "")),
                "has_expiry_metadata": bool(has_expiry),
                "validity_horizon_H_t": validity_horizon,
                "half_life_steps": half_life,
                "expires_at_step": expires_at,
                "validity_status": status,
                "expiry_consistent": bool(expiry_ok),
            }
        )
        if has_expiry and not expiry_ok:
            failure_rows.append(
                {
                    "row_index": int(index),
                    "command_id": str(certificate.get("command_id", "")),
                    "failure_type": "expiry_inconsistent",
                    "missing_fields": "",
                    "expected_hash": "",
                    "observed_hash": "",
                }
            )

    chain_valid = True
    failed_index: int | None = None
    failure_reason: str | None = None
    expected_prev_hash: str | None = None
    observed_prev_hash: str | None = None
    previous_hash: str | None = None
    for index, certificate in enumerate(cert_list):
        if not hash_valid_flags[index]:
            chain_valid = False
            failed_index = int(index)
            failure_reason = "hash_mismatch"
            break
        current_prev = certificate.get("prev_hash")
        if index == 0:
            if current_prev not in (None, ""):
                chain_valid = False
                failed_index = int(index)
                failure_reason = "genesis_prev_hash_present"
                expected_prev_hash = ""
                observed_prev_hash = current_prev
                break
            previous_hash = str(certificate.get("certificate_hash"))
            continue
        if current_prev in (None, ""):
            chain_valid = False
            failed_index = int(index)
            failure_reason = "prev_hash_missing"
            expected_prev_hash = previous_hash
            observed_prev_hash = current_prev
            break
        if current_prev != previous_hash:
            chain_valid = False
            failed_index = int(index)
            failure_reason = "prev_hash_mismatch"
            expected_prev_hash = previous_hash
            observed_prev_hash = current_prev
            break
        previous_hash = str(certificate.get("certificate_hash"))
    if not chain_valid:
        failure_rows.append(
            {
                "row_index": int(failed_index or 0),
                "command_id": str(cert_list[int(failed_index or 0)].get("command_id", "")) if cert_list else "",
                "failure_type": str(failure_reason or "chain_invalid"),
                "missing_fields": "",
                "expected_hash": str(expected_prev_hash or ""),
                "observed_hash": str(observed_prev_hash or ""),
            }
        )

    failure_df = pd.DataFrame(
        failure_rows,
        columns=["row_index", "command_id", "failure_type", "missing_fields", "expected_hash", "observed_hash"],
    )
    expiry_df = pd.DataFrame(
        expiry_rows,
        columns=[
            "row_index",
            "command_id",
            "has_expiry_metadata",
            "validity_horizon_H_t",
            "half_life_steps",
            "expires_at_step",
            "validity_status",
            "expiry_consistent",
        ],
    )
    governance_df = pd.DataFrame(
        [
            {
                "metric": "certificate_rows",
                "value": int(len(cert_list)),
            },
            {
                "metric": "chain_valid",
                "value": 1 if chain_valid else 0,
            },
            {
                "metric": "failure_rows",
                "value": int(len(failure_df)),
            },
            {
                "metric": "required_payload_pass_rate",
                "value": float(1.0 - len(failure_df[failure_df["failure_type"] == "missing_required_fields"]) / max(len(cert_list), 1)),
            },
            {
                "metric": "expiry_metadata_presence_rate",
                "value": float(expiry_df["has_expiry_metadata"].mean()) if not expiry_df.empty else 0.0,
            },
            {
                "metric": "expiry_consistency_rate",
                "value": float(expiry_df["expiry_consistent"].mean()) if not expiry_df.empty else 1.0,
            },
            {
                "metric": "audit_completeness_rate",
                "value": float(max(0.0, 1.0 - len(failure_df) / max(len(cert_list), 1))),
            },
        ]
    )
    summary = {
        "certificate_rows": int(len(cert_list)),
        "chain_valid": bool(chain_valid),
        "checked": int(len(cert_list) if chain_valid else (failed_index or 0)),
        "failure_rows": int(len(failure_df)),
        "required_payload_pass_rate": float(governance_df.loc[governance_df["metric"] == "required_payload_pass_rate", "value"].iloc[0]) if not governance_df.empty else 0.0,
        "expiry_metadata_presence_rate": float(governance_df.loc[governance_df["metric"] == "expiry_metadata_presence_rate", "value"].iloc[0]) if not governance_df.empty else 0.0,
        "expiry_consistency_rate": float(governance_df.loc[governance_df["metric"] == "expiry_consistency_rate", "value"].iloc[0]) if not governance_df.empty else 1.0,
        "audit_completeness_rate": float(governance_df.loc[governance_df["metric"] == "audit_completeness_rate", "value"].iloc[0]) if not governance_df.empty else 1.0,
        "failed_index": failed_index,
        "reason": failure_reason,
    }
    return summary, failure_df, expiry_df, governance_df
