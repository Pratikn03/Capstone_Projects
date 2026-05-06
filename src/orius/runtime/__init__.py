"""Canonical runtime release-contract surfaces."""

from .release_contract import (
    CANONICAL_RELEASE_EVIDENCE_FIELDS,
    PROMOTED_RELEASE_DOMAINS,
    RELEASE_CONTRACT_SCHEMA_VERSION,
    RuntimeReleaseEvidence,
    build_release_evidence,
    normalize_release_evidence,
    validate_release_evidence,
)

__all__ = [
    "CANONICAL_RELEASE_EVIDENCE_FIELDS",
    "PROMOTED_RELEASE_DOMAINS",
    "RELEASE_CONTRACT_SCHEMA_VERSION",
    "RuntimeReleaseEvidence",
    "build_release_evidence",
    "normalize_release_evidence",
    "validate_release_evidence",
]
