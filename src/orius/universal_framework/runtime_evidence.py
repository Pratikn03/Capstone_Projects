"""Registry-backed runtime evidence defaults for universal domain adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:  # pragma: no cover - fallback kept for non-repo packaging contexts
    from scripts._dataset_registry import get_runtime_dataset_config, get_runtime_source_label
except ImportError:  # pragma: no cover
    get_runtime_dataset_config = None
    get_runtime_source_label = None


_DOMAIN_ALIASES = {
    "av": "vehicle",
    "energy": "battery",
}

_MATURITY_TO_CLOSURE = {
    "reference": "reference",
    "proof_validated": "defended_bounded_row",
    "shadow_synthetic": "shadow_synthetic",
    "experimental": "experimental",
}

_SURFACE_LABELS: dict[str, dict[str, str]] = {
    "battery": {
        "canonical": "opsd_germany_reference_runtime",
        "support": "opsd_germany_reference_runtime",
        "missing": "opsd_germany_reference_runtime",
    },
    "vehicle": {
        "canonical": "waymo_motion_replay_surrogate",
        "support": "waymo_motion_replay_surrogate",
        "missing": "waymo_motion_replay_surrogate",
    },
    "industrial": {
        "canonical": "uci_ccpp_processed_replay_surrogate",
        "support": "uci_ccpp_processed_replay_surrogate",
        "missing": "uci_ccpp_processed_replay_surrogate",
    },
    "healthcare": {
        "canonical": "bidmc_processed_replay_surrogate",
        "support": "bidmc_processed_replay_surrogate",
        "missing": "bidmc_processed_replay_surrogate",
    },
    "navigation": {
        "canonical": "kitti_odometry_replay",
        "support": "kitti_runtime_shadow_support",
        "missing": "kitti_runtime_shadow_support",
    },
    "aerospace": {
        "canonical": "aerospace_realflight_runtime",
        "support": "public_adsb_runtime_support_lane",
        "missing": "aerospace_realflight_unstaged",
    },
}

_FALLBACK_EVIDENCE = {
    "battery": {
        "maturity_tier": "reference",
        "closure_tier": "reference",
        "fallback_policy": "paper6_runtime",
        "exact_blocker": "battery_reference_witness",
        "runtime_source_label": "canonical",
    },
    "vehicle": {
        "maturity_tier": "proof_validated",
        "closure_tier": "defended_bounded_row",
        "fallback_policy": "bounded_runtime_pass",
        "exact_blocker": "av_real_row_present",
        "runtime_source_label": "canonical",
    },
    "industrial": {
        "maturity_tier": "proof_validated",
        "closure_tier": "defended_bounded_row",
        "fallback_policy": "bounded_runtime_pass",
        "exact_blocker": "industrial_train_validation_chain_complete",
        "runtime_source_label": "canonical",
    },
    "healthcare": {
        "maturity_tier": "proof_validated",
        "closure_tier": "defended_bounded_row",
        "fallback_policy": "bounded_runtime_pass",
        "exact_blocker": "healthcare_train_validation_chain_complete",
        "runtime_source_label": "canonical",
    },
    "navigation": {
        "maturity_tier": "shadow_synthetic",
        "closure_tier": "shadow_synthetic",
        "fallback_policy": "shadow_synthetic_support_tier",
        "exact_blocker": "navigation_kitti_runtime_missing",
        "runtime_source_label": "missing",
    },
    "aerospace": {
        "maturity_tier": "experimental",
        "closure_tier": "experimental",
        "fallback_policy": "experimental_support_lane",
        "exact_blocker": "aerospace_realflight_runtime_missing",
        "runtime_source_label": "support",
    },
}


@dataclass(frozen=True)
class RuntimeEvidence:
    domain_id: str
    runtime_surface: str
    closure_tier: str
    maturity_tier: str
    fallback_policy: str
    exact_blocker: str
    runtime_source_label: str


def _normalize_domain_id(domain_id: str) -> str:
    normalized = str(domain_id).strip().lower()
    return _DOMAIN_ALIASES.get(normalized, normalized)


def _surface_label(domain_id: str, runtime_source_label: str) -> str:
    labels = _SURFACE_LABELS.get(domain_id, {})
    return labels.get(runtime_source_label, labels.get("missing", f"{domain_id}_runtime"))


def resolve_runtime_evidence(
    domain_id: str,
    cfg: Mapping[str, Any] | None = None,
) -> RuntimeEvidence:
    """Resolve default runtime truth for a domain adapter from the dataset registry."""

    normalized = _normalize_domain_id(domain_id)
    cfg_dict = dict(cfg or {})
    fallback = dict(_FALLBACK_EVIDENCE.get(normalized, {}))
    runtime_source_label = str(fallback.get("runtime_source_label", "missing"))
    maturity_tier = str(fallback.get("maturity_tier", "portability_only"))
    closure_tier = str(fallback.get("closure_tier", maturity_tier))
    fallback_policy = str(fallback.get("fallback_policy", "unspecified"))
    exact_blocker = str(fallback.get("exact_blocker", ""))

    if get_runtime_dataset_config is not None:
        try:
            registry_cfg = get_runtime_dataset_config(normalized)
        except KeyError:
            registry_cfg = None
        if registry_cfg is not None:
            maturity_tier = str(registry_cfg.maturity_tier or maturity_tier)
            closure_tier = _MATURITY_TO_CLOSURE.get(
                maturity_tier,
                str(registry_cfg.closure_target_tier or closure_tier),
            )
            fallback_policy = str(registry_cfg.fallback_policy or fallback_policy)
            exact_blocker = str(registry_cfg.exact_blocker or exact_blocker)
            if get_runtime_source_label is not None:
                runtime_source_label = str(get_runtime_source_label(normalized, allow_support_tier=True))

    runtime_surface = _surface_label(normalized, runtime_source_label)
    if cfg_dict.get("runtime_surface") not in (None, ""):
        runtime_surface = str(cfg_dict["runtime_surface"])
    if cfg_dict.get("closure_tier") not in (None, ""):
        closure_tier = str(cfg_dict["closure_tier"])

    return RuntimeEvidence(
        domain_id=normalized,
        runtime_surface=runtime_surface,
        closure_tier=closure_tier,
        maturity_tier=maturity_tier,
        fallback_policy=fallback_policy,
        exact_blocker=exact_blocker,
        runtime_source_label=runtime_source_label,
    )
