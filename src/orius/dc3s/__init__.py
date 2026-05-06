"""DC3S: Drift-Calibrated Conformal Safety Shield components."""

from .battery_adapter import BatteryDomainAdapter
from .calibration import (
    build_uncertainty_set,
    derived_inflation_factor,
    effective_sample_size,
    inflate_interval,
    inflate_q,
)
from .certificate import (
    CERTIFICATE_SCHEMA_VERSION,
    compute_config_hash,
    compute_model_hash,
    get_certificate,
    make_certificate,
    normalize_certificate_schema,
    recompute_certificate_hash,
    store_certificate,
    verify_certificate,
    verify_certificate_chain,
)
from .domain_adapter import DomainAdapter
from .drift import AdaptivePageHinkleyDetector, PageHinkleyDetector
from .ftit import FTIT_FAULT_KEYS, preview_fault_state
from .ftit import update as update_ftit_state
from .half_life import (
    compute_certificate_state,
    compute_half_life_from_horizon,
    compute_validity_horizon,
)
from .online_calibration import OnlineCalibrator, calibration_contract_check
from .pipeline import run_dc3s_step
from .quality import compute_reliability
from .rac_cert import (
    RACCertConfig,
    RACCertModel,
    compute_dispatch_sensitivity,
    compute_inflation,
    compute_q_multiplier,
    normalize_sensitivity,
)
from .reachability import (
    compute_expiration_bound,
    compute_validity_horizon_from_reachability,
    propagate_reachability_set,
)
from .safety_filter_theory import (
    check_tightened_soc_invariance,
    reliability_error_bound,
    safety_filter_projection_summary,
    tightened_soc_bounds,
)
from .shield import repair_action
from .state import DC3SStateStore
from .temporal_theorems import (
    certificate_expiration_bound,
    certificate_half_life,
    certificate_validity_horizon,
    certify_fallback_existence,
    evaluate_graceful_degradation_dominance,
    forward_tube,
    should_expire_certificate,
    should_renew_certificate,
    zero_dispatch_fallback,
)
from .theoretical_guarantees import (
    THEOREM_REGISTER,
    TransferContractResult,
    assert_finite_sample_bound,
    assert_separation,
    assert_sublinear_regret,
    compute_adaptive_regret_bound,
    compute_coverage_bound_surface,
    compute_finite_sample_coverage_bound,
    compute_separation_gap,
    compute_stylized_frontier_lower_bound,
    compute_universal_impossibility_bound,
    evaluate_structural_transfer,
    simulate_adaptive_tracking,
    simulate_separation_construction,
)


def __getattr__(name):
    """Lazy-load DeepOQE symbols (requires torch)."""
    if name in ("DeepOQEConfig", "DeepOQEModel", "DEEP_OQE_FEATURE_NAMES"):
        from .deep_oqe import FEATURE_NAMES as _FN
        from .deep_oqe import DeepOQEConfig, DeepOQEModel

        _map = {"DeepOQEConfig": DeepOQEConfig, "DeepOQEModel": DeepOQEModel, "DEEP_OQE_FEATURE_NAMES": _FN}
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CERTIFICATE_SCHEMA_VERSION",
    "DEEP_OQE_FEATURE_NAMES",
    "FTIT_FAULT_KEYS",
    "THEOREM_REGISTER",
    "AdaptivePageHinkleyDetector",
    "BatteryDomainAdapter",
    "DC3SStateStore",
    "DeepOQEConfig",
    "DeepOQEModel",
    "DomainAdapter",
    "OnlineCalibrator",
    "PageHinkleyDetector",
    "RACCertConfig",
    "RACCertModel",
    "TransferContractResult",
    "assert_finite_sample_bound",
    "assert_separation",
    "assert_sublinear_regret",
    "build_uncertainty_set",
    "calibration_contract_check",
    "certificate_expiration_bound",
    "certificate_half_life",
    "certificate_validity_horizon",
    "certify_fallback_existence",
    "check_tightened_soc_invariance",
    "compute_adaptive_regret_bound",
    "compute_certificate_state",
    "compute_config_hash",
    "compute_coverage_bound_surface",
    "compute_dispatch_sensitivity",
    "compute_expiration_bound",
    "compute_finite_sample_coverage_bound",
    "compute_half_life_from_horizon",
    "compute_inflation",
    "compute_model_hash",
    "compute_q_multiplier",
    "compute_reliability",
    "compute_separation_gap",
    "compute_stylized_frontier_lower_bound",
    "compute_universal_impossibility_bound",
    "compute_validity_horizon",
    "compute_validity_horizon_from_reachability",
    "derived_inflation_factor",
    "effective_sample_size",
    "evaluate_graceful_degradation_dominance",
    "evaluate_structural_transfer",
    "forward_tube",
    "get_certificate",
    "inflate_interval",
    "inflate_q",
    "make_certificate",
    "normalize_certificate_schema",
    "normalize_sensitivity",
    "preview_fault_state",
    "propagate_reachability_set",
    "recompute_certificate_hash",
    "reliability_error_bound",
    "repair_action",
    "run_dc3s_step",
    "safety_filter_projection_summary",
    "should_expire_certificate",
    "should_renew_certificate",
    "simulate_adaptive_tracking",
    "simulate_separation_construction",
    "store_certificate",
    "tightened_soc_bounds",
    "update_ftit_state",
    "verify_certificate",
    "verify_certificate_chain",
    "zero_dispatch_fallback",
]
