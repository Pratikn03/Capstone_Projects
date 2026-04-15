"""DC3S: Drift-Calibrated Conformal Safety Shield components."""

from .quality import compute_reliability
from .deep_oqe import DeepOQEConfig, DeepOQEModel, FEATURE_NAMES as DEEP_OQE_FEATURE_NAMES
from .ftit import FTIT_FAULT_KEYS, preview_fault_state, update as update_ftit_state
from .drift import PageHinkleyDetector, AdaptivePageHinkleyDetector
from .calibration import inflate_interval, inflate_q, build_uncertainty_set
from .shield import repair_action
from .certificate import (
    make_certificate,
    store_certificate,
    get_certificate,
    compute_model_hash,
    compute_config_hash,
    recompute_certificate_hash,
    verify_certificate,
    verify_certificate_chain,
)
from .state import DC3SStateStore
from .rac_cert import (
    RACCertConfig,
    RACCertModel,
    compute_dispatch_sensitivity,
    compute_q_multiplier,
    compute_inflation,
    normalize_sensitivity,
)
from .safety_filter_theory import (
    reliability_error_bound,
    tightened_soc_bounds,
    check_tightened_soc_invariance,
    safety_filter_projection_summary,
)
from .temporal_theorems import (
    forward_tube,
    certificate_validity_horizon,
    certificate_expiration_bound,
    zero_dispatch_fallback,
    certify_fallback_existence,
    evaluate_graceful_degradation_dominance,
    certificate_half_life,
    should_renew_certificate,
    should_expire_certificate,
)
from .domain_adapter import DomainAdapter
from .battery_adapter import BatteryDomainAdapter
from .pipeline import run_dc3s_step
from .half_life import (
    compute_validity_horizon,
    compute_half_life_from_horizon,
    compute_certificate_state,
)
from .reachability import (
    propagate_reachability_set,
    compute_validity_horizon_from_reachability,
    compute_expiration_bound,
)
from .theoretical_guarantees import (
    compute_finite_sample_coverage_bound,
    assert_finite_sample_bound,
    compute_coverage_bound_surface,
    compute_separation_gap,
    assert_separation,
    simulate_separation_construction,
    compute_adaptive_regret_bound,
    assert_sublinear_regret,
    simulate_adaptive_tracking,
    compute_universal_impossibility_bound,
    compute_stylized_frontier_lower_bound,
    evaluate_structural_transfer,
    TransferContractResult,
    THEOREM_REGISTER,
)

__all__ = [
    "compute_reliability",
    "DeepOQEConfig",
    "DeepOQEModel",
    "DEEP_OQE_FEATURE_NAMES",
    "FTIT_FAULT_KEYS",
    "preview_fault_state",
    "update_ftit_state",
    "PageHinkleyDetector",
    "AdaptivePageHinkleyDetector",
    "inflate_interval",
    "inflate_q",
    "build_uncertainty_set",
    "repair_action",
    "make_certificate",
    "store_certificate",
    "get_certificate",
    "compute_model_hash",
    "compute_config_hash",
    "recompute_certificate_hash",
    "verify_certificate",
    "verify_certificate_chain",
    "DC3SStateStore",
    "RACCertConfig",
    "RACCertModel",
    "compute_dispatch_sensitivity",
    "compute_q_multiplier",
    "compute_inflation",
    "normalize_sensitivity",
    "reliability_error_bound",
    "tightened_soc_bounds",
    "check_tightened_soc_invariance",
    "safety_filter_projection_summary",
    "forward_tube",
    "certificate_validity_horizon",
    "certificate_expiration_bound",
    "zero_dispatch_fallback",
    "certify_fallback_existence",
    "evaluate_graceful_degradation_dominance",
    "certificate_half_life",
    "should_renew_certificate",
    "should_expire_certificate",
    "compute_validity_horizon",
    "compute_half_life_from_horizon",
    "compute_certificate_state",
    "propagate_reachability_set",
    "compute_validity_horizon_from_reachability",
    "compute_expiration_bound",
    "DomainAdapter",
    "BatteryDomainAdapter",
    "run_dc3s_step",
    "compute_finite_sample_coverage_bound",
    "assert_finite_sample_bound",
    "compute_coverage_bound_surface",
    "compute_separation_gap",
    "assert_separation",
    "simulate_separation_construction",
    "compute_adaptive_regret_bound",
    "assert_sublinear_regret",
    "simulate_adaptive_tracking",
    "compute_universal_impossibility_bound",
    "compute_stylized_frontier_lower_bound",
    "evaluate_structural_transfer",
    "TransferContractResult",
    "THEOREM_REGISTER",
]
