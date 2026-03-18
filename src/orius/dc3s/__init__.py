"""DC3S: Drift-Calibrated Conformal Safety Shield components."""

from .quality import compute_reliability
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

__all__ = [
    "compute_reliability",
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
]
