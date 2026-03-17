"""CertOS runtime layer (Paper 6).

OS-like layer between modeling and actuation. Six lifecycle ops:
ISSUE, VALIDATE, EXPIRE, RENEW, REVOKE, FALLBACK.

Invariants (enforced as runtime assertions):
    INV-1: No dispatch without a valid or fallback certificate
    INV-2: Certificate hash chain is unbroken (prev_hash links)
    INV-3: Fallback triggers if and only if H_t ≤ 0
"""
from .belief_engine import get_belief as BeliefEngine
from .reliability_engine import compute_reliability as ReliabilityEngine
from .shift_engine import build_uncertainty_set as ShiftEngine
from .certificate_engine import CertificateEngine, LifecycleOp
from .reachability_engine import compute_validity_horizon as ReachabilityEngine
from .safe_action_filter import filter_action as SafeActionFilter
from .graceful_planner import plan_fallback as GracefulPlanner
from .audit_ledger import AuditLedger
from .recovery_manager import RecoveryManager
from .runtime import CertOSRuntime, CertOSConfig, CertOSState

__all__ = [
    "BeliefEngine",
    "ReliabilityEngine",
    "ShiftEngine",
    "CertificateEngine",
    "LifecycleOp",
    "ReachabilityEngine",
    "SafeActionFilter",
    "GracefulPlanner",
    "AuditLedger",
    "RecoveryManager",
    "CertOSRuntime",
    "CertOSConfig",
    "CertOSState",
]
