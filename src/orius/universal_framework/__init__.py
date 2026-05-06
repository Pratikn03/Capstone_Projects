"""ORIUS Universal Domain Physical Safety Framework.

Provides a domain-agnostic pipeline for:
  1. Detect  (OQE / reliability)
  2. Calibrate (uncertainty set)
  3. Constrain (tightened safe set)
  4. Shield  (repair)
  5. Certify (dispatch certificate)

 Domains: energy (battery), av (vehicle), healthcare.
"""

from .domain_registry import get_adapter, get_domain_capabilities, list_domains, register_domain
from .pipeline import PIPELINE_STAGES, run_universal_step
from .tables import DOMAIN_SAFETY_TABLE, DOMAIN_STATE_TABLE, FAULT_TAXONOMY_TABLE

__all__ = [
    "DOMAIN_SAFETY_TABLE",
    "DOMAIN_STATE_TABLE",
    "FAULT_TAXONOMY_TABLE",
    "PIPELINE_STAGES",
    "get_adapter",
    "get_domain_capabilities",
    "list_domains",
    "register_domain",
    "run_universal_step",
]
