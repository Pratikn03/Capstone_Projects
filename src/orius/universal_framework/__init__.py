"""ORIUS Universal Domain Physical Safety Framework.

Provides a domain-agnostic pipeline for:
  1. Detect  (OQE / reliability)
  2. Calibrate (uncertainty set)
  3. Constrain (tightened safe set)
  4. Shield  (repair)
  5. Certify (dispatch certificate)

 Domains: energy (battery), av (vehicle), healthcare.
"""
from .pipeline import run_universal_step, PIPELINE_STAGES
from .domain_registry import get_adapter, get_domain_capabilities, list_domains, register_domain
from .tables import DOMAIN_STATE_TABLE, DOMAIN_SAFETY_TABLE, FAULT_TAXONOMY_TABLE

__all__ = [
    "run_universal_step",
    "PIPELINE_STAGES",
    "get_adapter",
    "get_domain_capabilities",
    "list_domains",
    "register_domain",
    "DOMAIN_STATE_TABLE",
    "DOMAIN_SAFETY_TABLE",
    "FAULT_TAXONOMY_TABLE",
]
