"""ORIUS package root.

ORIUS is a research-to-runtime physical-safety framework whose battery-domain
reference implementation is DC3S. The repository now uses a canonical
top-level package shape while preserving legacy modules for compatibility.

Official backbone:
    - orius.adapters: canonical domain and benchmark entrypoints
    - orius.dc3s: battery-domain safety logic and theorem-linked helpers
    - orius.certos: runtime OS and certificate lifecycle
    - orius.orius_bench: cross-domain benchmark and export layer
    - orius.multi_agent: shared-constraint composition layer
    - orius.forecasting / optimizer / data_pipeline: upstream stack

Import discipline:
    New domain-facing code should import adapters from ``orius.adapters.*``.
    Legacy modules under ``orius.domain``, ``orius.universal_framework``, and
    ``orius.vehicles`` remain as compatibility implementations.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
