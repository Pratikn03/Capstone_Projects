"""Legacy domain adapter module — DEPRECATED.

Use ``orius.universal_framework`` adapters instead.
This module is retained only for backward compatibility.
"""

import warnings as _w

_w.warn(
    "orius.domain is deprecated; use orius.universal_framework adapters",
    DeprecationWarning,
    stacklevel=2,
)

from .adapter import DomainAdapter  # noqa: E402

__all__ = ["DomainAdapter"]
