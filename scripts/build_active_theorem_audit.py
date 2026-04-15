#!/usr/bin/env python3
"""Build the reconciled active T1--T11 theorem audit artifacts."""
from __future__ import annotations

from _active_theorem_program import write_active_theorem_audit_outputs


def main() -> int:
    write_active_theorem_audit_outputs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
