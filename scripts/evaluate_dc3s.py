#!/usr/bin/env python3
"""DC3S evaluation entry-point — one command to see results.

Usage::

    python evaluate_dc3s.py               # show usage menu
    python evaluate_dc3s.py quick         # 1-seed quick check (all domains)
    python evaluate_dc3s.py full          # 3-seed full validation
    python evaluate_dc3s.py sota          # SOTA comparison (Tube MPC, CBF, Lagrangian)
    python evaluate_dc3s.py tests         # run test suite
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

COMMANDS: dict[str, tuple[list[str], str]] = {
    "quick": (
        [sys.executable, "scripts/run_universal_orius_validation.py",
         "--seeds", "1", "--horizon", "24"],
        "Quick one-seed validation across all six domains (24 steps each)",
    ),
    "full": (
        [sys.executable, "scripts/run_universal_orius_validation.py",
         "--seeds", "3", "--horizon", "48"],
        "Full three-seed validation with evidence gate (48 steps each)",
    ),
    "sota": (
        [sys.executable, "scripts/run_sota_comparison.py",
         "--seeds", "3", "--rows", "100"],
        "SOTA comparison: DC3S vs Tube MPC, CBF, Lagrangian (100 steps each)",
    ),
    "proof": (
        [sys.executable, "scripts/build_orius_framework_proof.py",
         "--seeds", "1", "--horizon", "24"],
        "Build full framework proof bundle (theorem gate + training + SIL + validation)",
    ),
    "tests": (
        [sys.executable, "-m", "pytest", "tests/", "--no-cov", "-q"],
        "Run full test suite",
    ),
}


def _print_menu() -> None:
    print("\nDC3S / ORIUS Evaluation Entry-Point")
    print("=" * 45)
    print("Usage: python evaluate_dc3s.py <command>\n")
    print("Commands:")
    for cmd, (_, description) in COMMANDS.items():
        print(f"  {cmd:<8}  {description}")
    print()
    print("Examples:")
    print("  python evaluate_dc3s.py quick   # fast sanity check")
    print("  python evaluate_dc3s.py full    # thesis-grade validation")
    print("  python evaluate_dc3s.py sota    # SOTA baseline comparison")
    print()


def main() -> int:
    if len(sys.argv) < 2:
        _print_menu()
        return 0

    cmd_name = sys.argv[1].lower()
    if cmd_name not in COMMANDS:
        print(f"Unknown command: {cmd_name!r}")
        _print_menu()
        return 1

    args, description = COMMANDS[cmd_name]
    print(f"\n>>> {description}")
    print(f">>> Running: {' '.join(args)}\n")
    result = subprocess.run(args, cwd=REPO_ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
