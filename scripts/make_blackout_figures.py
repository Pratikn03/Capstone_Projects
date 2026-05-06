#!/usr/bin/env python3
from __future__ import annotations

from _battery_wrappers_common import run_script


def main() -> None:
    run_script("run_blackout_half_life.py")
    run_script("run_graceful_degradation.py")


if __name__ == "__main__":
    main()
