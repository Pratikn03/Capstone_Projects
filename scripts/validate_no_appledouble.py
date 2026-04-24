#!/usr/bin/env python3
"""Fail if AppleDouble sidecars remain outside active-run allowlists."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from cleanup_appledouble import default_exclude_parts, find_sidecars


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--exclude-active",
        action="store_true",
        help="Allow AppleDouble files in active max-freeze directories.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    exclude_parts = default_exclude_parts(root) if args.exclude_active else set()
    sidecars = find_sidecars(root, exclude_parts)
    if sidecars:
        print(f"[validate_no_appledouble] FAIL: {len(sidecars)} AppleDouble sidecars found")
        for path in sidecars[:50]:
            print(path.relative_to(root).as_posix())
        if len(sidecars) > 50:
            print(f"... {len(sidecars) - 50} more")
        return 1
    print("[validate_no_appledouble] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
