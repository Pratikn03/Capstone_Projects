#!/usr/bin/env python3
"""Count formal manuscript items from the active thesis include tree.

The canonical count is derived from the master manuscript include graph,
not from every standalone `.tex` file in the repository. This avoids
double-counting alternative chapter surfaces and archived material.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = REPO_ROOT / "orius_battery_409page_figures_upgraded_main.tex"
FORMAL_PATTERNS = {
    "theorem": re.compile(r"\\begin\{theorem\}"),
    "lemma": re.compile(r"\\begin\{lemma\}"),
    "proposition": re.compile(r"\\begin\{proposition\}"),
    "corollary": re.compile(r"\\begin\{corollary\}"),
    "definition": re.compile(r"\\begin\{definition\}"),
    "assumption": re.compile(r"\\begin\{assumption\}"),
}
DEFAULT_EXPECTED = {
    "theorem": 26,
    "lemma": 5,
    "proposition": 8,
    "corollary": 12,
    "definition": 15,
    "assumption": 16,
}
INCLUDE_PATTERN = re.compile(r"\\(?:include|input)\{([^}]+)\}")


def _resolve_include(target: str, base_dir: Path) -> Path | None:
    stem = Path(target)
    candidates = []
    if stem.suffix:
        candidates.append((base_dir / stem).resolve())
    else:
        candidates.append((base_dir / f"{target}.tex").resolve())
        candidates.append((base_dir / target).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _walk_include_tree(path: Path, visited: list[Path], seen: set[Path]) -> None:
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    visited.append(resolved)
    text = resolved.read_text(encoding="utf-8")
    for match in INCLUDE_PATTERN.finditer(text):
        child = _resolve_include(match.group(1), resolved.parent)
        if child is not None:
            _walk_include_tree(child, visited, seen)


def collect_formal_counts(master_path: Path) -> dict[str, object]:
    visited: list[Path] = []
    _walk_include_tree(master_path, visited, set())
    counts = {name: 0 for name in FORMAL_PATTERNS}
    per_file: dict[str, dict[str, int]] = {}
    for path in visited:
        text = path.read_text(encoding="utf-8")
        file_counts: dict[str, int] = {}
        for name, pattern in FORMAL_PATTERNS.items():
            count = len(pattern.findall(text))
            counts[name] += count
            if count:
                file_counts[name] = count
        if file_counts:
            per_file[str(path.relative_to(REPO_ROOT))] = file_counts
    return {
        "master": str(master_path.relative_to(REPO_ROOT)),
        "included_files": [str(path.relative_to(REPO_ROOT)) for path in visited],
        "counts": counts,
        "total": sum(counts.values()),
        "per_file": per_file,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count formal items from the active thesis include tree")
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER, help="Master manuscript file")
    parser.add_argument(
        "--expected-json",
        type=Path,
        default=None,
        help="Optional JSON file with expected counts keyed by item type",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the active count differs from expected values",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    master_path = args.master.resolve()
    if not master_path.exists():
        raise FileNotFoundError(f"Master manuscript not found: {master_path}")

    payload = collect_formal_counts(master_path)
    expected = dict(DEFAULT_EXPECTED)
    if args.expected_json is not None:
        loaded = json.loads(args.expected_json.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("Expected JSON must decode to an object")
        expected.update({str(k): int(v) for k, v in loaded.items()})
    payload["expected"] = expected
    payload["matches_expected"] = payload["counts"] == expected

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")

    if args.strict and not payload["matches_expected"]:
        print("Formal-item count mismatch against expected values.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
