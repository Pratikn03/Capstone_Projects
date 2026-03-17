#!/usr/bin/env python3
"""
Analyze agent artifact zip against repo thesis.

Compares theorem names, structure, and evidence mapping between the artifact zip
(ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md) and the canonical battery-8 register.
Reports mismatches and missing alignment.

Usage:
  python scripts/analyze_agent_artifact.py [path-to-artifact.zip]
  make analyze-artifact

If no path is given, finds the latest agent-artifacts-zip_*.zip in the repo root.
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

# Canonical battery-8 (from orius-plan/THEOREM_REGISTER_MAPPING.md)
CANONICAL_T1_T8 = {
    "T1": "OASG Existence",
    "T2": "One-Step Safety Preservation",
    "T3": "ORIUS Core Bound",
    "T4": "No Free Safety",
    "T5": "Certificate Validity Horizon",
    "T6": "Certificate Expiration Bound",
    "T7": "Feasible Fallback Existence",
    "T8": "Graceful Degradation Dominance",
}

# Expected artifact zip theorem names -> canonical ID
ZIP_TO_CANONICAL = {
    "Theorem 1: Observed-State Safety Guarantee (OASG)": "T1",
    "Theorem 1: OASG Illusion": "T1",
    "Theorem 2: Certificate Half-Life Bound": "T5,T6",
    "Theorem 3: Conformal Reachability Coverage": "T3",
    "Theorem 4: Graceful Degradation Under Sensor Loss": "T8",
    "Theorem 5: Multi-Domain Transfer Bound": None,  # out-of-scope
    "Theorem 6: Computational Complexity of DC3S": None,
    "Theorem 7: Optimal Control Under Uncertainty": None,
    "Theorem 8: System-Level Safety Convergence": None,
}


def find_artifact_zip(repo_root: Path) -> Path | None:
    """Find latest agent-artifacts-zip_*.zip in repo root."""
    candidates = sorted(repo_root.glob("agent-artifacts-zip_*.zip"), reverse=True)
    return candidates[0] if candidates else None


def extract_proofs_md(zip_path: Path) -> str | None:
    """Extract ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md from zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if "ORIUS_COMPLETE_MATHEMATICAL_PROOFS" in name and name.endswith(".md"):
                return zf.read(name).decode("utf-8", errors="replace")
    return None


def parse_zip_theorems(content: str) -> list[tuple[int, str]]:
    """Parse theorem numbers and names from proofs markdown."""
    theorems: list[tuple[int, str]] = []
    seen: set[int] = set()
    # ToC style: "1. [Theorem 1: Observed-State Safety Guarantee (OASG)](#theorem-1)"
    for m in re.finditer(
        r"^\d+\.\s*\[Theorem\s+(\d+):\s*([^\]]+)\]",
        content,
        re.MULTILINE | re.IGNORECASE,
    ):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            theorems.append((num, m.group(2).strip()))
    # Section headers: "# Theorem N: Full Name"
    for m in re.finditer(
        r"^#+\s*Theorem\s+(\d+):\s*(.+?)(?:\s*$|\s*\*\*)",
        content,
        re.MULTILINE | re.IGNORECASE,
    ):
        num = int(m.group(1))
        if num not in seen:
            seen.add(num)
            theorems.append((num, m.group(2).strip()))
    return sorted(theorems, key=lambda t: t[0])


def check_mapping_in_zip(zip_path: Path) -> bool:
    """Check if THEOREM_REGISTER_MAPPING.md is in the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if "THEOREM_REGISTER_MAPPING" in n]
        return len(names) > 0


def run_analysis(zip_path: Path, repo_root: Path) -> int:
    """Run full analysis. Returns 0 if aligned, 1 if gaps found."""
    print(f"Analyzing: {zip_path.name}\n")
    print("=" * 60)

    # 1. Extract proofs
    content = extract_proofs_md(zip_path)
    if not content:
        print("ERROR: ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md not found in zip.")
        return 1

    # 2. Parse zip theorems
    zip_theorems = parse_zip_theorems(content)
    if not zip_theorems:
        print("WARNING: Could not parse theorem names from proofs. Trying fallback.")
        # Fallback: look for "# Theorem N:"
        for m in re.finditer(r"#+\s*Theorem\s+(\d+):\s*(.+?)(?:\s*$|\s*\{)", content):
            zip_theorems.append((int(m.group(1)), m.group(2).strip()))
        zip_theorems = sorted(set(zip_theorems), key=lambda t: t[0])

    print("\n## 1. Artifact Zip Theorems (from ORIUS_COMPLETE_MATHEMATICAL_PROOFS.md)")
    for num, name in zip_theorems:
        print(f"   Zip T{num}: {name}")

    # 3. Compare to canonical
    print("\n## 2. Canonical Battery-8 (Repo Thesis Appendix M)")
    for tid, tname in CANONICAL_T1_T8.items():
        print(f"   {tid}: {tname}")

    # 4. Mapping analysis
    print("\n## 3. Mapping Analysis")
    gaps = []
    canonical_covered: set[str] = set()
    for num, name in zip_theorems:
        full_name = f"Theorem {num}: {name}"
        canonical = ZIP_TO_CANONICAL.get(full_name)
        if canonical is None:
            canonical = ZIP_TO_CANONICAL.get(f"Theorem {num}: " + name[:50])
        if canonical:
            if "," in str(canonical):
                for cid in canonical.split(","):
                    canonical_covered.add(cid.strip())
                print(f"   Zip T{num} -> {canonical} (partial: spans multiple canonical)")
            else:
                canonical_covered.add(canonical)
                print(f"   Zip T{num} -> {canonical} (aligned)")
        else:
            if num <= 4:
                gaps.append(f"Zip T{num} ({name}) has no canonical mapping")
            else:
                print(f"   Zip T{num} -> (out-of-scope: multi-domain extension)")

    # 5. Missing in zip
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]:
        if tid not in canonical_covered:
            gaps.append(f"Canonical {tid} ({CANONICAL_T1_T8[tid]}) not represented in zip")

    # 6. THEOREM_REGISTER_MAPPING in zip
    has_mapping = check_mapping_in_zip(zip_path)
    print("\n## 4. Alignment Files in Zip")
    if has_mapping:
        print("   THEOREM_REGISTER_MAPPING.md: present")
    else:
        print("   THEOREM_REGISTER_MAPPING.md: MISSING (add for alignment)")
        gaps.append("THEOREM_REGISTER_MAPPING.md not in zip")

    # 7. Summary
    print("\n" + "=" * 60)
    print("## Summary")
    if gaps:
        print("GAPS FOUND:")
        for g in gaps:
            print(f"  - {g}")
        print("\nRecommendation: Use repo thesis as source of edits. See orius-plan/SOURCE_OF_TRUTH_POLICY.md")
        return 1
    else:
        print("No critical gaps. Zip theorem set aligns with battery-8 mapping.")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze agent artifact zip vs repo thesis")
    parser.add_argument(
        "zip_path",
        nargs="?",
        help="Path to artifact zip (default: latest agent-artifacts-zip_*.zip)",
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repo root (default: current dir)",
    )
    args = parser.parse_args()
    repo_root = Path(args.repo).resolve()

    if args.zip_path:
        zip_path = Path(args.zip_path).resolve()
        if not zip_path.exists():
            print(f"ERROR: {zip_path} not found.")
            return 1
    else:
        zip_path = find_artifact_zip(repo_root)
        if not zip_path:
            print("ERROR: No agent-artifacts-zip_*.zip found in repo root.")
            return 1

    return run_analysis(zip_path, repo_root)


if __name__ == "__main__":
    sys.exit(main())
