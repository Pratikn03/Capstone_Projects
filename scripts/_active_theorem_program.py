"""Shared builders for the YAML-canonical theorem audit surface."""
from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports" / "publication"
REGISTRY_YAML = REPORTS_DIR / "theorem_registry.yml"
THEOREM_REGISTER_CSV = REPORTS_DIR / "theorem_surface_register.csv"
THEOREM_REGISTER_TEX = REPORTS_DIR / "theorem_surface_register.tex"
THEOREM_SURFACE_SUMMARY_CSV = REPORTS_DIR / "theorem_surface_summary.csv"
THEOREM_SURFACE_SUMMARY_TEX = REPORTS_DIR / "theorem_surface_summary.tex"
ASSUMPTION_REGISTER_FILE = REPO_ROOT / "appendices" / "app_b_assumptions.tex"

AUDIT_JSON = REPORTS_DIR / "active_theorem_audit.json"
AUDIT_CSV = REPORTS_DIR / "active_theorem_audit.csv"
AUDIT_MD = REPORTS_DIR / "active_theorem_audit.md"
REMEDIATION_MD = REPORTS_DIR / "active_theorem_remediation_plan.md"
DEFENDED_CORE_JSON = REPORTS_DIR / "defended_theorem_core.json"
DEFENDED_CORE_CSV = REPORTS_DIR / "defended_theorem_core.csv"
DEFENDED_CORE_MD = REPORTS_DIR / "defended_theorem_core.md"
ASSUMPTION_MAP_CSV = REPORTS_DIR / "defended_assumption_map.csv"
ASSUMPTION_MAP_MD = REPORTS_DIR / "defended_assumption_map.md"
LINEAR_READY_JSON = REPORTS_DIR / "theorem_registry_linear_ready.json"
BATTERY_CLAIM_EVIDENCE_REGISTER = REPORTS_DIR / "battery_claim_evidence_register.csv"
EXTERNAL_AUDIT_PACKET_MD = REPORTS_DIR / "external_proof_audit_packet.md"

CSV_COLUMNS = (
    "theorem_id",
    "title",
    "surface_kind",
    "defense_tier",
    "proof_tier",
    "program_role",
    "scope_note",
    "statement_location",
    "proof_location",
    "assumptions_used",
    "unresolved_assumptions",
    "dependencies",
    "weakest_step",
    "rigor_rating",
    "code_correspondence",
    "severity_if_broken",
    "remediation_class",
    "code_anchors",
    "test_anchors",
)
DEFENSE_TIER_ORDER = (
    "flagship_defended",
    "supporting_defended",
    "draft_non_defended",
)
ENV_ORDER = ("theorem", "lemma", "proposition", "corollary", "definition", "assumption", "remark")


def _load_registry() -> dict[str, Any]:
    payload = yaml.safe_load(REGISTRY_YAML.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{REGISTRY_YAML} must contain a mapping.")
    theorems = payload.get("theorems")
    if not isinstance(theorems, list) or not theorems:
        raise ValueError(f"{REGISTRY_YAML} must define a non-empty 'theorems' list.")
    return payload


def _find_text_line(path: Path, needle: str) -> int:
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return lineno
    raise ValueError(f"Unable to find '{needle}' in {path}")


def _find_py_symbol_line(path: Path, symbol: str) -> int:
    parts = symbol.split(".")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def find_in_nodes(nodes: list[ast.stmt], remaining: list[str]) -> int | None:
        target = remaining[0]
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                if len(remaining) == 1:
                    return int(node.lineno)
                if isinstance(node, ast.ClassDef):
                    nested = find_in_nodes(list(node.body), remaining[1:])
                    if nested is not None:
                        return nested
        return None

    line = find_in_nodes(list(tree.body), parts)
    if line is None and len(parts) == 1:
        target = parts[0]
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == target:
                line = int(node.lineno)
                break
    if line is None:
        raise ValueError(f"Unable to find symbol '{symbol}' in {path}")
    return line


def _resolve_location(file_path: str, needle: str) -> str:
    path = REPO_ROOT / file_path
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".py":
        line = _find_py_symbol_line(path, needle)
    else:
        line = _find_text_line(path, needle)
    return f"{file_path}:{line}"


def _resolve_anchor(anchor: dict[str, Any]) -> dict[str, str]:
    path = str(anchor["path"])
    symbol = str(anchor.get("symbol") or "")
    note = str(anchor.get("note") or "")
    full_path = REPO_ROOT / path
    if not full_path.exists():
        raise FileNotFoundError(full_path)
    line = 1
    if symbol:
        if full_path.suffix == ".py":
            line = _find_py_symbol_line(full_path, symbol)
        else:
            line = _find_text_line(full_path, symbol)
    return {
        "path": path,
        "symbol": symbol,
        "location": f"{path}:{line}",
        "note": note,
    }


ASSUMPTION_BEGIN_RE = re.compile(r"\\begin\{assumption\}\[(A\d+[a-z]?)\s+[-\u2014]+")


def _assumption_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for lineno, line in enumerate(ASSUMPTION_REGISTER_FILE.read_text(encoding="utf-8").splitlines(), start=1):
        match = ASSUMPTION_BEGIN_RE.search(line)
        if match:
            lookup[match.group(1)] = f"{ASSUMPTION_REGISTER_FILE.relative_to(REPO_ROOT).as_posix()}:{lineno}"
    return lookup


def _combine_assumptions(spec: dict[str, Any], assumption_lookup: dict[str, str]) -> tuple[list[str], list[dict[str, str]], list[str]]:
    assumptions = [str(value) for value in spec.get("assumptions", [])]
    extra_unresolved = [str(value) for value in spec.get("unresolved_assumptions", [])]
    locations: list[dict[str, str]] = []
    unresolved: list[str] = []
    for item in assumptions:
        if item in assumption_lookup:
            locations.append({"assumption": item, "location": assumption_lookup[item]})
        elif item.startswith("A"):
            unresolved.append(item)
    for item in extra_unresolved:
        if item not in unresolved:
            unresolved.append(item)
    return assumptions + [item for item in extra_unresolved if item not in assumptions], locations, unresolved


def _build_theorem_row(spec: dict[str, Any], assumption_lookup: dict[str, str]) -> dict[str, Any]:
    assumptions_used, assumption_locations, unresolved_assumptions = _combine_assumptions(spec, assumption_lookup)
    statement_source = spec["statement_source"]
    proof_source = spec["proof_source"]
    return {
        "theorem_id": spec["id"],
        "title": spec["title"],
        "surface_kind": spec["surface_kind"],
        "defense_tier": spec["defense_tier"],
        "proof_tier": spec["proof_tier"],
        "program_role": spec["program_role"],
        "scope_note": spec["scope_note"],
        "status": spec.get("status", "active"),
        "statement_location": _resolve_location(statement_source["file"], statement_source["needle"]),
        "proof_location": _resolve_location(proof_source["file"], proof_source["needle"]),
        "assumptions_used": assumptions_used,
        "assumption_locations": assumption_locations,
        "unresolved_assumptions": unresolved_assumptions,
        "dependencies": [str(value) for value in spec.get("dependencies", [])],
        "weakest_step": spec["weakest_step"],
        "rigor_rating": spec["rigor_rating"],
        "code_correspondence": spec["code_correspondence"],
        "code_correspondence_detail": spec["code_correspondence_detail"],
        "severity_if_broken": spec["severity_if_broken"],
        "remediation_class": spec["remediation_class"],
        "remediation_detail": spec["remediation_detail"],
        "legacy_aliases": [str(value) for value in spec.get("legacy_aliases", [])],
        "generator_targets": [str(value) for value in spec.get("generator_targets", [])],
        "code_anchors": [_resolve_anchor(anchor) for anchor in spec.get("code_anchors", [])],
        "test_anchors": [_resolve_anchor(anchor) for anchor in spec.get("test_anchors", [])],
    }


def build_active_theorem_audit_payload() -> dict[str, Any]:
    registry = _load_registry()
    assumption_lookup = _assumption_lookup()
    theorem_rows = [
        _build_theorem_row(spec, assumption_lookup)
        for spec in registry["theorems"]
        if str(spec.get("status", "active")) == "active"
    ]

    rigor_counts: dict[str, int] = Counter(row["rigor_rating"] for row in theorem_rows)
    code_counts: dict[str, int] = Counter(row["code_correspondence"] for row in theorem_rows)
    tier_counts: dict[str, int] = Counter(row["defense_tier"] for row in theorem_rows)
    flagship_rows = [row for row in theorem_rows if row["defense_tier"] == "flagship_defended"]
    supporting_rows = [row for row in theorem_rows if row["defense_tier"] == "supporting_defended"]
    draft_rows = [row for row in theorem_rows if row["defense_tier"] == "draft_non_defended"]
    flagship_gate_ready = all(
        row["rigor_rating"] not in {"broken", "has-a-hole"} and bool(row["scope_note"])
        for row in flagship_rows
    )
    return {
        "authoritative_surfaces": registry["authoritative_surfaces"],
        "theorems": theorem_rows,
        "namespace_drift": registry.get("namespace_drift", []),
        "summary": {
            "active_theorem_count": len(theorem_rows),
            "rigor_counts": dict(rigor_counts),
            "code_correspondence_counts": dict(code_counts),
            "defense_tier_counts": {tier: tier_counts.get(tier, 0) for tier in DEFENSE_TIER_ORDER},
            "flagship_gate_ready": flagship_gate_ready,
            "flagship_defended_ids": [row["theorem_id"] for row in flagship_rows],
            "supporting_defended_ids": [row["theorem_id"] for row in supporting_rows],
            "draft_non_defended_ids": [row["theorem_id"] for row in draft_rows],
        },
    }


def render_active_theorem_audit_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=False) + "\n"


def render_active_theorem_audit_csv(payload: dict[str, Any]) -> str:
    lines = [",".join(CSV_COLUMNS)]
    for theorem in payload["theorems"]:
        row = {
            "theorem_id": theorem["theorem_id"],
            "title": theorem["title"],
            "surface_kind": theorem["surface_kind"],
            "defense_tier": theorem["defense_tier"],
            "proof_tier": theorem["proof_tier"],
            "program_role": theorem["program_role"],
            "scope_note": theorem["scope_note"],
            "statement_location": theorem["statement_location"],
            "proof_location": theorem["proof_location"],
            "assumptions_used": " | ".join(theorem["assumptions_used"]),
            "unresolved_assumptions": " | ".join(theorem["unresolved_assumptions"]),
            "dependencies": " | ".join(theorem["dependencies"]),
            "weakest_step": theorem["weakest_step"],
            "rigor_rating": theorem["rigor_rating"],
            "code_correspondence": theorem["code_correspondence"],
            "severity_if_broken": theorem["severity_if_broken"],
            "remediation_class": theorem["remediation_class"],
            "code_anchors": " | ".join(anchor["location"] for anchor in theorem["code_anchors"]),
            "test_anchors": " | ".join(anchor["location"] for anchor in theorem["test_anchors"]),
        }
        escaped: list[str] = []
        for column in CSV_COLUMNS:
            value = str(row[column]).replace('"', '""')
            if any(ch in value for ch in {",", "\n", '"', "|"}):
                value = f'"{value}"'
            escaped.append(value)
        lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def _format_anchor_block(title: str, anchors: list[dict[str, str]]) -> list[str]:
    lines = [f"- {title}:"]
    for anchor in anchors:
        label = anchor["location"]
        if anchor["symbol"]:
            label += f" (`{anchor['symbol']}`)"
        if anchor["note"]:
            label += f" - {anchor['note']}"
        lines.append(f"  - {label}")
    return lines


def render_active_theorem_audit_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Active Theorem Audit",
        "",
        "This file is generated from `reports/publication/theorem_registry.yml` and is the reconciled active audit surface for the live theorem program.",
        "",
        "## Summary",
        "",
        f"- Active theorem rows: {payload['summary']['active_theorem_count']}",
        f"- Rigor counts: {payload['summary']['rigor_counts']}",
        f"- Code correspondence counts: {payload['summary']['code_correspondence_counts']}",
        f"- Defense-tier counts: {payload['summary']['defense_tier_counts']}",
        f"- Flagship gate ready: {payload['summary']['flagship_gate_ready']}",
        f"- Flagship defended IDs: {payload['summary']['flagship_defended_ids']}",
        f"- Supporting defended IDs: {payload['summary']['supporting_defended_ids']}",
        f"- Draft / non-defended IDs: {payload['summary']['draft_non_defended_ids']}",
        "",
        "## Namespace Drift",
        "",
    ]
    for drift in payload["namespace_drift"]:
        lines.extend(
            [
                f"### {drift['surface']}",
                "",
                f"- Issue: {drift['issue']}",
                f"- Impact: {drift['impact']}",
                f"- Status: {drift['status']}",
                f"- Remediation: {drift['remediation']}",
                "",
            ]
        )
    lines.append("## Per-Theorem Audit")
    lines.append("")
    for theorem in payload["theorems"]:
        lines.extend(
            [
                f"### {theorem['theorem_id']}: {theorem['title']}",
                "",
                f"- Surface kind: {theorem['surface_kind']}",
                f"- Defense tier: {theorem['defense_tier']}",
                f"- Proof tier: {theorem['proof_tier']}",
                f"- Program role: {theorem['program_role']}",
                f"- Scope note: {theorem['scope_note']}",
                f"- Statement location: {theorem['statement_location']}",
                f"- Proof location: {theorem['proof_location']}",
                f"- Assumptions used: {theorem['assumptions_used']}",
                f"- Unresolved assumptions: {theorem['unresolved_assumptions'] or '[]'}",
                f"- Dependencies: {theorem['dependencies']}",
                f"- Weakest step: {theorem['weakest_step']}",
                f"- Rigor rating: {theorem['rigor_rating']}",
                f"- Code correspondence: {theorem['code_correspondence']} - {theorem['code_correspondence_detail']}",
                f"- Severity if broken: {theorem['severity_if_broken']}",
                f"- Remediation class: {theorem['remediation_class']} - {theorem['remediation_detail']}",
                f"- Legacy aliases: {theorem['legacy_aliases'] or '[]'}",
            ]
        )
        lines.extend(_format_anchor_block("Code anchors", theorem["code_anchors"]))
        lines.extend(_format_anchor_block("Test anchors", theorem["test_anchors"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_active_theorem_remediation_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Active Theorem Remediation Plan",
        "",
        "This file is generated from `reports/publication/theorem_registry.yml` and converts the reconciled theorem registry into concrete follow-up work.",
        "",
        "## Immediate registry rules",
        "",
        "- Treat `reports/publication/theorem_registry.yml` as the only hand-edited theorem inventory.",
        "- Treat `reports/publication/theorem_surface_register.csv` as generated inventory only.",
        "- Keep theorem-facing assumptions synchronized with Appendix B and fail validation on unknown IDs.",
        "",
        "## Theorem-by-theorem actions",
        "",
    ]
    for theorem in payload["theorems"]:
        lines.append(
            f"- {theorem['theorem_id']} ({theorem['rigor_rating']}, {theorem['severity_if_broken']}): "
            f"{theorem['remediation_class']} - {theorem['remediation_detail']}"
        )
    lines.append("")
    return "\n".join(lines)


def render_defended_core_json(payload: dict[str, Any]) -> str:
    rows = [
        {
            "theorem_id": theorem["theorem_id"],
            "title": theorem["title"],
            "surface_kind": theorem["surface_kind"],
            "defense_tier": theorem["defense_tier"],
            "proof_tier": theorem["proof_tier"],
            "program_role": theorem["program_role"],
            "rigor_rating": theorem["rigor_rating"],
            "code_correspondence": theorem["code_correspondence"],
            "scope_note": theorem["scope_note"],
            "statement_location": theorem["statement_location"],
            "proof_location": theorem["proof_location"],
        }
        for theorem in payload["theorems"]
    ]
    return json.dumps({"summary": payload["summary"], "rows": rows}, indent=2, sort_keys=False) + "\n"


def render_defended_core_csv(payload: dict[str, Any]) -> str:
    columns = (
        "theorem_id",
        "title",
        "surface_kind",
        "defense_tier",
        "proof_tier",
        "program_role",
        "rigor_rating",
        "code_correspondence",
        "scope_note",
        "statement_location",
        "proof_location",
    )
    lines = [",".join(columns)]
    for theorem in payload["theorems"]:
        row = {column: theorem[column] for column in columns}
        escaped: list[str] = []
        for column in columns:
            value = str(row[column]).replace('"', '""')
            if any(ch in value for ch in {",", "\n", '"'}):
                value = f'"{value}"'
            escaped.append(value)
        lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def render_defended_core_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Defended Theorem Core",
        "",
        "This file is generated from `reports/publication/theorem_registry.yml` and is the strict defended-core theorem classification surface.",
        "",
        f"- Flagship defended: {payload['summary']['flagship_defended_ids']}",
        f"- Supporting defended: {payload['summary']['supporting_defended_ids']}",
        f"- Draft / non-defended: {payload['summary']['draft_non_defended_ids']}",
        f"- Flagship gate ready: {payload['summary']['flagship_gate_ready']}",
        "",
    ]
    for tier in DEFENSE_TIER_ORDER:
        lines.append(f"## {tier}")
        lines.append("")
        tier_rows = [theorem for theorem in payload["theorems"] if theorem["defense_tier"] == tier]
        if not tier_rows:
            lines.append("- none")
            lines.append("")
            continue
        for theorem in tier_rows:
            lines.append(
                f"- {theorem['theorem_id']} `{theorem['surface_kind']}` ({theorem['rigor_rating']}, {theorem['code_correspondence']}): {theorem['scope_note']}"
            )
        lines.append("")
    return "\n".join(lines)


def render_assumption_map_csv(payload: dict[str, Any]) -> str:
    columns = ("theorem_id", "defense_tier", "assumption", "resolved_in_register", "location")
    lines = [",".join(columns)]
    for theorem in payload["theorems"]:
        location_lookup = {item["assumption"]: item["location"] for item in theorem["assumption_locations"]}
        for assumption in theorem["assumptions_used"]:
            row = {
                "theorem_id": theorem["theorem_id"],
                "defense_tier": theorem["defense_tier"],
                "assumption": assumption,
                "resolved_in_register": str(assumption in location_lookup),
                "location": location_lookup.get(assumption, ""),
            }
            escaped: list[str] = []
            for column in columns:
                value = str(row[column]).replace('"', '""')
                if any(ch in value for ch in {",", "\n", '"'}):
                    value = f'"{value}"'
                escaped.append(value)
            lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def render_assumption_map_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Defended Assumption Map",
        "",
        "This file is generated from `reports/publication/theorem_registry.yml` and lists theorem-facing assumptions with their Appendix B resolution status.",
        "",
    ]
    for theorem in payload["theorems"]:
        lines.append(f"## {theorem['theorem_id']} ({theorem['defense_tier']})")
        lines.append("")
        location_lookup = {item["assumption"]: item["location"] for item in theorem["assumption_locations"]}
        for assumption in theorem["assumptions_used"]:
            if assumption in location_lookup:
                lines.append(f"- `{assumption}` - {location_lookup[assumption]}")
            else:
                lines.append(f"- `{assumption}` - theorem-local / unresolved")
        lines.append("")
    return "\n".join(lines)


def _render_registry_rows(registry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in registry.get("surface_register_rows", []):
        rows.append(
            {
                "register_id": row["register_id"],
                "group": row["group"],
                "environment": row["environment"],
                "title": row["title"],
                "context": row["context"],
                "source": row["source"],
                "line": int(_resolve_location(row["source"], row["needle"]).split(":")[-1]),
            }
        )
    for assumption, location in _assumption_lookup().items():
        source, line = location.rsplit(":", 1)
        rows.append(
            {
                "register_id": assumption,
                "group": "Assumption register",
                "environment": "assumption",
                "title": assumption,
                "context": "Detailed Statements",
                "source": source,
                "line": int(line),
            }
        )
    return rows


def render_theorem_surface_register_csv(registry: dict[str, Any]) -> str:
    rows = _render_registry_rows(registry)
    columns = ("register_id", "group", "environment", "title", "context", "source", "line")
    lines = [",".join(columns)]
    for row in rows:
        escaped: list[str] = []
        for column in columns:
            value = str(row[column]).replace('"', '""')
            if any(ch in value for ch in {",", "\n", '"'}):
                value = f'"{value}"'
            escaped.append(value)
        lines.append(",".join(escaped))
    return "\n".join(lines) + "\n"


def render_theorem_surface_summary_csv(registry: dict[str, Any]) -> str:
    rows = _render_registry_rows(registry)
    counts = Counter(row["environment"] for row in rows)
    lines = ["environment,count"]
    for environment in ENV_ORDER:
        lines.append(f"{environment},{counts.get(environment, 0)}")
    lines.append(f"total,{sum(counts.values())}")
    return "\n".join(lines) + "\n"


def render_theorem_surface_register_tex(registry: dict[str, Any]) -> str:
    rows = _render_registry_rows(registry)
    lines = [
        r"\begin{longtable}{p{0.10\linewidth}p{0.18\linewidth}p{0.10\linewidth}p{0.24\linewidth}p{0.26\linewidth}}",
        r"\caption{YAML-canonical theorem surface inventory.}\\",
        r"\toprule",
        r"ID & Group & Env & Title & Source \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"ID & Group & Env & Title & Source \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in rows:
        lines.append(
            f"{row['register_id']} & {row['group']} & {row['environment']} & {row['title']} & {row['source']}:{row['line']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines) + "\n"


def render_theorem_surface_summary_tex(registry: dict[str, Any]) -> str:
    rows = _render_registry_rows(registry)
    counts = Counter(row["environment"] for row in rows)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{YAML-canonical theorem surface summary.}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Environment & Count \\",
        r"\midrule",
    ]
    for environment in ENV_ORDER:
        lines.append(f"{environment} & {counts.get(environment, 0)} \\\\")
    lines.extend([r"\midrule", rf"total & {sum(counts.values())} \\\\", r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def render_battery_claim_evidence_register(payload: dict[str, Any]) -> str:
    columns = ("theorem_or_scope", "chapter", "token", "code_or_script", "artifact_path", "status", "code_exists", "artifact_exists")
    lines = [",".join(columns)]
    for theorem in payload["theorems"]:
        claim = next(
            (
                item for item in _load_registry()["theorems"]
                if item["id"] == theorem["theorem_id"] and "claim_evidence" in item
            ),
            None,
        )
        if claim is None:
            continue
        evidence = claim["claim_evidence"]
        code_or_script = theorem["code_anchors"][0]["path"] if theorem["code_anchors"] else ""
        artifact_path = str(evidence.get("artifact_path") or "")
        row = {
            "theorem_or_scope": theorem["theorem_id"],
            "chapter": evidence.get("chapter", ""),
            "token": evidence.get("token", ""),
            "code_or_script": code_or_script,
            "artifact_path": artifact_path,
            "status": theorem["defense_tier"],
            "code_exists": str(bool(code_or_script and (REPO_ROOT / code_or_script).exists())),
            "artifact_exists": str(bool(artifact_path and (REPO_ROOT / artifact_path).exists())),
        }
        escaped: list[str] = []
        for column in columns:
            value = str(row[column]).replace('"', '""')
            if any(ch in value for ch in {",", "\n", '"'}):
                value = f'"{value}"'
            escaped.append(value)
        lines.append(",".join(escaped))
    lines.append("A1-A13,ch15,ASM_REGISTER,appendices/app_b_assumptions.tex,appendices/app_b_assumptions.tex,locked,True,True")
    return "\n".join(lines) + "\n"


def render_linear_ready_json(payload: dict[str, Any]) -> str:
    epics = [
        {
            "title": "Canonical theorem registry",
            "description": "YAML schema, audit renderers, alias handling, and validator refactor.",
            "blocked_by": [],
            "theorems": ["T1", "T2", "T3a", "T3b", "T4", "T5", "T6", "T8", "T9", "T10", "T11"],
        },
        {
            "title": "Assumption register A1-A13",
            "description": "Appendix B rewrite and theorem-facing assumption validation.",
            "blocked_by": ["Canonical theorem registry"],
            "theorems": ["T1", "T2", "T3a", "T4", "T6", "T9", "T10", "T11", "T_trajectory_PAC"],
        },
        {
            "title": "T2 proof-to-runtime closure",
            "description": "Absorbed tightening, runtime assertions, and randomized safety checks.",
            "blocked_by": ["Canonical theorem registry", "Assumption register A1-A13"],
            "theorems": ["T2"],
        },
        {
            "title": "T6 direct theorem upgrade",
            "description": "Delta-aware expiration bound and migration of theorem-facing callers/tests.",
            "blocked_by": ["Canonical theorem registry", "Assumption register A1-A13"],
            "theorems": ["T6"],
        },
        {
            "title": "T3 split and migration",
            "description": "Keep T3 alias-only, defend T3a/T3b explicitly, and attach runtime theorem-contract reporting.",
            "blocked_by": ["Canonical theorem registry", "Assumption register A1-A13"],
            "theorems": ["T3a", "T3b"],
        },
        {
            "title": "Surface retiering",
            "description": "Keep T5 as a definition, T11 forward-only, T8 supporting unless both proof and runtime gates clear.",
            "blocked_by": ["Canonical theorem registry", "Assumption register A1-A13"],
            "theorems": ["T5", "T8", "T11"],
        },
        {
            "title": "Headline core rewrite",
            "description": "Appendix M/S and chapter-level defended-count surfaces synchronized to the registry.",
            "blocked_by": ["Canonical theorem registry", "Assumption register A1-A13", "T3 split and migration", "Surface retiering"],
            "theorems": ["T1", "T2", "T3a", "T3b", "T4", "T6", "T8", "T11", "T_trajectory_PAC"],
        },
        {
            "title": "External proof audit",
            "description": "Reviewer packet, structured findings register, and closure loop for flagship defended rows plus T6 and conditional T8 review.",
            "blocked_by": ["Headline core rewrite"],
            "theorems": ["T1", "T2", "T3a", "T4", "T6", "T8", "T11", "T_trajectory_PAC"],
        },
    ]
    return json.dumps({"epics": epics, "summary": payload["summary"]}, indent=2, sort_keys=False) + "\n"


def _external_audit_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    included_ids = set(payload["summary"]["flagship_defended_ids"]) | {"T6", "T8"}
    return [row for row in payload["theorems"] if row["theorem_id"] in included_ids]


def render_external_audit_packet_md(payload: dict[str, Any]) -> str:
    lines = [
        "# External Proof Audit Packet",
        "",
        "This packet is the bounded external-review surface for the current ORIUS theorem program.",
        "Reviewer task: identify the single weakest step for each row and classify it as one of",
        "`fixed in code/tests`, `narrowed in prose/registers`, or `left open as bounded future work`.",
        "",
        "Rows included in this packet:",
        f"- {', '.join(row['theorem_id'] for row in _external_audit_rows(payload))}",
        "",
    ]
    for row in _external_audit_rows(payload):
        code_anchor = row["code_anchors"][0]["location"] if row["code_anchors"] else ""
        lines.extend(
            [
                f"## {row['theorem_id']} — {row['title']}",
                f"- Defense tier: {row['defense_tier']}",
                f"- Proof tier: {row['proof_tier']}",
                f"- Statement location: `{row['statement_location']}`",
                f"- Proof location: `{row['proof_location']}`",
                f"- Assumptions: {', '.join(row['assumptions_used'])}",
                f"- Weakest known step: {row['weakest_step']}",
                f"- Scope note: {row['scope_note']}",
                f"- Primary code anchor: `{code_anchor}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_active_theorem_audit_outputs() -> None:
    registry = _load_registry()
    payload = build_active_theorem_audit_payload()
    AUDIT_JSON.write_text(render_active_theorem_audit_json(payload), encoding="utf-8")
    AUDIT_CSV.write_text(render_active_theorem_audit_csv(payload), encoding="utf-8")
    AUDIT_MD.write_text(render_active_theorem_audit_md(payload), encoding="utf-8")
    REMEDIATION_MD.write_text(render_active_theorem_remediation_md(payload), encoding="utf-8")
    DEFENDED_CORE_JSON.write_text(render_defended_core_json(payload), encoding="utf-8")
    DEFENDED_CORE_CSV.write_text(render_defended_core_csv(payload), encoding="utf-8")
    DEFENDED_CORE_MD.write_text(render_defended_core_md(payload), encoding="utf-8")
    ASSUMPTION_MAP_CSV.write_text(render_assumption_map_csv(payload), encoding="utf-8")
    ASSUMPTION_MAP_MD.write_text(render_assumption_map_md(payload), encoding="utf-8")
    THEOREM_REGISTER_CSV.write_text(render_theorem_surface_register_csv(registry), encoding="utf-8")
    THEOREM_REGISTER_TEX.write_text(render_theorem_surface_register_tex(registry), encoding="utf-8")
    THEOREM_SURFACE_SUMMARY_CSV.write_text(render_theorem_surface_summary_csv(registry), encoding="utf-8")
    THEOREM_SURFACE_SUMMARY_TEX.write_text(render_theorem_surface_summary_tex(registry), encoding="utf-8")
    BATTERY_CLAIM_EVIDENCE_REGISTER.write_text(render_battery_claim_evidence_register(payload), encoding="utf-8")
    LINEAR_READY_JSON.write_text(render_linear_ready_json(payload), encoding="utf-8")
    EXTERNAL_AUDIT_PACKET_MD.write_text(render_external_audit_packet_md(payload), encoding="utf-8")


__all__ = [
    "REGISTRY_YAML",
    "THEOREM_REGISTER_CSV",
    "THEOREM_REGISTER_TEX",
    "THEOREM_SURFACE_SUMMARY_CSV",
    "THEOREM_SURFACE_SUMMARY_TEX",
    "AUDIT_JSON",
    "AUDIT_CSV",
    "AUDIT_MD",
    "REMEDIATION_MD",
    "DEFENDED_CORE_JSON",
    "DEFENDED_CORE_CSV",
    "DEFENDED_CORE_MD",
    "ASSUMPTION_MAP_CSV",
    "ASSUMPTION_MAP_MD",
    "LINEAR_READY_JSON",
    "BATTERY_CLAIM_EVIDENCE_REGISTER",
    "EXTERNAL_AUDIT_PACKET_MD",
    "build_active_theorem_audit_payload",
    "render_active_theorem_audit_json",
    "render_active_theorem_audit_csv",
    "render_active_theorem_audit_md",
    "render_active_theorem_remediation_md",
    "render_defended_core_json",
    "render_defended_core_csv",
    "render_defended_core_md",
    "render_assumption_map_csv",
    "render_assumption_map_md",
    "render_theorem_surface_register_csv",
    "render_theorem_surface_register_tex",
    "render_theorem_surface_summary_csv",
    "render_theorem_surface_summary_tex",
    "render_battery_claim_evidence_register",
    "render_linear_ready_json",
    "render_external_audit_packet_md",
    "write_active_theorem_audit_outputs",
]
