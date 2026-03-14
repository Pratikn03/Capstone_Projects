"""Static code-health audit for publish gating."""
from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_GLOBS = [
    "src/gridpulse/**/*.py",
    "services/api/**/*.py",
    "scripts/**/*.py",
    "iot/**/*.py",
]


@dataclass
class FunctionHealth:
    name: str
    lineno: int
    line_count: int
    cyclomatic_estimate: int


def _load_publish_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("publish_audit", {}) if isinstance(payload, dict) else {}


def _iter_py_paths(globs: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(REPO_ROOT.glob(pattern)):
            if path.is_file():
                key = str(path)
                if key not in seen:
                    seen.add(key)
                    out.append(path)
    return out


def _cyclomatic_estimate(node: ast.AST) -> int:
    branch_types: tuple[type, ...] = (
        ast.If,
        ast.For,
        ast.While,
        ast.Try,
        ast.With,
        ast.BoolOp,
        ast.IfExp,
        ast.comprehension,
    )
    match_type = getattr(ast, "Match", None)
    if match_type is not None:
        branch_types = branch_types + (match_type,)

    score = 1
    for child in ast.walk(node):
        if isinstance(child, branch_types):
            score += 1
    return score


def _collect_functions(tree: ast.AST, source_lines: list[str]) -> list[FunctionHealth]:
    out: list[FunctionHealth] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            if end < start:
                end = start
            line_count = end - start + 1
            out.append(
                FunctionHealth(
                    name=str(node.name),
                    lineno=start,
                    line_count=line_count,
                    cyclomatic_estimate=_cyclomatic_estimate(node),
                )
            )
    return out


def _collect_unused_imports(tree: ast.AST) -> list[str]:
    imported: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imported[name] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname or alias.name
                imported[name] = f"{node.module}.{alias.name}" if node.module else alias.name

    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(str(node.id))
    unused = sorted([imported_name for imported_name in imported if imported_name not in used and not imported_name.startswith("_")])
    return unused


def run_code_health_audit(*, config_path: Path, include_globs: list[str] | None = None) -> dict[str, Any]:
    cfg = _load_publish_cfg(config_path)
    ch_cfg = cfg.get("code_health", {}) if isinstance(cfg.get("code_health"), dict) else {}
    max_function_lines = int(ch_cfg.get("max_function_lines", 180))
    max_cyclomatic = int(ch_cfg.get("max_cyclomatic_estimate", 30))
    max_file_lines = int(ch_cfg.get("max_file_lines", 1200))
    fail_on_bare_except = bool(ch_cfg.get("fail_on_bare_except", True))
    fail_on_except_pass = bool(ch_cfg.get("fail_on_except_pass", True))
    fail_on_unused_imports = bool(ch_cfg.get("fail_on_unused_imports", False))

    paths = _iter_py_paths(include_globs or DEFAULT_GLOBS)
    violations: list[dict[str, Any]] = []
    summary_files: list[dict[str, Any]] = []

    for path in paths:
        source = path.read_text(encoding="utf-8")
        source_lines = source.splitlines()
        file_row = {"path": str(path), "line_count": len(source_lines), "parse_ok": True}
        if len(source_lines) > max_file_lines:
            violations.append({"type": "file_too_large", "path": str(path), "line_count": len(source_lines), "max": max_file_lines})

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            file_row["parse_ok"] = False
            violations.append({"type": "syntax_error", "path": str(path), "error": str(exc)})
            summary_files.append(file_row)
            continue

        functions = _collect_functions(tree, source_lines)
        oversized = [f for f in functions if f.line_count > max_function_lines]
        complex_ = [f for f in functions if f.cyclomatic_estimate > max_cyclomatic]
        for item in oversized:
            violations.append(
                {
                    "type": "function_too_large",
                    "path": str(path),
                    "function": item.name,
                    "lineno": item.lineno,
                    "line_count": item.line_count,
                    "max": max_function_lines,
                }
            )
        for item in complex_:
            violations.append(
                {
                    "type": "function_too_complex",
                    "path": str(path),
                    "function": item.name,
                    "lineno": item.lineno,
                    "cyclomatic_estimate": item.cyclomatic_estimate,
                    "max": max_cyclomatic,
                }
            )

        bare_except_count = 0
        except_pass_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    bare_except_count += 1
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    except_pass_count += 1

        if fail_on_bare_except and bare_except_count > 0:
            violations.append({"type": "bare_except", "path": str(path), "count": bare_except_count})
        if fail_on_except_pass and except_pass_count > 0:
            violations.append({"type": "except_pass", "path": str(path), "count": except_pass_count})

        unused_imports = _collect_unused_imports(tree)
        if fail_on_unused_imports and unused_imports:
            violations.append({"type": "unused_imports", "path": str(path), "imports": unused_imports})

        file_row.update(
            {
                "functions": len(functions),
                "oversized_functions": len(oversized),
                "complex_functions": len(complex_),
                "bare_except": bare_except_count,
                "except_pass": except_pass_count,
                "unused_imports": unused_imports,
            }
        )
        summary_files.append(file_row)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "max_function_lines": max_function_lines,
            "max_cyclomatic_estimate": max_cyclomatic,
            "max_file_lines": max_file_lines,
            "fail_on_bare_except": fail_on_bare_except,
            "fail_on_except_pass": fail_on_except_pass,
            "fail_on_unused_imports": fail_on_unused_imports,
        },
        "files_scanned": len(summary_files),
        "files": summary_files,
        "violations": violations,
    }

    # Filter out known/pre-existing violations
    known_list = ch_cfg.get("known_violations", [])
    known_set: set[tuple[str, str]] = set()
    for entry in known_list if isinstance(known_list, list) else []:
        kp = entry.get("path", "")
        for vtype in entry.get("types", []):
            known_set.add((kp, vtype))

    new_violations = []
    known_violations = []
    for v in violations:
        rel_path = str(Path(v.get("path", "")).relative_to(REPO_ROOT)) if v.get("path", "").startswith(str(REPO_ROOT)) else v.get("path", "")
        if (rel_path, v.get("type", "")) in known_set:
            known_violations.append(v)
        else:
            new_violations.append(v)

    payload["known_violations_matched"] = len(known_violations)
    payload["new_violations"] = new_violations
    payload["fail"] = bool(new_violations)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Code-health audit")
    parser.add_argument("--config", default="configs/publish_audit.yaml")
    parser.add_argument("--out-json", default="reports/publish/code_health.json")
    parser.add_argument("--include-glob", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = run_code_health_audit(
        config_path=Path(args.config),
        include_globs=args.include_glob,
    )
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload.get("fail"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
