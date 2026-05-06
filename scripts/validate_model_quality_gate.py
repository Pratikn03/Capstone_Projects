#!/usr/bin/env python3
"""Validate the ML model-quality gate by recomputing it from source metrics."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

try:
    from scripts import build_model_quality_gate as builder
except ModuleNotFoundError:  # pragma: no cover
    import build_model_quality_gate as builder  # type: ignore[no-redef]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _drop_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(payload))
    clone.pop("generated_at_utc", None)
    return clone


def _recompute(
    *,
    gate_path: Path,
    metrics_paths: list[Path] | None,
    config_paths: list[Path] | None,
) -> dict[str, Any]:
    actual = _read_json(gate_path)
    if metrics_paths is None:
        metrics_paths = [Path(path) for path in actual.get("source_metrics", {})]
    if config_paths is None:
        config_paths = [Path(path) for path in actual.get("config_paths", [])]
    with tempfile.TemporaryDirectory(prefix="orius-model-quality-") as tmp:
        out_path = Path(tmp) / "model_quality_gate.json"
        return builder.build_model_quality_gate(
            metrics_paths=metrics_paths,
            config_paths=config_paths,
            out_path=out_path,
            policy=actual.get("policy") if isinstance(actual.get("policy"), dict) else None,
        )


def validate_model_quality_gate(
    gate_path: Path = builder.DEFAULT_OUT,
    *,
    metrics_paths: list[Path] | None = None,
    config_paths: list[Path] | None = None,
    require_pass: bool = False,
) -> dict[str, Any]:
    findings: list[str] = []
    try:
        actual = _read_json(gate_path)
    except FileNotFoundError as exc:
        return {"pass": False, "findings": [f"missing model quality gate: {exc.filename}"], "blockers": []}
    expected = _recompute(gate_path=gate_path, metrics_paths=metrics_paths, config_paths=config_paths)

    if _drop_volatile(actual) != _drop_volatile(expected):
        findings.append("model_quality_gate.json does not match recomputed gate from source metrics/configs")
    for path, expected_hash in expected.get("source_metrics", {}).items():
        if actual.get("source_metrics", {}).get(path) != expected_hash:
            findings.append(f"source metric hash mismatch for {path}")
    if require_pass and not bool(actual.get("pass")):
        findings.append("model quality gate is blocked under --require-pass")

    return {
        "pass": not findings,
        "findings": findings,
        "blockers": actual.get("blockers", []),
        "model_quality_pass": bool(actual.get("pass")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=builder.DEFAULT_OUT)
    parser.add_argument("--metrics", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path, action="append", default=[])
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = validate_model_quality_gate(
        args.gate.resolve(),
        metrics_paths=[path.resolve() for path in args.metrics] if args.metrics else None,
        config_paths=[path.resolve() for path in args.config] if args.config else None,
        require_pass=bool(args.require_pass),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        status = "PASS" if result["pass"] else "FAIL"
        quality = "PASS" if result["model_quality_pass"] else "BLOCKED"
        print(f"[validate_model_quality_gate] {status} model_quality={quality}")
        for finding in result["findings"]:
            print(f"- {finding}")
        if result["blockers"]:
            print("[validate_model_quality_gate] tracked blockers:")
            for blocker in result["blockers"]:
                print(f"- {blocker}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
