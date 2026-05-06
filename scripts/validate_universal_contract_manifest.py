#!/usr/bin/env python3
"""Validate the canonical ORIUS universal runtime-assurance contract manifest."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.universal_theory.domain_runtime_contracts import UNIVERSAL_CONTRACT_SLOTS

DEFAULT_MANIFEST = REPO_ROOT / "reports" / "publication" / "orius_universal_contract_manifest.json"
DEFAULT_SUMMARY = REPO_ROOT / "reports" / "publication" / "domain_runtime_contract_summary.json"
REQUIRED_DOMAINS = ("battery", "av", "healthcare")
PATH_REQUIRED_SLOTS = (
    "domain_data",
    "forecast_model",
    "uncertainty_estimate",
    "runtime_trace",
    "domain_contract_witness",
)
OVERCLAIM_PHRASES = (
    "road deployment",
    "road deployed",
    "carla completed",
    "carla closed loop completed",
    "clinical deployment",
    "clinical deployed",
    "clinical decision support approval",
    "unrestricted field deployment",
    "unrestricted field deployed",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, Mapping):
        return not value
    if isinstance(value, Iterable) and not isinstance(value, bytes | str):
        return not list(value)
    return False


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _iter_strings(item)
    elif isinstance(value, Iterable) and not isinstance(value, bytes):
        for item in value:
            yield from _iter_strings(item)


def _repo_path(raw: str) -> Path:
    path_text = raw.split("#", maxsplit=1)[0].split(":", maxsplit=1)[0].strip()
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _path_values(value: Any) -> list[Path]:
    if isinstance(value, str):
        return [_repo_path(value)]
    if isinstance(value, Iterable) and not isinstance(value, bytes | str):
        return [_repo_path(str(item)) for item in value]
    return []


def _has_unnegated_phrase(text: str, phrase: str) -> bool:
    normalized = " ".join(text.lower().replace("-", " ").split())
    phrase = " ".join(phrase.lower().replace("-", " ").split())
    start = 0
    while True:
        index = normalized.find(phrase, start)
        if index < 0:
            return False
        prefix_tokens = normalized[:index].split()[-5:]
        if not {"no", "not", "without", "never"}.intersection(prefix_tokens):
            return True
        start = index + len(phrase)


def _overclaim_failures(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for text in _iter_strings(manifest):
        for phrase in OVERCLAIM_PHRASES:
            if _has_unnegated_phrase(text, phrase):
                failures.append(f"deployment overclaim phrase detected: {phrase!r} in {text!r}")
    return failures


def validate_manifest(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    failures: list[str] = []
    manifest = _load_json(manifest_path)
    summary = _load_json(summary_path)
    if not manifest:
        failures.append(f"missing or invalid universal contract manifest: {manifest_path}")
    if not summary:
        failures.append(f"missing or invalid domain runtime contract summary: {summary_path}")
    if failures:
        return {"pass": False, "failures": failures}

    declared_slots = tuple(manifest.get("universal_pipeline_slots", ()))
    if declared_slots != UNIVERSAL_CONTRACT_SLOTS:
        failures.append("universal_pipeline_slots does not match the canonical ORIUS contract slot order")

    domains = manifest.get("domains", {})
    if not isinstance(domains, Mapping):
        failures.append("manifest domains must be an object")
        domains = {}
    summary_domains = summary.get("domains", {})
    if not isinstance(summary_domains, Mapping):
        failures.append("summary domains must be an object")
        summary_domains = {}

    for domain in REQUIRED_DOMAINS:
        payload = domains.get(domain)
        if not isinstance(payload, Mapping):
            failures.append(f"{domain}: missing manifest domain payload")
            continue
        summary_payload = summary_domains.get(domain)
        if not isinstance(summary_payload, Mapping):
            failures.append(f"{domain}: missing runtime contract summary")
        elif int(summary_payload.get("n_steps", 0) or 0) <= 0:
            failures.append(f"{domain}: runtime contract summary has no witness rows")
        for slot in UNIVERSAL_CONTRACT_SLOTS:
            if _is_missing(payload.get(slot)):
                failures.append(f"{domain}: missing required universal contract slot {slot}")
        for slot in PATH_REQUIRED_SLOTS:
            paths = _path_values(payload.get(slot))
            if not paths:
                failures.append(f"{domain}: slot {slot} has no source path")
                continue
            if not any(path.exists() for path in paths):
                missing = ", ".join(str(path) for path in paths)
                failures.append(f"{domain}: slot {slot} does not point to an existing source path: {missing}")
        if str(payload.get("contract_id", "")) != str(summary_payload.get("contract_id", "")):
            failures.append(f"{domain}: manifest contract_id does not match runtime contract summary")
        boundary = str(payload.get("claim_boundary", ""))
        if not boundary:
            failures.append(f"{domain}: missing claim boundary")

    failures.extend(_overclaim_failures(manifest))
    return {
        "pass": not failures,
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "domains": list(REQUIRED_DOMAINS),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    result = validate_manifest(manifest_path=args.manifest, summary_path=args.summary)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
