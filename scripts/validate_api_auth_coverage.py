#!/usr/bin/env python3
"""Validate that production API routes are authenticated unless allowlisted."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi.routing import APIRoute

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PUBLIC_ROUTES = {
    ("GET", "/health"),
    ("GET", "/ready"),
}


def _route_has_security(route: APIRoute) -> bool:
    dependant = route.dependant
    if getattr(dependant, "_security_dependencies", None):
        return True
    stack = list(getattr(dependant, "dependencies", []) or [])
    while stack:
        dep = stack.pop()
        call_name = getattr(getattr(dep, "call", None), "__name__", "")
        if call_name == "get_api_key":
            return True
        if getattr(dep, "_security_dependencies", None):
            return True
        stack.extend(getattr(dep, "dependencies", []) or [])
    return False


def _validate_app(app, label: str) -> list[str]:
    findings: list[str] = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        for method in sorted(route.methods or []):
            if method in {"HEAD", "OPTIONS"}:
                continue
            if (method, route.path) in PUBLIC_ROUTES:
                continue
            if not _route_has_security(route):
                findings.append(f"{label} {method} {route.path} has no API-key security dependency")
    return findings


def validate() -> list[str]:
    from orius.api.app import app as compatibility_app
    from services.api.main import app as canonical_app

    findings = _validate_app(canonical_app, "services.api.main")
    findings.extend(_validate_app(compatibility_app, "orius.api.app"))
    return findings


def main() -> int:
    findings = validate()
    if findings:
        print("[validate_api_auth_coverage] FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("[validate_api_auth_coverage] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
