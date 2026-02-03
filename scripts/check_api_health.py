"""Smoke test the FastAPI app without launching a server."""
from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(repo_root / "src") not in sys.path:
    sys.path.insert(0, str(repo_root / "src"))

from fastapi.testclient import TestClient

from services.api.main import app


def main() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "ok", f"Unexpected health payload: {data}"
    print("API health OK")


if __name__ == "__main__":
    main()
