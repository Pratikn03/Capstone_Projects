"""Smoke test the FastAPI app without launching a server."""
from __future__ import annotations

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
