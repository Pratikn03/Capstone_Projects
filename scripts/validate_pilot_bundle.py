#!/usr/bin/env python3
"""Validate future-pilot evidence bundles for schema and linkage integrity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_ndjson(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path} contains non-object JSON row")
        rows.append(payload)
    return rows


def validate_bundle(bundle_dir: str) -> dict[str, Any]:
    """Validate a bundle directory and return a compact pass/fail summary."""
    root = Path(bundle_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_files = ["telemetry.ndjson", "command_queue.ndjson", "acks.ndjson", "certificates.ndjson"]
    for rel in required_files:
        if not (root / rel).exists():
            raise FileNotFoundError(f"Missing bundle file: {root / rel}")

    telemetry = _load_ndjson(root / "telemetry.ndjson")
    queue = _load_ndjson(root / "command_queue.ndjson")
    acks = _load_ndjson(root / "acks.ndjson")
    certs = _load_ndjson(root / "certificates.ndjson")

    command_ids = [str(row.get("command_id")) for row in queue if row.get("command_id")]
    if len(command_ids) != len(set(command_ids)):
        raise ValueError("Duplicate command_id values found in command_queue.ndjson")

    ack_command_ids = [str(row.get("command_id")) for row in acks if row.get("command_id")]
    missing_queue_for_ack = sorted(set(ack_command_ids) - set(command_ids))
    if missing_queue_for_ack:
        raise ValueError(f"ACK rows reference missing command IDs: {', '.join(missing_queue_for_ack)}")

    cert_command_ids = {str(row.get("command_id")) for row in certs if row.get("command_id")}
    missing_cert_for_queue = sorted(set(command_ids) - cert_command_ids)

    telemetry_ts = [row.get("ts_utc") for row in telemetry if row.get("ts_utc")]
    if telemetry_ts != sorted(telemetry_ts):
        raise ValueError("Telemetry timestamps are not monotonic nondecreasing")

    queue_times = [row.get("queued_at") for row in queue if row.get("queued_at")]
    if queue_times and queue_times != sorted(queue_times):
        raise ValueError("Queued command timestamps are not monotonic nondecreasing")

    certificate_link_failures = []
    for ack in acks:
        cert_id = ack.get("certificate_id")
        if cert_id and str(cert_id) not in cert_command_ids:
            certificate_link_failures.append(str(cert_id))
    if certificate_link_failures:
        raise ValueError(
            "ACK rows reference missing certificate IDs: " + ", ".join(sorted(set(certificate_link_failures)))
        )

    return {
        "passed": True,
        "bundle_name": manifest.get("bundle_name", root.name),
        "telemetry_rows": len(telemetry),
        "command_rows": len(queue),
        "ack_rows": len(acks),
        "certificate_rows": len(certs),
        "commands_missing_certificates": missing_cert_for_queue,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a future-pilot evidence bundle")
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = validate_bundle(str(args.bundle_dir))
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(payload)


if __name__ == "__main__":
    main()
