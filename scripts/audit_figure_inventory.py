#!/usr/bin/env python3
"""Audit figure and publication artifact inventory + freshness."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_utc(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_manifest_lock_time(manifest_path: Path) -> datetime | None:
    if not manifest_path.exists():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return _parse_utc(payload.get("generated_at_utc"))


def _load_cpsbench_required_outputs() -> list[str]:
    try:
        import sys

        src = REPO_ROOT / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
        from orius.cpsbench_iot.runner import REQUIRED_OUTPUTS

        return [str(x) for x in REQUIRED_OUTPUTS]
    except Exception:
        return [
            "dc3s_main_table.csv",
            "dc3s_fault_breakdown.csv",
            "calibration_plot.png",
            "violation_vs_cost_curve.png",
            "dc3s_run_summary.json",
        ]


def build_inventory(*, run_start_utc: datetime | None, manifest_lock_utc: datetime | None) -> dict[str, Any]:
    roots = [REPO_ROOT / "reports" / "figures", REPO_ROOT / "reports" / "publication"]
    files: list[dict[str, Any]] = []

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            rel_path = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
            size_bytes = int(stat.st_size)
            non_empty = size_bytes > 0
            files.append(
                {
                    "path": rel_path,
                    "size_bytes": size_bytes,
                    "non_empty": non_empty,
                    "mtime_utc": mtime.isoformat(),
                    "sha256": _sha256_file(path),
                    "fresh_since_run_start": (mtime >= run_start_utc) if run_start_utc else None,
                    "newer_than_manifest_lock": (mtime >= manifest_lock_utc) if manifest_lock_utc else None,
                }
            )

    required_outputs = _load_cpsbench_required_outputs()
    critical_required_paths = [f"reports/publication/{name}" for name in required_outputs]

    file_map = {row["path"]: row for row in files}
    missing_critical: list[str] = []
    zero_size_critical: list[str] = []
    for rel in critical_required_paths:
        row = file_map.get(rel)
        if row is None:
            missing_critical.append(rel)
            continue
        if not bool(row.get("non_empty", False)):
            zero_size_critical.append(rel)

    stale_before_run_start = [
        row["path"]
        for row in files
        if run_start_utc is not None and isinstance(row.get("fresh_since_run_start"), bool) and not row["fresh_since_run_start"]
    ]
    newer_than_lock = [
        row["path"]
        for row in files
        if manifest_lock_utc is not None and isinstance(row.get("newer_than_manifest_lock"), bool) and row["newer_than_manifest_lock"]
    ]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_start_utc": run_start_utc.isoformat() if run_start_utc else None,
        "manifest_lock_utc": manifest_lock_utc.isoformat() if manifest_lock_utc else None,
        "roots": [str(x.relative_to(REPO_ROOT)).replace("\\", "/") for x in roots],
        "required_cpsbench_outputs": critical_required_paths,
        "summary": {
            "files_total": len(files),
            "critical_missing": len(missing_critical),
            "critical_zero_size": len(zero_size_critical),
            "stale_before_run_start": len(stale_before_run_start),
            "newer_than_manifest_lock": len(newer_than_lock),
            "critical_ok": len(missing_critical) == 0 and len(zero_size_critical) == 0,
        },
        "missing_critical": missing_critical,
        "zero_size_critical": zero_size_critical,
        "stale_before_run_start": stale_before_run_start,
        "newer_than_manifest_lock": newer_than_lock,
        "files": files,
    }
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines: list[str] = []
    lines.append("# Figure Freshness Report")
    lines.append("")
    lines.append(f"- Generated at: `{payload.get('generated_at')}`")
    lines.append(f"- Run start UTC: `{payload.get('run_start_utc')}`")
    lines.append(f"- Manifest lock UTC: `{payload.get('manifest_lock_utc')}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Files inventoried: `{summary.get('files_total', 0)}`")
    lines.append(f"- Critical missing: `{summary.get('critical_missing', 0)}`")
    lines.append(f"- Critical zero-size: `{summary.get('critical_zero_size', 0)}`")
    lines.append(f"- Stale before run start: `{summary.get('stale_before_run_start', 0)}`")
    lines.append(f"- Newer than manifest lock: `{summary.get('newer_than_manifest_lock', 0)}`")
    lines.append(f"- Critical OK: `{summary.get('critical_ok')}`")
    lines.append("")

    lines.append("## Missing Critical")
    missing = payload.get("missing_critical", [])
    if missing:
        for item in missing:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Zero-Size Critical")
    zeros = payload.get("zero_size_critical", [])
    if zeros:
        for item in zeros:
            lines.append(f"- `{item}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Inventory")
    lines.append("| Path | Size (bytes) | Non Empty | MTime UTC | SHA256 |")
    lines.append("|---|---:|:---:|---|---|")
    for row in payload.get("files", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {row.get('path')} | {row.get('size_bytes')} | {'yes' if row.get('non_empty') else 'no'} | "
            f"{row.get('mtime_utc')} | `{row.get('sha256')}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit figure/publication artifact inventory and freshness")
    parser.add_argument("--out-dir", default="reports/publish")
    parser.add_argument("--manifest", default="paper/metrics_manifest.json")
    parser.add_argument("--run-start-utc", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path

    run_start = _parse_utc(args.run_start_utc)
    lock_time = _load_manifest_lock_time(manifest_path)
    payload = build_inventory(run_start_utc=run_start, manifest_lock_utc=lock_time)

    inventory_json = out_dir / "figure_inventory.json"
    freshness_md = out_dir / "figure_freshness_report.md"
    inventory_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    freshness_md.write_text(_render_markdown(payload), encoding="utf-8")

    print(json.dumps(payload.get("summary", {}), indent=2))


if __name__ == "__main__":
    main()
