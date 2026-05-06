#!/usr/bin/env python3
"""Attempt bounded local CARLA closed-loop validation without overclaiming.

The local development target is macOS ARM64. CARLA Linux archives are useful
provenance, but they are not completed simulator evidence unless the binary is
runnable and produces closed-loop traces. This script therefore fails closed by
emitting blocker artifacts when the local platform cannot execute CARLA.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import tarfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CARLA_ROOT = REPO_ROOT / "data" / "orius_av" / "carla" / "0.9.16"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "predeployment_external_validation" / "carla_local_95"
CLAIM_BOUNDARY = (
    "CARLA local preflight/stress-test surface only; completed CARLA closed-loop "
    "simulation is claimed only when runnable simulator traces are emitted. This "
    "does not claim road deployment or full autonomous-driving field closure."
)
TRACE_FIELDS = [
    "episode_id",
    "step",
    "stress_family",
    "controller",
    "certificate_valid",
    "true_constraint_violated",
    "postcondition_passed",
]


def _path_ref(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_members(path: Path, *, limit: int = 2000) -> list[str]:
    if not path.exists():
        return []
    members: list[str] = []
    try:
        with tarfile.open(path, "r:*") as archive:
            for index, member in enumerate(archive):
                if index >= limit:
                    break
                members.append(member.name)
    except (tarfile.TarError, OSError):
        return []
    return members


def inspect_carla_local_preflight(carla_root: Path = DEFAULT_CARLA_ROOT) -> dict[str, Any]:
    """Inspect the local CARLA archive and decide whether execution is possible."""
    carla_root = carla_root.resolve()
    archive = carla_root / "CARLA_0.9.16.tar.gz"
    maps_archive = carla_root / "AdditionalMaps_0.9.16.tar.gz"
    system = platform.system().lower()
    machine = platform.machine().lower()
    members = _tar_members(archive)
    has_linux_launcher = any(name.endswith(("CarlaUE4.sh", "CarlaUnreal.sh")) for name in members)
    has_macos_app = any(".app/" in name or name.endswith(".app") for name in members)
    extracted_launchers = [
        path
        for path in (
            carla_root / "CarlaUE4.sh",
            carla_root / "CarlaUnreal.sh",
            carla_root / "CARLA_0.9.16" / "CarlaUE4.sh",
            carla_root / "CARLA_0.9.16" / "CarlaUnreal.sh",
        )
        if path.exists()
    ]

    if not archive.exists():
        status = "missing_archive"
        reason = f"Missing CARLA archive at {archive}"
        runnable = False
    elif system == "darwin" and not has_macos_app:
        status = "blocked_by_local_platform"
        reason = "Local host is macOS and the CARLA archive appears to contain Linux launchers only."
        runnable = False
    elif system != "linux" and not has_macos_app:
        status = "blocked_by_local_platform"
        reason = f"Unsupported local platform for this CARLA archive: {system}/{machine}."
        runnable = False
    elif system == "linux" and (has_linux_launcher or extracted_launchers):
        status = "runnable_preflight_passed"
        reason = "Linux launcher detected."
        runnable = True
    elif has_macos_app:
        status = "runnable_preflight_passed"
        reason = "macOS app bundle detected."
        runnable = True
    else:
        status = "blocked_no_launcher"
        reason = "No runnable CARLA launcher was detected."
        runnable = False

    return {
        "carla_root": str(carla_root),
        "archive": str(archive),
        "archive_exists": archive.exists(),
        "archive_sha256": _sha256(archive),
        "additional_maps_archive": str(maps_archive),
        "additional_maps_exists": maps_archive.exists(),
        "additional_maps_sha256": _sha256(maps_archive),
        "platform": {"system": system, "machine": machine},
        "has_linux_launcher": has_linux_launcher,
        "has_macos_app": has_macos_app,
        "sample_members": members[:50],
        "runnable": runnable,
        "status": status,
        "blocked_reason": "" if runnable else reason,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _write_summary(out_dir: Path, row: dict[str, Any]) -> Path:
    path = out_dir / "carla_closed_loop_summary.csv"
    fields = [
        "validation_surface",
        "status",
        "carla_completed",
        "episodes_completed",
        "steps_completed",
        "safety_violations",
        "certificate_valid_rate",
        "blocked_reason",
        "claim_boundary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "passed", "pass"}


def _read_trace_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _summarize_trace_rows(
    rows: list[dict[str, str]],
    *,
    min_completed_episodes: int,
    min_completed_steps: int,
) -> dict[str, Any]:
    episodes = {
        str(row.get("episode_id", "")).strip() for row in rows if str(row.get("episode_id", "")).strip()
    }
    steps_completed = len(rows)
    safety_violations = sum(1 for row in rows if _boolish(row.get("true_constraint_violated", "")))
    certificate_count = sum(1 for row in rows if "certificate_valid" in row)
    certificate_valid = sum(1 for row in rows if _boolish(row.get("certificate_valid", "")))
    postcondition_failures = sum(1 for row in rows if not _boolish(row.get("postcondition_passed", "")))
    certificate_valid_rate = certificate_valid / certificate_count if certificate_count else 0.0
    completed = bool(
        len(episodes) >= min_completed_episodes
        and steps_completed >= min_completed_steps
        and safety_violations == 0
        and postcondition_failures == 0
    )
    blockers: list[str] = []
    if len(episodes) < min_completed_episodes:
        blockers.append(
            f"episodes_completed={len(episodes)} < min_completed_episodes={min_completed_episodes}"
        )
    if steps_completed < min_completed_steps:
        blockers.append(f"steps_completed={steps_completed} < min_completed_steps={min_completed_steps}")
    if safety_violations:
        blockers.append(f"safety_violations={safety_violations}")
    if postcondition_failures:
        blockers.append(f"postcondition_failures={postcondition_failures}")
    return {
        "completed": completed,
        "episodes_completed": len(episodes),
        "steps_completed": steps_completed,
        "safety_violations": safety_violations,
        "certificate_valid_rate": certificate_valid_rate,
        "blockers": blockers,
    }


def _write_traces(out_dir: Path, rows: list[dict[str, str]] | None = None) -> Path:
    path = out_dir / "carla_closed_loop_traces.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACE_FIELDS)
        writer.writeheader()
        for row in rows or []:
            writer.writerow({field: row.get(field, "") for field in TRACE_FIELDS})
    return path


def run_carla_closed_loop_validation(
    *,
    carla_root: Path = DEFAULT_CARLA_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_episodes: int = 24,
    max_steps_per_episode: int = 600,
    stress: tuple[str, ...] = ("nominal", "dropout", "stale_sensor", "latency", "emergency_brake", "weather"),
    trace_input: Path | None = None,
    min_completed_episodes: int = 1,
    min_completed_steps: int = 1,
) -> dict[str, Any]:
    """Run preflight and emit completed or blocked local CARLA artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight = inspect_carla_local_preflight(carla_root)
    trace_rows = _read_trace_rows(trace_input) if trace_input is not None else []
    trace_summary = _summarize_trace_rows(
        trace_rows,
        min_completed_episodes=min_completed_episodes,
        min_completed_steps=min_completed_steps,
    )
    completed = bool(trace_input and trace_summary["completed"])
    status = "completed_closed_loop" if completed else preflight["status"]
    blocked_reason = (
        preflight.get("blocked_reason") or "CARLA simulator loop not implemented for this local platform."
    )
    if trace_input is not None and not completed:
        blocked_reason = (
            "; ".join(trace_summary["blockers"]) or f"No usable CARLA trace rows in {trace_input}"
        )
    row = {
        "validation_surface": "carla_local_closed_loop",
        "status": status,
        "carla_completed": completed,
        "episodes_completed": trace_summary["episodes_completed"],
        "steps_completed": trace_summary["steps_completed"],
        "safety_violations": trace_summary["safety_violations"] if trace_input is not None else "",
        "certificate_valid_rate": trace_summary["certificate_valid_rate"] if trace_input is not None else "",
        "blocked_reason": "" if completed else blocked_reason,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    summary_path = _write_summary(out_dir, row)
    traces_path = _write_traces(out_dir, trace_rows if trace_input is not None else None)
    manifest = {
        "validation_surface": "carla_local_closed_loop",
        "status": status,
        "carla_completed": completed,
        "episodes_requested": int(max_episodes),
        "max_steps_per_episode": int(max_steps_per_episode),
        "episodes_completed": trace_summary["episodes_completed"],
        "steps_completed": trace_summary["steps_completed"],
        "safety_violations": trace_summary["safety_violations"] if trace_input is not None else None,
        "certificate_valid_rate": trace_summary["certificate_valid_rate"]
        if trace_input is not None
        else None,
        "stress_families": list(stress),
        "summary": _path_ref(summary_path),
        "traces": _path_ref(traces_path),
        "trace_input": _path_ref(trace_input) if trace_input is not None else "",
        "preflight": preflight,
        "blockers": [] if completed else trace_summary["blockers"],
        "claim_boundary": CLAIM_BOUNDARY,
    }
    manifest_path = out_dir / "carla_closed_loop_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-root", type=Path, default=DEFAULT_CARLA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-episodes", type=int, default=24)
    parser.add_argument("--max-steps-per-episode", type=int, default=600)
    parser.add_argument(
        "--stress", type=str, default="nominal,dropout,stale_sensor,latency,emergency_brake,weather"
    )
    parser.add_argument(
        "--trace-input",
        type=Path,
        default=None,
        help="Real CARLA closed-loop trace CSV from a runnable simulator environment",
    )
    parser.add_argument("--min-completed-episodes", type=int, default=1)
    parser.add_argument("--min-completed-steps", type=int, default=1)
    parser.add_argument("--fail-as-blocked-on-platform", action="store_true")
    args = parser.parse_args()
    manifest = run_carla_closed_loop_validation(
        carla_root=args.carla_root,
        out_dir=args.out_dir,
        max_episodes=args.max_episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        stress=tuple(item.strip() for item in args.stress.split(",") if item.strip()),
        trace_input=args.trace_input,
        min_completed_episodes=args.min_completed_episodes,
        min_completed_steps=args.min_completed_steps,
    )
    print(
        json.dumps({"status": manifest["status"], "carla_completed": manifest["carla_completed"]}, indent=2)
    )
    if args.fail_as_blocked_on_platform and not manifest["carla_completed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
