#!/usr/bin/env python3
"""Scaffold and verify an external SSD layout for ORIUS raw datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from orius.data_pipeline.external_raw import (
    DEFAULT_STRICT_EXTERNAL_ROOT,
    EXTERNAL_DATASETS,
    STRICT_EXTERNAL_ROOT_ENV,
)

DATASET_DIRS = tuple(spec.directory_name for spec in EXTERNAL_DATASETS.values())

ENV_VAR = "ORIUS_EXTERNAL_DATA_ROOT"
DEFAULT_EXTERNAL_ROOT_NAME = "orius_external_data"
DEFAULT_STRICT_LINK = DEFAULT_STRICT_EXTERNAL_ROOT
BLOCK_START = "# >>> ORIUS external SSD >>>"
BLOCK_END = "# <<< ORIUS external SSD <<<"


def _resolve_external_root(volume_root: Path, external_root_name: str) -> Path:
    return volume_root.expanduser().resolve() / external_root_name


def _required_dirs(external_root: Path) -> list[Path]:
    return [external_root, *(external_root / name for name in DATASET_DIRS)]


def _render_shell_block(external_root: Path) -> str:
    return (
        "\n".join(
            [
                BLOCK_START,
                f"export {ENV_VAR}={external_root}",
                BLOCK_END,
            ]
        )
        + "\n"
    )


def _upsert_shell_block(shell_profile: Path, external_root: Path) -> None:
    shell_profile = shell_profile.expanduser()
    shell_profile.parent.mkdir(parents=True, exist_ok=True)
    existing = shell_profile.read_text(encoding="utf-8") if shell_profile.exists() else ""
    block = _render_shell_block(external_root)
    if BLOCK_START in existing and BLOCK_END in existing:
        before, remainder = existing.split(BLOCK_START, 1)
        _, after = remainder.split(BLOCK_END, 1)
        rewritten = before.rstrip() + "\n\n" + block
        if after.strip():
            rewritten += "\n" + after.lstrip("\n")
        shell_profile.write_text(rewritten.rstrip() + "\n", encoding="utf-8")
        return

    new_text = existing.rstrip()
    if new_text:
        new_text += "\n\n"
    new_text += block
    shell_profile.write_text(new_text, encoding="utf-8")


def _symlink_status(strict_link: Path, external_root: Path) -> dict[str, object]:
    if not strict_link.exists() and not strict_link.is_symlink():
        return {
            "path": str(strict_link),
            "exists": False,
            "is_symlink": False,
            "points_to_external_root": False,
        }
    resolved = strict_link.resolve()
    return {
        "path": str(strict_link),
        "exists": True,
        "is_symlink": strict_link.is_symlink(),
        "resolved_path": str(resolved),
        "points_to_external_root": resolved == external_root.resolve(),
    }


def _ensure_symlink(strict_link: Path, external_root: Path) -> dict[str, object]:
    status = _symlink_status(strict_link, external_root)
    if not status["exists"]:
        strict_link.parent.mkdir(parents=True, exist_ok=True)
        strict_link.symlink_to(external_root)
        return _symlink_status(strict_link, external_root)
    if strict_link.is_symlink() and strict_link.resolve() == external_root.resolve():
        return status
    raise RuntimeError(
        f"Strict link path {strict_link} already exists and does not point to {external_root}. "
        "Move or remove it manually before rerunning setup."
    )


def _verify_layout(external_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in _required_dirs(external_root):
        rows.append({"path": str(path), "exists": path.exists(), "is_dir": path.is_dir()})
    return rows


def _print_next_steps(external_root: Path, strict_link: Path, *, shell_profile: Path | None) -> None:
    lines = [
        "",
        "Next steps:",
        f"1. Put raw datasets under {external_root}",
        f"2. Reload your shell and verify {ENV_VAR}:",
        "   source ~/.zshrc",
        f'   echo "${ENV_VAR}"',
        "3. Run the repo checks:",
        f"   PYTHONPATH=src .venv/bin/python scripts/verify_real_data_preflight.py --external-root {external_root}",
        "   PYTHONPATH=src .venv/bin/python scripts/refresh_real_data_manifests.py",
        "4. Build individual real-data rows:",
        f"   PYTHONPATH=src .venv/bin/python scripts/build_navigation_real_dataset.py --external-root {external_root}",
        f"   PYTHONPATH=src .venv/bin/python scripts/build_aerospace_real_flight_dataset.py --external-root {external_root}",
    ]
    if shell_profile is not None:
        lines.insert(2, f"   shell profile updated: {shell_profile}")
    lines.insert(2, f"   strict final-gate link: {strict_link}")
    print("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scaffold and verify an ORIUS external SSD layout.")
    parser.add_argument(
        "--volume-root", type=Path, required=True, help="Mounted SSD root for the external SSD volume."
    )
    parser.add_argument(
        "--external-root-name",
        default=DEFAULT_EXTERNAL_ROOT_NAME,
        help="Folder name created under the SSD root",
    )
    parser.add_argument(
        "--strict-link",
        type=Path,
        default=Path(str(DEFAULT_STRICT_LINK)),
        help=f"Symlink path used by strict equal-domain gate (default: ${STRICT_EXTERNAL_ROOT_ENV} or ~/orius_external_data)",
    )
    parser.add_argument(
        "--skip-symlink", action="store_true", help="Do not create or verify the strict-link path"
    )
    parser.add_argument(
        "--append-shell-profile",
        action="store_true",
        help="Append or update the env var block in the selected shell profile",
    )
    parser.add_argument(
        "--shell-profile",
        type=Path,
        default=Path("~/.zshrc"),
        help="Shell profile to update when --append-shell-profile is set",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify the SSD layout; do not create missing directories or links",
    )
    args = parser.parse_args()

    volume_root = args.volume_root.expanduser().resolve()
    if not volume_root.exists() or not volume_root.is_dir():
        print(f"Mounted SSD root does not exist: {volume_root}", file=sys.stderr)
        return 1

    external_root = _resolve_external_root(volume_root, args.external_root_name)
    created_dirs: list[str] = []
    if not args.verify_only:
        for path in _required_dirs(external_root):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created_dirs.append(str(path))

    layout = _verify_layout(external_root)
    missing_paths = [row["path"] for row in layout if not row["exists"] or not row["is_dir"]]
    if missing_paths:
        print("External SSD layout is incomplete:", file=sys.stderr)
        for path in missing_paths:
            print(f"- {path}", file=sys.stderr)
        return 1

    shell_profile_path: Path | None = None
    if args.append_shell_profile:
        shell_profile_path = args.shell_profile.expanduser()
        if not args.verify_only:
            _upsert_shell_block(shell_profile_path, external_root)

    symlink_report: dict[str, object] | None = None
    if not args.skip_symlink:
        try:
            symlink_report = (
                _symlink_status(args.strict_link, external_root)
                if args.verify_only
                else _ensure_symlink(args.strict_link, external_root)
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if not symlink_report.get("points_to_external_root", False):
            print(
                f"Strict-link path {args.strict_link} does not point to {external_root}.",
                file=sys.stderr,
            )
            return 1

    report = {
        "volume_root": str(volume_root),
        "external_root": str(external_root),
        "dataset_dirs": list(DATASET_DIRS),
        "created_dirs": created_dirs,
        "layout": layout,
        "strict_link": symlink_report,
        "shell_profile": str(shell_profile_path) if shell_profile_path is not None else None,
        "env_var": ENV_VAR,
    }
    print(json.dumps(report, indent=2))
    _print_next_steps(external_root, args.strict_link, shell_profile=shell_profile_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
