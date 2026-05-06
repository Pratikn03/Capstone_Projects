#!/usr/bin/env python3
"""Upload completed local nuPlan archives to a private Hugging Face dataset."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from orius.av_nuplan import DEFAULT_TRAIN_GLOB, resolve_nuplan_train_archives

DEFAULT_REPO_ID = "pratikn03/orius-nuplan-private"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync completed nuPlan zip archives to a private HF dataset repo"
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--train-zip", type=Path, action="append", default=None)
    parser.add_argument("--train-dir", type=Path, action="append", default=[REPO_ROOT])
    parser.add_argument("--train-glob", default=DEFAULT_TRAIN_GLOB)
    parser.add_argument("--maps-zip", type=Path, default=REPO_ROOT / "nuplan-maps-v1.0.zip")
    parser.add_argument("--include-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-incomplete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _manifest_payload(repo_id: str, archives, skipped, maps_zip: Path | None) -> dict[str, object]:
    return {
        "repo_id": repo_id,
        "source": "nuplan",
        "train_archives": [
            {
                "archive_id": archive.archive_id,
                "filename": archive.path.name,
                "sha256": archive.sha256,
                "size_bytes": archive.size_bytes,
                "db_count": archive.db_count,
                "hf_path": f"raw/{archive.archive_id}/{archive.path.name}",
            }
            for archive in archives
        ],
        "skipped_train_archives": skipped,
        "maps_archive": (
            {
                "filename": maps_zip.name,
                "hf_path": f"maps/{maps_zip.name}",
                "size_bytes": maps_zip.stat().st_size,
            }
            if maps_zip is not None and maps_zip.exists()
            else None
        ),
    }


def main() -> int:
    args = _parse_args()
    archives, skipped = resolve_nuplan_train_archives(
        train_zips=args.train_zip,
        train_dirs=args.train_dir,
        train_glob=args.train_glob,
        skip_incomplete=args.skip_incomplete,
    )
    maps_zip = args.maps_zip if args.include_maps else None
    manifest = _manifest_payload(args.repo_id, archives, skipped, maps_zip)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to upload nuPlan archives") from exc

    create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    api = HfApi()
    for archive in archives:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(archive.path),
            path_in_repo=f"raw/{archive.archive_id}/{archive.path.name}",
        )
    if maps_zip is not None and maps_zip.exists():
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(maps_zip),
            path_in_repo=f"maps/{maps_zip.name}",
        )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as handle:
        json.dump(manifest, handle, indent=2)
        manifest_path = Path(handle.name)
    try:
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=str(manifest_path),
            path_in_repo="nuplan_source_manifest.json",
        )
    finally:
        manifest_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
