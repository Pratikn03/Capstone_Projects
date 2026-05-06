#!/usr/bin/env python3
"""Build a clean three-domain ORIUS artifact release folder.

The builder intentionally uses an allowlist. It does not mirror the repository,
raw datasets, or historical report trees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:  # pragma: no cover - direct script execution uses the fallback branch.
    from scripts.clean_artifact_release_common import (
        AV_EVIDENCE_DIRS,
        BATTERY_EVIDENCE_DIRS,
        CODE_SCRIPTS,
        CODE_TESTS,
        CONFIG_FILES,
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        ENVIRONMENT_FILES,
        HEALTHCARE_EVIDENCE_FILES,
        MANIFEST_FILES,
        MANUSCRIPT_FILES,
        REPRODUCTION_COMMANDS,
        REQUIRED_RELEASE_PATHS,
        SENSITIVE_NAME_MARKERS,
        TABLE_DIRS,
        TABLE_FILES,
        THREE_DOMAIN_CLAIM_BOUNDARY,
        posix,
    )
except ModuleNotFoundError:  # pragma: no cover
    from clean_artifact_release_common import (  # type: ignore
        AV_EVIDENCE_DIRS,
        BATTERY_EVIDENCE_DIRS,
        CODE_SCRIPTS,
        CODE_TESTS,
        CONFIG_FILES,
        DISALLOWED_PARTS,
        DISALLOWED_SUFFIXES,
        ENVIRONMENT_FILES,
        HEALTHCARE_EVIDENCE_FILES,
        MANIFEST_FILES,
        MANUSCRIPT_FILES,
        REPRODUCTION_COMMANDS,
        REQUIRED_RELEASE_PATHS,
        SENSITIVE_NAME_MARKERS,
        TABLE_DIRS,
        TABLE_FILES,
        THREE_DOMAIN_CLAIM_BOUNDARY,
        posix,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
HASH_CHUNK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class CopySpec:
    source: Path
    dest: Path
    category: str
    claim_scope: str


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_disallowed_path(path: Path) -> bool:
    if path.name.startswith("._"):
        return True
    parts = set(path.parts)
    if parts & DISALLOWED_PARTS:
        return True
    suffix = path.suffix.lower()
    if suffix in DISALLOWED_SUFFIXES:
        return True
    lowered_name = path.name.lower()
    return any(marker in lowered_name for marker in SENSITIVE_NAME_MARKERS)


def _iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in DISALLOWED_PARTS or part.startswith("._") for part in rel.parts):
            continue
        if path.is_symlink():
            try:
                if not path.resolve().is_file():
                    continue
            except OSError:
                continue
        elif path.is_dir():
            continue
        if _is_disallowed_path(rel):
            continue
        files.append(path)
    return files


def _copy_file(src: Path, dest: Path, *, copy_mode: str) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if copy_mode == "hardlink":
        try:
            os.link(src, dest)
            shutil.copystat(src, dest, follow_symlinks=False)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dest)
            return "copy"
    shutil.copy2(src, dest)
    return "copy"


def _relative_source(repo_root: Path, path: Path) -> str:
    try:
        return posix(path.relative_to(repo_root))
    except ValueError:
        return posix(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _remove_appledouble(root: Path) -> None:
    for path in sorted(root.rglob("._*"), reverse=True):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _record_entry(
    *,
    release_root: Path,
    repo_root: Path,
    src: Path | None,
    dest: Path,
    category: str,
    claim_scope: str,
    copy_method: str,
) -> dict[str, Any]:
    rel = posix(dest.relative_to(release_root))
    return {
        "relative_path": rel,
        "source_path": f"generated:{rel}" if src is None else _relative_source(repo_root, src),
        "category": category,
        "claim_scope": claim_scope,
        "size_bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "copy_method": copy_method,
    }


def _copy_spec(spec: CopySpec, *, repo_root: Path, release_root: Path, copy_mode: str) -> dict[str, Any]:
    src = repo_root / spec.source
    if not src.exists():
        raise FileNotFoundError(f"Required release source is missing: {spec.source}")
    if not src.is_file():
        raise ValueError(f"Release source is not a file: {spec.source}")
    if _is_disallowed_path(spec.source):
        raise ValueError(f"Refusing disallowed release source: {spec.source}")
    dest = release_root / spec.dest
    method = _copy_file(src, dest, copy_mode=copy_mode)
    return _record_entry(
        release_root=release_root,
        repo_root=repo_root,
        src=src,
        dest=dest,
        category=spec.category,
        claim_scope=spec.claim_scope,
        copy_method=method,
    )


def _copy_tree(
    *,
    repo_root: Path,
    release_root: Path,
    source_rel: str,
    dest_rel: str,
    category: str,
    claim_scope: str,
    copy_mode: str,
) -> list[dict[str, Any]]:
    src_root = repo_root / source_rel
    if not src_root.exists():
        raise FileNotFoundError(f"Required release directory is missing: {source_rel}")
    if not src_root.is_dir():
        raise ValueError(f"Release source is not a directory: {source_rel}")

    entries: list[dict[str, Any]] = []
    for src in _iter_source_files(src_root):
        src_rel = src.relative_to(src_root)
        dest = release_root / dest_rel / src_rel
        method = _copy_file(src, dest, copy_mode=copy_mode)
        entries.append(
            _record_entry(
                release_root=release_root,
                repo_root=repo_root,
                src=src,
                dest=dest,
                category=category,
                claim_scope=claim_scope,
                copy_method=method,
            )
        )
    return entries


def _manifest_dest(source_rel: str) -> Path:
    source = Path(source_rel)
    name = source.name
    if "data/orius_av" in source_rel:
        return Path("manifests/av_nuplan") / name
    if "data/healthcare" in source_rel:
        prefix = "mimic3" if "mimic3" in source.parts else "healthcare"
        return Path("manifests/healthcare") / f"{prefix}_{name}"
    if "reports/battery_av_healthcare" in source_rel:
        return Path("manifests/three_domain") / name
    if "reports/predeployment_external_validation" in source_rel:
        if "healthcare_site_splits" in source.parts:
            return Path("manifests/predeployment/healthcare_site_splits_manifest.json")
        return Path("manifests/predeployment") / name
    if "reports/publication" in source_rel:
        return Path("manifests/publication") / name
    return Path("manifests") / name


def _manuscript_dest(source_rel: str) -> Path:
    source = Path(source_rel)
    if source_rel == "paper.pdf":
        return Path("manuscripts/paper.pdf")
    if source.parts[:1] == ("paper",) and len(source.parts) >= 2 and source.parts[1] == "ieee":
        return Path("manuscripts/ieee") / source.name
    if source.parts[:1] == ("paper",):
        return Path("manuscripts/monograph") / source.name
    return Path("manuscripts") / source.name


def _healthcare_dest(source_rel: str) -> Path:
    source = Path(source_rel)
    if "healthcare_site_splits" in source.parts:
        return Path("evidence/healthcare/site_splits") / source.name
    if "predeployment_external_validation" in source.parts:
        return Path("evidence/healthcare/predeployment") / source.name
    return Path("evidence/healthcare") / source.name


def _generate_readme(release_id: str, include_manuscripts: bool, copy_mode: str) -> str:
    manuscript_line = "included" if include_manuscripts else "not included"
    return f"""# ORIUS Three-Domain Clean Artifact Release

Release id: `{release_id}`

This folder is the clean reproducibility surface for the promoted ORIUS
three-domain package. It contains derived evidence, code snapshots, configs,
manifests, tables, manuscript outputs, and reproduction commands.

Claim boundary:

{THREE_DOMAIN_CLAIM_BOUNDARY}

## Contents

- `environment/`: dependency and runtime snapshots.
- `code/`: selected source scripts, ORIUS package code, and focused tests.
- `configs/`: configs needed to identify the promoted Battery, nuPlan AV, and Healthcare lanes.
- `manifests/`: source, validation, publication, and freeze manifests.
- `tables/`: generated LaTeX and CSV tables used by the paper package.
- `evidence/`: derived evidence for Battery, nuPlan AV, and Healthcare.
- `manuscripts/`: PDF/DOCX outputs are {manuscript_line}.

Large derived files are stored with copy method `{copy_mode}` when possible.
Raw nuPlan archives, incomplete browser downloads, private healthcare source
rows, caches, secrets, stale historical report bundles, and AppleDouble files
are intentionally excluded.
"""


def _generate_reproduce() -> str:
    commands = "\n".join(f"{command}" for command in REPRODUCTION_COMMANDS)
    fenced = "\n".join(f"```bash\n{command}\n```" for command in REPRODUCTION_COMMANDS)
    return f"""# Reproduce ORIUS Three-Domain Closure

Run these commands from the ORIUS repository root with the project virtual
environment already created.

## Command List

{fenced}

## Copy/Paste Block

```bash
{commands}
```

The promoted closure is bounded to Battery + nuPlan AV + Healthcare. The AV
row is completed all-zip nuPlan replay/surrogate evidence, not CARLA or road
deployment. The Healthcare row is retrospective/source-holdout evidence, not a
prospective clinical trial or live deployment.
"""


def _generated_environment_files(release_root: Path) -> list[tuple[Path, str]]:
    python_text = "\n".join(
        [
            f"python_executable={sys.executable}",
            f"python_version={platform.python_version()}",
            f"platform={platform.platform()}",
            "",
        ]
    )
    generated = [(release_root / "environment/python_runtime.txt", python_text)]
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode == 0:
            generated.append((release_root / "environment/pip_freeze.txt", proc.stdout))
    except (OSError, subprocess.TimeoutExpired):
        return generated
    return generated


def build_release(
    *,
    repo_root: Path = REPO_ROOT,
    out_root: Path | None = None,
    release_id: str | None = None,
    mode: str = "full-derived",
    include_manuscripts: bool = True,
    copy_mode: str = "hardlink",
    force: bool = False,
    verify: bool = False,
) -> Path:
    if mode != "full-derived":
        raise ValueError("Only mode='full-derived' is currently supported")
    if copy_mode not in {"hardlink", "copy"}:
        raise ValueError("copy_mode must be 'hardlink' or 'copy'")

    repo_root = repo_root.resolve()
    release_id = release_id or f"orius-three-domain-artifact-{_utc_stamp()}"
    out_root = (out_root or repo_root / "artifacts/releases").resolve()
    release_root = out_root / release_id

    if release_root.exists():
        if not force:
            raise FileExistsError(f"Release directory already exists: {release_root}")
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True)

    entries: list[dict[str, Any]] = []

    for rel in ENVIRONMENT_FILES:
        src = repo_root / rel
        if src.exists():
            entries.append(
                _copy_spec(
                    CopySpec(
                        Path(rel), Path("environment") / Path(rel).name, "environment", "reproducibility"
                    ),
                    repo_root=repo_root,
                    release_root=release_root,
                    copy_mode=copy_mode,
                )
            )

    for generated_path, text in _generated_environment_files(release_root):
        _write_text(generated_path, text)
        entries.append(
            _record_entry(
                release_root=release_root,
                repo_root=repo_root,
                src=None,
                dest=generated_path,
                category="environment",
                claim_scope="reproducibility",
                copy_method="generated",
            )
        )

    for rel in CODE_SCRIPTS:
        entries.append(
            _copy_spec(
                CopySpec(Path(rel), Path("code") / rel, "code", "reproduction_commands"),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )
    for rel in CODE_TESTS:
        entries.append(
            _copy_spec(
                CopySpec(Path(rel), Path("code") / rel, "code", "focused_regression_tests"),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )
    entries.extend(
        _copy_tree(
            repo_root=repo_root,
            release_root=release_root,
            source_rel="src/orius",
            dest_rel="code/src/orius",
            category="code",
            claim_scope="runtime_package_snapshot",
            copy_mode=copy_mode,
        )
    )

    for rel in CONFIG_FILES:
        entries.append(
            _copy_spec(
                CopySpec(Path(rel), Path(rel), "config", "three_domain_reproduction"),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )

    for rel in MANIFEST_FILES:
        entries.append(
            _copy_spec(
                CopySpec(Path(rel), _manifest_dest(rel), "manifest", "three_domain_claim_boundary"),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )

    for rel in TABLE_DIRS:
        entries.extend(
            _copy_tree(
                repo_root=repo_root,
                release_root=release_root,
                source_rel=rel,
                dest_rel="tables/generated",
                category="table",
                claim_scope="paper_claim_surface",
                copy_mode=copy_mode,
            )
        )
    for rel in TABLE_FILES:
        entries.append(
            _copy_spec(
                CopySpec(
                    Path(rel), Path("tables/publication") / Path(rel).name, "table", "paper_claim_surface"
                ),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )

    for rel in BATTERY_EVIDENCE_DIRS:
        dest = "evidence/battery/battery" if rel.endswith("/battery") else "evidence/battery/hil"
        entries.extend(
            _copy_tree(
                repo_root=repo_root,
                release_root=release_root,
                source_rel=rel,
                dest_rel=dest,
                category="evidence",
                claim_scope="battery",
                copy_mode=copy_mode,
            )
        )

    for rel in AV_EVIDENCE_DIRS:
        if "models_orius" in rel:
            dest = "evidence/av_nuplan/models"
        elif "uncertainty" in rel:
            dest = "evidence/av_nuplan/uncertainty"
        else:
            dest = "evidence/av_nuplan/runtime"
        entries.extend(
            _copy_tree(
                repo_root=repo_root,
                release_root=release_root,
                source_rel=rel,
                dest_rel=dest,
                category="evidence",
                claim_scope="av_nuplan",
                copy_mode=copy_mode,
            )
        )

    for rel in HEALTHCARE_EVIDENCE_FILES:
        entries.append(
            _copy_spec(
                CopySpec(Path(rel), _healthcare_dest(rel), "evidence", "healthcare"),
                repo_root=repo_root,
                release_root=release_root,
                copy_mode=copy_mode,
            )
        )

    if include_manuscripts:
        for rel in MANUSCRIPT_FILES:
            entries.append(
                _copy_spec(
                    CopySpec(Path(rel), _manuscript_dest(rel), "manuscript", "submission_outputs"),
                    repo_root=repo_root,
                    release_root=release_root,
                    copy_mode=copy_mode,
                )
            )

    readme_path = release_root / "README.md"
    reproduce_path = release_root / "REPRODUCE.md"
    _write_text(readme_path, _generate_readme(release_id, include_manuscripts, copy_mode))
    _write_text(reproduce_path, _generate_reproduce())
    for generated_path, category in [(readme_path, "release_doc"), (reproduce_path, "release_doc")]:
        entries.append(
            _record_entry(
                release_root=release_root,
                repo_root=repo_root,
                src=None,
                dest=generated_path,
                category=category,
                claim_scope="three_domain_claim_boundary",
                copy_method="generated",
            )
        )

    entries = sorted(entries, key=lambda item: item["relative_path"])
    manifest = {
        "bundle_version": 1,
        "bundle_id": release_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "copy_mode_requested": copy_mode,
        "include_manuscripts": include_manuscripts,
        "claim_boundary": THREE_DOMAIN_CLAIM_BOUNDARY,
        "promoted_domains": [
            "Battery Energy Storage",
            "nuPlan Autonomous Vehicles",
            "Medical and Healthcare Monitoring",
        ],
        "excluded": [
            "raw nuPlan zip archives",
            "incomplete .crdownload files",
            "private healthcare source rows",
            "patient-level healthcare split parquet files",
            "secrets/tokens/credentials",
            "caches and dependency directories",
            "AppleDouble files",
            "legacy/stale AV report bundles",
        ],
        "required_release_paths": REQUIRED_RELEASE_PATHS,
        "file_count": len(entries),
        "total_size_bytes": sum(int(entry["size_bytes"]) for entry in entries),
        "files": entries,
    }
    manifest_path = release_root / "manifest.json"
    _write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    _remove_appledouble(release_root)
    sha_lines: list[str] = []
    for path in sorted(p for p in release_root.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256"):
        sha_lines.append(f"{_sha256(path)}  {posix(path.relative_to(release_root))}")
    _write_text(release_root / "MANIFEST.sha256", "\n".join(sha_lines) + "\n")
    _remove_appledouble(release_root)

    if verify:
        try:
            from scripts.validate_clean_artifact_release import validate_release
        except ModuleNotFoundError:  # pragma: no cover
            from validate_clean_artifact_release import validate_release  # type: ignore

        validate_release(release_root)

    return release_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a clean ORIUS three-domain artifact release")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--release-id", default=None)
    parser.add_argument("--mode", default="full-derived", choices=["full-derived"])
    parser.add_argument("--include-manuscripts", action="store_true")
    parser.add_argument("--copy-mode", default="hardlink", choices=["hardlink", "copy"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    release_root = build_release(
        repo_root=args.repo_root,
        out_root=args.out_root,
        release_id=args.release_id,
        mode=args.mode,
        include_manuscripts=args.include_manuscripts,
        copy_mode=args.copy_mode,
        force=args.force,
        verify=args.verify,
    )
    print(release_root)


if __name__ == "__main__":
    main()
