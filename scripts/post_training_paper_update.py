#!/usr/bin/env python3
"""Canonical finalization flow for the manuscript submission freeze.

This script freezes one release family across publication artifacts and manuscript
outputs. It does not train models itself; it expects a verified single
release family to exist already.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER_DIR = REPO / "paper"
CANONICAL_PDF = REPO / "paper.pdf"
PUBLICATION_DIR = REPO / "reports" / "publication"
PYTHON_BIN = sys.executable or "python3"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd or REPO), check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_pdf(pdf_path: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = out_dir / pdf_path.stem
    _run(
        [
            "gs",
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=png16m",
            "-r144",
            f"-sOutputFile={prefix}-%03d.png",
            str(pdf_path),
        ]
    )
    return sorted(str(path.relative_to(REPO)) for path in out_dir.glob(f"{pdf_path.stem}-*.png"))


def _compile_papers() -> dict[str, Path]:
    outputs: dict[str, Path] = {}

    _run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"], cwd=PAPER_DIR)
    built_manuscript = PAPER_DIR / "paper.pdf"
    shutil.copy2(built_manuscript, CANONICAL_PDF)
    outputs["manuscript"] = CANONICAL_PDF

    ieee_dir = PAPER_DIR / "ieee"
    for surface, tex_name in (
        ("ieee_main", "orius_ieee_main.tex"),
        ("ieee_appendix", "orius_ieee_appendix.tex"),
    ):
        _run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", tex_name], cwd=ieee_dir)
        outputs[surface] = ieee_dir / tex_name.replace(".tex", ".pdf")

    return outputs


def _freeze_pdfs(
    *,
    release_id: str,
    publication_dir: Path,
    outputs: dict[str, Path],
) -> dict[str, dict[str, object]]:
    freeze_root = publication_dir / "frozen" / release_id
    review_dir = freeze_root / "review"
    freeze_root.mkdir(parents=True, exist_ok=True)

    frozen_payload: dict[str, dict[str, object]] = {}
    for surface, pdf_path in outputs.items():
        frozen_pdf = freeze_root / f"{release_id}_{surface}.pdf"
        shutil.copy2(pdf_path, frozen_pdf)
        images = _render_pdf(frozen_pdf, review_dir / surface)
        frozen_payload[surface] = {
            "canonical_path": str(pdf_path.relative_to(REPO)),
            "frozen_path": str(frozen_pdf.relative_to(REPO)),
            "sha256": _sha256(frozen_pdf),
            "rendered_pages": images,
        }
    return frozen_payload


def _update_release_manifest(
    *,
    release_id: str,
    publication_dir: Path,
    frozen_pdfs: dict[str, dict[str, object]],
) -> Path:
    manifest_path = publication_dir / "release_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["release_id"] = release_id
    payload["paper_assets"] = payload.get("paper_assets", {})
    payload.pop("frozen_pdfs", None)
    for key in [
        "FINAL_THESIS_PDF",
        "FINAL_CONFERENCE_PDF",
        "FINAL_MANUSCRIPT_PDF",
        "FINAL_IEEE_MAIN_PDF",
        "FINAL_IEEE_APPENDIX_PDF",
    ]:
        payload["paper_assets"].pop(key, None)
    payload["paper_assets"]["CANONICAL_MONOGRAPH_PDF"] = {
        "paper_path": frozen_pdfs["manuscript"]["canonical_path"],
        "source_artifact": frozen_pdfs["manuscript"]["canonical_path"],
        "build_command": f"{Path(__file__).relative_to(REPO)} --release-id {release_id}",
        "sha256": frozen_pdfs["manuscript"]["sha256"],
    }
    payload["paper_assets"]["IEEE_MAIN_PDF"] = {
        "paper_path": frozen_pdfs["ieee_main"]["canonical_path"],
        "source_artifact": frozen_pdfs["ieee_main"]["canonical_path"],
        "build_command": f"{Path(__file__).relative_to(REPO)} --release-id {release_id}",
        "sha256": frozen_pdfs["ieee_main"]["sha256"],
    }
    payload["paper_assets"]["IEEE_APPENDIX_PDF"] = {
        "paper_path": frozen_pdfs["ieee_appendix"]["canonical_path"],
        "source_artifact": frozen_pdfs["ieee_appendix"]["canonical_path"],
        "build_command": f"{Path(__file__).relative_to(REPO)} --release-id {release_id}",
        "sha256": frozen_pdfs["ieee_appendix"]["sha256"],
    }
    payload["archive_policy"] = {
        "historical_frozen_bundle": f"reports/publication/frozen/{release_id}/freeze_summary.json",
        "active_manifest_policy": "canonical pdfs only; frozen bundles are historical provenance and not active truth surfaces",
    }
    payload["frozen_at_utc"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _write_freeze_summary(
    *,
    release_id: str,
    publication_dir: Path,
    frozen_pdfs: dict[str, dict[str, object]],
    manifest_path: Path,
) -> Path:
    summary = {
        "release_id": release_id,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "publication_manifest": str(manifest_path.relative_to(REPO)),
        "frozen_pdfs": frozen_pdfs,
    }
    summary_path = publication_dir / "frozen" / release_id / "freeze_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze a final release family into manuscript and publication outputs"
    )
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--out-dir", default=str(PUBLICATION_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)

    _run(
        [
            PYTHON_BIN,
            "scripts/build_baseline_comparison_table.py",
            "--release-id",
            args.release_id,
            "--out-dir",
            str(out_dir),
        ]
    )
    _run(
        [
            PYTHON_BIN,
            "scripts/build_publication_artifact.py",
            "--release-id",
            args.release_id,
            "--out-dir",
            str(out_dir),
        ]
    )
    _run(["bash", "scripts/export_paper_assets.sh"])
    _run([PYTHON_BIN, "scripts/update_paper_metrics.py"])
    _run([PYTHON_BIN, "scripts/validate_paper_claims.py"])
    _run([PYTHON_BIN, "scripts/sync_paper_assets.py", "--check"])

    compiled = _compile_papers()
    frozen_pdfs = _freeze_pdfs(
        release_id=args.release_id,
        publication_dir=out_dir,
        outputs=compiled,
    )
    manifest_path = _update_release_manifest(
        release_id=args.release_id,
        publication_dir=out_dir,
        frozen_pdfs=frozen_pdfs,
    )
    summary_path = _write_freeze_summary(
        release_id=args.release_id,
        publication_dir=out_dir,
        frozen_pdfs=frozen_pdfs,
        manifest_path=manifest_path,
    )

    print("\nFreeze complete:")
    print(f"  release_id: {args.release_id}")
    print(f"  manifest:   {manifest_path.relative_to(REPO)}")
    print(f"  summary:    {summary_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
