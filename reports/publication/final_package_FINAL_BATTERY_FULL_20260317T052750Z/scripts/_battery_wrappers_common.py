#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
PYTHON = str(REPO_ROOT / ".venv" / "bin" / "python3") if (REPO_ROOT / ".venv" / "bin" / "python3").exists() else sys.executable


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_script(script_name: str, *args: str) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "mpl"))
    env.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    ensure_dir(Path(env["MPLCONFIGDIR"]))
    ensure_dir(Path(env["XDG_CACHE_HOME"]))
    cmd = [PYTHON, str(SCRIPTS_DIR / script_name), *args]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)


def copy_outputs(items: list[tuple[Path, Path]]) -> None:
    for src, dst in items:
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def write_manifest(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
