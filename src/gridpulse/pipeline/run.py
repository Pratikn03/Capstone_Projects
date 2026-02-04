"""Pipeline orchestration: run."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import yaml

from gridpulse.utils.logging import setup_logging
from gridpulse.utils.registry import register_models

def _repo_root() -> Path:
    # Key: orchestrate end-to-end pipeline steps
    # run.py -> src/gridpulse/pipeline/run.py, so repo root is parents[3]
    return Path(__file__).resolve().parents[3]


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _hash_paths(paths: Iterable[Path], base: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        if p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file():
                    rel = fp.relative_to(base)
                    h.update(str(rel).encode("utf-8"))
                    h.update(_hash_file(fp).encode("utf-8"))
        elif p.is_file():
            rel = p.relative_to(base)
            h.update(str(rel).encode("utf-8"))
            h.update(_hash_file(p).encode("utf-8"))
    return h.hexdigest()


def _load_cache(cache_path: Path) -> dict:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _pip_freeze(repo_root: Path, run_dir: Path, log: logging.Logger) -> None:
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=repo_root)
        (run_dir / "pip_freeze.txt").write_bytes(out)
        log.info("Saved pip freeze to %s", run_dir / "pip_freeze.txt")
    except Exception as exc:
        log.warning("pip freeze failed: %s", exc)


def _snapshot_configs(repo_root: Path, run_dir: Path) -> None:
    cfg_dir = repo_root / "configs"
    if not cfg_dir.exists():
        return
    out_dir = run_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fp in cfg_dir.glob("*.yaml"):
        shutil.copy2(fp, out_dir / fp.name)


@dataclass
class SignalConfig:
    enabled: bool
    path: Path | None


@dataclass
class WeatherConfig:
    enabled: bool
    path: Path | None


@dataclass
class HolidaysConfig:
    enabled: bool
    path: Path | None
    country: str


def _load_signal_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> SignalConfig:
    if not cfg_path.exists():
        return SignalConfig(False, None)
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return SignalConfig(False, None)
    signals_cfg = payload.get("signals", {}) if isinstance(payload, dict) else {}
    enabled = bool(signals_cfg.get("enabled", False))
    if not enabled:
        return SignalConfig(False, None)
    signal_path = signals_cfg.get("file")
    if not signal_path:
        log.warning("signals.enabled is true but signals.file is missing in %s", cfg_path)
        return SignalConfig(True, None)
    path = (repo_root / signal_path).resolve() if not Path(signal_path).is_absolute() else Path(signal_path)
    return SignalConfig(True, path)


def _load_weather_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> WeatherConfig:
    if not cfg_path.exists():
        return WeatherConfig(False, None)
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return WeatherConfig(False, None)
    weather_cfg = payload.get("weather", {}) if isinstance(payload, dict) else {}
    enabled = bool(weather_cfg.get("enabled", False))
    if not enabled:
        return WeatherConfig(False, None)
    weather_path = weather_cfg.get("file")
    if not weather_path:
        log.warning("weather.enabled is true but weather.file is missing in %s", cfg_path)
        return WeatherConfig(True, None)
    path = (repo_root / weather_path).resolve() if not Path(weather_path).is_absolute() else Path(weather_path)
    return WeatherConfig(True, path)


def _load_holidays_config(cfg_path: Path, repo_root: Path, log: logging.Logger) -> HolidaysConfig:
    if not cfg_path.exists():
        return HolidaysConfig(False, None, "DE")
    try:
        payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.warning("Failed to parse %s: %s", cfg_path, exc)
        return HolidaysConfig(False, None, "DE")
    holidays_cfg = payload.get("holidays", {}) if isinstance(payload, dict) else {}
    enabled = bool(holidays_cfg.get("enabled", False))
    if not enabled:
        return HolidaysConfig(False, None, str(holidays_cfg.get("country", "DE")))
    holiday_path = holidays_cfg.get("file")
    country = str(holidays_cfg.get("country", "DE"))
    if not holiday_path:
        log.warning("holidays.enabled is true but holidays.file is missing in %s", cfg_path)
        return HolidaysConfig(True, None, country)
    path = (repo_root / holiday_path).resolve() if not Path(holiday_path).is_absolute() else Path(holiday_path)
    return HolidaysConfig(True, path, country)


def _run(cmd: list[str], log: logging.Logger) -> None:
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _snapshot_artifacts(repo_root: Path, run_dir: Path, log: logging.Logger) -> None:
    # copy models
    models_src = repo_root / "artifacts" / "models"
    models_dst = run_dir / "models"
    if models_src.exists():
        models_dst.mkdir(parents=True, exist_ok=True)
        for fp in models_src.glob("*"):
            if fp.is_file():
                shutil.copy2(fp, models_dst / fp.name)

    # copy reports
    reports_dst = run_dir / "reports"
    reports_dst.mkdir(parents=True, exist_ok=True)

    report_files = [
        repo_root / "reports" / "formal_evaluation_report.md",
        repo_root / "reports" / "ml_vs_dl_comparison.md",
        repo_root / "reports" / "multi_horizon_backtest.json",
        repo_root / "reports" / "walk_forward_report.json",
        repo_root / "reports" / "impact_comparison.md",
        repo_root / "reports" / "impact_comparison.json",
        repo_root / "reports" / "impact_summary.csv",
    ]
    for fp in report_files:
        if fp.exists():
            shutil.copy2(fp, reports_dst / fp.name)

    figures_src = repo_root / "reports" / "figures"
    if figures_src.exists():
        figures_dst = reports_dst / "figures"
        figures_dst.mkdir(parents=True, exist_ok=True)
        for fp in figures_src.glob("*"):
            if fp.is_file():
                shutil.copy2(fp, figures_dst / fp.name)

    cards_src = repo_root / "reports" / "model_cards"
    if cards_src.exists():
        cards_dst = reports_dst / "model_cards"
        cards_dst.mkdir(parents=True, exist_ok=True)
        for fp in cards_src.glob("*.md"):
            shutil.copy2(fp, cards_dst / fp.name)

    log.info("Snapshot saved to %s", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GridPulse pipeline orchestrator")
    parser.add_argument("--steps", default="data,train,reports", help="Comma-separated steps: data,train,reports")
    parser.add_argument("--all", action="store_true", help="Run all steps (data,train,reports)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if cache is unchanged")
    parser.add_argument("--run-id", default=None, help="Override run id (default: timestamp)")
    args = parser.parse_args()

    repo_root = _repo_root()
    steps = ["data", "train", "reports"] if args.all else [s.strip() for s in args.steps.split(",") if s.strip()]

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "artifacts" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(file_path=str(run_dir / "pipeline.log"))
    log = logging.getLogger("gridpulse.pipeline")

    cache_path = repo_root / ".cache" / "pipeline.json"
    cache = _load_cache(cache_path)

    raw_csv = repo_root / "data" / "raw" / "time_series_60min_singleindex.csv"
    data_cfg = repo_root / "configs" / "data.yaml"
    train_cfg = repo_root / "configs" / "train_forecast.yaml"
    signal_cfg = _load_signal_config(data_cfg, repo_root, log)
    weather_cfg = _load_weather_config(data_cfg, repo_root, log)
    holidays_cfg = _load_holidays_config(data_cfg, repo_root, log)

    raw_hash = _hash_paths([raw_csv], repo_root) if raw_csv.exists() else None
    data_cfg_hash = _hash_paths([data_cfg], repo_root) if data_cfg.exists() else None
    train_cfg_hash = _hash_paths([train_cfg], repo_root) if train_cfg.exists() else None
    signals_hash = (
        _hash_paths([signal_cfg.path], repo_root) if signal_cfg.enabled and signal_cfg.path and signal_cfg.path.exists() else None
    )

    features_path = repo_root / "data" / "processed" / "features.parquet"

    if "data" in steps:
        if not raw_csv.exists():
            raise FileNotFoundError(f"Missing raw CSV: {raw_csv}")
        if (
            not args.force
            and cache.get("raw_hash") == raw_hash
            and cache.get("data_cfg_hash") == data_cfg_hash
            and cache.get("signals_hash") == signals_hash
            and features_path.exists()
        ):
            log.info("Data step skipped (cache hit)")
        else:
            _run([sys.executable, "-m", "gridpulse.data_pipeline.validate_schema", "--in", "data/raw", "--report", "reports/data_quality_report.md"], log)
            build_cmd = [
                sys.executable,
                "-m",
                "gridpulse.data_pipeline.build_features",
                "--in",
                "data/raw",
                "--out",
                "data/processed",
            ]
            if signal_cfg.enabled:
                if signal_cfg.path and signal_cfg.path.exists():
                    build_cmd += ["--signals", str(signal_cfg.path)]
                else:
                    log.warning("Signals enabled but file missing; proceeding without signals.")
            if weather_cfg.enabled:
                if weather_cfg.path and weather_cfg.path.exists():
                    build_cmd += ["--weather", str(weather_cfg.path)]
                else:
                    log.warning("Weather enabled but file missing; proceeding without weather.")
            if holidays_cfg.enabled:
                if holidays_cfg.path and holidays_cfg.path.exists():
                    build_cmd += ["--holidays", str(holidays_cfg.path)]
                else:
                    log.warning("Holidays enabled but file missing; proceeding without holidays.")
                build_cmd += ["--holiday-country", holidays_cfg.country]
            _run(build_cmd, log)
            _run([sys.executable, "-m", "gridpulse.data_pipeline.split_time_series", "--in", "data/processed/features.parquet", "--out", "data/processed/splits"], log)

        cache["raw_hash"] = raw_hash
        cache["data_cfg_hash"] = data_cfg_hash
        cache["signals_hash"] = signals_hash
        if features_path.exists():
            cache["features_hash"] = _hash_paths([features_path], repo_root)

    if "train" in steps:
        if not features_path.exists():
            raise FileNotFoundError("Missing data/processed/features.parquet. Run data step first.")
        train_hash = _hash_paths([features_path, train_cfg], repo_root)
        if not args.force and cache.get("train_hash") == train_hash and (repo_root / "artifacts" / "models").exists():
            log.info("Train step skipped (cache hit)")
        else:
            _run([sys.executable, "-m", "gridpulse.forecasting.train", "--config", "configs/train_forecast.yaml"], log)
            register_models(repo_root / "artifacts" / "models", repo_root / "artifacts" / "registry" / "models.json", run_id=run_id)
        cache["train_hash"] = train_hash

    if "reports" in steps:
        if not args.force and cache.get("reports_hash") == cache.get("train_hash"):
            log.info("Reports step skipped (cache hit)")
        else:
            _run([sys.executable, "scripts/build_reports.py"], log)
        cache["reports_hash"] = cache.get("train_hash")

    # snapshot outputs
    _snapshot_artifacts(repo_root, run_dir, log)
    _snapshot_configs(repo_root, run_dir)
    _pip_freeze(repo_root, run_dir, log)

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.utcnow().isoformat(),
        "git_commit": _git_commit(repo_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "raw_hash": cache.get("raw_hash"),
        "data_cfg_hash": cache.get("data_cfg_hash"),
        "features_hash": cache.get("features_hash"),
        "train_hash": cache.get("train_hash"),
        "reports_hash": cache.get("reports_hash"),
        "signals_hash": cache.get("signals_hash"),
        "steps": steps,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    _save_cache(cache_path, cache)
    log.info("Pipeline complete: %s", run_id)


if __name__ == "__main__":
    main()
