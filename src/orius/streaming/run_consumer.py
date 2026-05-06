"""CLI entrypoint to run the streaming ingest consumer from YAML config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def _load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config payload in {path}: expected mapping")
    return payload


def _build_app_config(payload: dict[str, Any]) -> Any:
    from orius.streaming.consumer import (
        AppConfig,
        ConsumerConfig,
        StorageConfig,
        ValidationConfig,
    )

    kafka_cfg = payload.get("kafka", {})
    storage_cfg = payload.get("storage", {})
    validation_cfg = payload.get("validation", {})
    checkpoint_cfg = payload.get("checkpoint", {})

    return AppConfig(
        kafka=ConsumerConfig(
            bootstrap_servers=str(kafka_cfg.get("bootstrap_servers", "localhost:9092")),
            topic=str(kafka_cfg.get("topic", "orius.opsd.v1")),
            group_id=str(kafka_cfg.get("group_id", "orius-consumer")),
            auto_offset_reset=str(kafka_cfg.get("auto_offset_reset", "earliest")),
        ),
        storage=StorageConfig(
            mode=str(storage_cfg.get("mode", "duckdb")),
            duckdb_path=str(storage_cfg.get("duckdb_path", "data/interim/streaming.duckdb")),
            table_name=str(storage_cfg.get("table_name", "telemetry_events")),
            parquet_dir=str(storage_cfg.get("parquet_dir", "data/interim/streaming_parquet")),
        ),
        checkpoint_path=str(checkpoint_cfg.get("path", "artifacts/checkpoints/streaming_checkpoint.json")),
        validation=ValidationConfig(
            strict=bool(validation_cfg.get("strict", True)),
            cadence_seconds=int(validation_cfg.get("cadence_seconds", 3600)),
            cadence_tolerance_seconds=int(validation_cfg.get("cadence_tolerance_seconds", 120)),
            min_mw=float(validation_cfg.get("min_mw", 0.0)),
            max_mw=float(validation_cfg.get("max_mw", 200000.0)),
            max_delta_mw=(
                float(validation_cfg["max_delta_mw"])
                if validation_cfg.get("max_delta_mw") is not None
                else None
            ),
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run streaming ingest consumer from YAML config")
    ap.add_argument("--config", required=True, help="Path to streaming YAML config")
    ap.add_argument("--max-messages", type=int, default=None, help="Stop after N messages")
    args = ap.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    payload = _load_yaml_config(config_path)
    app_config = _build_app_config(payload)

    Path(app_config.storage.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(app_config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    if app_config.storage.parquet_dir:
        Path(app_config.storage.parquet_dir).mkdir(parents=True, exist_ok=True)

    from orius.streaming.consumer import StreamingIngestConsumer

    consumer = StreamingIngestConsumer(app_config)
    try:
        consumer.run_forever(max_messages=args.max_messages)
    except KeyboardInterrupt:
        return
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
