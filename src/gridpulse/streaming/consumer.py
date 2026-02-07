"""Streaming consumer for Kafka/Redpanda ingestion."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import duckdb
from kafka import KafkaConsumer

from .schemas import OPSDTelemetryEvent
from .checkpoint import load_checkpoint, save_checkpoint


@dataclass
class ConsumerConfig:
    bootstrap_servers: str
    topic: str
    group_id: str
    auto_offset_reset: str = "earliest"


@dataclass
class StorageConfig:
    mode: str  # duckdb|parquet
    duckdb_path: str
    table_name: str
    parquet_dir: str


@dataclass
class ValidationConfig:
    strict: bool = True
    cadence_seconds: int = 3600
    cadence_tolerance_seconds: int = 120
    min_mw: float = 0.0
    max_mw: float = 200000.0
    max_delta_mw: float | None = None


@dataclass
class AppConfig:
    kafka: ConsumerConfig
    storage: StorageConfig
    checkpoint_path: str
    validation: ValidationConfig


class StreamingIngestConsumer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ckpt = load_checkpoint(cfg.checkpoint_path)
        self._last_ts: Optional[datetime] = None
        self._last_values: Dict[str, float] = {}
        self.validation = cfg.validation
        self.consumer = KafkaConsumer(
            cfg.kafka.topic,
            bootstrap_servers=cfg.kafka.bootstrap_servers,
            group_id=cfg.kafka.group_id,
            auto_offset_reset=cfg.kafka.auto_offset_reset,
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )

        if cfg.storage.mode == "duckdb":
            self.con = duckdb.connect(cfg.storage.duckdb_path)
            self._init_duckdb()

    def _init_duckdb(self) -> None:
        # Create a simple wide table. DuckDB is flexible; you can also normalize later.
        self.con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.cfg.storage.table_name} (
              utc_timestamp VARCHAR,
              payload JSON,
              ingested_at DOUBLE
            )
            """
        )

    def _validate(self, event_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            evt = OPSDTelemetryEvent(**event_dict)
            normalized = evt.model_dump()
            self._check_event(normalized)
            return normalized
        except Exception:
            if self.cfg.validation.strict:
                raise
            return None

    def _parse_ts(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None

    def _check_event(self, event_dict: Dict[str, Any]) -> None:
        ts = self._parse_ts(event_dict.get("utc_timestamp"))
        if ts is None:
            raise ValueError("Invalid utc_timestamp; expected ISO-8601 string.")

        if self._last_ts is not None and self.validation.cadence_seconds > 0:
            delta = (ts - self._last_ts).total_seconds()
            tol = self.validation.cadence_tolerance_seconds
            if abs(delta - self.validation.cadence_seconds) > tol:
                raise ValueError(
                    f"Cadence violation: delta={delta}s expected ~{self.validation.cadence_seconds}s"
                )

        # Basic unit/outlier sanity checks in MW.
        for key in [
            "DE_load_actual_entsoe_transparency",
            "DE_wind_generation_actual",
            "DE_solar_generation_actual",
        ]:
            val = event_dict.get(key)
            if val is None:
                continue
            if val < self.validation.min_mw or val > self.validation.max_mw:
                raise ValueError(f"{key} out of bounds: {val}")
            if self.validation.max_delta_mw is not None and key in self._last_values:
                if abs(val - self._last_values[key]) > self.validation.max_delta_mw:
                    raise ValueError(f"{key} jump too large: {val} vs {self._last_values[key]}")

        self._last_ts = ts
        for key in [
            "DE_load_actual_entsoe_transparency",
            "DE_wind_generation_actual",
            "DE_solar_generation_actual",
        ]:
            val = event_dict.get(key)
            if val is not None:
                self._last_values[key] = float(val)

    def _write(self, event_dict: Dict[str, Any]) -> None:
        norm = self._validate(event_dict)
        if norm is None:
            return

        if self.cfg.storage.mode == "duckdb":
            self.con.execute(
                f"INSERT INTO {self.cfg.storage.table_name} VALUES (?, ?, ?)",
                [norm["utc_timestamp"], json.dumps(event_dict), time.time()],
            )
        else:
            # Parquet write can be added later: append partitioned parquet by date/hour.
            pass

    def run_forever(self, max_messages: Optional[int] = None) -> None:
        count = 0
        for msg in self.consumer:
            event_dict = msg.value
            self._write(event_dict)

            count += 1
            if count % 200 == 0:
                save_checkpoint(self.cfg.checkpoint_path, {"last_count": count, "last_ts": time.time()})
                self.consumer.commit()

            if max_messages is not None and count >= max_messages:
                break

        save_checkpoint(self.cfg.checkpoint_path, {"last_count": count, "last_ts": time.time()})
        try:
            self.consumer.commit()
        except Exception:
            pass
