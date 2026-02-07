"""Streaming consumer for Kafka/Redpanda ingestion."""
import json
import time
from dataclasses import dataclass
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
class AppConfig:
    kafka: ConsumerConfig
    storage: StorageConfig
    checkpoint_path: str
    strict_validation: bool = True


class StreamingIngestConsumer:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ckpt = load_checkpoint(cfg.checkpoint_path)
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
            return evt.model_dump()
        except Exception:
            if self.cfg.strict_validation:
                raise
            return None

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
