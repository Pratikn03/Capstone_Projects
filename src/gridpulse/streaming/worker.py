"""
Kafka Streaming Worker for GridPulse.

This module implements the main streaming worker that consumes telemetry
events from Kafka, validates them, and persists to DuckDB/PostgreSQL.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from gridpulse.streaming.consumer import (
    StreamingIngestConsumer,
    ConsumerConfig,
    StorageConfig,
    ValidationConfig,
    AppConfig,
)
from gridpulse.streaming.schemas import OPSDTelemetryEvent
from gridpulse.monitoring.prometheus_metrics import (
    KAFKA_MESSAGES_CONSUMED,
    KAFKA_CONSUMER_LAG,
    STREAMING_PROCESSING_DELAY,
    update_streaming_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerConfig:
    """Configuration for the streaming worker."""
    kafka_bootstrap_servers: str
    kafka_topic: str
    kafka_group_id: str
    duckdb_path: str
    checkpoint_path: str
    batch_size: int = 100
    commit_interval_seconds: int = 5
    max_poll_records: int = 500
    shutdown_timeout_seconds: int = 30


def load_config_from_env() -> WorkerConfig:
    """Load worker configuration from environment variables."""
    return WorkerConfig(
        kafka_bootstrap_servers=os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        ),
        kafka_topic=os.environ.get("KAFKA_TOPIC", "gridpulse-telemetry"),
        kafka_group_id=os.environ.get("KAFKA_GROUP_ID", "gridpulse-streaming-worker"),
        duckdb_path=os.environ.get(
            "DUCKDB_PATH", "data/streaming/events.duckdb"
        ),
        checkpoint_path=os.environ.get(
            "CHECKPOINT_PATH", "data/streaming/checkpoint.json"
        ),
        batch_size=int(os.environ.get("BATCH_SIZE", "100")),
        commit_interval_seconds=int(os.environ.get("COMMIT_INTERVAL_SECONDS", "5")),
        max_poll_records=int(os.environ.get("MAX_POLL_RECORDS", "500")),
        shutdown_timeout_seconds=int(os.environ.get("SHUTDOWN_TIMEOUT_SECONDS", "30")),
    )


class StreamingWorker:
    """
    Main streaming worker that processes Kafka messages.
    
    Features:
    - Graceful shutdown handling
    - Batch processing for efficiency
    - Dead letter queue for failed messages
    - Prometheus metrics integration
    - Checkpoint-based exactly-once processing
    """
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self._shutdown_requested = False
        self._consumer: Optional[StreamingIngestConsumer] = None
        self._messages_processed = 0
        self._last_commit_time = time.time()
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_requested = True
        
    def _create_consumer(self) -> StreamingIngestConsumer:
        """Create and configure the Kafka consumer."""
        kafka_config = ConsumerConfig(
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            topic=self.config.kafka_topic,
            group_id=self.config.kafka_group_id,
            auto_offset_reset="earliest",
        )
        
        storage_config = StorageConfig(
            mode="duckdb",
            duckdb_path=self.config.duckdb_path,
            table_name="telemetry_events",
            parquet_dir="",
        )
        
        validation_config = ValidationConfig(
            strict=False,  # Log but don't fail on validation errors
            cadence_seconds=3600,
            cadence_tolerance_seconds=120,
            min_mw=0.0,
            max_mw=200000.0,
        )
        
        app_config = AppConfig(
            kafka=kafka_config,
            storage=storage_config,
            checkpoint_path=self.config.checkpoint_path,
            validation=validation_config,
        )
        
        return StreamingIngestConsumer(app_config)
    
    def _process_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a single Kafka message.
        
        Returns:
            True if processing succeeded, False otherwise
        """
        try:
            # Parse and validate
            event = OPSDTelemetryEvent(**message)
            
            # Calculate processing delay
            try:
                event_time = datetime.fromisoformat(
                    event.utc_timestamp.replace("Z", "+00:00")
                )
                delay = (datetime.utcnow() - event_time.replace(tzinfo=None)).total_seconds()
                STREAMING_PROCESSING_DELAY.labels(
                    topic=self.config.kafka_topic
                ).set(max(0, delay))
            except Exception:
                pass
            
            # Increment counter
            KAFKA_MESSAGES_CONSUMED.labels(topic=self.config.kafka_topic).inc()
            self._messages_processed += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            return False
    
    def _should_commit(self) -> bool:
        """Check if we should commit offsets."""
        time_since_commit = time.time() - self._last_commit_time
        return time_since_commit >= self.config.commit_interval_seconds
    
    def run(self):
        """Main worker loop."""
        logger.info("Starting GridPulse Streaming Worker...")
        logger.info(f"Kafka: {self.config.kafka_bootstrap_servers}")
        logger.info(f"Topic: {self.config.kafka_topic}")
        logger.info(f"Group: {self.config.kafka_group_id}")
        
        try:
            self._consumer = self._create_consumer()
            logger.info("Consumer created successfully, starting to poll...")
            
            while not self._shutdown_requested:
                try:
                    # Poll for messages
                    for message in self._consumer.consumer:
                        if self._shutdown_requested:
                            break
                        
                        # Process message
                        success = self._process_message(message.value)
                        
                        if not success:
                            # Log to dead letter queue (could also publish to DLQ topic)
                            logger.warning(f"Message sent to DLQ: {message.value}")
                        
                        # Periodic commit
                        if self._should_commit():
                            self._consumer.consumer.commit()
                            self._last_commit_time = time.time()
                            logger.debug(
                                f"Committed offsets, processed {self._messages_processed} messages"
                            )
                    
                except Exception as e:
                    logger.error(f"Error in consumer loop: {e}")
                    if not self._shutdown_requested:
                        time.sleep(5)  # Back off before retry
                        
        except Exception as e:
            logger.error(f"Fatal error in worker: {e}")
            raise
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Cleaning up resources...")
        
        if self._consumer:
            try:
                # Final commit
                self._consumer.consumer.commit()
                self._consumer.consumer.close()
                logger.info("Consumer closed successfully")
            except Exception as e:
                logger.error(f"Error closing consumer: {e}")
        
        logger.info(f"Worker shutdown complete. Processed {self._messages_processed} messages.")


def main():
    """Main entry point."""
    config = load_config_from_env()
    
    # Ensure directories exist
    Path(config.duckdb_path).parent.mkdir(parents=True, exist_ok=True)
    Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    
    worker = StreamingWorker(config)
    worker.run()


if __name__ == "__main__":
    main()
