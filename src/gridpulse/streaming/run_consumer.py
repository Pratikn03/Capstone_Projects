"""CLI entrypoint for the streaming consumer."""
import argparse

import yaml

from .consumer import ConsumerConfig, StorageConfig, AppConfig, StreamingIngestConsumer, ValidationConfig


def load_config(path: str) -> AppConfig:
    cfg = yaml.safe_load(open(path, "r", encoding="utf-8"))
    kafka = ConsumerConfig(**cfg["kafka"])
    storage = StorageConfig(**cfg["storage"])
    validation = ValidationConfig(**(cfg.get("validation", {}) or {}))
    ckpt = cfg.get("checkpoint", {}).get("path", "artifacts/checkpoints/streaming_checkpoint.json")
    return AppConfig(kafka=kafka, storage=storage, checkpoint_path=ckpt, validation=validation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/streaming.yaml")
    ap.add_argument("--max-messages", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    consumer = StreamingIngestConsumer(cfg)
    consumer.run_forever(max_messages=args.max_messages)


if __name__ == "__main__":
    main()
