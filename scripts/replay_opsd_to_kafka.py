"""
Replay OPSD (or your processed features) as Kafka events to simulate real-time streaming.

Example:
  python scripts/replay_opsd_to_kafka.py \
    --csv data/raw/time_series_60min_singleindex.csv \
    --topic gridpulse.opsd.v1 \
    --rate 50
"""
import argparse
import json
import time

import pandas as pd
from kafka import KafkaProducer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--topic", default="gridpulse.opsd.v1")
    ap.add_argument("--bootstrap", default="localhost:9092")
    ap.add_argument("--rate", type=float, default=20.0, help="rows/sec")
    ap.add_argument("--timestamp-col", default="utc_timestamp")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.timestamp_col not in df.columns:
        raise ValueError(
            f"timestamp col '{args.timestamp_col}' not found in CSV columns={list(df.columns)[:10]}..."
        )

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=5,
    )

    delay = 1.0 / max(args.rate, 0.1)
    for _, row in df.iterrows():
        event = row.to_dict()
        producer.send(args.topic, event)
        time.sleep(delay)

    producer.flush()
    print(f"Replayed {len(df)} rows to topic={args.topic}")


if __name__ == "__main__":
    main()
