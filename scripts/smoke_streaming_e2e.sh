#!/usr/bin/env bash
set -e

docker compose -f docker/docker-compose.streaming.yml up -d
python -m pip install -q kafka-python duckdb pandas pydantic pyyaml

# Start consumer in background
python -m gridpulse.streaming.run_consumer --config configs/streaming.yaml &
CONSUMER_PID=$!

# Replay
python scripts/replay_opsd_to_kafka.py --csv data/raw/time_series_60min_singleindex.csv --rate 100

# Give consumer time to flush
sleep 3
kill $CONSUMER_PID || true

echo "Streaming smoke test completed."
