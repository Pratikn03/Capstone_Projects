# IoT Pi Validation

This runbook is a 3-day checklist for validating DC3S on a Raspberry Pi or similar edge target. Keep the API host, edge-agent host, and benchmark host aligned where possible so the Day 3 latency numbers reflect the same environment used in Days 1-2.

## Day 1: Shadow Mode Run

Start the API:

```bash
source .venv/bin/activate
uvicorn services.api.main:app --host 0.0.0.0 --port 8000
```

In a separate shell, start the edge agent in shadow mode:

```bash
source .venv/bin/activate
export GRIDPULSE_IOT_API_KEY='<gridpulse_rw_key>'
python iot/edge_agent/run_agent.py --config configs/iot.yaml --mode shadow --iterations 24
```

Explicit override example:

```bash
python iot/edge_agent/run_agent.py \
  --config configs/iot.yaml \
  --mode shadow \
  --iterations 24 \
  --device-id edge-device-001 \
  --zone-id DE \
  --api-base-url http://<pi-or-lan-host>:8000
```

## Day 2: Fault Injection with `tc netem`

Select the active network interface:

```bash
export IFACE=eth0
```

Inject packet loss:

```bash
sudo tc qdisc add dev "$IFACE" root netem loss 5%
```

Inject delay and jitter:

```bash
sudo tc qdisc change dev "$IFACE" root netem delay 200ms 50ms distribution normal
```

Combined example:

```bash
sudo tc qdisc add dev "$IFACE" root netem loss 5% delay 200ms 50ms
```

Inspect current rules:

```bash
sudo tc qdisc show dev "$IFACE"
```

Clean up after the fault-injection session:

```bash
sudo tc qdisc del dev "$IFACE" root
```

## Day 3: Run the DC3S Micro-Benchmark

Run the benchmark on the same Pi or edge target used for Days 1-2:

```bash
source .venv/bin/activate
python scripts/benchmark_dc3s_steps.py --iterations 10000 --out reports/dc3s_latency_benchmark.json
```

Inspect the recorded artifact:

```bash
cat reports/dc3s_latency_benchmark.json
```

Expected artifact:

- `reports/dc3s_latency_benchmark.json`
