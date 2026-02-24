# CODEX IMPLEMENTATION UPDATE — GridPulse → DC³S + CPSBench-IoT + Closed-Loop IoT Validation
**Audience:** Codex / coding agent implementing changes in this repo.  
**Repo root:** `Capstone_Projects-main/`  
**Primary deliverable:** DC³S (Drift-Calibrated Conformal Safety Shield) + CPSBench-IoT benchmark + closed-loop IoT validation.

> **Do not “re-architect” the whole repo.** Keep existing forecasting/optimization code as-is whenever possible.
> Add new components in additive folders and integrate via FastAPI routers + scripts + tests.

---

## 0) Guardrails (must follow)
1) **No fake outputs** in production routes. If demo mode exists, label it clearly.
2) **Fresh clone runnable:** `make pre-release` must pass.
3) **No absolute local paths** (`/Users/...`) in tracked docs.
4) **Deterministic benchmark runs:** seed all scenario generation.
5) **Backwards compatible API:** existing endpoints remain unchanged.

---

## 1) Immediate publish blockers (fix these first)

### 1.1 Replace mocked chat tool outputs with real FastAPI calls
**File:** `frontend/src/app/api/chat/route.ts`  
Currently, tool `execute` functions return hardcoded objects.

**Change:** For each tool, call backend via fetch (server-side route can use `process.env.FASTAPI_URL`).
- `get_load_forecast` → GET `/forecast/with-intervals?target=...&horizon=...`
- `get_dispatch_forecast` → GET `/forecast?targets=...&horizon=...` + format into chart payload
- `get_battery_schedule` → POST `/optimize` with forecast arrays then extract schedule
- `get_grid_status` → GET `/monitor` (and/or `/anomaly`) + health endpoints
- `get_cost_carbon_tradeoff` → either call existing optimizer with varying carbon weight OR label as “not available” until implemented
- `get_model_info` → add a small FastAPI endpoint `/monitor/model-info` (preferred) OR load from registry artifacts

**Acceptance tests:**
- `cd frontend && npm run build`
- Chat tool calls produce values sourced from backend (verify by temporarily disabling backend and observing failure rather than returning fake data).

### 1.2 Fix local absolute paths in reports
**File:** `reports/formal_evaluation_report.md`  
Line contains: `![](/Users/pratik_n/Downloads/gridpulse/reports/figures/multi_horizon_backtest.png)`

**Change:** Replace with repo-relative path:
`![](figures/multi_horizon_backtest.png)`

### 1.3 Remove tracked coverage artifacts
**Files (tracked):**
- `reports/.coverage`
- `reports/coverage.xml`

**Change:**
- Delete both files from repo.
- Add to root `.gitignore`:
  - `reports/.coverage`
  - `reports/coverage.xml`
  - `reports/coverage/` (if generated)

### 1.4 Fix dashboard port mismatches (8501 → 3000)
**Files:**
- `deploy/k8s/gridpulse-dashboard-deployment.yaml` (containerPort + probes)
- `deploy/k8s/gridpulse-dashboard-service.yaml` (targetPort)
- `deploy/aws/ecs-task-def-dashboard.json` (containerPort mapping)

**Change:** Set to **3000** everywhere.

### 1.5 Fix deploy workflow Dockerfile reference
**File:** `.github/workflows/deploy.yml`  
References `docker/Dockerfile.app` which does not exist.

**Change:** Use `docker/Dockerfile.frontend` (or create `Dockerfile.app` as an alias).

---

## 2) Streaming correctness (release smoke must work)

### 2.1 Add missing module `gridpulse.streaming.run_consumer`
**Script expects:**
`scripts/release_check.sh` runs:
`python -m gridpulse.streaming.run_consumer --config configs/streaming.yaml --max-messages 500`

**Add new file:** `src/gridpulse/streaming/run_consumer.py`  
Implementation:
- Parse args: `--config`, `--max-messages`
- Load yaml config into `AppConfig` (use existing `gridpulse.streaming.consumer` config classes)
- Instantiate `StreamingIngestConsumer(app_config)`
- Run `.run(max_messages=...)` or loop consumer to process `max_messages` then exit cleanly.

**Acceptance:**
- `bash scripts/release_check.sh` reaches step (2) and does not error on missing module.

### 2.2 Make streaming worker persist validated messages
**File:** `src/gridpulse/streaming/worker.py`  
Currently validates but does not write to DuckDB.

**Change:**
- After parsing `OPSDTelemetryEvent(**message)`, call a persistence function.
Preferred: expose a public method in `StreamingIngestConsumer` like `.write_event(event_dict)` that writes to DuckDB.
Fallback: call existing internal `_write(...)` if available.

**Acceptance:**
- Run streaming smoke: replay script creates rows in DuckDB table `telemetry_events`.

---

## 3) Add DC³S (primary novelty method)

### 3.1 New config
**Add:** `configs/dc3s.yaml`

Minimum parameters:
```yaml
dc3s:
  alpha0: 0.10
  alpha_min: 0.02
  k_quality: 0.8
  k_drift: 0.6
  reliability:
    lambda_delay: 0.002
    spike_beta: 0.25
    ooo_gamma: 0.35
    min_w: 0.05
  drift:
    detector: page_hinkley
    ph_delta: 0.01
    ph_lambda: 5.0
    warmup_steps: 48
    cooldown_steps: 24
  shield:
    mode: projection   # projection | robust_resolve
    reserve_soc_pct_drift: 0.08
  audit:
    duckdb_path: data/audit/dc3s_audit.duckdb
    table_name: dispatch_certificates
```

### 3.2 New package
**Add folder:** `src/gridpulse/dc3s/`

Files to create:

#### `quality.py`
- `compute_reliability(event, last_event, expected_cadence_s) -> (w_t: float, flags: dict)`
- Input can be dict-like telemetry: `ts_utc`, key features, etc.
- Compute: missing fraction, delay, out-of-order, spike.
- Output: `w_t in [min_w, 1.0]`.

#### `drift.py`
- Implement Page-Hinkley drift on residual magnitude `r_t = |y_t - yhat_t|`.
- Class `PageHinkleyDetector` with `update(r_t) -> {drift: bool, score: float}`.
- Include warmup/cooldown behavior.

#### `calibration.py`
- DC³S uses existing conformal intervals (from `gridpulse.forecasting.uncertainty.conformal`).
- Provide:
  - `inflate_interval(lower, upper, inflation)` OR `inflate_q(q, inflation)`
  - `build_uncertainty_set(yhat, q, w_t, drift_flag, cfg) -> (lower, upper, meta)`
- Optional: integrate `AdaptiveConformal` for online alpha updates using last-step miss.

#### `shield.py`
- Core: `repair_action(a_star, state, uncertainty_set, constraints, cfg) -> (a_safe, repair_meta)`
- Support two modes:
  1) **projection**: clip charge/discharge to satisfy SOC + power + ramp under worst-case bounds (fast).
  2) **robust_resolve**: if infeasible, call `optimize_robust_dispatch(...)` with interval bounds derived from DC³S uncertainty.

#### `certificate.py`
- `make_certificate(...) -> dict`
- Compute `model_hash`, `config_hash` (sha256 of bytes).
- Maintain optional hash-chain: include `prev_hash`.
- Store to DuckDB.

#### `state.py`
- Persist online state per (zone/device/target):
  - last timestamp
  - last yhat, last y_true
  - drift detector state
  - any adaptive alpha state
  - last prev_hash for audit chain

### 3.3 FastAPI router for DC³S
**Add:** `services/api/routers/dc3s.py`  
Mount with: `app.include_router(dc3s.router, prefix="/dc3s", tags=["dc3s"])`

Endpoints:

#### `POST /dc3s/step`
Inputs (Pydantic):
- `device_id`, `zone_id` (DE/US)
- `current_soc_mwh`
- telemetry quality info (or `telemetry_event` dict)
- optional: `last_actual_load_mw`, `last_pred_load_mw` (for residual update)
- `horizon` (default 24)
- `controller` = `deterministic|robust|heuristic` (start with deterministic/robust)
Outputs:
- `proposed_action` (charge/discharge for step 0)
- `safe_action` (after shield)
- `dispatch_plan` (optional full plan)
- `uncertainty` (bounds used)
- `certificate_id` / `command_id`
- `certificate` (optional inline)

Implementation sketch:
1) Load `configs/dc3s.yaml`
2) Compute `w_t` from telemetry
3) Update drift detector using residual input if present
4) Obtain yhat + conformal interval from existing forecast endpoints (internal call) OR call `predict_next_24h(...)` directly.
5) Inflate interval using `w_t` + drift.
6) Call optimizer to get dispatch plan.
7) Run shield repair; run BMS validate.
8) Create certificate; persist; return response.

#### `GET /dc3s/audit/{command_id}`
Return stored certificate JSON.

**Acceptance:**
- `curl -X POST /dc3s/step ...` returns safe action + non-empty certificate.
- `GET /dc3s/audit/<id>` returns matching record.

---

## 4) Add CPSBench-IoT (benchmark signal)

### 4.1 New package
**Add:** `src/gridpulse/cpsbench_iot/`

Required modules:

#### `scenarios.py`
- Generate episodes with deterministic seed.
- Fault injectors: dropout, delay/jitter, out-of-order, spikes, stale sensor.
- Drift injectors: covariate drift, label drift at time T.
- Must return:
  - observed stream `x_obs[t]`
  - ground truth `x_true[t]`
  - fault/drift metadata log.

#### `metrics.py`
Forecast:
- MAE/RMSE
- Coverage (PICP) at 90/95
- Mean interval width
Control:
- violation rate (SOC/power)
- violation severity (integral or sum of margin)
- recovery time after drift/fault
- intervention rate (% repaired actions)
Trace:
- % certificates present, missing-field count

#### `baselines.py`
- deterministic MILP (existing optimize)
- robust fixed-interval (existing robust dispatch with fixed bounds)
- heuristic controller (simple price-based)
- DC³S wrapper around any controller (calls your `dc3s` components)

#### `runner.py`
- CLI-friendly runner: `python -m gridpulse.cpsbench_iot.runner --scenario ... --seed ...`
- Write outputs to `reports/publication/`:
  - `dc3s_main_table.csv`
  - `dc3s_fault_breakdown.csv`
  - `calibration_plot.png`
  - `violation_vs_cost_curve.png`

### 4.2 Script entrypoint
**Add:** `scripts/run_cpsbench.py`
- Runs default suite (e.g., 6 scenarios × 5 seeds)
- Produces publication outputs

### 4.3 Makefile targets
Add:
- `cpsbench:`
  - `python scripts/run_cpsbench.py`
- `dc3s-demo:`
  - runs a short scenario and prints certificate
- `iot-sim:`
  - runs simulator loop (below)

**Acceptance:**
- `make cpsbench` produces tables/plots with deterministic results (same seed → same numbers).

---

## 5) Add IoT verification + closed-loop validation

### 5.1 Device contract
**Add:** `iot/DEVICE_CONTRACT.md`
Must include:
- telemetry schema + units + valid ranges + cadence
- command schema
- ACK/NACK schema
- transport choices:
  - HTTP endpoints (below) OR MQTT topics (optional)

### 5.2 Edge agent (sim first, hardware later)
**Add:** `iot/edge_agent/agent.py`
Responsibilities:
- send telemetry on cadence
- poll for next command
- apply command via driver (sim or real stub)
- send ACK/NACK

Also add:
- `iot/edge_agent/drivers/sim.py` (digital twin)
- `iot/edge_agent/drivers/real_stub.py` (placeholder for Modbus/BMS/inverter)

### 5.3 IoT API router
**Add:** `services/api/routers/iot.py`  
Mount: `app.include_router(iot.router, prefix="/iot", tags=["iot"])`

Endpoints:
- `POST /iot/telemetry` → store telemetry + compute `w_t` + update latest state
- `GET /iot/command/next?device_id=...` → returns queued command (or none)
- `POST /iot/ack` → store ack and link to certificate/command_id
- `GET /iot/state?device_id=...` → latest telemetry + last command + last ack
- `GET /iot/audit/{command_id}` → returns certificate (proxy to DC³S audit store)

### 5.4 Closed-loop simulator
**Add:** `iot/simulator/run_closed_loop.py`
Loop:
1) simulate telemetry
2) POST `/iot/telemetry`
3) POST `/dc3s/step` to get safe action + certificate and queue command
4) agent applies and POST `/iot/ack`
5) record outcomes + violations + trace completeness

**Acceptance:**
- `make iot-sim` runs end-to-end and prints summary:
  - 0 violations in normal scenario
  - conservative behavior under dropout scenario

---

## 6) Tests (must add)
Add tests to `tests/`:

- `tests/test_dc3s_quality.py` (w_t bounds, flags correct)
- `tests/test_dc3s_drift.py` (Page-Hinkley triggers drift)
- `tests/test_dc3s_shield.py` (unsafe action repaired)
- `tests/test_dc3s_api_smoke.py` (POST /dc3s/step returns certificate)
- `tests/test_iot_loop_smoke.py` (telemetry→command→ack works)
- `tests/test_cpsbench_smoke.py` (runner creates expected output files)

**Acceptance:** `make test` passes.

---

## 7) Documentation updates (keep trust high)
Update:
- `docs/ARCHITECTURE.md` to include DC³S + CPSBench + IoT closed loop
- `docs/EVALUATION.md` to document CPSBench metrics + baselines
- `README.md` to add “Reproduce paper figures” section:
  - `make cpsbench`
  - `make iot-sim`
  - `make dc3s-demo`

---

## 8) Final definition of done (release gate)
All must pass:
1) `make pre-release`
2) `make frontend-build`
3) `make cpsbench` (creates publication outputs)
4) `make iot-sim` (closed loop works)
5) Chat assistant tools return backend-derived data (no hardcoded demo values).

---

## Appendix: Quick grep checks (for trust)
- No local paths:
  - `grep -R "/Users/" -n reports docs notebooks || true`
- No mocked tool returns:
  - `grep -R "peak_load_mw" -n frontend/src/app/api/chat/route.ts` should show formatting only, not constants.
