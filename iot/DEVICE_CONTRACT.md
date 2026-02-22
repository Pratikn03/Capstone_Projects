# GridPulse IoT Device Contract

This contract defines telemetry, command, and ACK/NACK payloads for the closed-loop IoT validation path.

## 0) Authentication and Scopes

All `/iot/*` endpoints require header:

```text
X-GridPulse-Key: <api-key>
```

Scope policy:

- Read scope: `GET /iot/state`, `GET /iot/command/next`, `GET /iot/audit/{command_id}`
- Write scope: `POST /iot/telemetry`, `POST /iot/ack`, `POST /iot/control/reset-hold`

## 1) Telemetry Contract (`POST /iot/telemetry`)

Required envelope:

```json
{
  "device_id": "string",
  "zone_id": "DE|US",
  "telemetry_event": {
    "ts_utc": "2026-02-22T12:00:00Z",
    "load_mw": 50000.0,
    "renewables_mw": 18000.0,
    "soc_mwh": 10000.0
  }
}
```

Telemetry value guidance:

- `load_mw`: MW, non-negative, expected range `[0, 200000]`
- `renewables_mw`: MW, non-negative, expected range `[0, 120000]`
- `soc_mwh`: MWh, bounded by device battery limits
- Cadence: 1 telemetry event per 1h in simulator mode (configurable)

## 2) Command Contract (queue payload consumed via `GET /iot/command/next`)

Returned shape:

```json
{
  "command_id": "uuid-or-stable-id",
  "device_id": "string",
  "zone_id": "DE|US",
  "status": "queued|dispatched",
  "certificate_id": "string|null",
  "command": {
    "safe_action": {
      "charge_mw": 0.0,
      "discharge_mw": 1250.0
    }
  }
}
```

## 3) ACK/NACK Contract (`POST /iot/ack`)

Request:

```json
{
  "device_id": "string",
  "command_id": "string",
  "status": "acked|nacked",
  "certificate_id": "string|null",
  "reason": "string|null",
  "payload": {
    "accepted": true,
    "violation": false
  }
}
```

Behavior:

- `acked`: command applied within safety limits.
- `nacked`: command rejected or clipped due to safety violation.

Shadow-mode ACK semantics:

- status remains `acked` for compatibility.
- payload must include:
  - `shadow_mode: true`
  - `applied: false`
  - `recommended_action` (charge/discharge recommendation).

## 4) Hold/Timeout Behavior

Command queue TTL defaults to 30 seconds (`queue_ttl_seconds`).

- Expired queued/dispatched commands are marked `status=timeout`.
- On timeout, device enters hold mode and dequeue is blocked.
- `GET /iot/command/next` returns:
  - `status=hold`
  - `hold_reason` (for example `ack_timeout`).

Reset endpoint:

```json
POST /iot/control/reset-hold
{
  "device_id": "string",
  "reason": "operator_clearance"
}
```

## 5) Transport

Current implementation transport:

- HTTP API endpoints (`/iot/*`, `/dc3s/*`)

Optional future extension:

- MQTT topic bridge for telemetry/commands/acks (not required for current release).
