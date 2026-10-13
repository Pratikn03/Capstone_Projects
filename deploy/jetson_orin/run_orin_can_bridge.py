#!/usr/bin/env python3
import time
import numpy as np
from orius.kbound import CANBusInterface, VelocityAwareUniversalShield
from datetime import datetime, timezone

print("ORIUS + K-Bound Jetson Orin CAN-Bus Hardware Loop Active.")
bus = CANBusInterface(baud_kbps=500, drop_rate=0.0)
shield = VelocityAwareUniversalShield(
    name="Orin_AV_Longitudinal",
    hard_limit=120.0,
    limit_type="upper",
    max_force=4.0,
    min_force=-8.0,
    robust_decel_floor=3.8,
    safety_margin=4.5,
)
now = datetime.now(timezone.utc)
for frame in range(10):
    t0 = time.perf_counter()
    pkt = bus.pack_telemetry(85.0 - frame * 2.0, 22.0, status=1)
    wire_delay_s, _ = bus.transmit_frame(pkt)
    dist, vel = bus.unpack_telemetry(pkt)
    safe_u, w_t, intervened = shield.step(now, wire_delay_s, dist, vel, nominal_u=2.0)
    dt_us = (time.perf_counter() - t0) * 1e6
    print(f"  Frame {frame+1}: safe_u={safe_u:.3f} m/s^2 | w_t={w_t:.4f} | intervened={intervened} | Latency={dt_us:.1f} us")
print("Hardware verification loop complete.")
