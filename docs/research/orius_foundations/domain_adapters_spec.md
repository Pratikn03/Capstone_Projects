# Domain Adapter Specifications: Battery, Grid, and Robotics
**Target Quality: 9.8 / 10**  
*Physical Closed-Loop Contracts for ORIUS Action Release*

---

## 1. Universal Domain Adapter Interface Contract

All physical plants interact with ORIUS through a typed `DomainAdapter` interface adhering to Theorem E:

```python
class UniversalDomainAdapter(ABC):
    @abstractmethod
    def parse_telemetry(self, raw_packet: bytes | dict[str, Any]) -> TelemetrySnapshot:
        """Parse raw telemetry, timestamp, and metadata."""
        ...

    @abstractmethod
    def compute_reliability(self, history: list[TelemetrySnapshot]) -> float:
        """Stage 1 (Detect): Compute observation reliability w_t in [0, 1]."""
        ...

    @abstractmethod
    def calibrate_uncertainty_set(self, snapshot: TelemetrySnapshot, w_t: float) -> StateUncertaintySet:
        """Stage 2 (Calibrate): Inflate state uncertainty set U_t(H_t)."""
        ...

    @abstractmethod
    def tighten_admissible_actions(self, U_t: StateUncertaintySet) -> AdmissibleActionSet:
        """Stage 3 (Constrain): Compute robust safe action set A_robust(U_t)."""
        ...

    @abstractmethod
    def shield_action(self, a_candidate: Action, admissible_set: AdmissibleActionSet) -> Action:
        """Stage 4 (Shield): Convex projection or fallback selection."""
        ...

    @abstractmethod
    def emit_certificate(self, w_t: float, U_t: StateUncertaintySet, a_released: Action) -> ReleaseCertificate:
        """Stage 5 (Certify): Cryptographic signing and TTL issuance."""
        ...
```

---

## 2. Domain 1: Physical Battery Storage (BESS) Specification

* **Physical Plant**: Lithium Iron Phosphate (LFP) / NMC battery bank with programmable bi-directional DC power supply.
* **Physical State**: $x_t = [\mathrm{SoC}_t, T_{\mathrm{cell}, t}, V_{\mathrm{terminal}, t}, I_{\mathrm{pack}, t}]^T$.
* **Telemetry**: Voltage, current, thermocouple array, BMS status words over Modbus TCP / CAN bus.
* **Degraded Telemetry Injections**:
  * Packet drops (simulating wireless gateway timeout).
  * Temperature sensor stuck-at-previous and linear drift.
  * Stale state-of-charge estimator during high C-rate cycling.
* **Safety Predicate $\mathcal{S}_{\mathrm{bess}}$**:
  * $0.10 \le \mathrm{SoC}_t \le 0.90$
  * $T_{\mathrm{cell}, t} \le 50.0^\circ\mathrm{C}$
  * $V_{\min} \le V_{\mathrm{terminal}, t} \le V_{\max}$
* **Candidate Action**: Active power dispatch setpoint $P_{\mathrm{dispatch}}$ (kW/MW) requested by market optimizer.
* **Robust Shielding**: Linear program projecting $P_{\mathrm{dispatch}}$ into:
  $$\mathcal{A}_{\mathrm{robust}}(U_t) = \left[ P_{\min}(U_t), \, P_{\max}(U_t) \right]$$
* **Fallback Authority**: Zero net active power ($P = 0.0\text{ kW}$) or physical contactor trip.

---

## 3. Domain 2: Power Grid & Microgrid Inverter Specification

* **Physical Plant**: Grid-forming / grid-following inverter connected to distribution bus.
* **Physical State**: $x_t = [\Delta f_t, \Delta V_{\mathrm{bus}, t}, P_{\mathrm{flow}, t}, Q_{\mathrm{flow}, t}]^T$.
* **Telemetry**: Phasor Measurement Unit (PMU) streaming positive-sequence voltage/current at 60 Hz.
* **Degraded Telemetry Injections**:
  * GPS-spoofed timestamp jitter and phase angle drift.
  * Sudden communication latency exceeding the 100 ms dispatch window.
* **Safety Predicate $\mathcal{S}_{\mathrm{grid}}$**:
  * NEMA/IEEE 1547 standards: $59.5\text{ Hz} \le f_t \le 60.5\text{ Hz}$
  * Bus voltage envelope: $0.95\text{ p.u.} \le V_{\mathrm{bus}, t} \le 1.05\text{ p.u.}$
  * Thermal line limit: $\sqrt{P^2 + Q^2} \le S_{\max}$
* **Candidate Action**: Real and reactive power setpoints $(P_{\mathrm{inv}}, Q_{\mathrm{inv}})$.
* **Robust Shielding**: Second-Order Cone Program (SOCP) projecting power setpoints into safe voltage/frequency boundaries.
* **Fallback Authority**: Local autonomous droop control curve without remote supervisory dispatch.

---

## 4. Domain 3: Physical Mobile Robot (ROS 2) Specification

* **Physical Plant**: Differential-drive / Ackermann ground robot in an obstacle-populated environment.
* **Physical State**: $x_t = [p_{x, t}, p_{y, t}, \theta_t, v_t, \omega_t]^T$.
* **Telemetry**: 2D LiDAR point cloud, wheel encoders, 9-DOF IMU, depth camera streaming via ROS 2 topics (`/scan`, `/odom`, `/imu`).
* **Degraded Telemetry Injections**:
  * LiDAR ray dropouts and multipath reflections.
  * Wheel odometry slippage on low-friction terrain.
  * ROS 2 executor scheduling latency and message queue drops.
* **Safety Predicate $\mathcal{S}_{\mathrm{robot}}$**:
  * Non-collision with static and dynamic obstacles:
    $$\min_{j} \| p_{\mathrm{robot}, t} - p_{\mathrm{obs}, j, t} \|_2 \ge d_{\mathrm{stop}}(v_t) + d_{\mathrm{margin}}$$
  * Kinematic limits: $|v_t| \le v_{\max}$, $|\omega_t| \le \omega_{\max}$.
* **Candidate Action**: Velocity command $a_t = [v_{\mathrm{cmd}}, \omega_{\mathrm{cmd}}]^T$ from nominal navigation stack (e.g. Nav2, DWA, TEB local planner).
* **Robust Shielding**: Velocity obstacle / Control Barrier Function projection over the inflated localization set $U_t(H_t)$.
* **Fallback Authority**: Immediate maximum emergency deceleration to zero velocity ($v = 0, \omega = 0$) and mechanical brake lock.
