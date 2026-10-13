# ORIUS Evaluation Metrics & Experimental Unit Specification
**Issue OR-008 Milestone Specification**

---

## 1. Statistical Framework & Independent Experimental Units (IEUs)

A critical flaw in physical AI literature is the calculation of $p$-values and hypothesis tests across individual timesteps within a closed-loop trajectory. Because states $x_t$ and disturbances $w_t$ exhibit severe temporal autocorrelation, individual steps violate the independent and identically distributed (i.i.d.) assumption.

### The Independent Experimental Unit (IEU)
In ORIUS, the fundamental unit of statistical replication is the **independent closed-loop episode** $\mathcal{E}^{(i)}$:
$$\mathcal{E}^{(i)} = \left( x_0^{(i)}, \{w_t^{(i)}\}_{t=0}^{T-1}, \{\nu_t^{(i)}\}_{t=0}^{T-1}, \{d_t^{(i)}\}_{t=0}^{T-1} \right), \quad i \in \{1, \dots, N\},$$
where initial states, disturbance realizations, and telemetry dropout sequences are drawn independently from their respective non-stationary distributions.

---

## 2. Core Quantitative Metrics

### 2.1 Trajectory Safety Violation Rate (TSVR)
Measures the proportion of independent rollouts where the true physical state penetrated the true safety set $\mathcal{S}_d$ at least once:
$$\text{TSVR} \triangleq \frac{1}{N} \sum_{i=1}^N \mathbf{1}_{\left\{ \exists t \in \{0, \dots, T\} : x_t^{(i)} \notin \mathcal{S}_d \right\}}.$$

* **Exact Binomial Confidence Bounds**: For $N$ independent rollouts with $k$ observed violations, we report the two-sided $95\%$ Clopper-Pearson exact confidence interval $[p_L, p_U]$:
  $$p_L = \text{Beta}\left(\frac{\alpha}{2}; k, N - k + 1\right), \quad p_U = \text{Beta}\left(1 - \frac{\alpha}{2}; k + 1, N - k\right).$$
  For ORIUS with $k = 0$ over $N = 1000$, the 95% upper confidence bound on true trajectory violation rate is:
  $$\text{TSVR}_{95\%} \le 1 - (0.05)^{1/1000} \approx 0.298\%.$$

### 2.2 Normalized Task Utility (NTU)
Quantifies task completion performance normalized against unconstrained nominal execution and conservative fallback:
$$\text{NTU} \triangleq \frac{J(\pi) - J(\pi_{\mathrm{fallback}})}{J(\pi_{\mathrm{nominal}}) - J(\pi_{\mathrm{fallback}})} \in [0, 1],$$
where $J(\pi) = \mathbb{E}\left[\sum_{t=0}^T \gamma^t r(x_t, u_t)\right]$.

### 2.3 Intervention Rate (IR) and Fallback Rate (FR)
Measures the frequency of safety filter modifications and emergency fallback activations:
$$\text{IR} \triangleq \frac{1}{N \cdot T} \sum_{i=1}^N \sum_{t=0}^{T-1} \mathbf{1}_{\left\{ u_t^{(i)} \ne u_{\mathrm{nom}, t}^{(i)} \right\}}, \quad \text{FR} \triangleq \frac{1}{N \cdot T} \sum_{i=1}^N \sum_{t=0}^{T-1} \mathbf{1}_{\left\{ u_t^{(i)} = u_{\mathrm{fb}} \right\}}.$$

### 2.4 Mean and Max Recovery Time (MRT)
Following a burst of sensor degradation or packet loss of duration $\Delta \tau_{\mathrm{drop}}$, MRT measures the number of steps required for the controller to restore nominal performance after observation telemetry is restored:
$$\tau_{\mathrm{recovery}} \triangleq \min \left\{ k \ge 0 \;\middle|\; \|u_{t_{\mathrm{restore}} + k} - u_{\mathrm{nom}, t_{\mathrm{restore}} + k}\| < \epsilon \right\}.$$

### 2.5 Real-Time Latency Envelope
Reported in microseconds ($\mu\text{s}$) across wall-clock time spent in `DomainAdapter.evaluate_and_filter`:
- Median Latency ($p_{50}$)
- 95th Percentile Latency ($p_{95}$)
- Maximum Peak Latency ($p_{\max}$)
*Hard constraint*: $p_{95} \le 1000\,\mu\text{s}$ (1 ms) for 1 kHz robotics / 60 Hz power grid loops.
