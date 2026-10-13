# ORIUS Theorem Dependency Graph
**Issue OR-003 Foundational Milestone**

---

## 1. Theorem Architecture and Grounding

The mathematical core of ORIUS establishes an unassailable logical chain addressing the four foundational Physical AI safety questions:

```mermaid
flowchart TD
    Q1["1. Knowability: Can we distinguish safe from hazard states?"] --> ThmB["Theorem B: Minimax Information Lower Bound\n(Indistinguishability -> Mandatory Fallback)"]
    Q2["2. Possibility: Under what conditions does a safe action exist?"] --> ThmA["Theorem A: Viability-Preserving Release Feasibility\n(Recursive Forward Invariance of Viable Domain)"]
    Q3["3. Achievability: Can online release guarantee trajectory safety?"] --> ThmC["Theorem C: Trajectory Safety Achievability Bound\n(Pr(Violation) <= sum alpha_t + delta_model + delta_act)"]
    Q4["4. Usefulness: What is the cost of conservatism?"] --> ThmD["Theorem D: Pareto Safety-Utility Regret Frontier\n(Utility Regret <= L_J * r)"]
    
    ThmA --> ThmC
    ThmB --> ThmA
    ThmC --> ThmE["Theorem E: Universal Plant Transferability\n(DomainAdapter Interface Contract)"]
    ThmD --> ThmE
    
    subgraph MachineVerification["Machine Verification in Lean 4 (formal/Orius/TheoremsAE.lean)"]
        LeanThmA["Theorem A Mechanization\n(ViableSafetyDomain.safe_action_release_feasibility)"]
        LeanThmE["Theorem E Mechanization\n(UniversalTransferObligations.transfer_safety)"]
    end
    
    ThmA -.-> LeanThmA
    ThmE -.-> LeanThmE
    
    subgraph ConcretePlants["Heterogeneous Physical Adapters"]
        Plant1["Battery Storage (BESS)\n(Thermal & SoC Limits)"]
        Plant2["Power Grid Inverter\n(IEEE 1547 Droop Stability)"]
        Plant3["ROS 2 Robotics\n(Stopping-Distance Viability Barrier)"]
    end
    
    ThmE --> Plant1
    ThmE --> Plant2
    ThmE --> Plant3
```

---

## 2. Logical Interdependence of Theorems

### Step 1: Theorem B (Minimax Information Lower Bound) $\rightarrow$ Theorem A
- **Logical Function**: Theorem B proves that when observation uncertainty obscures state identity across disjoint viable control sets, **no universal release can guess safely**. Thus, any release mechanism *must* incorporate a certified fallback path.
- **Connection to Theorem A**: Theorem A takes this necessity and proves that whenever the uncertainty set $\mathcal{X}_t$ is non-empty, the safe action set $\mathcal{U}_{\mathrm{safe}}(\mathcal{X}_t)$ contains at least the certified fallback action $u_{\mathrm{fb}}$, guaranteeing feasibility.

### Step 2: Theorem A (Viability Invariance) $\rightarrow$ Theorem C (Trajectory Achievability)
- **Logical Function**: Theorem A establishes one-step recursive invariance ($x_t \in \mathcal{V} \implies x_{t+1} \in \mathcal{V}$).
- **Connection to Theorem C**: By applying a union bound over a finite horizon $T$ and bounding the probability that the true state escapes the calibrated uncertainty set $\mathcal{X}_t(\alpha_t)$ by $\alpha_t$, Theorem C proves cumulative trajectory safety:
  $$\Pr(\exists t \le T: x_t \notin \mathcal{S}_d) \le \sum_{t=1}^T \alpha_t + \delta_{\mathrm{model}} + \delta_{\mathrm{act}}.$$

### Step 3: Theorem C $\rightarrow$ Theorem D (Pareto Regret Envelope)
- **Logical Function**: Theorem D analyzes the price of Theorem C's guarantee. When uncertainty radius $r_t$ inflates due to sensor degradation, the admissible control set tightens, incurring an objective utility penalty bounded by Lipschitz constant $L_J$.

### Step 4: Theorems A--D $\rightarrow$ Theorem E (Universal Transferability)
- **Logical Function**: Proves that the safety guarantee is not plant-specific. Any physical plant $\mathcal{P} = (\mathcal{X}, \mathcal{U}, f, g, \mathcal{S}_d, \mathcal{V}, u_{\mathrm{fb}})$ implementing the three interface obligations (`viability_invariance`, `conformal_coverage`, `fallback_reachability`) inherits the complete safety envelope unconditionally.
