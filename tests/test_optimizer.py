"""
Tests for Battery Dispatch Optimization Module.

This test suite validates the optimizer's ability to generate feasible and
physically realistic battery dispatch schedules. Key validation areas:

1. **Output Structure**: Correct array shapes and keys
2. **Physical Feasibility**: No negative power flows, SoC within bounds
3. **Energy Conservation**: Power balance equations satisfied
4. **Constraint Satisfaction**: Respects battery capacity and power limits

Test Strategy:
    We use small, interpretable scenarios (3-hour horizons) where the
    optimal solution can be reasoned about manually. This makes debugging
    easier when tests fail.

Running Tests:
    pytest tests/test_optimizer.py -v
    pytest tests/test_optimizer.py -k "shapes"  # Run only shape tests

See Also:
    - test_optimizer_constraints.py: More detailed constraint tests
    - src/gridpulse/optimizer/lp_dispatch.py: The module being tested
"""
import numpy as np

from gridpulse.optimizer import optimize_dispatch


def test_optimize_dispatch_shapes():
    """
    Test that optimizer returns arrays of correct shape.
    
    This is a basic sanity check that the optimizer produces outputs
    matching the input horizon length. Each output should have exactly
    3 elements for a 3-hour scenario.
    """
    # Arrange: Create a tiny 3-hour scenario for easy validation
    load = [10.0, 12.0, 9.0]  # MW of demand each hour
    ren = [3.0, 4.0, 2.0]     # MW of renewable generation
    
    # Battery configuration: 5 MWh capacity, 2 MW max power
    cfg = {
        "battery": {
            "capacity_mwh": 5.0,
            "max_power_mw": 2.0,
            "efficiency": 0.95,
            "min_soc_mwh": 0.5,
            "initial_soc_mwh": 2.5,
        },
        "grid": {
            "max_import_mw": 50.0, 
            "price_per_mwh": 50.0, 
            "carbon_kg_per_mwh": 0.0
        },
        "penalties": {
            "curtailment_per_mw": 500.0,    # High penalty for wasting renewables
            "unmet_load_per_mw": 10000.0,   # Very high penalty for blackouts
        },
        "objective": {
            "cost_weight": 1.0, 
            "carbon_weight": 0.0
        },
    }

    # Act: Run the optimizer
    out = optimize_dispatch(load, ren, cfg)
    
    # Assert: Each output array has correct length (matches horizon)
    assert len(out["grid_mw"]) == 3, "Grid import should match horizon length"
    assert len(out["battery_charge_mw"]) == 3, "Charge should match horizon"
    assert len(out["battery_discharge_mw"]) == 3, "Discharge should match horizon"
    assert len(out["soc_mwh"]) == 3, "State of charge should match horizon"

    # Assert: Physical feasibility - no negative power flows
    assert np.all(np.asarray(out["grid_mw"]) >= 0.0), "Grid import cannot be negative"
    assert np.all(np.asarray(out["battery_charge_mw"]) >= 0.0), "Charge cannot be negative"
    assert np.all(np.asarray(out["battery_discharge_mw"]) >= 0.0), "Discharge cannot be negative"
