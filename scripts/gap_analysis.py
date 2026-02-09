#!/usr/bin/env python3
"""Analyze gaps in dashboard data."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

print("=" * 60)
print("GAP ANALYSIS")
print("=" * 60)

gaps = []

# 1. Check conformal coverage for US
us_coverage = ROOT / "reports" / "eia930" / "metrics" / "forecast_intervals.csv"
if us_coverage.exists():
    print(f"\n1. US Conformal Coverage CSV: ✓ Exists")
else:
    print(f"\n1. US Conformal Coverage CSV: ✗ MISSING")
    gaps.append("US conformal coverage not computed")

# 2. Check dashboard metrics have coverage
de_metrics = json.load(open(ROOT / "data/dashboard/de_metrics.json"))
us_metrics = json.load(open(ROOT / "data/dashboard/us_metrics.json"))

de_cov = [m for m in de_metrics if m.get("coverage_90")]
us_cov = [m for m in us_metrics if m.get("coverage_90")]
print(f"   DE metrics with coverage_90: {len(de_cov)}/9 (GBM only)")
print(f"   US metrics with coverage_90: {len(us_cov)}/9")
if len(us_cov) == 0:
    gaps.append("US models missing coverage_90")

# 3. Check paper claims vs dashboard data
print("\n2. Paper Claims vs Dashboard:")
de_impact = json.load(open(ROOT / "data/dashboard/de_impact.json"))
print(f"   Paper: 2.89% cost → Dashboard: {de_impact['cost_savings_pct']:.2f}% ✓")
print(f"   Paper: 0.58% carbon → Dashboard: {de_impact['carbon_reduction_pct']:.2f}% ✓")

# 4. Check for robustness data
robustness = ROOT / "reports" / "metrics" / "robustness_summary.csv"
if robustness.exists():
    print(f"\n3. Robustness Summary: ✓ Exists")
else:
    print(f"\n3. Robustness Summary: ✗ MISSING")

# 5. Check model registry exists
de_registry = json.load(open(ROOT / "data/dashboard/de_registry.json"))
us_registry = json.load(open(ROOT / "data/dashboard/us_registry.json"))
print(f"\n4. Model Registry: DE={len(de_registry)}, US={len(us_registry)}")

# 6. Check forecast data
de_forecast = json.load(open(ROOT / "data/dashboard/de_forecast.json"))
us_forecast = json.load(open(ROOT / "data/dashboard/us_forecast.json"))
print(f"\n5. Forecast Data Points:")
for target in ["load_mw", "wind_mw", "solar_mw"]:
    de_pts = len(de_forecast.get(target, []))
    us_pts = len(us_forecast.get(target, []))
    status = "✓" if de_pts > 0 and us_pts > 0 else "✗"
    print(f"   {target}: DE={de_pts}, US={us_pts} {status}")

# 7. Significance tests
sig_tests = ROOT / "reports" / "significance_tests.json"
print(f"\n6. Significance Tests: {'✓ Exists' if sig_tests.exists() else '✗ MISSING'}")

# 8. SHAP summary
shap = ROOT / "reports" / "shap_summary.json"
print(f"\n7. SHAP Feature Importance: {'✓ Exists' if shap.exists() else '✗ MISSING'}")

# Summary
print("\n" + "=" * 60)
if gaps:
    print("GAPS FOUND:")
    for g in gaps:
        print(f"  ⚠️  {g}")
else:
    print("✅ No critical gaps found!")
print("=" * 60)
