#!/usr/bin/env python3
"""Validate dashboard data against actual training reports."""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Dashboard data
de_dash_metrics = json.load(open(ROOT / 'data/dashboard/de_metrics.json'))
us_dash_metrics = json.load(open(ROOT / 'data/dashboard/us_metrics.json'))
de_dash_impact = json.load(open(ROOT / 'data/dashboard/de_impact.json'))
us_dash_impact = json.load(open(ROOT / 'data/dashboard/us_impact.json'))

# Reports data
de_report_metrics = json.load(open(ROOT / 'reports/week2_metrics.json'))
us_report_metrics = json.load(open(ROOT / 'reports/eia930/week2_metrics.json'))

print("=" * 60)
print("GERMANY (DE) METRICS VALIDATION")
print("=" * 60)

issues = []

for target in ['load_mw', 'wind_mw', 'solar_mw']:
    print(f"\n{target.upper()}:")
    for model in ['gbm', 'lstm', 'tcn']:
        # Dashboard
        dash_m = [m for m in de_dash_metrics if m['target'] == target and model.upper() in m['model'].upper()]
        dash_rmse = dash_m[0]['rmse'] if dash_m else None
        
        # Report
        report_m = de_report_metrics['targets'].get(target, {}).get(model, {})
        report_rmse = round(report_m.get('rmse', 0), 2) if report_m else None
        
        if dash_rmse is not None and report_rmse is not None:
            diff = abs(float(dash_rmse) - float(report_rmse))
            match = "✓" if diff < 1 else "✗"
            if diff >= 1:
                issues.append(f"DE {target} {model}: Dashboard={dash_rmse}, Report={report_rmse}")
        else:
            match = "?"
        print(f"  {model.upper():6s}: Dashboard={str(dash_rmse):>10}, Report={str(report_rmse):>10} {match}")

print("\n" + "=" * 60)
print("USA (US) METRICS VALIDATION")
print("=" * 60)

for target in ['load_mw', 'wind_mw', 'solar_mw']:
    print(f"\n{target.upper()}:")
    for model in ['gbm', 'lstm', 'tcn']:
        # Dashboard
        dash_m = [m for m in us_dash_metrics if m['target'] == target and model.upper() in m['model'].upper()]
        dash_rmse = dash_m[0]['rmse'] if dash_m else None
        
        # Report
        report_m = us_report_metrics['targets'].get(target, {}).get(model, {})
        report_rmse = round(report_m.get('rmse', 0), 2) if report_m else None
        
        if dash_rmse is not None and report_rmse is not None:
            diff = abs(float(dash_rmse) - float(report_rmse))
            match = "✓" if diff < 1 else "✗"
            if diff >= 1:
                issues.append(f"US {target} {model}: Dashboard={dash_rmse}, Report={report_rmse}")
        else:
            match = "?"
        print(f"  {model.upper():6s}: Dashboard={str(dash_rmse):>10}, Report={str(report_rmse):>10} {match}")

print("\n" + "=" * 60)
print("IMPACT VALIDATION")
print("=" * 60)

print("\nGERMANY:")
print(f"  Cost Savings: Dashboard={de_dash_impact['cost_savings_pct']:.3f}%")
print(f"  Carbon Red:   Dashboard={de_dash_impact['carbon_reduction_pct']:.3f}%")
print(f"  Peak Shave:   Dashboard={de_dash_impact['peak_shaving_pct']:.3f}%")

print("\nUSA:")
print(f"  Cost Savings: Dashboard={us_dash_impact['cost_savings_pct']:.3f}%")
print(f"  Carbon Red:   Dashboard={us_dash_impact['carbon_reduction_pct']:.3f}%")
print(f"  Peak Shave:   Dashboard={us_dash_impact['peak_shaving_pct']:.3f}%")

if issues:
    print("\n" + "=" * 60)
    print("ISSUES FOUND:")
    print("=" * 60)
    for issue in issues:
        print(f"  ⚠️  {issue}")
else:
    print("\n✅ All metrics match!")
