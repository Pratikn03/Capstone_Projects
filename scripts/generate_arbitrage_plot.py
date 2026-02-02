"""
Standalone script to generate the Arbitrage Optimization plot for the final report.
Run this to produce 'reports/figures/arbitrage_optimization_demo.png'.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def main():
    # Key: CLI/reporting helper
    # --- 1. MOCK DATA (As requested) ---
    # Create a 24-hour timeline
    hours = np.arange(24)

    # Real-world style data
    price_curve = np.array([30, 25, 20, 15, 15, 20, 40, 60, 80, 70, 60, 50, 45, 40, 45, 55, 90, 120, 110, 80, 60, 50, 40, 35])
    grid_load = np.array([10, 10, 10, 10, 12, 15, 20, 25, 28, 30, 32, 35, 35, 34, 33, 35, 40, 45, 42, 38, 30, 25, 20, 15])

    # GridPulse Logic (The "Level-4" behavior)
    # Charge (negative) when price is low, Discharge (positive) when price is high
    battery_flow = np.array([-5, -5, -5, -5, -5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0])

    # Calculate the "New" Grid Load (Load - Battery)
    # When charging (flow < 0), we draw MORE from grid -> Load - (-5) = Load + 5
    # When discharging (flow > 0), we draw LESS from grid -> Load - (5) = Load - 5
    optimized_load = grid_load - battery_flow

    # --- 2. THE WINNING PLOT CODE ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Market Price (The "Signal")
    color = 'tab:red'
    ax1.set_xlabel('Hour of Day (0-23)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Electricity Price ($/MWh)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(hours, price_curve, color=color, linestyle='--', linewidth=2, label='Market Price', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Create a second y-axis for Power (The "Action")
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Power (MW)', color=color, fontsize=12, fontweight='bold')

    # Plot 1: Baseline Load (The "Before")
    ax2.plot(hours, grid_load, color='gray', alpha=0.4, linewidth=2, label='Baseline Grid Load')

    # Plot 2: Optimized Load (The "After")
    ax2.plot(hours, optimized_load, color=color, linewidth=3, label='GridPulse Optimized Load')

    # Highlight the "Arbitrage" (The Level-4 Magic)
    # Green area = Charging (Money saved later)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(battery_flow < 0), color='green', alpha=0.3, label='Charging (Low Price)')
    # Orange area = Discharging (Cost Avoided)
    ax2.fill_between(hours, grid_load, optimized_load, 
                     where=(battery_flow > 0), color='orange', alpha=0.5, label='Discharging (High Price)')

    # --- 3. POLISH ---
    plt.title('GridPulse Decision Logic: Arbitrage Optimization', fontsize=16, fontweight='bold', pad=20)
    
    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', frameon=True, shadow=True)

    plt.tight_layout()
    
    # Save
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "arbitrage_optimization_demo.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Graph saved to: {out_path}")

if __name__ == "__main__":
    main()
