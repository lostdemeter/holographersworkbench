"""
Example: Using the Convergence Analyzer
========================================

Demonstrates automatic convergence analysis and stopping recommendations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from workbench.analysis.convergence import ConvergenceAnalyzer

print("=" * 70)
print("EXAMPLE 1: Fast Zetas Convergence (Real Data)")
print("=" * 70)

# Actual RMSE history from fast_zetas optimization
rmse_history = [0.512, 0.261, 0.145, 0.089, 0.056, 0.034]

analyzer = ConvergenceAnalyzer(
    metric_history=rmse_history,
    metric_name="RMSE",
    lower_is_better=True,
    target_metric=0.01
)

report = analyzer.analyze()
report.print_summary()

print("\n" + "=" * 70)
print("EXAMPLE 2: With Cost Analysis")
print("=" * 70)

# Include iteration costs
costs = [1.0, 1.2, 1.5, 2.0, 2.8, 4.0]

analyzer2 = ConvergenceAnalyzer(
    metric_history=rmse_history,
    metric_name="RMSE",
    lower_is_better=True,
    iteration_costs=costs
)

report2 = analyzer2.analyze()
print(f"\nTotal cost: {report2.total_cost:.2f} seconds")
if report2.cost_benefit_ratios is not None:
    print(f"Latest cost-benefit ratio: {report2.cost_benefit_ratios[-1]:.2f}")

print("\n" + "=" * 70)
print("EXAMPLE 3: Predict Future Improvements")
print("=" * 70)

future_iters, future_rmse = analyzer.predict_future_improvements(n_future=5)
print(f"Current RMSE: {rmse_history[-1]:.6f}")
print(f"Predicted after 5 more layers: {future_rmse[-1]:.6f}")
print(f"Additional improvement: {(rmse_history[-1] - future_rmse[-1]) / rmse_history[-1]:.1%}")

print("\n" + "=" * 70)
print("All examples completed!")
print("=" * 70)
