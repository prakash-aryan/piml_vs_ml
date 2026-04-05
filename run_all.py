#!/usr/bin/env python3
"""
ML vs Physics-Informed ML: Comprehensive Comparison
=====================================================

This project demonstrates the differences between standard machine learning
and physics-informed machine learning (PIML) using REAL data from the
UCI Combined Cycle Power Plant dataset (9568 measurements from a real plant).

Experiments:
1. Full Data Comparison - Shows similar accuracy but better physics consistency
2. Sparse Data Regime - Shows PIML advantage with limited training data
3. Extrapolation - Shows PIML robustness outside training distribution

All data is REAL (not simulated). Physics constraints are derived from
thermodynamic principles governing gas turbine power plants.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data import load_raw_data
from src.plot import plot_summary_dashboard


def main():
    print("=" * 70)
    print("   ML vs PHYSICS-INFORMED ML: COMPREHENSIVE COMPARISON")
    print("   Using Real Data: UCI Combined Cycle Power Plant Dataset")
    print("=" * 70)

    t_start = time.time()

    # Load data once
    print("\n[1/4] Loading real-world dataset...")
    df = load_raw_data()
    print(f"  Dataset: {len(df)} real measurements from a gas turbine power plant")
    print(f"  Features: Ambient Temp, Exhaust Vacuum, Ambient Pressure, Rel. Humidity")
    print(f"  Target: Net electrical energy output (MW)")
    print(f"  Source: UCI Machine Learning Repository")

    # Run experiments
    print("\n[2/4] Running Experiment 1: Full Data Comparison...")
    from experiments.exp1_full_comparison import run as run_exp1
    exp1_results, exp1_physics = run_exp1(df)

    print("\n[3/4] Running Experiment 2: Sparse Data Regime...")
    from experiments.exp2_sparse_data import run as run_exp2
    exp2_results = run_exp2(df)

    print("\n[4/4] Running Experiment 3: Extrapolation Test...")
    from experiments.exp3_extrapolation import run as run_exp3
    exp3_results = run_exp3(df)

    # Summary dashboard
    print("\n--- Generating Summary Dashboard ---")
    plot_summary_dashboard(exp1_results, exp2_results, exp3_results, exp1_physics=exp1_physics)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print(f"Results saved to: {os.path.join(os.path.dirname(__file__), 'results')}/")
    print(f"{'=' * 70}")

    # List generated files
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    print("\nGenerated files:")
    for f in sorted(os.listdir(results_dir)):
        fpath = os.path.join(results_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
