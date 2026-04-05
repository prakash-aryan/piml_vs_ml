"""
Experiment 2: Sparse + Noisy Data Regime
==========================================
Compare ML vs PIML when training data is scarce AND noisy (realistic sensors).

This is where PIML truly shines: physical constraints act as regularizers
AND denoisers — they prevent the model from fitting noise and maintain
physically plausible predictions even with very few noisy training samples.

Noise: 5 MW Gaussian noise on training targets (~1% of output range).
This is realistic for industrial power plant sensor measurements.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

from src.data import load_raw_data, prepare_sparse_data
from src.models import StandardNN, PhysicsInformedNN, PhysicsHybridNN
from src.train import train_standard, train_physics_informed, train_hybrid
from src.evaluate import evaluate_model
from src.plot import plot_data_efficiency


FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = [42, 123, 456]
NOISE_STD = 10.0  # 10 MW noise (~2% of output, realistic for industrial sensors)


def run(df=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Sparse + Noisy Data Regime")
    print(f"  Sensor noise: {NOISE_STD} MW std (~2% of output, realistic for industrial sensors)")
    print("=" * 70)

    if df is None:
        df = load_raw_data()

    all_results = {
        "Standard NN": {},
        "PINN": {},
        "Hybrid PIML": {},
    }

    for frac in FRACTIONS:
        epochs = min(500, max(300, int(200 / max(frac, 0.01))))
        seed_results = {"Standard NN": [], "PINN": [], "Hybrid PIML": []}

        for seed in SEEDS:
            sparse_data = prepare_sparse_data(
                df, [frac], random_state=seed, noise_std=NOISE_STD
            )
            d = sparse_data[frac]
            n = d["n_train"]

            # Standard NN
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_std = StandardNN(input_dim=4, hidden_dims=(64, 64, 32))
            train_standard(
                model_std, d["X_train"], d["y_train"],
                d["X_test"], d["y_test"],
                epochs=epochs, lr=1e-3, verbose=False
            )
            seed_results["Standard NN"].append(
                evaluate_model(model_std, d["X_test"], d["y_test"], d["scaler_y"])
            )

            # PINN — same architecture + physics
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_pinn = PhysicsInformedNN(input_dim=4, hidden_dims=(64, 64, 32))
            train_physics_informed(
                model_pinn, d["X_train"], d["y_train"],
                d["X_test"], d["y_test"],
                physics_weight=0.5, epochs=epochs, lr=1e-3, verbose=False
            )
            seed_results["PINN"].append(
                evaluate_model(model_pinn, d["X_test"], d["y_test"], d["scaler_y"])
            )

            # Hybrid PIML
            torch.manual_seed(seed)
            np.random.seed(seed)
            model_hybrid = PhysicsHybridNN(input_dim=4, hidden_dims=(32, 32))
            train_hybrid(
                model_hybrid, d["X_train"], d["y_train"],
                d["X_test"], d["y_test"],
                physics_weight=0.5, epochs=epochs, lr=1e-3, verbose=False
            )
            seed_results["Hybrid PIML"].append(
                evaluate_model(model_hybrid, d["X_test"], d["y_test"], d["scaler_y"])
            )

        # Average over seeds
        for name in all_results:
            avg = {}
            for metric in ["RMSE", "MAE", "R2"]:
                vals = [r[metric] for r in seed_results[name]]
                avg[metric] = np.mean(vals)
                avg[f"{metric}_std"] = np.std(vals)
            avg["n_train"] = sparse_data[frac]["n_train"]
            all_results[name][frac] = avg

        n = all_results["Standard NN"][frac]["n_train"]
        print(f"\n  {frac:.1%} ({n} samples):")
        for name in all_results:
            r = all_results[name][frac]
            print(f"    {name:<15s}: RMSE={r['RMSE']:.3f} +/- {r['RMSE_std']:.3f}  "
                  f"R²={r['R2']:.4f}")

    # --- Summary ---
    print("\n\n--- SUMMARY: Mean RMSE (with noise) ---")
    print(f"{'Frac':<8} {'N':<6} {'Standard NN':<14} {'PINN':<14} {'Hybrid PIML':<14} {'PIML Gain':<10}")
    print("-" * 70)
    for frac in FRACTIONS:
        n = all_results["Standard NN"][frac]["n_train"]
        r1 = all_results["Standard NN"][frac]["RMSE"]
        r2 = all_results["PINN"][frac]["RMSE"]
        r3 = all_results["Hybrid PIML"][frac]["RMSE"]
        best_piml = min(r2, r3)
        gain = (r1 - best_piml) / r1 * 100
        print(f"{frac:<8.3f} {n:<6d} {r1:<14.3f} {r2:<14.3f} {r3:<14.3f} {gain:>+8.1f}%")

    print("\n--- Generating Plots ---")
    plot_data_efficiency(all_results, filename="exp2_data_efficiency.png")

    return all_results


if __name__ == "__main__":
    run()
