"""
Experiment 3: Extrapolation Beyond Training Distribution
=========================================================
Train models only on data where ambient temperature (AT) is below a cutoff,
then test on the full range including higher temperatures never seen in training.

The Hybrid model's physics backbone (linear with guaranteed negative AT/V
coefficients) extrapolates correctly because it encodes the right monotonic
trend. The standard NN has no such guidance and can produce arbitrary
predictions in unseen temperature ranges.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

from src.data import load_raw_data, prepare_extrapolation_data
from src.models import StandardNN, PhysicsInformedNN, PhysicsHybridNN
from src.train import train_standard, train_physics_informed, train_hybrid
from src.evaluate import evaluate_model
from src.plot import plot_extrapolation


def run(df=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Extrapolation Beyond Training Distribution")
    print("=" * 70)

    if df is None:
        df = load_raw_data()

    data = prepare_extrapolation_data(df, temp_cutoff_percentile=60)
    print(f"\nTemperature cutoff: {data['temp_cutoff']:.1f}°C (60th percentile)")
    print(f"Training samples: {len(data['X_train'])} (AT ≤ {data['temp_cutoff']:.1f}°C)")
    print(f"Interpolation test: {len(data['X_interp_test'])} samples")
    print(f"Extrapolation test: {len(data['X_extrap_test'])} samples "
          f"(AT > {data['temp_cutoff']:.1f}°C)")

    results = {}

    # --- Standard NN ---
    print("\n--- Training Standard NN ---")
    torch.manual_seed(42)
    np.random.seed(42)
    model_std = StandardNN(input_dim=4, hidden_dims=(128, 64, 32))
    train_standard(
        model_std, data["X_train"], data["y_train"],
        data["X_extrap_test"], data["y_extrap_test"],
        epochs=400, lr=1e-3
    )

    # --- PINN ---
    print("\n--- Training Physics-Informed NN ---")
    torch.manual_seed(42)
    np.random.seed(42)
    model_pinn = PhysicsInformedNN(input_dim=4, hidden_dims=(128, 64, 32))
    train_physics_informed(
        model_pinn, data["X_train"], data["y_train"],
        data["X_extrap_test"], data["y_extrap_test"],
        physics_weight=0.5, epochs=400, lr=1e-3
    )

    # --- Hybrid ---
    print("\n--- Training Hybrid PIML ---")
    torch.manual_seed(42)
    np.random.seed(42)
    model_hybrid = PhysicsHybridNN(input_dim=4, hidden_dims=(32, 32))
    train_hybrid(
        model_hybrid, data["X_train"], data["y_train"],
        data["X_extrap_test"], data["y_extrap_test"],
        physics_weight=0.5, epochs=400, lr=1e-3
    )

    # --- Evaluate ---
    print("\n--- Evaluation ---")
    models = {
        "Standard NN": model_std,
        "PINN": model_pinn,
        "Hybrid PIML": model_hybrid,
    }

    for name, model in models.items():
        interp_res = evaluate_model(
            model, data["X_interp_test"], data["y_interp_test"], data["scaler_y"]
        )
        extrap_res = evaluate_model(
            model, data["X_extrap_test"], data["y_extrap_test"], data["scaler_y"]
        )

        all_X = torch.cat([data["X_interp_test"], data["X_extrap_test"]], dim=0)
        all_y = torch.cat([data["y_interp_test"], data["y_extrap_test"]], dim=0)
        all_res = evaluate_model(model, all_X, all_y, data["scaler_y"])

        all_temps = np.concatenate([data["X_interp_raw"][:, 0], data["X_extrap_raw"][:, 0]])

        results[name] = {
            "interp_RMSE": interp_res["RMSE"],
            "interp_R2": interp_res["R2"],
            "extrap_RMSE": extrap_res["RMSE"],
            "extrap_R2": extrap_res["R2"],
            "y_pred": all_res["y_pred"],
            "y_true": all_res["y_true"],
            "temperatures": all_temps,
        }

        degradation = extrap_res["RMSE"] / max(interp_res["RMSE"], 1e-6)
        print(f"\n  {name}:")
        print(f"    Interpolation  (AT ≤ {data['temp_cutoff']:.1f}°C): "
              f"RMSE={interp_res['RMSE']:.3f} MW, R²={interp_res['R2']:.4f}")
        print(f"    Extrapolation  (AT > {data['temp_cutoff']:.1f}°C): "
              f"RMSE={extrap_res['RMSE']:.3f} MW, R²={extrap_res['R2']:.4f}")
        print(f"    Degradation ratio: {degradation:.2f}x")

    print("\n--- Generating Plots ---")
    plot_extrapolation(
        results, data["temp_cutoff"], data["all_temps"],
        filename="exp3_extrapolation.png"
    )

    return results


if __name__ == "__main__":
    run()
