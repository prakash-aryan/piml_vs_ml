"""
Experiment 1: Full Data Comparison
===================================
Compare Standard NN, Physics-Informed NN, and Hybrid PIML on the complete
UCI Combined Cycle Power Plant dataset.

This experiment shows that with abundant data, all models perform similarly,
but PIML models additionally maintain physical consistency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np

from src.data import load_raw_data, prepare_data
from src.models import StandardNN, PhysicsInformedNN, PhysicsHybridNN
from src.train import train_standard, train_physics_informed, train_hybrid
from src.evaluate import evaluate_model, compute_physics_consistency, print_comparison
from src.plot import plot_training_curves, plot_predictions_scatter, plot_residuals, plot_monotonicity_check


def run(df=None):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Full Data Comparison (ML vs PIML)")
    print("=" * 70)

    if df is None:
        df = load_raw_data()

    data = prepare_data(df, test_size=0.2)
    print(f"\nTraining samples: {len(data['X_train'])}")
    print(f"Test samples:     {len(data['X_test'])}")

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Model 1: Standard Neural Network ---
    print("\n--- Training Standard NN ---")
    model_std = StandardNN(input_dim=4, hidden_dims=(128, 64, 32))
    hist_std = train_standard(
        model_std, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        epochs=300, lr=1e-3
    )

    # --- Model 2: Physics-Informed Neural Network ---
    print("\n--- Training Physics-Informed NN (PINN) ---")
    torch.manual_seed(42)
    model_pinn = PhysicsInformedNN(input_dim=4, hidden_dims=(128, 64, 32))
    hist_pinn = train_physics_informed(
        model_pinn, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        physics_weight=0.5, epochs=300, lr=1e-3
    )

    # --- Model 3: Hybrid Physics + ML ---
    print("\n--- Training Hybrid PIML ---")
    torch.manual_seed(42)
    model_hybrid = PhysicsHybridNN(input_dim=4, hidden_dims=(32, 32))
    hist_hybrid = train_hybrid(
        model_hybrid, data["X_train"], data["y_train"],
        data["X_test"], data["y_test"],
        physics_weight=0.5, epochs=300, lr=1e-3
    )

    # --- Evaluate all models ---
    print("\n--- Evaluation on Test Set ---")
    results = {
        "Standard NN": evaluate_model(model_std, data["X_test"], data["y_test"], data["scaler_y"]),
        "PINN": evaluate_model(model_pinn, data["X_test"], data["y_test"], data["scaler_y"]),
        "Hybrid PIML": evaluate_model(model_hybrid, data["X_test"], data["y_test"], data["scaler_y"]),
    }
    print_comparison(results)

    # --- Physics consistency check ---
    print("\n--- Physics Consistency Check ---")
    physics_results = {
        "Standard NN": compute_physics_consistency(
            model_std, data["X_test"], data["scaler_X"], data["scaler_y"]
        ),
        "PINN": compute_physics_consistency(
            model_pinn, data["X_test"], data["scaler_X"], data["scaler_y"]
        ),
        "Hybrid PIML": compute_physics_consistency(
            model_hybrid, data["X_test"], data["scaler_X"], data["scaler_y"]
        ),
    }

    for name, pr in physics_results.items():
        print(f"\n  {name}:")
        print(f"    AT monotonicity violations: {pr['AT_monotonicity_violations']:.1%}")
        print(f"    V  monotonicity violations: {pr['V_monotonicity_violations']:.1%}")
        print(f"    Output range: [{pr['pred_min']:.1f}, {pr['pred_max']:.1f}] MW")

    # --- Generate plots ---
    print("\n--- Generating Plots ---")
    histories = {"Standard NN": hist_std, "PINN": hist_pinn, "Hybrid PIML": hist_hybrid}
    plot_training_curves(histories, title="Exp 1: Training Curves (Full Data)",
                         filename="exp1_training_curves.png")
    plot_predictions_scatter(results, title="Exp 1: Predictions vs Actual (Full Data)",
                             filename="exp1_predictions.png")
    plot_residuals(results, title="Exp 1: Residual Distributions",
                   filename="exp1_residuals.png")
    plot_monotonicity_check(physics_results, title="Exp 1: Monotonicity Compliance",
                            filename="exp1_monotonicity.png")

    return results, physics_results


if __name__ == "__main__":
    run()
