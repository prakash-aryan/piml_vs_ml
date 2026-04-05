"""
Evaluation metrics and analysis utilities.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(model, X, y, scaler_y):
    """
    Evaluate a model and return metrics in original scale.

    Returns dict with MSE, RMSE, MAE, R², and predictions.
    """
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y.numpy())

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "y_pred": y_pred.flatten(),
        "y_true": y_true.flatten(),
    }


def check_monotonicity(model, X, feature_idx=0, n_samples=500):
    """
    Check if model predictions are monotonically decreasing w.r.t. a feature.

    Generates a sweep of the target feature while holding others at mean.
    Returns the fraction of points where monotonicity is violated.
    """
    model.eval()
    x_base = X.mean(dim=0, keepdim=True).repeat(n_samples, 1)
    sweep = torch.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), n_samples)
    x_base[:, feature_idx] = sweep

    with torch.no_grad():
        preds = model(x_base).numpy().flatten()

    diffs = np.diff(preds)
    violations = np.sum(diffs > 0)  # Should be decreasing
    violation_fraction = violations / len(diffs)

    return {
        "sweep_values": sweep.numpy(),
        "predictions": preds,
        "violation_fraction": violation_fraction,
        "n_violations": violations,
    }


def compute_physics_consistency(model, X, scaler_X, scaler_y):
    """
    Evaluate physical consistency of predictions:
    1. Monotonicity w.r.t. temperature (AT)
    2. Monotonicity w.r.t. vacuum (V)
    3. Output range validity
    """
    results = {}

    # 1. Temperature monotonicity
    at_mono = check_monotonicity(model, X, feature_idx=0)
    results["AT_monotonicity_violations"] = at_mono["violation_fraction"]
    results["AT_sweep"] = at_mono

    # 2. Vacuum monotonicity
    v_mono = check_monotonicity(model, X, feature_idx=1)
    results["V_monotonicity_violations"] = v_mono["violation_fraction"]
    results["V_sweep"] = v_mono

    # 3. Output range check
    model.eval()
    with torch.no_grad():
        preds_scaled = model(X).numpy()
    preds = scaler_y.inverse_transform(preds_scaled).flatten()

    # Physical bounds for a CCPP: roughly 420-500 MW
    below_min = np.sum(preds < 400)
    above_max = np.sum(preds > 510)
    results["out_of_range_fraction"] = (below_min + above_max) / len(preds)
    results["pred_min"] = preds.min()
    results["pred_max"] = preds.max()

    return results


def print_comparison(results_dict):
    """Print a formatted comparison table of model results."""
    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'RMSE (MW)':>10} {'MAE (MW)':>10} {'R²':>10}")
    print("-" * 70)
    for name, res in results_dict.items():
        print(f"{name:<20} {res['RMSE']:>10.4f} {res['MAE']:>10.4f} {res['R2']:>10.6f}")
    print("=" * 70)
