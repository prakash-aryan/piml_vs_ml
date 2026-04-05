"""
Data loading utilities for real-world datasets.

Uses the UCI Combined Cycle Power Plant dataset:
- 9568 real measurements collected over 6 years (2006-2011)
- Features: Ambient Temperature (AT), Exhaust Vacuum (V),
            Ambient Pressure (AP), Relative Humidity (RH)
- Target: Net hourly electrical energy output (PE) in MW

Source: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant
"""

import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_URL = "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+plant.zip"


def download_dataset():
    """Download the Combined Cycle Power Plant dataset from UCI."""
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "ccpp.zip")
    csv_path = os.path.join(DATA_DIR, "ccpp.csv")

    if os.path.exists(csv_path):
        return csv_path

    print("Downloading Combined Cycle Power Plant dataset from UCI...")
    urllib.request.urlretrieve(DATASET_URL, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find the xlsx or csv file inside
        names = zf.namelist()
        print(f"  Archive contents: {names}")

        # Extract all
        zf.extractall(DATA_DIR)

    # Try to find and convert the data file
    # The dataset comes as an xlsx file inside a nested zip
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            if f.endswith(".xlsx"):
                print(f"  Found Excel file: {fpath}")
                df = pd.read_excel(fpath)
                df.to_csv(csv_path, index=False)
                print(f"  Converted to CSV: {csv_path}")
                return csv_path
            elif f.endswith(".zip") and f != "ccpp.zip":
                # Nested zip
                with zipfile.ZipFile(fpath, "r") as inner_zf:
                    inner_zf.extractall(DATA_DIR)
                # Recurse to find xlsx
                for root2, dirs2, files2 in os.walk(DATA_DIR):
                    for f2 in files2:
                        if f2.endswith(".xlsx"):
                            fpath2 = os.path.join(root2, f2)
                            print(f"  Found Excel file: {fpath2}")
                            df = pd.read_excel(fpath2)
                            df.to_csv(csv_path, index=False)
                            print(f"  Converted to CSV: {csv_path}")
                            return csv_path
                        elif f2.endswith(".csv"):
                            fpath2 = os.path.join(root2, f2)
                            df = pd.read_csv(fpath2)
                            df.to_csv(csv_path, index=False)
                            return csv_path

    raise FileNotFoundError("Could not find data file in the downloaded archive.")


def load_raw_data():
    """Load the raw dataset as a pandas DataFrame."""
    csv_path = download_dataset()
    df = pd.read_csv(csv_path)
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample:\n{df.head()}")
    return df


def prepare_data(df, test_size=0.2, random_state=42):
    """
    Prepare data for training: split, scale, return tensors.

    Returns dict with train/test tensors, scalers, and raw arrays.
    """
    feature_cols = ["AT", "V", "AP", "RH"]
    target_col = "PE"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    X_train_s = scaler_X.transform(X_train).astype(np.float32)
    X_test_s = scaler_X.transform(X_test).astype(np.float32)
    y_train_s = scaler_y.transform(y_train).astype(np.float32)
    y_test_s = scaler_y.transform(y_test).astype(np.float32)

    return {
        "X_train": torch.tensor(X_train_s),
        "X_test": torch.tensor(X_test_s),
        "y_train": torch.tensor(y_train_s),
        "y_test": torch.tensor(y_test_s),
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train_raw": y_train,
        "y_test_raw": y_test,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "feature_names": feature_cols,
        "target_name": target_col,
    }


def prepare_sparse_data(df, fractions, random_state=42, noise_std=0.0):
    """
    Prepare multiple train sets with varying sizes (for data efficiency experiment).

    Args:
        noise_std: Standard deviation of Gaussian noise added to training targets
                   (in original scale, MW). Simulates real sensor measurement noise.

    Returns a dict mapping fraction -> data dict, all sharing the same test set.
    """
    feature_cols = ["AT", "V", "AP", "RH"]
    target_col = "PE"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # Fixed test set (20%)
    X_trainall, X_test, y_trainall, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Scaler fit on full training data
    scaler_X = StandardScaler().fit(X_trainall)
    scaler_y = StandardScaler().fit(y_trainall)

    X_test_s = scaler_X.transform(X_test).astype(np.float32)
    y_test_s = scaler_y.transform(y_test).astype(np.float32)

    results = {}
    for frac in fractions:
        n = max(10, int(len(X_trainall) * frac))
        idx = np.random.RandomState(random_state).choice(len(X_trainall), n, replace=False)
        X_tr = X_trainall[idx]
        y_tr = y_trainall[idx].copy()

        # Add measurement noise to training targets (test stays clean)
        if noise_std > 0:
            rng = np.random.RandomState(random_state + 1)
            y_tr = y_tr + rng.randn(*y_tr.shape).astype(np.float32) * noise_std

        X_tr_s = scaler_X.transform(X_tr).astype(np.float32)
        y_tr_s = scaler_y.transform(y_tr).astype(np.float32)

        results[frac] = {
            "X_train": torch.tensor(X_tr_s),
            "y_train": torch.tensor(y_tr_s),
            "X_test": torch.tensor(X_test_s),
            "y_test": torch.tensor(y_test_s),
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "n_train": n,
            "X_test_raw": X_test,
            "y_test_raw": y_test,
            "X_train_raw": X_tr,
            "y_train_raw": y_tr,
        }

    return results


def prepare_extrapolation_data(df, temp_cutoff_percentile=60, random_state=42):
    """
    Split data for extrapolation experiment:
    - Train on samples where AT (ambient temperature) is below the cutoff
    - Test on ALL data (interpolation + extrapolation regions)

    This tests whether the model can predict energy output at temperatures
    it has never seen during training.
    """
    feature_cols = ["AT", "V", "AP", "RH"]
    target_col = "PE"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # AT is the first feature column
    at_values = X[:, 0]
    cutoff = np.percentile(at_values, temp_cutoff_percentile)

    train_mask = at_values <= cutoff
    extrap_mask = at_values > cutoff

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_interp_test = X[train_mask]  # in-distribution test
    y_interp_test = y[train_mask]
    X_extrap_test = X[extrap_mask]  # out-of-distribution test
    y_extrap_test = y[extrap_mask]

    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)

    # Subsample training data for a fair experiment
    rng = np.random.RandomState(random_state)
    n_train = min(2000, len(X_train))
    idx = rng.choice(len(X_train), n_train, replace=False)
    X_train_sub = X_train[idx]
    y_train_sub = y_train[idx]

    return {
        "X_train": torch.tensor(scaler_X.transform(X_train_sub).astype(np.float32)),
        "y_train": torch.tensor(scaler_y.transform(y_train_sub).astype(np.float32)),
        "X_interp_test": torch.tensor(scaler_X.transform(X_interp_test).astype(np.float32)),
        "y_interp_test": torch.tensor(scaler_y.transform(y_interp_test).astype(np.float32)),
        "X_extrap_test": torch.tensor(scaler_X.transform(X_extrap_test).astype(np.float32)),
        "y_extrap_test": torch.tensor(scaler_y.transform(y_extrap_test).astype(np.float32)),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "temp_cutoff": cutoff,
        "X_train_raw": X_train_sub,
        "y_train_raw": y_train_sub,
        "X_interp_raw": X_interp_test,
        "y_interp_raw": y_interp_test,
        "X_extrap_raw": X_extrap_test,
        "y_extrap_raw": y_extrap_test,
        "all_temps": at_values,
    }


def make_dataloader(X, y, batch_size=256, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
