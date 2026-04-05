"""
Training loops for standard ML and physics-informed ML models.
"""

import time
import torch
import torch.nn as nn
import numpy as np


def train_standard(model, X_train, y_train, X_val, y_val,
                   epochs=300, lr=1e-3, batch_size=256, verbose=True):
    """Train a standard neural network with MSE loss only."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    n = len(X_train)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        # Mini-batch training
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)
        dt = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["epoch_time"].append(dt)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [Standard] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.1e}")

    return history


def train_physics_informed(model, X_train, y_train, X_val, y_val,
                           physics_weight=0.1, epochs=300, lr=1e-3,
                           batch_size=256, verbose=True):
    """
    Train a physics-informed neural network.

    Loss = MSE(data) + physics_weight * sum(physics_losses)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    history = {
        "train_loss": [], "val_loss": [], "data_loss": [],
        "physics_loss": [], "epoch_time": [],
    }
    n = len(X_train)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        perm = torch.randperm(n)
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx].clone().requires_grad_(True)
            yb = y_train[idx]

            pred = model(xb)
            data_loss = criterion(pred, yb)

            # Physics losses
            phys_losses = model.physics_loss(xb, pred)
            phys_total = sum(phys_losses.values())

            total_loss = data_loss + physics_weight * phys_total

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_total.item()
            n_batches += 1

        avg_data = epoch_data_loss / n_batches
        avg_phys = epoch_phys_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)
        dt = time.time() - t0

        history["train_loss"].append(avg_data + physics_weight * avg_phys)
        history["val_loss"].append(val_loss)
        history["data_loss"].append(avg_data)
        history["physics_loss"].append(avg_phys)
        history["epoch_time"].append(dt)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [PIML]     Epoch {epoch+1:3d}/{epochs} | "
                  f"Data: {avg_data:.6f} | Phys: {avg_phys:.6f} | "
                  f"Val: {val_loss:.6f}")

    return history


def train_hybrid(model, X_train, y_train, X_val, y_val,
                 physics_weight=0.1, epochs=300, lr=1e-3,
                 batch_size=256, verbose=True):
    """Train the hybrid physics+ML model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5, min_lr=1e-5
    )
    criterion = nn.MSELoss()

    history = {
        "train_loss": [], "val_loss": [], "data_loss": [],
        "physics_loss": [], "epoch_time": [],
    }
    n = len(X_train)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()

        perm = torch.randperm(n)
        epoch_data_loss = 0.0
        epoch_phys_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X_train[idx], y_train[idx]

            pred = model(xb)
            data_loss = criterion(pred, yb)

            phys_losses = model.physics_loss(xb, pred)
            phys_total = sum(phys_losses.values())

            total_loss = data_loss + physics_weight * phys_total

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_data_loss += data_loss.item()
            epoch_phys_loss += phys_total.item()
            n_batches += 1

        avg_data = epoch_data_loss / n_batches
        avg_phys = epoch_phys_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)
        dt = time.time() - t0

        history["train_loss"].append(avg_data + physics_weight * avg_phys)
        history["val_loss"].append(val_loss)
        history["data_loss"].append(avg_data)
        history["physics_loss"].append(avg_phys)
        history["epoch_time"].append(dt)

        if verbose and (epoch + 1) % 50 == 0:
            print(f"  [Hybrid]   Epoch {epoch+1:3d}/{epochs} | "
                  f"Data: {avg_data:.6f} | Phys: {avg_phys:.6f} | "
                  f"Val: {val_loss:.6f}")

    return history
