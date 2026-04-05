"""
Model definitions: Standard ML and Physics-Informed ML.

Physics knowledge for the Combined Cycle Power Plant:
1. Higher ambient temp (AT) → lower Carnot efficiency → lower power: ∂PE/∂AT < 0
2. Higher exhaust vacuum (V) → lower condenser efficiency → lower power: ∂PE/∂V < 0
3. Energy output bounded by plant capacity (~420-500 MW)
4. Physical systems produce smooth, continuous responses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardNN(nn.Module):
    """Standard fully-connected neural network (pure data-driven)."""

    def __init__(self, input_dim=4, hidden_dims=(64, 64, 32), output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PhysicsInformedNN(nn.Module):
    """
    Physics-Informed Neural Network.

    Same architecture as StandardNN but with SiLU activations (smooth,
    non-saturating) and trained with physics loss terms enforcing
    thermodynamic monotonicity and smoothness.
    """

    def __init__(self, input_dim=4, hidden_dims=(64, 64, 32), output_dim=1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())  # Smooth + non-saturating (unlike Tanh)
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def physics_loss(self, x, y_pred):
        losses = {}
        if x.requires_grad:
            grad = torch.autograd.grad(
                y_pred.sum(), x, create_graph=True, retain_graph=True
            )[0]

            # ∂PE/∂AT < 0: penalize positive gradients
            losses["mono_AT"] = torch.mean(torch.relu(grad[:, 0]) ** 2)
            # ∂PE/∂V < 0: penalize positive gradients
            losses["mono_V"] = torch.mean(torch.relu(grad[:, 1]) ** 2)

            # Smoothness: penalize large second derivatives
            grad2 = torch.autograd.grad(
                grad.sum(), x, create_graph=True, retain_graph=True
            )[0]
            losses["smoothness"] = torch.mean(grad2 ** 2) * 0.1

        return losses


class PhysicsHybridNN(nn.Module):
    """
    Hybrid physics + ML model.

    Physics backbone: linear model with structurally guaranteed signs.
    AT and V coefficients are forced negative via -softplus.
    This guarantees correct monotonicity in ANY regime (including extrapolation).

    A small NN learns the residual (nonlinear corrections the linear
    physics misses). The correction is lightly penalized to keep the
    physics backbone dominant.
    """

    def __init__(self, input_dim=4, hidden_dims=(32, 32), output_dim=1):
        super().__init__()

        # --- Physics backbone (linear, guaranteed monotonic) ---
        self.base = nn.Parameter(torch.tensor(0.0))
        self.at_raw = nn.Parameter(torch.tensor(1.0))   # → -softplus(at_raw) * AT
        self.v_raw = nn.Parameter(torch.tensor(0.5))    # → -softplus(v_raw) * V
        self.ap_scale = nn.Parameter(torch.tensor(0.05))
        self.rh_scale = nn.Parameter(torch.tensor(-0.02))

        # --- NN correction ---
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.correction = nn.Sequential(*layers)

    def physics_forward(self, x):
        at, v, ap, rh = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        return (
            self.base
            - F.softplus(self.at_raw) * at
            - F.softplus(self.v_raw) * v
            + self.ap_scale * ap
            + self.rh_scale * rh
        )

    def forward(self, x):
        return self.physics_forward(x) + self.correction(x)

    def physics_loss(self, x, y_pred):
        losses = {}
        correction = y_pred - self.physics_forward(x)
        losses["correction"] = torch.mean(correction ** 2) * 0.05
        return losses
