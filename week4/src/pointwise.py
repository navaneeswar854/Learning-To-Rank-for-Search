"""
pointwise.py
============
Loss function and training utilities for the Pointwise RankNet approach.

Extracted from: notebooks/Pointwise.ipynb
Dataset      : LETOR4 / MQ2008
Model        : RankNet — Deep Pointwise Scoring Network
Loss         : MSE (Mean Squared Error) on relevance labels
Metric       : NDCG@K
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ── Model ──────────────────────────────────────────────────────────────────────

class RankNet(nn.Module):
    """
    A Pointwise Scoring Network for Learning-to-Rank.

    Supports four architecture variants controlled by `architecture_type`:
      - 'linear'      : Pure linear baseline (46 → 1), no hidden layers.
      - 'baseline'    : Standard MLP without regularization (46 → 64 → 32 → 1).
      - 'regularized' : Standard MLP with 20 % Dropout (46 → 64 → 32 → 1).  [default]
      - 'deep'        : Over-parameterized deep network (46 → 128 → 64 → 32 → 16 → 1).

    Parameters
    ----------
    input_dim : int
        Number of input IR features (default: 46 for MQ2008).
    hidden_dim : int
        Base hidden layer width (default: 64).
    architecture_type : str
        One of {'linear', 'baseline', 'regularized', 'deep'}.
    """

    def __init__(self, input_dim=46, hidden_dim=64, architecture_type='regularized'):
        super(RankNet, self).__init__()

        if architecture_type == 'linear':
            # Pure linear baseline: no activation functions, no hidden layers (46 -> 1)
            self.scorer = nn.Linear(input_dim, 1)

        elif architecture_type == 'baseline':
            # Our standard structure but with zero regularization (46 -> 64 -> 32 -> 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif architecture_type == 'regularized':
            # Our standard structure protected by a 20% Dropout Rate (46 -> 64 -> 32 -> 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim // 2, 1)
            )

        elif architecture_type == 'deep':
            # Over-parameterized Deep Network for Ablation (46 -> 128 -> 64 -> 32 -> 16 -> 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),

                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),

                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),

                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=0.2),

                nn.Linear(hidden_dim // 4, 1)
            )

    def forward(self, x):
        return self.scorer(x)


# ── Loss Function ──────────────────────────────────────────────────────────────

def pointwise_loss(scores, labels):
    """
    Computes the Mean Squared Error (MSE) between predicted scores and
    ground-truth relevance labels.

    Parameters
    ----------
    scores : torch.Tensor
        Model output scores, shape (n_docs, 1).
    labels : torch.Tensor
        Ground-truth relevance labels, shape (n_docs,).

    Returns
    -------
    torch.Tensor
        Scalar MSE loss.
    """
    return F.mse_loss(scores.squeeze(), labels.float())


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_ranknet(model, train_loader, val_loader,
                  epochs=15, lr=0.001, device="cpu", verbose=True):
    """
    A generalised training engine for the Pointwise RankNet approach.

    Processes any given model architecture and data split configuration.
    Implements best-model checkpointing based on validation loss.

    Parameters
    ----------
    model : RankNet
        The scoring network to train.
    train_loader : DataLoader
        Query-grouped DataLoader for training data.
    val_loader : DataLoader
        Query-grouped DataLoader for validation data.
    epochs : int
        Number of training epochs (default: 15).
    lr : float
        Adam learning rate (default: 0.001).
    device : str or torch.device
        Device to train on, e.g. "cpu" or "cuda".
    verbose : bool
        If True, prints per-epoch loss statistics.

    Returns
    -------
    model : RankNet
        The model loaded with the best-validation-loss weights.
    train_loss_history : list[float]
        Per-epoch average training loss.
    val_loss_history : list[float]
        Per-epoch average validation loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    best_model_weights = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        # ═════════════════ TRAINING PHASE ═════════════════
        model.train()
        total_train_loss = 0.0
        train_queries_count = 0

        for batch_qids, batch_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0

            for i in range(len(batch_feats)):
                feats = batch_feats[i].to(device)
                labels = batch_labels[i].to(device)
                scores = model(feats)
                batch_loss += pointwise_loss(scores, labels)
                train_queries_count += 1

            batch_loss = batch_loss / len(batch_feats)
            batch_loss.backward()
            optimizer.step()
            total_train_loss += batch_loss.item() * len(batch_feats)

        avg_train_loss = total_train_loss / max(1, train_queries_count)
        train_loss_history.append(avg_train_loss)

        # ═════════════════ VALIDATION PHASE ═════════════════
        model.eval()
        total_val_loss = 0.0
        val_queries_count = 0

        with torch.no_grad():
            for batch_qids, batch_feats, batch_labels in val_loader:
                for i in range(len(batch_feats)):
                    feats = batch_feats[i].to(device)
                    labels = batch_labels[i].to(device)
                    scores = model(feats)
                    total_val_loss += pointwise_loss(scores, labels).item()
                    val_queries_count += 1

        avg_val_loss = total_val_loss / max(1, val_queries_count)
        val_loss_history.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f}")

    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, train_loss_history, val_loss_history
