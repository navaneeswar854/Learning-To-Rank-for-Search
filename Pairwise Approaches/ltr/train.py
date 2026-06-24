"""
ltr/train.py
------------
Unified training loop for all three LTR modes.

Public API
----------
set_seed        — Fix all random seeds for reproducibility.
train           — Single-run training with early stopping on val NDCG@k (Fix #2).
train_multiseed — Multi-seed wrapper reporting Mean ± Std (Fix #8).
train_lambdamart_multiseed — Multi-seed wrapper for LambdaMART.

Fix #2 — Early stopping on validation NDCG@k
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The original pointwise and RankNet notebooks saved the best model based on
**validation loss**.  This was inconsistent with LambdaRank (which already
used val NDCG) and theoretically incorrect — the goal metric is NDCG, not
loss.  All three modes now use **validation NDCG@k** for best-model
selection and early stopping.

Fix #8 — Multi-seed training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``train_multiseed`` runs the full training procedure across three random
seeds (default: 42, 123, 456) and returns per-seed results alongside
Mean ± Std aggregates.  This gives an honest estimate of result variance
and avoids lucky/unlucky single-seed runs.
"""

import copy
import random
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from .losses  import pointwise_mse, ranknet_loss, lambda_gradients
from .metrics import mean_ndcg


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Core training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    mode: str = "pointwise",
    epochs: int = 50,
    lr: float = 0.001,
    k: int = 10,
    patience: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Tuple[torch.nn.Module, List[float]]:
    """
    Unified training loop for Pointwise, RankNet, and LambdaRank.

    Fix #2 — Early stopping & best-model selection are ALWAYS based on
    **validation NDCG@k** across all three modes, never on validation loss.

    Parameters
    ----------
    model        : ScoringMLP instance (will be moved to ``device``).
    train_loader : DataLoader yielding ``(qids, feats_list, labels_list)``.
    val_loader   : Validation DataLoader (same format).
    mode         : ``'pointwise'``, ``'ranknet'``, or ``'lambdarank'``.
    epochs       : Maximum number of training epochs.
    lr           : Adam learning rate.
    k            : NDCG cutoff used for validation and early stopping.
    patience     : Stop training if val NDCG@k does not improve for this
                   many consecutive epochs.  Set to 0 to disable early stopping.
    device       : ``'cpu'`` or ``'cuda'``.
    verbose      : Print per-epoch metrics when True.

    Returns
    -------
    (best_model, val_ndcg_history)
        ``best_model``        — model restored to best-val-NDCG@k weights.
        ``val_ndcg_history``  — list of val NDCG@k values, one per epoch.
    """
    _VALID_MODES = ("pointwise", "ranknet", "lambdarank")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got '{mode}'.")

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_ndcg: float = -1.0
    best_weights = None
    val_ndcg_history: List[float] = []
    epochs_no_improve: int = 0

    for epoch in range(epochs):

        # ──────────────────── TRAINING PHASE ──────────────────────────────
        model.train()

        for batch_qids, batch_feats, batch_labels in train_loader:

            if mode == "lambdarank":
                # ── LambdaRank: one optimizer step per query ───────────────
                # Gradients are injected directly via scores.backward(lambdas),
                # so we cannot accumulate across a batch in the usual sense.
                for feats, labels in zip(batch_feats, batch_labels):
                    feats  = feats.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    scores = model(feats)
                    scores.retain_grad()  # keep grad on non-leaf tensor

                    with torch.no_grad():
                        lambdas = lambda_gradients(scores, labels, k=k)

                    scores.backward(lambdas)
                    optimizer.step()

            else:
                # ── Pointwise / RankNet: accumulate loss, one step per batch
                optimizer.zero_grad()
                batch_losses = []

                for feats, labels in zip(batch_feats, batch_labels):
                    feats  = feats.to(device)
                    labels = labels.to(device)
                    scores = model(feats)

                    if mode == "pointwise":
                        loss = pointwise_mse(scores, labels)
                    else:  # ranknet
                        loss = ranknet_loss(scores, labels)

                    batch_losses.append(loss)

                if batch_losses:
                    avg_loss = sum(batch_losses) / len(batch_losses)
                    avg_loss.backward()
                    optimizer.step()

        # ──────────────────── VALIDATION PHASE (Fix #2) ───────────────────
        model.eval()
        val_ndcg = mean_ndcg(model, val_loader, k_list=(k,), device=device)[k]
        val_ndcg_history.append(val_ndcg)

        improved = val_ndcg > best_val_ndcg
        if improved:
            best_val_ndcg = val_ndcg
            best_weights  = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose:
            marker = "  ← best" if improved else ""
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Val NDCG@{k}: {val_ndcg:.4f}{marker}"
            )

        # ── Early stopping ─────────────────────────────────────────────────
        if patience > 0 and epochs_no_improve >= patience:
            if verbose:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {patience} consecutive epochs)."
                )
            break

    # Restore weights from the best epoch
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, val_ndcg_history


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed wrapper (Fix #8)
# ─────────────────────────────────────────────────────────────────────────────

def train_multiseed(
    model_fn: Callable,
    train_loader,
    val_loader,
    test_loader,
    mode: str = "pointwise",
    seeds: Tuple[int, ...] = (42, 123, 456),
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    device: str = "cpu",
    **train_kwargs,
) -> Dict:
    """
    Fix #8 — Run training across multiple random seeds and report Mean ± Std.

    Each seed produces an independently initialised and independently trained
    model.  Final test-set NDCG@k values are aggregated to give a more
    reliable estimate of model performance and its variance.

    Parameters
    ----------
    model_fn    : Zero-argument callable that returns a **fresh** model instance.
                  Called once per seed, e.g. ``lambda: ScoringMLP(46, [64,32], 0.2)``.
    train_loader: Training DataLoader.
    val_loader  : Validation DataLoader.
    test_loader : Test DataLoader used for final evaluation only.
    mode        : ``'pointwise'``, ``'ranknet'``, or ``'lambdarank'``.
    seeds       : Random seeds to run over.  Default: ``(42, 123, 456)``.
    k_list      : NDCG cutoffs to evaluate and report.
    device      : ``'cpu'`` or ``'cuda'``.
    **train_kwargs : Additional keyword arguments forwarded to ``train()``,
                     e.g. ``epochs=50``, ``lr=0.001``, ``patience=10``.

    Returns
    -------
    dict with keys:
        ``'per_seed'`` — list of per-seed NDCG dicts ``{k: score}``.
        ``'summary'``  — ``{k: {'mean': float, 'std': float}}``.
    """
    per_seed_results: List[Dict[int, float]] = []

    for seed in seeds:
        set_seed(seed)
        model = model_fn().to(device)

        trained_model, _ = train(
            model, train_loader, val_loader,
            mode=mode, device=device, **train_kwargs
        )

        ndcg_scores = mean_ndcg(
            trained_model, test_loader, k_list=k_list, device=device
        )
        per_seed_results.append(ndcg_scores)

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        values = [r[k] for r in per_seed_results]
        summary[k] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
        }

    return {
        "per_seed": per_seed_results,
        "summary":  summary,
    }


def train_lambdamart_multiseed(
    model_fn: Callable,
    train_loader,
    val_loader,
    test_loader,
    seeds: Tuple[int, ...] = (42, 123, 456),
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    device: str = "cpu",
    **train_kwargs,
) -> Dict:
    """
    Multi-seed wrapper specifically for LambdaMART.
    """
    per_seed_results: List[Dict[int, float]] = []

    for seed in seeds:
        set_seed(seed)
        model = model_fn()

        trained_model, _ = model.fit(
            train_loader=train_loader, 
            val_loader=val_loader,
            device=device, 
            **train_kwargs
        )

        ndcg_scores = mean_ndcg(
            trained_model, test_loader, k_list=k_list, device=device
        )
        per_seed_results.append(ndcg_scores)

    # ── Aggregate ─────────────────────────────────────────────────────────
    summary: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        values = [r[k] for r in per_seed_results]
        summary[k] = {
            "mean": float(np.mean(values)),
            "std":  float(np.std(values)),
        }

    return {
        "per_seed": per_seed_results,
        "summary":  summary,
    }
