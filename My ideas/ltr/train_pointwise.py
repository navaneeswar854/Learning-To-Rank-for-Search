"""
ltr/train_pointwise.py
----------------------
Training script for PointwiseScorer using custom lambda-gradient injection.

This is NOT a traditional loss-minimization loop.
Instead of computing a scalar loss and calling loss.backward(),
we directly compute a per-document gradient signal (lambda) and
inject it via scores.backward(gradient=lambdas).

Custom Lambda Formula (per document i, within one query)
---------------------------------------------------------
    lambda_i = s_i - s_min - (r_i * (s_max - s_min)) / 4

Where:
    s_i    = predicted score for document i          (from PointwiseScorer)
    s_min  = min predicted score across all docs in the query
    s_max  = max predicted score across all docs in the query
    r_i    = ground truth relevance label (0-4 for MSLR)

Intuition
---------
    lambda_i = 0   -> doc i is exactly where it should be in the score range
    lambda_i > 0   -> doc i is scored TOO HIGH  -> optimizer pushes s_i DOWN
    lambda_i < 0   -> doc i is scored TOO LOW   -> optimizer pushes s_i UP

    For a perfectly relevant doc (r_i = 4):
        target position = s_min + (s_max - s_min) = s_max
        lambda_i = s_i - s_max <= 0  -> always pushes toward the top

    For an irrelevant doc (r_i = 0):
        target position = s_min
        lambda_i = s_i - s_min >= 0  -> always pushes toward the bottom

Proxy metric (reported as "Lambda RMS")
----------------------------------------
Since there is no scalar loss to report, we report the Root Mean Square of
the lambdas across all docs in the batch.  A smaller RMS means the predicted
scores are closer to their target positions in the ranking.

Returns
-------
train_pointwise() returns a tuple of three items:
    model            : PointwiseScorer loaded with the best checkpoint weights
    train_ndcg_hist  : List[float]  -- train NDCG@10 recorded every epoch
    val_ndcg_hist    : List[float]  -- val   NDCG@10 recorded every epoch

Usage
-----
    from ltr.train_pointwise import train_pointwise, PointwiseTrainConfig

    model, train_hist, val_hist = train_pointwise(PointwiseTrainConfig(
        dataset     = "mslr",
        data_path   = "/content/MSLR",
        fold        = 1,
        input_dim   = 111,
        hidden_dims = [64, 32],
        epochs      = 100,
        patience    = 10,
    ))
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import List, Tuple

import torch
import torch.nn as nn

from ltr.model_pointwise import PointwiseScorer
from ltr.metrics         import mean_ndcg


# ---------------------------------------------------------------------------
# Custom lambda function
# ---------------------------------------------------------------------------

def compute_lambdas(
    scores: torch.Tensor,
    labels: torch.Tensor,
    max_relevance: int = 4,
) -> torch.Tensor:
    """
    Compute the per-document lambda (gradient signal) for one query.

    Formula
    -------
        k        = max_relevance  (e.g. 4 for MSLR, 2 for MQ2008)
        x        = abs(s_max - s_min - k)
        t        = 2*(k - x) / ((k - 2) * x)      -- x is in the DENOMINATOR
        lambda_i = (1 - t)*(s_i - r_i)
                 - t*(s_i - s_min - r_i*(s_max - s_min)/k)
                 + (s_max + s_min - k)

    Term breakdown
    --------------
        (1-t)*(s_i - r_i)                     MSE-like pull toward raw label
        t*(s_i - s_min - r_i*(s_max-s_min)/k) proportional position pull
        (s_max + s_min - k)                    centering shift (scalar)

    Behaviour of t
    --------------
        x = 0 (spread == k, ideal) -> t -> inf  (clipped by eps)
        x large (poor spread)      -> t -> 0    (MSE pull dominates)

    Parameters
    ----------
    scores        : shape (N,), requires_grad=True
    labels        : shape (N,), ground truth relevance labels
    max_relevance : highest possible relevance label (k)

    Returns
    -------
    torch.Tensor, shape (N,)  -- normalized by N
    """
    k   = float(max_relevance)
    s   = scores.detach()
    r   = labels.float()

    s_min = s.min()
    s_max = s.max()

    # x: how far the score spread deviates from the ideal spread k
    x = (s_max - s_min - k).abs()

    # t = 2*(k-x) / ((k-2)*x)   -- x in denominator
    # eps on both (k-2) and x to prevent division by zero:
    #   x = 0 when spread = k exactly
    #   k = 2 for MQ2008
    eps = 1e-8
    t   = 2.0 * (k - x) / ((k - 2.0 + eps) * (x + eps))

    # Term 1: MSE-like pull -- push each score toward its raw relevance label
    term1 = (1.0 - t) * (s - r)

    # Term 2: proportional position pull
    term2 = t * (s - s_min - r * (s_max - s_min) / k)

    # Term 3: centering shift (scalar, same for all docs in this query)
    term3 = s_max + s_min - k

    lambdas = term1 - term2 + term3

    return lambdas / len(labels)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PointwiseTrainConfig:
    """All hyperparameters for one pointwise lambda-gradient training run."""

    # Data
    dataset    : str       = "mslr"         # "mslr" or "mq2008"
    data_path  : str       = "/content/MSLR"
    fold       : int       = 1
    batch_size : int       = 4

    # Architecture (PointwiseScorer always has output_dim=1, no activation)
    input_dim     : int       = 111            # 111 for MSLR, 46 for MQ2008
    hidden_dims   : List[int] = dataclasses.field(default_factory=lambda: [64, 32])
    dropout       : float     = 0.0

    # Lambda formula
    max_relevance : int       = 4              # perfect label: 4 for MSLR, 2 for MQ2008

    # Optimiser
    lr            : float     = 1e-4
    epochs     : int       = 100

    # Early stopping (based on val NDCG@10)
    patience   : int       = 10

    # Checkpointing
    save_path  : str       = "best_pointwise.pt"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train_pointwise(
    cfg: PointwiseTrainConfig,
) -> Tuple[PointwiseScorer, List[float], List[float]]:
    """
    Train PointwiseScorer using custom lambda-gradient injection.

    At each batch step:
        1. Forward pass  : scores = model(feats)              -- (N,)
        2. Compute lambda: lambdas = compute_lambdas(s, r)    -- (N,) detached
        3. Inject grads  : scores.backward(gradient=lambdas)
        4. Clip & step   : clip_grad_norm_ + optimizer.step()

    Parameters
    ----------
    cfg : PointwiseTrainConfig

    Returns
    -------
    model : PointwiseScorer
        Loaded with the weights from the epoch with the best val NDCG@10.
    train_ndcg_hist : List[float]
        Train NDCG@10 recorded at the end of every completed epoch.
    val_ndcg_hist : List[float]
        Val NDCG@10 recorded at the end of every completed epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*57}")
    print(f"  Mode        : Pointwise (custom lambda injection)")
    print(f"  Device      : {device}")
    print(f"  Fold        : {cfg.fold}")
    print(f"  Max epochs  : {cfg.epochs}  (patience={cfg.patience})")
    print(f"  Batch size  : {cfg.batch_size} queries")
    print(f"  Architecture: {cfg.input_dim} -> {cfg.hidden_dims} -> 1 (linear out)")
    print(f"  Dropout     : {cfg.dropout}   LR: {cfg.lr}")
    print(f"{'='*57}")

    # -- Data -----------------------------------------------------------------
    if cfg.dataset.lower() == "mslr":
        from ltr.data_mslr import load_fold
    else:
        from ltr.data import load_fold

    train_loader, val_loader, _ = load_fold(
        base_path  = cfg.data_path,
        fold_num   = cfg.fold,
        batch_size = cfg.batch_size,
    )

    # -- Model ----------------------------------------------------------------
    model = PointwiseScorer(
        input_dim   = cfg.input_dim,
        hidden_dims = cfg.hidden_dims,
        dropout     = cfg.dropout,
    ).to(device)

    print(f"\nParameters  : {model.count_parameters():,}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # -- History & early-stopping state ---------------------------------------
    train_ndcg_hist: List[float] = []
    val_ndcg_hist:   List[float] = []

    best_val_ndcg  = -1.0
    best_epoch     = 0
    patience_count = 0

    # -- Header ---------------------------------------------------------------
    # "Lambda RMS" is the proxy training signal:
    #   small RMS -> predicted scores are close to their target positions
    print(f"{'Epoch':>6}  {'Lambda RMS':>10}  {'Train NDCG@10':>13}  {'Val NDCG@10':>11}")
    print(f"{'-'*49}")

    # -- Epoch loop -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):

        # ---- Training -------------------------------------------------------
        model.train()
        total_rms   = 0.0
        num_batches = 0

        for qids, feats_list, labels_list in train_loader:
            optimizer.zero_grad()   # zero once per batch, accumulate per query

            batch_rms = 0.0

            for feats, labels in zip(feats_list, labels_list):
                feats  = feats.to(device)
                labels = labels.to(device)

                # Forward: raw unbounded score per document
                scores  = model(feats)                        # (N,)

                # Compute lambda signal (detached from graph)
                lambdas = compute_lambdas(scores, labels, cfg.max_relevance)  # (N,)

                # Inject lambda as the gradient of an imaginary loss.
                # PyTorch treats this as: dL/ds_i = lambda_i,
                # and chain-rules backward through the MLP.
                scores.backward(gradient=lambdas)

                batch_rms += lambdas.pow(2).mean().sqrt().item()

            # Safety: clip accumulated gradients before stepping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_rms   += batch_rms / len(feats_list)
            num_batches += 1

        avg_rms = total_rms / max(num_batches, 1)

        # ---- NDCG@10 evaluation (train + val) -------------------------------
        model.eval()
        train_scores = mean_ndcg(model, train_loader, k_list=(10,), device=str(device))
        val_scores   = mean_ndcg(model, val_loader,   k_list=(10,), device=str(device))

        tr_ndcg = train_scores[10]
        vl_ndcg = val_scores[10]

        train_ndcg_hist.append(tr_ndcg)
        val_ndcg_hist.append(vl_ndcg)

        # ---- Checkpoint & early stopping ------------------------------------
        improved = vl_ndcg > best_val_ndcg
        if improved:
            best_val_ndcg  = vl_ndcg
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), cfg.save_path)
        else:
            patience_count += 1

        # ---- Logging --------------------------------------------------------
        marker = " *" if improved else ""
        print(
            f"{epoch:>6}  {avg_rms:>10.4f}  {tr_ndcg:>13.4f}  {vl_ndcg:>11.4f}{marker}"
        )

        # ---- Early stop check -----------------------------------------------
        if patience_count >= cfg.patience:
            print(f"\n[Early stop] No improvement for {cfg.patience} epochs.")
            break

    # -- Summary --------------------------------------------------------------
    print(f"{'-'*49}")
    print(f"\nBest val NDCG@10 : {best_val_ndcg:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint saved : {cfg.save_path}\n")

    # -- Load best weights back into model ------------------------------------
    model.load_state_dict(
        torch.load(cfg.save_path, map_location=device, weights_only=True)
    )
    model.eval()

    return model, train_ndcg_hist, val_ndcg_hist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> PointwiseTrainConfig:
    p = argparse.ArgumentParser(
        description="Train PointwiseScorer with custom lambda-gradient injection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",     type=str,   default="mslr",
                   choices=["mslr", "mq2008"])
    p.add_argument("--data-path",   type=str,   default="/content/MSLR")
    p.add_argument("--fold",        type=int,   default=1)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--input-dim",   type=int,   default=111)
    p.add_argument("--hidden-dims",    type=int,   nargs="+", default=[64, 32])
    p.add_argument("--dropout",        type=float, default=0.0)
    p.add_argument("--max-relevance",  type=int,   default=4,
                   help="Perfect relevance label (4 for MSLR, 2 for MQ2008)")
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--save-path",   type=str,   default="best_pointwise.pt")

    a = p.parse_args()
    return PointwiseTrainConfig(
        dataset       = a.dataset,
        data_path     = a.data_path,
        fold          = a.fold,
        batch_size    = a.batch_size,
        input_dim     = a.input_dim,
        hidden_dims   = a.hidden_dims,
        dropout       = a.dropout,
        max_relevance = a.max_relevance,
        lr            = a.lr,
        epochs        = a.epochs,
        patience      = a.patience,
        save_path     = a.save_path,
    )


if __name__ == "__main__":
    model, train_hist, val_hist = train_pointwise(_parse_args())
