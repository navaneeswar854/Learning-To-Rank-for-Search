"""
ltr/train.py
------------
End-to-end training script for ScoringMLP with WeightedRelevanceBCE loss.

Training objective  : WeightedRelevanceBCE (custom per-neuron weighted loss)
Ranking metric      : NDCG@10 via expected-relevance scoring
Batch strategy      : Option A -- loss computed per query, averaged over batch
Early stopping      : Patience on val NDCG@10

Expected-relevance scoring
--------------------------
The model outputs softmax probabilities over k classes, but NDCG requires
a single scalar score per document for ranking.  We convert:

    score(doc) = sum_{i=0}^{k-1}  i * p_i

This is the expected relevance label under the model's predicted distribution.
A more confident prediction of label 2 scores higher than label 1, etc.

Returns
-------
train() returns a tuple of three items:
    model            : ScoringMLP loaded with the best checkpoint weights
    train_ndcg_hist  : List[float]  -- train NDCG@10 recorded every epoch
    val_ndcg_hist    : List[float]  -- val   NDCG@10 recorded every epoch

Usage (command line)
--------------------
    python -m ltr.train --data-path /content/MQ2008 --fold 1 --epochs 100

Usage (import)
--------------
    from ltr.train import train, TrainConfig
    model, train_hist, val_hist = train(TrainConfig(data_path="..."))
"""

from __future__ import annotations

import argparse
import dataclasses
from typing import List, Tuple

import torch
import torch.nn as nn

from ltr.data    import load_fold
from ltr.loss    import WeightedRelevanceBCE
from ltr.metrics import mean_ndcg
from ltr.model   import ScoringMLP


# ---------------------------------------------------------------------------
# Evaluation wrapper
# ---------------------------------------------------------------------------

class _ExpectedRelevanceScorer(nn.Module):
    """
    Thin wrapper around ScoringMLP that converts the (N, k) softmax output
    to a (N, 1) expected-relevance scalar, compatible with metrics.mean_ndcg.

        score(doc) = sum_{i=0}^{k-1}  i * p_i

    Used ONLY for NDCG evaluation; training always uses the raw ScoringMLP.
    """

    def __init__(self, model: ScoringMLP) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "class_idx",
            torch.arange(model.output_dim, dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.model(x)                                     # (N, k)
        return (probs * self.class_idx).sum(dim=1, keepdim=True)  # (N, 1)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    """All hyperparameters for one training run."""

    # Data
    data_path  : str       = "/content/MQ2008"
    fold       : int       = 1
    batch_size : int       = 4

    # Architecture
    input_dim  : int       = 46
    hidden_dims: List[int] = dataclasses.field(default_factory=lambda: [64, 32])
    num_classes: int       = 3       # MQ2008: labels {0, 1, 2}
    dropout    : float     = 0.0

    # Optimiser
    lr         : float     = 1e-3
    epochs     : int       = 100

    # Early stopping
    patience   : int       = 10     # stop if val NDCG@10 does not improve for
                                    # this many consecutive epochs

    # Checkpointing
    save_path  : str       = "best_model.pt"


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(
    cfg: TrainConfig,
) -> Tuple[ScoringMLP, List[float], List[float]]:
    """
    Train ScoringMLP and return the best model together with NDCG histories.

    Parameters
    ----------
    cfg : TrainConfig

    Returns
    -------
    model : ScoringMLP
        Loaded with the weights from the epoch with the best val NDCG@10.
    train_ndcg_hist : List[float]
        Train NDCG@10 recorded at the end of every completed epoch.
    val_ndcg_hist : List[float]
        Val NDCG@10 recorded at the end of every completed epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'='*55}")
    print(f"  Device      : {device}")
    print(f"  Fold        : {cfg.fold}")
    print(f"  Max epochs  : {cfg.epochs}  (patience={cfg.patience})")
    print(f"  Batch size  : {cfg.batch_size} queries")
    print(f"  Architecture: {cfg.input_dim} -> {cfg.hidden_dims} -> {cfg.num_classes}")
    print(f"  Dropout     : {cfg.dropout}   LR: {cfg.lr}")
    print(f"{'='*55}")

    # -- Data -----------------------------------------------------------------
    train_loader, val_loader, _ = load_fold(
        base_path  = cfg.data_path,
        fold_num   = cfg.fold,
        batch_size = cfg.batch_size,
    )

    # -- Model ----------------------------------------------------------------
    model = ScoringMLP(
        input_dim   = cfg.input_dim,
        hidden_dims = cfg.hidden_dims,
        output_dim  = cfg.num_classes,
        dropout     = cfg.dropout,
    ).to(device)

    print(f"\nParameters  : {model.count_parameters():,}\n")

    scorer    = _ExpectedRelevanceScorer(model).to(device)
    criterion = WeightedRelevanceBCE(num_classes=cfg.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # -- History & early-stopping state ---------------------------------------
    train_ndcg_hist: List[float] = []
    val_ndcg_hist:   List[float] = []

    best_val_ndcg   = -1.0
    best_epoch      = 0
    patience_count  = 0

    # -- Header ---------------------------------------------------------------
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train NDCG@10':>13}  {'Val NDCG@10':>11}")
    print(f"{'-'*47}")

    # -- Epoch loop -----------------------------------------------------------
    for epoch in range(1, cfg.epochs + 1):

        # ---- Training -------------------------------------------------------
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for qids, feats_list, labels_list in train_loader:
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=device)

            for feats, labels in zip(feats_list, labels_list):
                feats  = feats.to(device)
                labels = labels.long().to(device)

                probs      = model(feats)              # (num_docs, num_classes)
                query_loss = criterion(probs, labels)  # scalar
                batch_loss = batch_loss + query_loss

            # Average loss over queries in this batch
            batch_loss = batch_loss / len(feats_list)
            batch_loss.backward()
            optimizer.step()

            total_loss  += batch_loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        # ---- NDCG@10 evaluation ---------------------------------------------
        model.eval()
        train_scores = mean_ndcg(scorer, train_loader, k_list=(10,), device=str(device))
        val_scores   = mean_ndcg(scorer, val_loader,   k_list=(10,), device=str(device))

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
            f"{epoch:>6}  {avg_loss:>10.4f}  {tr_ndcg:>13.4f}  {vl_ndcg:>11.4f}{marker}"
        )

        # ---- Early stop check -----------------------------------------------
        if patience_count >= cfg.patience:
            print(f"\n[Early stop] No improvement for {cfg.patience} epochs.")
            break

    # -- Summary --------------------------------------------------------------
    print(f"{'-'*47}")
    print(f"\nBest val NDCG@10 : {best_val_ndcg:.4f}  (epoch {best_epoch})")
    print(f"Checkpoint saved : {cfg.save_path}\n")

    # -- Load best weights back into the model --------------------------------
    model.load_state_dict(
        torch.load(cfg.save_path, map_location=device, weights_only=True)
    )
    model.eval()

    return model, train_ndcg_hist, val_ndcg_hist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description="Train ScoringMLP on LETOR / MQ2008 data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-path",   type=str,   default="/content/MQ2008")
    p.add_argument("--fold",        type=int,   default=1)
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--input-dim",   type=int,   default=46)
    p.add_argument("--hidden-dims", type=int,   nargs="+", default=[64, 32])
    p.add_argument("--num-classes", type=int,   default=3)
    p.add_argument("--dropout",     type=float, default=0.0)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=10)
    p.add_argument("--save-path",   type=str,   default="best_model.pt")

    a = p.parse_args()
    return TrainConfig(
        data_path   = a.data_path,
        fold        = a.fold,
        batch_size  = a.batch_size,
        input_dim   = a.input_dim,
        hidden_dims = a.hidden_dims,
        num_classes = a.num_classes,
        dropout     = a.dropout,
        lr          = a.lr,
        epochs      = a.epochs,
        patience    = a.patience,
        save_path   = a.save_path,
    )


if __name__ == "__main__":
    model, train_hist, val_hist = train(_parse_args())
