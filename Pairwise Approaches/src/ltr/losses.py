"""
ltr/losses.py
-------------
Loss functions and gradient computation for all three LTR training modes.

Functions
---------
pointwise_mse       — MSE between predicted scores and relevance labels.
ranknet_loss        — Vectorized pairwise BCE loss on (i > j) pairs.
lambda_gradients    — Vectorized LambdaRank gradient computation (Fix #3).
lambda_gradients_and_hessians — Computes both lambdas and hessians for LambdaMART.
"""

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Pointwise
# ─────────────────────────────────────────────────────────────────────────────

def pointwise_mse(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error between predicted relevance scores and ground-truth labels.

    Parameters
    ----------
    scores : torch.Tensor, shape (num_docs, 1)
    labels : torch.Tensor, shape (num_docs,)

    Returns
    -------
    Scalar loss tensor.
    """
    return F.mse_loss(scores.squeeze(), labels.float())


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise (RankNet)
# ─────────────────────────────────────────────────────────────────────────────

def ranknet_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Pairwise RankNet loss using Binary Cross-Entropy on (i > j) pairs.

    For each valid pair where ``label_i > label_j``:

        P_ij = σ(s_i − s_j)
        L    = −log P_ij  =  BCE(s_i − s_j,  target=1)

    The loss is averaged over all valid pairs in the query group.
    Returns a zero-gradient placeholder tensor if no valid pairs exist
    (e.g. all documents have identical relevance labels).

    Parameters
    ----------
    scores : torch.Tensor, shape (num_docs, 1)
    labels : torch.Tensor, shape (num_docs,)

    Returns
    -------
    Scalar loss tensor.
    """
    scores = scores.squeeze()  # (num_docs,)

    # Build all pairwise score differences: diff[i, j] = s_i − s_j
    scores_diff = scores.unsqueeze(1) - scores.unsqueeze(0)  # (N, N)

    # Build all pairwise label differences
    labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # (N, N)

    # Keep only valid pairs: doc i is strictly more relevant than doc j
    i_idx, j_idx = torch.where(labels_diff > 0)

    if len(i_idx) == 0:
        # No valid pairs — return differentiable zero
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    valid_diffs = scores_diff[i_idx, j_idx]
    targets = torch.ones_like(valid_diffs)

    return F.binary_cross_entropy_with_logits(valid_diffs, targets)


# ─────────────────────────────────────────────────────────────────────────────
# LambdaRank gradients (Fix #3 — vectorized, O(N²) loop removed)
# ─────────────────────────────────────────────────────────────────────────────

def _dcg_np(relevance: np.ndarray, k: int) -> float:
    """Compute DCG@k on a numpy array (helper, not exported)."""
    rel = np.asarray(relevance, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    denominators = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0 ** rel - 1.0) / denominators))


def lambda_gradients(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """
    Vectorized LambdaRank gradient computation (Fix #3).

    Replaces the original O(N²) double Python ``for`` loop with NumPy
    matrix broadcasting, computing all pairwise quantities simultaneously.

    For each valid pair (i, j) where ``label_i > label_j``:

        ρ_ij        = σ(−(s_i − s_j))        # sigmoid of negated score diff
        |ΔNDCG_ij|  = |gain_diff · disc_diff| / IDCG
        λ_ij        = ρ_ij · |ΔNDCG_ij|

    Net lambda per document (used as a surrogate gradient):

        λ_i = −Σ_{j: rel_i > rel_j} λ_ij  +  Σ_{j: rel_j > rel_i} λ_ij

    A negative λ_i means "push this document's score up";
    a positive λ_i means "push it down".

    Parameters
    ----------
    scores : torch.Tensor, shape (num_docs, 1)
        Raw predicted scores from the model (with grad).
    labels : torch.Tensor, shape (num_docs,)
        Ground-truth relevance labels.
    k : int
        NDCG cutoff — only positions ≤ k contribute to |ΔNDCG|.

    Returns
    -------
    torch.Tensor, shape (num_docs, 1)
        Lambda values to pass directly to ``scores.backward(lambdas)``.
        Returns zeros if IDCG == 0 (no relevant documents).
    """
    scores_sq  = scores.squeeze()                           # (N,)
    labels_np  = labels.cpu().numpy().astype(np.float64)   # (N,)
    scores_np  = scores_sq.detach().cpu().numpy().astype(np.float64)  # (N,)
    N = len(labels_np)

    if N == 0:
        return torch.zeros_like(scores)

    # ── Ideal DCG ─────────────────────────────────────────────────────────
    ideal_labels = np.sort(labels_np)[::-1]
    idcg = _dcg_np(ideal_labels, k)

    if idcg == 0.0:
        # No relevant documents — all lambdas are zero
        return torch.zeros_like(scores_sq).unsqueeze(1)

    # ── Current ranking ────────────────────────────────────────────────────
    ranked_order = np.argsort(scores_np)[::-1]   # doc indices sorted by score desc
    rank_of = np.empty(N, dtype=np.int64)
    for pos, doc_idx in enumerate(ranked_order):
        rank_of[doc_idx] = pos                   # 0-indexed rank of each document

    # ── Position discounts: 1/log2(rank+2), zero for ranks ≥ k ───────────
    disc = np.where(rank_of < k, 1.0 / np.log2(rank_of + 2.0), 0.0)  # (N,)

    # ── Per-document gain: 2^rel − 1 ──────────────────────────────────────
    gain = (2.0 ** labels_np) - 1.0  # (N,)

    # ── Vectorized (N×N) matrices ─────────────────────────────────────────
    # |ΔNDCG| for every pair (i, j)
    gain_diff  = gain[:, None] - gain[None, :]          # (N, N)
    disc_diff  = disc[:, None] - disc[None, :]          # (N, N)
    delta_ndcg = np.abs(gain_diff * disc_diff) / idcg   # (N, N)

    # ρ_ij = σ(−(s_i − s_j)) — numerically clipped to avoid overflow
    s_diff = scores_np[:, None] - scores_np[None, :]            # (N, N)
    rho    = 1.0 / (1.0 + np.exp(np.clip(s_diff, -500, 500)))  # (N, N)

    # Valid pair mask: only where label_i > label_j
    mask = (labels_np[:, None] > labels_np[None, :]).astype(np.float64)  # (N, N)

    # Lambda matrix: λ_ij for each valid pair
    lam_matrix = rho * delta_ndcg * mask  # (N, N)

    # Net lambda: doc i accumulates negative lambdas for pairs it wins,
    # and positive lambdas for pairs it loses.
    lambdas = -lam_matrix.sum(axis=1) + lam_matrix.sum(axis=0)  # (N,)

    return torch.tensor(
        lambdas, dtype=torch.float32, device=scores_sq.device
    ).unsqueeze(1)


def lambda_gradients_and_hessians(
    scores: torch.Tensor,
    labels: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized LambdaMART gradient and hessian computation.

    Returns both the first-order gradients (lambdas) and second-order
    derivatives (hessians) needed for Newton leaf steps in LambdaMART.

    Parameters
    ----------
    scores : torch.Tensor, shape (num_docs, 1)
        Raw predicted scores from the model.
    labels : torch.Tensor, shape (num_docs,)
        Ground-truth relevance labels.
    k : int
        NDCG cutoff — only positions ≤ k contribute to |ΔNDCG|.

    Returns
    -------
    (lambdas, hessians) : Tuple[torch.Tensor, torch.Tensor]
        Both tensors have shape (num_docs, 1).
    """
    scores_sq  = scores.squeeze()                           # (N,)
    labels_np  = labels.cpu().numpy().astype(np.float64)   # (N,)
    scores_np  = scores_sq.detach().cpu().numpy().astype(np.float64)  # (N,)
    N = len(labels_np)

    if N == 0:
        return torch.zeros_like(scores), torch.zeros_like(scores)

    # ── Ideal DCG ─────────────────────────────────────────────────────────
    ideal_labels = np.sort(labels_np)[::-1]
    idcg = _dcg_np(ideal_labels, k)

    if idcg == 0.0:
        # No relevant documents — all lambdas and hessians are zero
        zeros = torch.zeros_like(scores_sq).unsqueeze(1)
        return zeros, zeros

    # ── Current ranking ────────────────────────────────────────────────────
    ranked_order = np.argsort(scores_np)[::-1]   # doc indices sorted by score desc
    rank_of = np.empty(N, dtype=np.int64)
    for pos, doc_idx in enumerate(ranked_order):
        rank_of[doc_idx] = pos                   # 0-indexed rank of each document

    # ── Position discounts: 1/log2(rank+2), zero for ranks ≥ k ───────────
    disc = np.where(rank_of < k, 1.0 / np.log2(rank_of + 2.0), 0.0)  # (N,)

    # ── Per-document gain: 2^rel − 1 ──────────────────────────────────────
    gain = (2.0 ** labels_np) - 1.0  # (N,)

    # ── Vectorized (N×N) matrices ─────────────────────────────────────────
    # |ΔNDCG| for every pair (i, j)
    gain_diff  = gain[:, None] - gain[None, :]          # (N, N)
    disc_diff  = disc[:, None] - disc[None, :]          # (N, N)
    delta_ndcg = np.abs(gain_diff * disc_diff) / idcg   # (N, N)

    # ρ_ij = σ(−(s_i − s_j)) — numerically clipped to avoid overflow
    s_diff = scores_np[:, None] - scores_np[None, :]            # (N, N)
    rho    = 1.0 / (1.0 + np.exp(np.clip(s_diff, -500, 500)))  # (N, N)

    # Valid pair mask: only where label_i > label_j
    mask = (labels_np[:, None] > labels_np[None, :]).astype(np.float64)  # (N, N)

    # Lambda matrix: λ_ij for each valid pair
    lam_matrix = rho * delta_ndcg * mask  # (N, N)

    # Net lambda: doc i accumulates negative lambdas for pairs it wins,
    # and positive lambdas for pairs it loses.
    lambdas = -lam_matrix.sum(axis=1) + lam_matrix.sum(axis=0)  # (N,)

    # Hessian matrix: w_ij = ρ_ij * (1 - ρ_ij) * |ΔNDCG_ij| * mask
    hess_matrix = rho * (1.0 - rho) * delta_ndcg * mask  # (N, N)

    # Net hessian: symmetric, so doc i accumulates from both sides
    hessians = hess_matrix.sum(axis=1) + hess_matrix.sum(axis=0)  # (N,)

    lam_t = torch.tensor(lambdas, dtype=torch.float32, device=scores_sq.device).unsqueeze(1)
    hess_t = torch.tensor(hessians, dtype=torch.float32, device=scores_sq.device).unsqueeze(1)

    return lam_t, hess_t
