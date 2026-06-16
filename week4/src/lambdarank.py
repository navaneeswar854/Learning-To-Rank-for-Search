"""
lambdarank.py
=============
Gradient computation and training utilities for the LambdaRank approach.

Extracted from: notebooks/LambdaRank.ipynb
Dataset      : LETOR4 / MQ2008
Model        : RankNet — Deep Pointwise Scoring Network
Gradients    : Lambda (λ) gradients weighted by |ΔNDCG|
Metric       : NDCG@K
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Evaluation Metric Helpers ──────────────────────────────────────────────────

def compute_dcg(relevance_scores, k):
    """
    Calculates Discounted Cumulative Gain (DCG) up to position k.

    Parameters
    ----------
    relevance_scores : array-like
        Ordered relevance labels (highest first is not required; the caller
        must sort if an ideal ranking is desired).
    k : int
        Cutoff position.

    Returns
    -------
    float
        DCG value.
    """
    relevance_scores = np.asarray(relevance_scores, dtype=float)[:k]
    if relevance_scores.size == 0:
        return 0.0
    denominators = np.log2(np.arange(2, relevance_scores.size + 2))
    return np.sum((2 ** relevance_scores - 1) / denominators)


def evaluate_model_ndcg(model, data_loader, k_list=[1, 3, 5], device="cpu"):
    """
    Evaluates a RankNet model and returns Mean NDCG scores at multiple cutoffs.

    Processes query groups individually to support variable document counts.

    Parameters
    ----------
    model : nn.Module
        The scoring model to evaluate.
    data_loader : DataLoader
        Query-grouped DataLoader (train, val, or test).
    k_list : list[int]
        List of cutoff positions for NDCG (default: [1, 3, 5]).
    device : str or torch.device
        Device the model resides on.

    Returns
    -------
    dict[int, float]
        Mapping from cutoff k → mean NDCG@k across all queries.
    """
    model.eval()

    qid_to_true_labels = {}
    qid_to_pred_scores = {}

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                qids_list = batch['qids']
                feats_list = batch['feats']
                labels_list = batch['labels']
            else:
                qids_list, feats_list, labels_list = batch

            # Loop through each individual query group inside the mini-batch
            for qid, feats, labels in zip(qids_list, feats_list, labels_list):

                # Ensure the feature group is a proper PyTorch tensor
                if not isinstance(feats, torch.Tensor):
                    feats = torch.tensor(feats, dtype=torch.float32)

                feats = feats.to(device)

                # Forward pass for this single query's documents
                scores = model(feats).squeeze().cpu().numpy()

                # Edge case: if a query has only 1 document, squeeze removes all dimensions
                if scores.ndim == 0:
                    scores = np.array([scores])

                # Clean up labels to a numpy array
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                else:
                    labels = np.array(labels)

                # Extract a single scalar query ID for dictionary mapping
                if isinstance(qid, torch.Tensor):
                    qid_val = qid.cpu().numpy().flatten()[0]
                elif isinstance(qid, (list, np.ndarray)):
                    qid_val = np.array(qid).flatten()[0]
                else:
                    qid_val = qid

                # Initialize tracking buckets if this is a new query ID
                if qid_val not in qid_to_true_labels:
                    qid_to_true_labels[qid_val] = []
                    qid_to_pred_scores[qid_val] = []

                # Collect predictions and ground truths for this query profile
                qid_to_true_labels[qid_val].extend(labels)
                qid_to_pred_scores[qid_val].extend(scores)

    # ─── NDCG Core Metric Aggregation Loop ───
    ndcg_results = {k: [] for k in k_list}

    for qid in qid_to_true_labels.keys():
        true_rels = np.array(qid_to_true_labels[qid])
        pred_scores = np.array(qid_to_pred_scores[qid])

        if len(true_rels) < 2 or np.max(true_rels) == 0:
            continue

        predicted_sort_order = np.argsort(pred_scores)[::-1]
        model_ordered_labels = true_rels[predicted_sort_order]
        ideal_ordered_labels = sorted(true_rels, reverse=True)

        for k in k_list:
            idcg = compute_dcg(ideal_ordered_labels, k)
            dcg = compute_dcg(model_ordered_labels, k)

            if idcg > 0:
                ndcg_results[k].append(dcg / idcg)
            else:
                ndcg_results[k].append(0.0)

    mean_ndcg_scores = {
        k: np.mean(ndcg_results[k]) if ndcg_results[k] else 0.0 for k in k_list
    }

    return mean_ndcg_scores


# ── Lambda Gradient Computation ────────────────────────────────────────────────

def compute_lambda_gradients(scores, labels, k=10):
    """
    Computes per-document LambdaRank gradients (λᵢ) for a single query group.

    For every valid pair (i, j) where document i is more relevant than j:

        λᵢⱼ = ρᵢⱼ × |ΔNDCG_{ij}|

    where  ρᵢⱼ = 1 / (1 + exp(sᵢ − sⱼ))  (the RankNet probability signal).

    The net lambda for each document accumulates over all its pairs:
        λᵢ = −Σ_{j: relᵢ > relⱼ} λᵢⱼ  +  Σ_{j: relⱼ > relᵢ} λᵢⱼ

    A negative λᵢ means "increase the document's score";
    a positive value means "decrease it".

    Parameters
    ----------
    scores : torch.Tensor
        Model output scores for each document in the query group, shape (n,) or (n, 1).
    labels : torch.Tensor
        Ground-truth relevance labels for the query group, shape (n,).
    k : int
        NDCG cutoff position used for |ΔNDCG| weighting (default: 10).

    Returns
    -------
    torch.Tensor
        Lambda gradients, shape (n, 1), on the same device as `scores`.
    """
    scores = scores.squeeze()
    labels_np = labels.cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    N = len(labels_np)

    ranked_order = np.argsort(scores_np)[::-1]   # indices sorted by score descending

    # IDCG
    ideal_labels = np.sort(labels_np)[::-1]
    idcg = compute_dcg(ideal_labels, k)
    if idcg == 0:
        return torch.zeros_like(scores).unsqueeze(1)

    # discount[i] = 1 / log2(i+2) for i = 0 to N-1
    positions = np.arange(1, N + 1)
    discounts = 1.0 / np.log2(positions + 1)

    # Store the current ranks
    rank_of = np.empty(N, dtype=int)
    for pos, doc_idx in enumerate(ranked_order):
        rank_of[doc_idx] = pos

    lambdas = np.zeros(N, dtype=np.float64)

    for i in range(N):
        for j in range(N):
            # only process pairs where document i is more relevant than j
            if labels_np[i] <= labels_np[j]:
                continue

            # |ΔNDCG|
            rank_i = rank_of[i]
            rank_j = rank_of[j]

            gain_i = (2 ** labels_np[i] - 1)
            gain_j = (2 ** labels_np[j] - 1)

            # Only positions within the cutoff k matter
            disc_i = discounts[rank_i] if rank_i < k else 0.0
            disc_j = discounts[rank_j] if rank_j < k else 0.0

            delta_ndcg = abs(
                (gain_i - gain_j) * (disc_i - disc_j)
            ) / idcg

            # RankNet probability for pair i,j
            s_diff = scores_np[i] - scores_np[j]
            rho = 1.0 / (1.0 + np.exp(s_diff))

            # Lambda calculation
            lam = rho * delta_ndcg

            lambdas[i] -= lam
            lambdas[j] += lam

    lambda_tensor = torch.tensor(lambdas, dtype=torch.float32,
                                  device=scores.device).unsqueeze(1)
    return lambda_tensor


# ── Training Loop ──────────────────────────────────────────────────────────────

def train_lambdarank(model, train_loader, val_loader,
                     epochs=15, lr=0.001, k=10, device="cpu", verbose=True):
    """
    Training engine for the LambdaRank approach.

    Uses the efficient O(n)-per-query gradient formulation: one forward pass
    produces scores for all n documents; then λ-gradients are computed and
    used directly in a single backward pass.

    Implements best-model checkpointing based on validation NDCG@k.

    Parameters
    ----------
    model : nn.Module
        The scoring model (e.g. RankNet from pointwise.py).
    train_loader : DataLoader
        Query-grouped DataLoader for training data.
    val_loader : DataLoader
        Query-grouped DataLoader for validation data.
    epochs : int
        Number of training epochs (default: 15).
    lr : float
        Adam learning rate (default: 0.001).
    k : int
        NDCG cutoff used both for lambda weighting and validation metric (default: 10).
    device : str or torch.device
        Device to train on.
    verbose : bool
        If True, prints per-epoch NDCG statistics.

    Returns
    -------
    model : nn.Module
        The model loaded with the best-validation-NDCG weights.
    train_ndcg_history : list[float]
        Per-epoch average Train NDCG@k.
    val_ndcg_history : list[float]
        Per-epoch average Validation NDCG@k.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_ndcg = 0.0
    best_weights = None
    train_ndcg_history = []
    val_ndcg_history = []

    for epoch in range(epochs):
        # ═════════════════ TRAINING PHASE ═════════════════
        model.train()
        for batch_qids, batch_feats, batch_labels in train_loader:
            for feats, labels in zip(batch_feats, batch_labels):
                feats = feats.to(device)
                labels = labels.to(device)

                # One forward pass — get scores for all n docs
                optimizer.zero_grad()
                scores = model(feats)
                scores.retain_grad()

                # Compute λᵢ for each doc
                with torch.no_grad():
                    lambdas = compute_lambda_gradients(scores, labels, k=k)

                # n backprops through the network using lambdas
                scores.backward(lambdas)
                optimizer.step()

        # ═════════════════ VALIDATION PHASE ═════════════════
        model.eval()

        val_ndcg = evaluate_model_ndcg(
            model, val_loader, k_list=[k], device=device
        )[k]

        train_ndcg = evaluate_model_ndcg(
            model, train_loader, k_list=[k], device=device
        )[k]

        train_ndcg_history.append(train_ndcg)
        val_ndcg_history.append(val_ndcg)

        if val_ndcg > best_val_ndcg:
            best_val_ndcg = val_ndcg
            best_weights = copy.deepcopy(model.state_dict())

        if verbose:
            print(f"Epoch {epoch+1:02d}/{epochs} | "
                  f"Train NDCG@{k}: {train_ndcg:.4f} | "
                  f"Val NDCG@{k}: {val_ndcg:.4f}")

    # Restoring the best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_ndcg_history, val_ndcg_history
