"""
evaluate.py
-----------
Evaluation utilities for Learning-to-Rank models.

Metric implemented
------------------
NDCG@k  (Normalised Discounted Cumulative Gain at cut-off k)

    DCG@k  = Σ_{i=1}^{k}  (2^{rel_i} − 1) / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

where IDCG@k is the DCG of the ideal (perfectly sorted) ranking.

Reference
---------
Järvelin & Kekäläinen, "Cumulated gain-based evaluation of IR techniques",
ACM TOIS 2002.
"""

import torch
import numpy as np
from typing import Dict, List


def ndcg_at_k(relevance: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k for a single query given a relevance-sorted list.

    Parameters
    ----------
    relevance : 1-D array of ground-truth relevance labels ordered by the
                *model's* predicted scores (highest score first).
    k         : Cut-off position.

    Returns
    -------
    float in [0, 1].
    """
    relevance = np.asarray(relevance, dtype=float)
    k = min(k, len(relevance))

    # DCG of the model ranking
    gains   = 2.0 ** relevance[:k] - 1.0
    discounts = np.log2(np.arange(2, k + 2))   # log2(2), log2(3), ..., log2(k+1)
    dcg     = np.sum(gains / discounts)

    # Ideal DCG (sort by true relevance, descending)
    ideal   = np.sort(relevance)[::-1]
    ideal_gains = 2.0 ** ideal[:k] - 1.0
    idcg    = np.sum(ideal_gains / discounts)

    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_model_ndcg(
    model,
    data_loader,
    k_list: List[int] = (1, 3, 5, 10),
    device: str = "cpu",
) -> Dict[int, float]:
    """
    Evaluate a RankNet model on an entire DataLoader and return NDCG@k scores.

    Parameters
    ----------
    model       : Trained RankNet model (in eval mode after this call).
    data_loader : DataLoader yielding (qids, features_list, labels_list).
    k_list      : List of cut-off positions to evaluate.
    device      : Torch device string or torch.device.

    Returns
    -------
    dict mapping k → mean NDCG@k across all queries.
    """
    model.eval()

    ndcg_scores: Dict[int, List[float]] = {k: [] for k in k_list}

    with torch.no_grad():
        for _, batch_feats, batch_labels in data_loader:
            for feats, labels in zip(batch_feats, batch_labels):
                feats  = feats.to(device)
                labels = labels.numpy()

                # Score all documents for this query
                scores = model(feats).squeeze(-1).cpu().numpy()  # (num_docs,)

                # Sort by predicted score (descending)
                ranked_indices = np.argsort(scores)[::-1]
                ranked_labels  = labels[ranked_indices]

                for k in k_list:
                    ndcg_scores[k].append(ndcg_at_k(ranked_labels, k))

    return {k: float(np.mean(v)) for k, v in ndcg_scores.items()}
