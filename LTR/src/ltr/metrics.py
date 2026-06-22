"""
ltr/metrics.py
--------------
NDCG evaluation and paired significance testing.

Functions
---------
compute_dcg         — DCG@k on a single ranked list.
ndcg_at_k           — NDCG@k for a single query (Fix #1).
per_query_ndcg      — Raw per-query NDCG list from a DataLoader.
mean_ndcg           — Mean NDCG@k over all queries in a DataLoader.
paired_significance — Paired t-test p-value between two models (Fix #7).

Fix #1 — Zero-relevance queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
When a query has no relevant documents (IDCG == 0), the original notebooks
skipped it entirely, inflating the reported mean NDCG.  This module assigns
those queries an NDCG of **0.0** and includes them in the denominator, which
produces an honest estimate of ranking quality across the full query set.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
from scipy.stats import ttest_rel


# ─────────────────────────────────────────────────────────────────────────────
# Low-level DCG helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_dcg(relevance_scores, k: int) -> float:
    """
    Discounted Cumulative Gain at position k.

    Parameters
    ----------
    relevance_scores : array-like
        Ground-truth relevance labels in the order they are ranked
        (index 0 = top-ranked document).
    k : int
        Ranking cutoff.

    Returns
    -------
    float
    """
    rel = np.asarray(relevance_scores, dtype=np.float64)[:k]
    if rel.size == 0:
        return 0.0
    denominators = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2.0 ** rel - 1.0) / denominators))


# ─────────────────────────────────────────────────────────────────────────────
# Per-query NDCG (Fix #1)
# ─────────────────────────────────────────────────────────────────────────────

def ndcg_at_k(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    k: int,
    empty_query: str = "zero",
) -> float:
    """
    Normalized Discounted Cumulative Gain at position k for a **single query**.

    Fix #1 — Zero-relevance queries
    --------------------------------
    If the query has no relevant documents (IDCG = 0), the function returns
    **0.0** regardless of the model's ranking.  The caller receives a defined
    value for every query, so zero-relevance queries are always counted in the
    denominator when computing mean NDCG.

    Parameters
    ----------
    true_labels : np.ndarray, shape (num_docs,)
        Ground-truth relevance labels.
    pred_scores : np.ndarray, shape (num_docs,)
        Predicted relevance scores from the model.
    k : int
        NDCG cutoff position.
    empty_query : str
        Strategy for IDCG == 0. Only ``"zero"`` is supported — returns 0.0.

    Returns
    -------
    float in [0, 1].
    """
    true_labels = np.asarray(true_labels, dtype=np.float64)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)

    idcg = compute_dcg(sorted(true_labels, reverse=True), k)

    if idcg == 0.0:
        # Fix #1: assign 0.0 — do NOT skip this query
        return 0.0

    sorted_labels = true_labels[np.argsort(pred_scores)[::-1]]
    return compute_dcg(sorted_labels, k) / idcg


# ─────────────────────────────────────────────────────────────────────────────
# Batch evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collect_scores_and_labels(
    model: torch.nn.Module,
    data_loader,
    device: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """
    Run inference over data_loader and collect per-query predictions and labels.

    Returns
    -------
    (scores_per_qid, labels_per_qid) : two dicts mapping qid → list of floats.
    """
    model.eval()
    scores_per_qid: Dict[str, List[float]] = {}
    labels_per_qid: Dict[str, List[float]] = {}

    with torch.no_grad():
        for batch in data_loader:
            qids_list, feats_list, labels_list = batch

            for qid, feats, labels in zip(qids_list, feats_list, labels_list):
                if not isinstance(feats, torch.Tensor):
                    feats = torch.tensor(feats, dtype=torch.float32)
                feats = feats.to(device)

                pred = model(feats).squeeze().cpu().numpy()
                # Guard against scalar output when a query has exactly 1 document
                if pred.ndim == 0:
                    pred = np.array([float(pred)])

                if isinstance(labels, torch.Tensor):
                    labs = labels.cpu().numpy().astype(np.float64)
                else:
                    labs = np.asarray(labels, dtype=np.float64)

                # Normalise qid to a consistent string key
                if isinstance(qid, torch.Tensor):
                    qkey = str(qid.item())
                elif isinstance(qid, (list, np.ndarray)):
                    qkey = str(np.asarray(qid).flatten()[0])
                else:
                    qkey = str(qid)

                if qkey not in scores_per_qid:
                    scores_per_qid[qkey] = []
                    labels_per_qid[qkey] = []

                scores_per_qid[qkey].extend(pred.tolist())
                labels_per_qid[qkey].extend(labs.tolist())

    return scores_per_qid, labels_per_qid


def per_query_ndcg(
    model: torch.nn.Module,
    data_loader,
    k: int = 10,
    device: str = "cpu",
) -> List[float]:
    """
    Return a raw list of NDCG@k values — one per query in the DataLoader.

    Includes 0.0 for queries with no relevant documents (Fix #1).
    This list is the input expected by ``paired_significance``.

    Parameters
    ----------
    model       : Trained ScoringMLP.
    data_loader : DataLoader (train / val / test).
    k           : NDCG cutoff.
    device      : 'cpu' or 'cuda'.

    Returns
    -------
    List[float], length == total queries in data_loader.
    """
    scores_per_qid, labels_per_qid = _collect_scores_and_labels(
        model, data_loader, device
    )

    results = []
    for qkey in labels_per_qid:
        true_labs = np.array(labels_per_qid[qkey])
        pred_sc   = np.array(scores_per_qid[qkey])
        results.append(ndcg_at_k(true_labs, pred_sc, k=k))

    return results


def mean_ndcg(
    model: torch.nn.Module,
    data_loader,
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    device: str = "cpu",
) -> Dict[int, float]:
    """
    Compute mean NDCG@k for multiple cutoff values simultaneously.

    Parameters
    ----------
    model       : Trained ScoringMLP.
    data_loader : DataLoader (train / val / test).
    k_list      : Tuple of cutoff values.
    device      : 'cpu' or 'cuda'.

    Returns
    -------
    Dict[int, float] mapping each k to its mean NDCG score.
    """
    scores_per_qid, labels_per_qid = _collect_scores_and_labels(
        model, data_loader, device
    )

    ndcg_by_k: Dict[int, List[float]] = {k: [] for k in k_list}

    for qkey in labels_per_qid:
        true_labs = np.array(labels_per_qid[qkey])
        pred_sc   = np.array(scores_per_qid[qkey])
        for k in k_list:
            ndcg_by_k[k].append(ndcg_at_k(true_labs, pred_sc, k=k))

    return {
        k: float(np.mean(vals)) if vals else 0.0
        for k, vals in ndcg_by_k.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance (Fix #7)
# ─────────────────────────────────────────────────────────────────────────────

def paired_significance(
    scores_a: List[float],
    scores_b: List[float],
) -> float:
    """
    Paired two-tailed t-test comparing per-query NDCG arrays of two models.

    Fix #7 — uses ``scipy.stats.ttest_rel`` on the raw per-query NDCG
    vectors (one value per query), which correctly accounts for the
    paired structure of the data (same queries evaluated by both models).

    Parameters
    ----------
    scores_a : List[float]
        Per-query NDCG@k values for model A (from ``per_query_ndcg``).
    scores_b : List[float]
        Per-query NDCG@k values for model B (must be same length as scores_a).

    Returns
    -------
    float
        Two-tailed p-value.  A value < 0.05 indicates a statistically
        significant difference between the two models at the 5% level.

    Raises
    ------
    ValueError
        If the two lists have different lengths.
    """
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)

    if a.shape != b.shape:
        raise ValueError(
            f"scores_a and scores_b must have the same length; "
            f"got {len(a)} and {len(b)}."
        )

    _, p_value = ttest_rel(a, b)
    return float(p_value)
