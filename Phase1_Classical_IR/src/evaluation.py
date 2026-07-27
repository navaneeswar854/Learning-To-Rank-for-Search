"""
src/evaluation.py
-----------------
Shared evaluation utilities for the Week 2 Information Retrieval notebooks.

Provides a single, authoritative implementation of the three core IR metrics
used across all experiments:
    - Precision@k
    - Recall@k
    - NDCG@k  (Normalized Discounted Cumulative Gain)
"""

import math


def evaluate_rankings(rankings: dict, qrels: dict, k: int = 10) -> tuple:
    """
    Evaluate a set of ranked retrieval results against ground-truth relevance.

    Parameters
    ----------
    rankings : dict
        { query_id : [(doc_id, score), ...] }
        Each value is a ranked list of (document_id, score) pairs, ordered
        from most to least relevant. Only the top-k entries are used.

    qrels : dict
        { query_id : { doc_id : relevance_score } }
        Ground-truth relevance judgements. A relevance_score > 0 means relevant.

    k : int, optional (default=10)
        Cutoff depth for all metrics.

    Returns
    -------
    tuple of float
        (precision_at_k, recall_at_k, ndcg_at_k)
        Each value is the macro-average across all queries that appear in qrels.

    Notes
    -----
    Metric definitions:
    - Precision@k  : fraction of the top-k retrieved docs that are relevant.
    - Recall@k     : fraction of all relevant docs that appear in the top-k.
    - NDCG@k       : DCG@k / IDCG@k, where DCG weights relevant docs by their
                     rank position using a log2 discount. IDCG is the ideal
                     (best possible) DCG for this query's relevance judgements.
    """
    total_queries  = 0
    sum_precision  = 0.0
    sum_recall     = 0.0
    sum_ndcg       = 0.0

    for query_id, true_docs in qrels.items():
        if query_id not in rankings:
            continue

        # Count truly relevant documents (score > 0)
        total_relevant = sum(1 for s in true_docs.values() if s > 0)
        if total_relevant == 0:
            continue

        total_queries += 1
        predicted_docs = [doc_id for doc_id, _ in rankings[query_id][:k]]

        relevant_retrieved = 0
        dcg = 0.0

        # ── Step 1: Precision, Recall, DCG ─────────────────────────────────
        for rank_idx, doc_id in enumerate(predicted_docs):
            if doc_id in true_docs and true_docs[doc_id] > 0:
                relevant_retrieved += 1
                relevance_score = true_docs[doc_id]
                # rank_idx is 0-based → rank 1 divides by log2(2) = 1
                dcg += relevance_score / math.log2(rank_idx + 2)

        precision_at_k = relevant_retrieved / k
        recall_at_k    = relevant_retrieved / total_relevant

        # ── Step 2: Ideal DCG (IDCG) ───────────────────────────────────────
        ideal_scores = sorted(true_docs.values(), reverse=True)
        idcg = sum(
            rel / math.log2(rank_idx + 2)
            for rank_idx, rel in enumerate(ideal_scores[:k])
        )

        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0

        sum_precision += precision_at_k
        sum_recall    += recall_at_k
        sum_ndcg      += ndcg_at_k

    if total_queries == 0:
        return 0.0, 0.0, 0.0

    return (
        sum_precision / total_queries,
        sum_recall    / total_queries,
        sum_ndcg      / total_queries,
    )


def print_results(results: dict, k: int = 10) -> None:
    """
    Pretty-print a leaderboard of evaluation results.

    Parameters
    ----------
    results : dict
        { model_name : (precision, recall, ndcg) }
    k : int
        The cutoff used (for display only).
    """
    col_w = max(len(name) for name in results) + 2
    header = f"{'Model':<{col_w}} {'P@' + str(k):>10}  {'R@' + str(k):>10}  {'NDCG@' + str(k):>10}"
    print(header)
    print("-" * len(header))
    for name, (p, r, n) in results.items():
        print(f"{name:<{col_w}} {p:>10.4f}  {r:>10.4f}  {n:>10.4f}")
