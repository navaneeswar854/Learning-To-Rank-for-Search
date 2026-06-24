"""
ltr/evaluate.py
---------------
Cross-fold evaluation logic using the standard LETOR author-provided folds.

Public API
----------
cross_fold_eval — Evaluate a model across all 5 LETOR folds with multi-seed
                  training, returning per-fold and overall Mean ± Std results.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .data  import load_fold
from .train import train_multiseed, train_lambdamart_multiseed


def cross_fold_eval(
    model_fn: Callable,
    mode: str,
    base_path: str = "/content/MQ2008",
    folds: Tuple[int, ...] = (1, 2, 3, 4, 5),
    seeds: Tuple[int, ...] = (42, 123, 456),
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    batch_size: int = 4,
    device: str = "cpu",
    verbose: bool = False,
    **train_kwargs,
) -> Dict:
    """
    Cross-fold evaluation using the standard LETOR author-provided splits.

    We use the **exact 5 test splits provided by the dataset authors** rather
    than shuffling and creating our own cross-validation splits.  This ensures
    results are directly comparable to published LETOR benchmark papers.

    For each fold, ``train_multiseed`` is called with the provided seeds,
    and the per-seed test NDCG scores are collected.  The function returns
    both per-fold summaries and a global aggregate across all folds × seeds.

    Parameters
    ----------
    model_fn   : Zero-argument callable returning a fresh ``ScoringMLP``
                 instance, e.g. ``lambda: ScoringMLP(46, [64, 32], 0.2)``.
    mode       : ``'pointwise'``, ``'ranknet'``, or ``'lambdarank'``.
    base_path  : Path to MQ2008 directory containing ``Fold1/`` … ``Fold5/``.
    folds      : Which fold numbers to include.  Default: all 5.
    seeds      : Random seeds for multi-seed training (Fix #8).
    k_list     : NDCG cutoffs to compute.
    batch_size : Queries per mini-batch.
    device     : ``'cpu'`` or ``'cuda'``.
    verbose    : Pass to ``train()`` — print per-epoch metrics if True.
    **train_kwargs : Extra keyword arguments forwarded to ``train()``.

    Returns
    -------
    dict with keys:
        ``'fold_results'`` — list of per-fold dicts, each containing:
            ``'fold'``     — fold number.
            ``'summary'``  — ``{k: {'mean': float, 'std': float}}``.
            ``'per_seed'`` — list of ``{k: score}`` dicts per seed.
        ``'overall'``  — global ``{k: {'mean': float, 'std': float}}``
                         aggregated across all folds and seeds.
    """
    fold_results: List[Dict] = []

    for fold_num in folds:
        print(f"\n{'═' * 55}")
        print(f"  FOLD {fold_num} / {len(folds)}")
        print(f"{'═' * 55}")

        train_loader, val_loader, test_loader = load_fold(
            base_path=base_path,
            fold_num=fold_num,
            batch_size=batch_size,
        )

        result = train_multiseed(
            model_fn=model_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            mode=mode,
            seeds=seeds,
            k_list=k_list,
            device=device,
            verbose=verbose,
            **train_kwargs,
        )

        fold_results.append({
            "fold":     fold_num,
            "summary":  result["summary"],
            "per_seed": result["per_seed"],
        })

        # Print per-fold summary
        for k in k_list:
            s = result["summary"][k]
            print(f"  NDCG@{k:<3}: {s['mean']:.4f} ± {s['std']:.4f}")

    # ── Global aggregation across all folds × seeds ───────────────────────
    overall: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        all_values: List[float] = []
        for fr in fold_results:
            all_values.extend([ps[k] for ps in fr["per_seed"]])
        overall[k] = {
            "mean": float(np.mean(all_values)),
            "std":  float(np.std(all_values)),
        }

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  OVERALL RESULTS ({mode.upper()})")
    print(f"{'═' * 55}")
    for k in k_list:
        o = overall[k]
        print(f"  NDCG@{k:<3}: {o['mean']:.4f} ± {o['std']:.4f}")
    print(f"{'═' * 55}\n")

    return {
        "fold_results": fold_results,
        "overall":      overall,
    }


def cross_fold_eval_lambdamart(
    model_fn: Callable,
    base_path: str = "/content/MQ2008",
    folds: Tuple[int, ...] = (1, 2, 3, 4, 5),
    seeds: Tuple[int, ...] = (42, 123, 456),
    k_list: Tuple[int, ...] = (1, 3, 5, 10),
    batch_size: int = 4,
    device: str = "cpu",
    verbose: bool = False,
    **train_kwargs,
) -> Dict:
    """
    Cross-fold evaluation specifically for LambdaMART.
    """
    fold_results: List[Dict] = []

    for fold_num in folds:
        print(f"\n{'═' * 55}")
        print(f"  FOLD {fold_num} / {len(folds)}")
        print(f"{'═' * 55}")

        train_loader, val_loader, test_loader = load_fold(
            base_path=base_path,
            fold_num=fold_num,
            batch_size=batch_size,
        )

        result = train_lambdamart_multiseed(
            model_fn=model_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            seeds=seeds,
            k_list=k_list,
            device=device,
            verbose=verbose,
            **train_kwargs,
        )

        fold_results.append({
            "fold":     fold_num,
            "summary":  result["summary"],
            "per_seed": result["per_seed"],
        })

        # Print per-fold summary
        for k in k_list:
            s = result["summary"][k]
            print(f"  NDCG@{k:<3}: {s['mean']:.4f} ± {s['std']:.4f}")

    # ── Global aggregation across all folds × seeds ───────────────────────
    overall: Dict[int, Dict[str, float]] = {}
    for k in k_list:
        all_values: List[float] = []
        for fr in fold_results:
            all_values.extend([ps[k] for ps in fr["per_seed"]])
        overall[k] = {
            "mean": float(np.mean(all_values)),
            "std":  float(np.std(all_values)),
        }

    # ── Final summary table ───────────────────────────────────────────────
    print(f"\n{'═' * 55}")
    print(f"  OVERALL RESULTS (LAMBDAMART)")
    print(f"{'═' * 55}")
    for k in k_list:
        o = overall[k]
        print(f"  NDCG@{k:<3}: {o['mean']:.4f} ± {o['std']:.4f}")
    print(f"{'═' * 55}\n")

    return {
        "fold_results": fold_results,
        "overall":      overall,
    }
