"""
ltr -- Learning-to-Rank package
================================
A clean, modular LTR library built on PyTorch.

Modules
-------
ltr.model    : ScoringMLP           - flexible MLP scorer
ltr.loss     : WeightedRelevanceBCE - custom weighted per-neuron BCE loss
ltr.data     : load_fold            - LETOR / MQ2008 data loading
ltr.metrics  : mean_ndcg etc.       - NDCG evaluation & significance tests
ltr.train    : train, TrainConfig   - end-to-end training with early stopping

Quick start
-----------
    from ltr import ScoringMLP, WeightedRelevanceBCE, load_fold, train, TrainConfig

    model, train_hist, val_hist = train(TrainConfig(
        data_path   = "/content/MQ2008",
        fold        = 1,
        hidden_dims = [64, 32],
        num_classes = 3,
        epochs      = 100,
        patience    = 10,
    ))
"""

from ltr.model   import ScoringMLP
from ltr.loss    import WeightedRelevanceBCE
from ltr.data    import LETORQueryDataset, load_fold
from ltr.metrics import (
    compute_dcg,
    ndcg_at_k,
    per_query_ndcg,
    mean_ndcg,
    paired_significance,
)
from ltr.train   import TrainConfig, train

__version__ = "0.1.0"

__all__ = [
    # Model
    "ScoringMLP",
    # Loss
    "WeightedRelevanceBCE",
    # Data
    "LETORQueryDataset",
    "load_fold",
    # Metrics
    "compute_dcg",
    "ndcg_at_k",
    "per_query_ndcg",
    "mean_ndcg",
    "paired_significance",
    # Training
    "TrainConfig",
    "train",
]
