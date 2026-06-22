"""
ltr — Learning-to-Rank package for LETOR4 / MQ2008.

Public API
----------
from ltr.data    import load_fold
from ltr.models  import ScoringMLP
from ltr.losses  import pointwise_mse, ranknet_loss, lambda_gradients
from ltr.metrics import ndcg_at_k, per_query_ndcg, mean_ndcg, paired_significance
from ltr.train   import train, train_multiseed, set_seed
from ltr.evaluate import cross_fold_eval
"""

from .data     import load_fold
from .models   import ScoringMLP
from .losses   import pointwise_mse, ranknet_loss, lambda_gradients
from .metrics  import ndcg_at_k, per_query_ndcg, mean_ndcg, paired_significance
from .train    import train, train_multiseed, set_seed
from .evaluate import cross_fold_eval

__all__ = [
    "load_fold",
    "ScoringMLP",
    "pointwise_mse",
    "ranknet_loss",
    "lambda_gradients",
    "ndcg_at_k",
    "per_query_ndcg",
    "mean_ndcg",
    "paired_significance",
    "train",
    "train_multiseed",
    "set_seed",
    "cross_fold_eval",
]
