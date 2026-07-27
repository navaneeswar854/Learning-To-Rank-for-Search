"""
src — Shared utility library for Learning-to-Rank experiments.

Public API
----------
from src.data    import load_fold
from src.models  import ScoringMLP
from src.losses  import pointwise_mse, ranknet_loss, lambda_gradients
from src.metrics import ndcg_at_k, per_query_ndcg, mean_ndcg, paired_significance
from src.train   import train, train_multiseed, train_lambdamart_multiseed, set_seed
from src.evaluate import cross_fold_eval, cross_fold_eval_lambdamart
from src.lambdamart import LambdaMART
"""

from .data     import load_fold
from .models   import ScoringMLP
from .losses   import pointwise_mse, ranknet_loss, lambda_gradients
from .metrics  import ndcg_at_k, per_query_ndcg, mean_ndcg, paired_significance
from .train    import train, train_multiseed, train_lambdamart_multiseed, set_seed
from .evaluate import cross_fold_eval, cross_fold_eval_lambdamart
from .lambdamart import LambdaMART

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
    "train_lambdamart_multiseed",
    "set_seed",
    "cross_fold_eval",
    "cross_fold_eval_lambdamart",
    "LambdaMART",
]
