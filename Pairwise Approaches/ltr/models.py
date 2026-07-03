"""
ltr/models.py
-------------
Unified ScoringMLP — the single pointwise scoring network used across
all three LTR modes (Pointwise, RankNet, LambdaRank).

Replaces the ``RankNet`` class with its four-way ``architecture_type``
string selector. Depth and regularization are now controlled by explicit
constructor arguments, making the architecture transparent and auditable.

Canonical configurations for MQ2008 (46 features)
---------------------------------------------------
Linear baseline : ``ScoringMLP(46, hidden_dims=[],           dropout=0.0)``
Baseline (no reg): ``ScoringMLP(46, hidden_dims=[64, 32],    dropout=0.0)``
Regularized     : ``ScoringMLP(46, hidden_dims=[64, 32],     dropout=0.2)``
Deep (ablation) : ``ScoringMLP(46, hidden_dims=[128,64,32,16], dropout=0.2)``
"""

from typing import List
import torch
import torch.nn as nn


class ScoringMLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron for document relevance scoring.

    The network maps a single document's feature vector to a scalar
    relevance score. The same architecture is used as-is for all three
    LTR training objectives (pointwise MSE, pairwise BCE, lambda gradients).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector. 46 for LETOR4 / MQ2008.
    hidden_dims : list of int
        Hidden layer sizes in order. An empty list produces a pure linear
        model (no hidden layers, no activations).
    dropout : float
        Dropout probability applied **after each hidden layer's ReLU**.
        Set to 0.0 to disable dropout entirely.

    Examples
    --------
    >>> model = ScoringMLP(input_dim=46, hidden_dims=[64, 32], dropout=0.2)
    >>> scores = model(features_tensor)   # (num_docs, 46) → (num_docs, 1)
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim

        # Final scoring head — single scalar output per document
        layers.append(nn.Linear(prev_dim, 1))

        self.scorer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (num_docs, input_dim)

        Returns
        -------
        torch.Tensor, shape (num_docs, 1)
        """
        return self.scorer(x)
