"""
ltr/model.py
------------
Unified ScoringMLP -- the single pointwise scoring network used across
all three LTR modes (Pointwise, RankNet, LambdaRank).

The number of input neurons, hidden neurons (per layer), and output neurons
are all supplied as constructor arguments, giving full architectural control
with no hard-coded magic numbers.

Canonical configurations for MQ2008 (46 features)
---------------------------------------------------
Linear baseline : ``ScoringMLP(input_dim=46, hidden_dims=[],             output_dim=1, dropout=0.0)``
Baseline (no reg): ``ScoringMLP(input_dim=46, hidden_dims=[64, 32],      output_dim=1, dropout=0.0)``
Regularized     : ``ScoringMLP(input_dim=46, hidden_dims=[64, 32],       output_dim=1, dropout=0.2)``
Deep (ablation) : ``ScoringMLP(input_dim=46, hidden_dims=[128,64,32,16], output_dim=1, dropout=0.2)``

Compatibility
-------------
``output_dim`` defaults to 1 so that existing callers (``metrics.py``,
training loops) that rely on a *scalar* score per document continue to
work without modification.  Set ``output_dim > 1`` only when the task
requires a multi-dimensional output head.
"""

from typing import List

import torch
import torch.nn as nn


class ScoringMLP(nn.Module):
    """
    Flexible Multi-Layer Perceptron for document relevance scoring.

    The network maps a document's feature vector to a relevance score
    (or a multi-dimensional output when ``output_dim > 1``).  The same
    architecture is used as-is for all three LTR training objectives:

    * **Pointwise** -- MSE against integer relevance grades.
    * **Pairwise / RankNet** -- Binary cross-entropy on document pairs.
    * **LambdaRank** -- Lambda-gradient updates on document pairs.

    Architecture
    ------------
    ::

        Input (input_dim)
          +-- [Linear -> ReLU -> (Dropout)] x len(hidden_dims)
               +-- Linear -> Output (output_dim)

    An empty ``hidden_dims`` list collapses the network to a pure linear
    projection from ``input_dim`` to ``output_dim`` with no activations.

    Parameters
    ----------
    input_dim : int
        Number of input neurons -- dimensionality of one document's
        feature vector.  46 for LETOR4 / MQ2008, 136 for MSLR-WEB10K.
    hidden_dims : list of int
        Number of neurons in each hidden layer, in order from input to
        output.  An empty list produces a linear model.
    output_dim : int
        Number of output neurons.  Defaults to 1 (scalar relevance score),
        which is the correct setting for all standard LTR objectives.
    dropout : float
        Dropout probability applied **after each hidden layer's ReLU**.
        Must be in [0, 1).  Set to 0.0 to disable dropout entirely.

    Raises
    ------
    ValueError
        If ``input_dim`` or ``output_dim`` is not a positive integer, or
        if ``dropout`` is not in the half-open interval [0, 1).

    Examples
    --------
    Standard scoring head for MQ2008:

    >>> model = ScoringMLP(input_dim=46, hidden_dims=[64, 32], dropout=0.2)
    >>> features = torch.randn(20, 46)   # 20 candidate documents
    >>> scores = model(features)         # -> shape (20, 1)

    Linear (no hidden layers) baseline:

    >>> linear_model = ScoringMLP(input_dim=46, hidden_dims=[], output_dim=1)
    >>> scores = linear_model(features)  # -> shape (20, 1)

    Custom deep architecture:

    >>> deep_model = ScoringMLP(
    ...     input_dim=136,
    ...     hidden_dims=[256, 128, 64, 32],
    ...     output_dim=1,
    ...     dropout=0.3,
    ... )
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dims: List[int] = None,
        output_dim: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # -- Argument validation ----------------------------------------------
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError(
                f"input_dim must be a positive integer; got {input_dim!r}."
            )
        if not isinstance(output_dim, int) or output_dim < 1:
            raise ValueError(
                f"output_dim must be a positive integer; got {output_dim!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1); got {dropout!r}."
            )

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # -- Build layer stack ------------------------------------------------
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim

        # Final projection head — linear then sigmoid so outputs are
        # probabilities in the range [0, 1].
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.scorer = nn.Sequential(*layers)

        # -- Store hyperparameters for repr / serialisation -------------------
        self.input_dim   = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim  = output_dim
        self.dropout     = dropout

    # -- Forward pass ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of documents.

        Parameters
        ----------
        x : torch.Tensor, shape (num_docs, input_dim)
            Feature matrix for a set of candidate documents.

        Returns
        -------
        torch.Tensor, shape (num_docs, output_dim)
            Relevance scores.  When ``output_dim=1`` (the default), callers
            typically squeeze the last dimension with ``.squeeze(-1)`` or
            ``.squeeze()`` before computing losses or NDCG.
        """
        return self.scorer(x)

    # -- Utility helpers ------------------------------------------------------

    def count_parameters(self) -> int:
        """
        Return the total number of **trainable** parameters in the model.

        Useful for logging and sanity-checking ablation experiments.

        Returns
        -------
        int
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, "
            f"dropout={self.dropout}"
            f")"
        )
