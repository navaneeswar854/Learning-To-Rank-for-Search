"""
ltr/model_pointwise.py
----------------------
PointwiseScorer -- a single-output MLP for custom lambda-gradient training.

Key difference from model.py (ScoringMLP)
------------------------------------------
ScoringMLP (model.py)        : output_dim = k, final activation = Sigmoid
                               Outputs a probability vector (0, 1)^k.

PointwiseScorer (this file)  : output_dim = 1, NO final activation.
                               Outputs a single unbounded real-valued score
                               per document, i.e. s in (-inf, +inf).

Why no output activation?
--------------------------
When injecting custom lambda gradients via scores.backward(gradient=lambdas),
the gradient must be able to push the score in either direction without
hitting a saturation ceiling.  Sigmoid saturates near 0 and 1, killing the
gradient (vanishing gradient problem).  A linear output keeps the gradient
signal clean and proportional to your lambda values.

Architecture
------------
    Input (input_dim)
      +-- [Linear -> ReLU -> (Dropout)] x len(hidden_dims)
           +-- Linear -> scalar score (no activation)

Usage
-----
    from ltr.model_pointwise import PointwiseScorer

    model = PointwiseScorer(input_dim=111, hidden_dims=[64, 32], dropout=0.1)
    features = torch.randn(120, 111)   # 120 candidate documents
    scores   = model(features)         # shape (120,) -- one score per doc
"""

from typing import List

import torch
import torch.nn as nn


class PointwiseScorer(nn.Module):
    """
    Single-output MLP that maps a document feature vector to one unbounded
    real-valued relevance score.

    Designed specifically for custom lambda-gradient training where you
    compute the gradient for each document manually and inject it via::

        scores = model(feats)                    # (N,)  range: (0, 1)
        lambdas = your_lambda_fn(scores, labels) # (N,)
        scores.backward(gradient=lambdas)
        optimizer.step()

    Output range
    ------------
    The final layer is followed by Sigmoid, so all outputs are in (0, 1).
    This gives a natural, bounded relevance score per document.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input feature vector.
        46 for MQ2008, 111 for MSLR-WEB10K (after feature dropping).
    hidden_dims : list of int
        Number of neurons in each hidden layer (in order).
        Pass an empty list [] for a pure linear model.
    dropout : float
        Dropout probability applied after each hidden ReLU.
        Set to 0.0 to disable. Must be in [0, 1).

    Output
    ------
    forward() returns a 1-D tensor of shape (N,) -- one raw score per document.
    There is no sigmoid, softmax, or any other output activation.

    Examples
    --------
    >>> model = PointwiseScorer(input_dim=111, hidden_dims=[64, 32])
    >>> feats  = torch.randn(120, 111)
    >>> scores = model(feats)
    >>> scores.shape
    torch.Size([120])
    """

    def __init__(
        self,
        input_dim: int = 111,
        hidden_dims: List[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # -- Argument validation ----------------------------------------------
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError(
                f"input_dim must be a positive integer; got {input_dim!r}."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(
                f"dropout must be in [0, 1); got {dropout!r}."
            )

        if hidden_dims is None:
            hidden_dims = [64, 32]

        # -- Build hidden layer stack -----------------------------------------
        layers: List[nn.Module] = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = dim

        # Final linear projection to a single scalar, followed by Sigmoid.
        # Output range: (0, 1)  -- a soft probability-like relevance score.
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.scorer = nn.Sequential(*layers)

        # -- Store hyperparameters for repr / checkpointing -------------------
        self.input_dim   = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout     = dropout

    # -- Forward pass ---------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score a batch of documents.

        Parameters
        ----------
        x : torch.Tensor, shape (N, input_dim)

        Returns
        -------
        torch.Tensor, shape (N,)
            One unbounded relevance score per document.
            Squeeze is applied internally so callers never need to .squeeze().
        """
        return self.scorer(x).squeeze(-1)   # (N, 1) -> (N,)

    # -- Utility helpers ------------------------------------------------------

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"hidden_dims={self.hidden_dims}, "
            f"dropout={self.dropout})"
        )
