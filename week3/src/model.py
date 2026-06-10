"""
model.py
--------
RankNet neural network architecture.

RankNet is a pairwise Learning-to-Rank model introduced by Burges et al. (2005).
The neural network itself is a **pointwise scorer** — it maps a single document's
feature vector to a single relevance score.  The pairwise ranking logic lives in
the loss function (see loss.py).

Supported architecture variants
--------------------------------
``linear``
    Pure linear baseline: 46 → 1  (no hidden layers, no activations).

``baseline``
    Standard MLP without regularisation: 46 → 64 → 32 → 1 with ReLU activations.

``regularized``  *(default)*
    Same MLP with 20 % Dropout after each hidden layer.
    This is the final production architecture used in the experiments.

``deep``
    Over-parameterised deep network for ablation: 46 → 128 → 64 → 32 → 16 → 1
    with ReLU + Dropout(0.2) after each hidden layer.
"""

import torch.nn as nn


class RankNet(nn.Module):
    """
    Pointwise document scorer used as the backbone of RankNet.

    Parameters
    ----------
    input_dim         : Number of input features per document (default: 46 for MQ2008).
    hidden_dim        : Base hidden dimension (default: 64).
    architecture_type : One of ``'linear'``, ``'baseline'``, ``'regularized'``, ``'deep'``.
    """

    def __init__(
        self,
        input_dim: int = 46,
        hidden_dim: int = 64,
        architecture_type: str = "regularized",
    ) -> None:
        super().__init__()

        if architecture_type == "linear":
            # Pure linear baseline: no activations, no hidden layers (46 → 1)
            self.scorer = nn.Linear(input_dim, 1)

        elif architecture_type == "baseline":
            # Standard MLP — no regularisation (46 → 64 → 32 → 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        elif architecture_type == "regularized":
            # Standard MLP with 20 % Dropout (46 → 64 → 32 → 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim // 2, 1),
            )

        elif architecture_type == "deep":
            # Over-parameterised deep network for ablation (46 → 128 → 64 → 32 → 16 → 1)
            self.scorer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim // 4, 1),
            )

        else:
            raise ValueError(
                f"Unknown architecture_type '{architecture_type}'. "
                "Choose from: 'linear', 'baseline', 'regularized', 'deep'."
            )

    def forward(self, x):
        """
        Parameters
        ----------
        x : FloatTensor of shape (num_docs, input_dim)

        Returns
        -------
        FloatTensor of shape (num_docs, 1) — relevance scores
        """
        return self.scorer(x)
