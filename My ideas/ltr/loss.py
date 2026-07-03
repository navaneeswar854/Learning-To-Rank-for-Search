"""
ltr/loss.py
-----------
WeightedRelevanceBCE -- custom loss function for multi-output LTR scoring.

Motivation
----------
When the model has ``output_dim = k`` output neurons and neuron ``i``
represents relevance label ``i``, a plain BCE treats every label boundary
equally.  This loss up-weights higher-relevance boundaries so that
mislabelling a highly relevant document is penalised more than
mislabelling an irrelevant one.

Loss formula (per neuron ``i``, per document)
---------------------------------------------
    L_i = (1 / 2^(k - i)) * [ -t * log(p) - (1 - t) * log(1 - p) ]

where
    k  = number of output neurons (= output_dim)
    i  = neuron index, 0-based  (i = 0, 1, ..., k-1)
    t  = 1 if ground-truth label == i, else 0
    p  = softmax(z)_i              [neuron i output — already a probability]

Note: the model (model.py) now applies Softmax at the output layer, so
``p`` is directly the model's output.  No sigmoid is needed here.

Weight schedule
---------------
    neuron 0  -> weight = 1 / 2^k        (lowest, label "irrelevant")
    neuron 1  -> weight = 1 / 2^(k-1)
    ...
    neuron k-1 -> weight = 1 / 2^1 = 0.5 (highest, most relevant label)

The total per-document loss is the sum of L_i over all k neurons.
The batch loss is the mean of per-document losses.

Integration with model.py
-------------------------
    from ltr.model import ScoringMLP
    from ltr.loss  import WeightedRelevanceBCE

    model     = ScoringMLP(input_dim=46, hidden_dims=[64, 32], output_dim=4)
    criterion = WeightedRelevanceBCE(num_classes=4)

    probs = model(features)               # (num_docs, 4) -- softmax probs
    loss  = criterion(probs, labels)      # labels: LongTensor (num_docs,)
    loss.backward()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedRelevanceBCE(nn.Module):
    """
    Weighted per-neuron Binary Cross-Entropy for multi-label relevance scoring.

    Expects the model to output **softmax probabilities** (as produced by
    ``ScoringMLP`` with its ``nn.Softmax`` output layer).  Each neuron ``i``
    receives a one-vs-rest binary target derived from the integer ground-truth
    label.  The BCE at neuron ``i`` is scaled by ``1 / 2^(k - i)`` so that
    higher-relevance label boundaries contribute more to the total loss.

    Parameters
    ----------
    num_classes : int
        Number of output neurons ``k``.  Must match ``model.output_dim``.
    reduction : str
        How to aggregate the per-document losses across the batch.
        ``'mean'`` (default) divides by the number of documents.
        ``'sum'`` returns the un-normalised total.
        ``'none'`` returns a loss value for each document (shape: ``(N,)``).
    eps : float
        Probabilities are clamped to ``[eps, 1 - eps]`` before taking ``log``
        to prevent ``log(0)`` producing ``-inf``.  Default ``1e-8``.
    use_stable_bce : bool
        When ``True`` (default), uses ``F.binary_cross_entropy`` with
        ``eps``-clamping for numerical safety.
        Set to ``False`` to use the formula letter-for-letter without
        clamping — useful for debugging on small, controlled inputs.

    Raises
    ------
    ValueError
        If ``num_classes < 1`` or ``reduction`` is not one of the three
        supported strings.

    Examples
    --------
    >>> criterion = WeightedRelevanceBCE(num_classes=4)
    >>> probs   = torch.softmax(torch.randn(10, 4), dim=-1)  # model output
    >>> labels  = torch.randint(0, 4, (10,))                 # ground truth
    >>> loss    = criterion(probs, labels)
    >>> loss.backward()

    Verifying the weight schedule:

    >>> criterion.weights   # tensor([0.0625, 0.1250, 0.2500, 0.5000]) for k=4
    """

    def __init__(
        self,
        num_classes: int,
        reduction: str = "mean",
        eps: float = 1e-8,
        use_stable_bce: bool = True,
    ) -> None:
        super().__init__()

        # -- Argument validation ----------------------------------------------
        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError(
                f"num_classes must be a positive integer; got {num_classes!r}."
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction must be 'mean', 'sum', or 'none'; got {reduction!r}."
            )
        if not (0.0 < eps < 1.0):
            raise ValueError(
                f"eps must be a small positive float in (0, 1); got {eps!r}."
            )

        self.num_classes    = num_classes
        self.reduction      = reduction
        self.eps            = eps
        self.use_stable_bce = use_stable_bce

        # -- Pre-compute weight vector ----------------------------------------
        # weight[i] = 1 / 2^(k - i)  for i in 0 .. k-1
        # Registered as a buffer so it moves to the correct device with .to()
        # and is included in state_dict for reproducibility.
        exponents = torch.arange(num_classes, 0, -1, dtype=torch.float32)
        # exponents[i] = k - i  =>  weight[i] = 2^(-(k-i))
        weights = torch.pow(2.0, -exponents)          # shape: (k,)
        self.register_buffer("weights", weights)

    # -- Forward pass ---------------------------------------------------------

    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the weighted relevance BCE loss.

        Parameters
        ----------
        probs : torch.Tensor, shape (N, k)
            Softmax probabilities from the model's output layer.  ``N`` is
            the number of documents in the batch; ``k`` must equal
            ``self.num_classes``.  Each row must sum to 1 and all values
            must be in (0, 1) — guaranteed when the model uses Softmax.
        labels : torch.Tensor, shape (N,)
            Integer ground-truth relevance label for each document.
            Each value must be in ``{0, 1, ..., k-1}``.

        Returns
        -------
        torch.Tensor
            Scalar loss (when ``reduction`` is ``'mean'`` or ``'sum'``), or
            a 1-D tensor of per-document losses (when ``reduction='none'``).

        Raises
        ------
        ValueError
            If ``probs`` or ``labels`` have unexpected shapes.
        """
        if probs.dim() != 2 or probs.size(1) != self.num_classes:
            raise ValueError(
                f"Expected probs of shape (N, {self.num_classes}); "
                f"got {tuple(probs.shape)}."
            )
        if labels.dim() != 1 or labels.size(0) != probs.size(0):
            raise ValueError(
                f"Expected labels of shape ({probs.size(0)},); "
                f"got {tuple(labels.shape)}."
            )

        # -- Build one-hot binary target matrix t: (N, k) --------------------
        # t[n, i] = 1 if labels[n] == i else 0
        targets = F.one_hot(labels.long(), num_classes=self.num_classes).float()

        # -- Compute per-element BCE ------------------------------------------
        if self.use_stable_bce:
            # Clamp probabilities away from 0 and 1 before taking log.
            # p is already a probability (from Softmax) so no sigmoid needed.
            # bce shape: (N, k)
            probs_clamped = probs.clamp(self.eps, 1.0 - self.eps)
            bce = F.binary_cross_entropy(
                probs_clamped, targets, reduction="none"
            )
        else:
            # Exact formula letter-for-letter (no clamping):
            #   L_i = -t * log(p) - (1-t) * log(1-p)
            # p comes directly from the Softmax output — no sigmoid required.
            bce = -(
                targets         * torch.log(probs + self.eps)
                + (1 - targets) * (torch.log(1 - probs + self.eps) + probs)
            )

        # -- Apply per-neuron weights: weights shape (k,) -> broadcast (N, k) -
        weighted_bce = bce * self.weights          # (N, k)

        # -- Sum over neurons -> per-document loss: (N,) ----------------------
        doc_loss = weighted_bce.sum(dim=1)         # (N,)

        # -- Reduce over documents -------------------------------------------
        if self.reduction == "mean":
            return doc_loss.mean()
        elif self.reduction == "sum":
            return doc_loss.sum()
        else:  # "none"
            return doc_loss

    # -- Utility helpers ------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"reduction='{self.reduction}', "
            f"use_stable_bce={self.use_stable_bce}"
        )
