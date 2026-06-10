"""
loss.py
-------
Pairwise ranking loss functions for RankNet.

Two variants are supported:

1. **Standard RankNet Loss (BCE)**
   Binary Cross-Entropy applied to all valid pairs (i, j) where doc i is
   strictly more relevant than doc j.  The target probability is always 1.0
   because we only form pairs where the correct ordering is known.

   Loss = BCE(sigmoid(s_i − s_j), 1)

2. **Fidelity Loss (FRank)**
   A bounded alternative to BCE that uses the geometric mean of the predicted
   probability instead of its logarithm, avoiding extreme gradient magnitudes.

   Loss = 1 − sqrt(sigmoid(s_i − s_j))

Reference
---------
Burges et al., "Learning to Rank using Gradient Descent", ICML 2005.
"""

import torch
import torch.nn.functional as F


def ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    fidelity: bool = False,
) -> torch.Tensor:
    """
    Compute the pairwise ranking loss for a single query group.

    Parameters
    ----------
    scores   : FloatTensor of shape (num_docs, 1) — predicted scores from the model.
    labels   : FloatTensor of shape (num_docs,)   — ground-truth relevance labels.
    fidelity : If False (default) use BCE loss; if True use Fidelity (FRank) loss.

    Returns
    -------
    Scalar loss tensor with gradient tracking.
    """
    # 1. All pairwise score differences:  s_i − s_j  →  (num_docs, num_docs)
    scores_diff = scores - scores.T

    # 2. All pairwise ground-truth label differences
    labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)

    # 3. Keep only pairs where doc i is *strictly* more relevant than doc j
    i_idx, j_idx = torch.where(labels_diff > 0)

    # Edge case: no valid pairs (all docs share the same relevance label)
    if len(i_idx) == 0:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    # 4. Extract score differences for valid pairs
    valid_scores_diff = scores_diff[i_idx, j_idx]

    if not fidelity:
        # ── Standard RankNet: BCE Loss ──────────────────────────────────────
        # Target is always 1.0 because we only form pairs where i > j
        targets = torch.ones_like(valid_scores_diff)
        loss = F.binary_cross_entropy_with_logits(valid_scores_diff, targets)
    else:
        # ── FRank: Fidelity Loss ────────────────────────────────────────────
        # Convert raw score differences to probabilities via Sigmoid
        pred_probs = torch.sigmoid(valid_scores_diff)
        # Fidelity Loss for target = 1:  1 − sqrt(P)
        # A small epsilon inside sqrt prevents NaN if prob == 0
        loss = torch.mean(1.0 - torch.sqrt(pred_probs + 1e-7))

    return loss
