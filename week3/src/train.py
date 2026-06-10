"""
train.py
--------
Training engine for RankNet.

The training loop follows a query-level mini-batch strategy:
  - Each mini-batch contains ``batch_size`` queries.
  - For every query, all documents are scored in one forward pass.
  - The pairwise BCE (or Fidelity) loss is computed for each query
    and averaged across the batch before back-propagation.

Optimisation defaults
---------------------
  Optimizer : Adam
  LR        : 0.001
  Epochs    : 15
"""

import copy
import torch
import torch.optim as optim

from .loss import ranknet_loss


def train_ranknet(
    model,
    train_loader,
    val_loader,
    epochs: int = 15,
    lr: float = 0.001,
    device: str = "cpu",
    verbose: bool = True,
    fidelity: bool = False,
):
    """
    Train a RankNet model with Adam optimisation.

    The best model weights (lowest validation loss) are restored at the end.

    Parameters
    ----------
    model        : RankNet (or any nn.Module with a compatible forward).
    train_loader : DataLoader yielding (qids, features_list, labels_list).
    val_loader   : DataLoader for validation (same format).
    epochs       : Number of training epochs.
    lr           : Learning rate for Adam.
    device       : Torch device string or torch.device.
    verbose      : If True, print per-epoch loss summary.
    fidelity     : If True, use Fidelity Loss instead of BCE.

    Returns
    -------
    (model, train_loss_history, val_loss_history)
        model               — best checkpoint restored
        train_loss_history  — list of avg training losses per epoch
        val_loss_history    — list of avg validation losses per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_model_weights = None

    train_loss_history: list = []
    val_loss_history:   list = []

    for epoch in range(epochs):

        # ══════════════════════════ TRAINING PHASE ════════════════════════════
        model.train()
        total_train_loss = 0.0
        train_queries_count = 0

        for batch_qids, batch_feats, batch_labels in train_loader:
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_queries_in_batch = 0

            for i in range(len(batch_feats)):
                feats  = batch_feats[i].to(device)
                labels = batch_labels[i].to(device)

                scores = model(feats)
                loss   = ranknet_loss(scores, labels, fidelity=fidelity)

                if loss.item() > 0:
                    batch_loss += loss
                    valid_queries_in_batch += 1
                    train_queries_count    += 1

            if valid_queries_in_batch > 0:
                batch_loss = batch_loss / valid_queries_in_batch
                batch_loss.backward()
                optimizer.step()
                total_train_loss += batch_loss.item() * valid_queries_in_batch

        avg_train_loss = total_train_loss / max(1, train_queries_count)
        train_loss_history.append(avg_train_loss)

        # ═════════════════════════ VALIDATION PHASE ═══════════════════════════
        model.eval()
        total_val_loss = 0.0
        val_queries_count = 0

        with torch.no_grad():
            for batch_qids, batch_feats, batch_labels in val_loader:
                for i in range(len(batch_feats)):
                    feats  = batch_feats[i].to(device)
                    labels = batch_labels[i].to(device)

                    scores = model(feats)
                    loss   = ranknet_loss(scores, labels, fidelity=fidelity)

                    if loss.item() > 0:
                        total_val_loss   += loss.item()
                        val_queries_count += 1

        avg_val_loss = total_val_loss / max(1, val_queries_count)
        val_loss_history.append(avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss      = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {avg_val_loss:.4f}"
            )

    # Restore best weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)

    return model, train_loss_history, val_loss_history
