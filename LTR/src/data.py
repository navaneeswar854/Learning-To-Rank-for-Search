"""
ltr/data.py
-----------
LETOR libsvm data loading, query grouping, and DataLoader construction.

The standard MQ2008 dataset uses the libsvm format::

    <relevance> qid:<query_id> 1:<f1> 2:<f2> ... 46:<f46> #docid=...

Each DataLoader item is a tuple: (qid, features_tensor, labels_tensor),
where features_tensor is (num_docs, 46) and labels_tensor is (num_docs,).
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LETORQueryDataset(Dataset):
    """
    Parses a single LETOR libsvm split file and groups documents by Query ID.

    Each ``__getitem__`` returns::

        (qid: str, features: FloatTensor[num_docs, 46], labels: FloatTensor[num_docs])

    Parameters
    ----------
    filepath : str
        Absolute or relative path to ``train.txt``, ``vali.txt``, or ``test.txt``.
    """

    def __init__(self, filepath: str) -> None:
        self.queries = []
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        current_qid = None
        current_features = []
        current_labels = []

        with open(filepath, "r") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(filepath)}", leave=False):
                parts = line.strip().split()
                if not parts:
                    continue

                label = int(parts[0])
                qid = parts[1].split(":")[1]
                # Parse exactly 46 features (indices 1–46 in libsvm format)
                features = [float(x.split(":")[1]) for x in parts[2:48]]

                # Flush current query when qid changes
                if current_qid is not None and qid != current_qid:
                    if current_labels:
                        self.queries.append((
                            current_qid,
                            torch.tensor(current_features, dtype=torch.float32),
                            torch.tensor(current_labels, dtype=torch.float32),
                        ))
                    current_features = []
                    current_labels = []

                current_qid = qid
                current_features.append(features)
                current_labels.append(label)

        # Flush the last query in the file
        if current_qid is not None and current_labels:
            self.queries.append((
                current_qid,
                torch.tensor(current_features, dtype=torch.float32),
                torch.tensor(current_labels, dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


def _collate_fn(batch):
    """
    Collate variable-length query groups into parallel lists.

    Returns ``(qids, features_list, labels_list)`` where each element
    is a list with one entry per query in the mini-batch.
    """
    qids = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return qids, features, labels


def load_fold(
    base_path: str = "/content/MQ2008",
    fold_num: int = 1,
    batch_size: int = 4,
):
    """
    Load Train, Validation, and Test DataLoaders for a given LETOR fold.

    Uses the standard author-provided test splits (Fold1–Fold5), so results
    are directly comparable to published papers.

    Parameters
    ----------
    base_path : str
        Path to the directory containing ``Fold1/`` … ``Fold5/`` subdirectories.
        Defaults to the standard Colab extraction path ``/content/MQ2008``.
    fold_num : int
        Which fold to load (1–5).
    batch_size : int
        Number of queries per mini-batch.

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    """
    fold_dir = os.path.join(base_path, f"Fold{fold_num}")

    train_ds = LETORQueryDataset(os.path.join(fold_dir, "train.txt"))
    val_ds   = LETORQueryDataset(os.path.join(fold_dir, "vali.txt"))
    test_ds  = LETORQueryDataset(os.path.join(fold_dir, "test.txt"))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn
    )

    print(f"  Fold {fold_num}: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test queries")

    return train_loader, val_loader, test_loader
