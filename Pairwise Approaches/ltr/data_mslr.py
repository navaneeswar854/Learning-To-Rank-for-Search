"""
ltr/data_mslr.py
----------------
MSLR-WEB10K libsvm data loading, query grouping, and DataLoader construction.

The MSLR-WEB10K dataset uses the libsvm format:

    <relevance> qid:<query_id> 1:<f1> 2:<f2> ... 136:<f136>

Preprocessing applied:
  1. Drop 25 highly-correlated features identified in the EDA notebook.
  12. Per-query z-score normalization. Each query's features are normalized
     to mean 0 and std 1 individually. Constant features are left at 0.

Each DataLoader item is a tuple: (qid, features_tensor, labels_tensor),
where features_tensor is (num_docs, 111) and labels_tensor is (num_docs,).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ── Features to drop (1-indexed, as they appear in the libsvm file) ──────────
# These 25 features were identified as having pairwise correlation > 0.98
# with another feature that has a higher correlation to the relevance label.
# Identified via EDA in the MSLR.ipynb notebook.
COLS_TO_DROP = {
    1, 15, 16, 17, 19, 21, 22, 25, 26, 31, 36, 45,
    71, 76, 81, 86, 95, 111, 112, 113, 114, 116, 119, 121, 125
}

# Total features in MSLR = 136. After dropping 25, we have 111.
NUM_TOTAL_FEATURES = 136
KEEP_INDICES = [i for i in range(1, NUM_TOTAL_FEATURES + 1) if i not in COLS_TO_DROP]
NUM_FEATURES = len(KEEP_INDICES)  # 111


class MSLRQueryDataset(Dataset):
    """
    Parses a single MSLR-WEB10K libsvm split file and groups documents by
    Query ID. Applies feature dropping and global z-score normalization.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to train.txt, vali.txt, or test.txt.
    normalize : bool
        If True (default), apply per-query z-score normalization.
        Set to False for tree-based models (e.g. LambdaMART) that are
        scale-invariant and process documents across all queries jointly.
    """

    def __init__(self, filepath: str, normalize: bool = True) -> None:
        self.queries = []
        self.normalize = normalize
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        current_qid = None
        current_features = []
        current_labels = []
        all_raw_docs = []   # used to compute mean/std if not provided

        with open(filepath, "r") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(filepath)}", leave=False):
                parts = line.strip().split()
                if not parts:
                    continue

                label = int(parts[0])
                qid = parts[1].split(":")[1]

                # Parse all 136 features, keeping only the 111 we want
                raw_feats = {
                    int(kv.split(":")[0]): float(kv.split(":")[1])
                    for kv in parts[2:138]   # feature indices 1..136
                }
                features = [raw_feats.get(idx, 0.0) for idx in KEEP_INDICES]

                # Flush current query when qid changes
                if current_qid is not None and qid != current_qid:
                    if current_labels:
                        all_raw_docs.extend(current_features)
                        self.queries.append((current_qid, current_features, current_labels))
                    current_features = []
                    current_labels = []

                current_qid = qid
                current_features.append(features)
                current_labels.append(label)

        # Flush the last query in the file
        if current_qid is not None and current_labels:
            self.queries.append((current_qid, current_features, current_labels))

        # ── Optionally apply per-query normalization ─────────────────────────────────
        normalized = []
        for qid, feats, labels in self.queries:
            feat_array = np.array(feats, dtype=np.float32)          # (num_docs, 111)

            if self.normalize:
                q_mean = feat_array.mean(axis=0)
                q_std  = feat_array.std(axis=0)
                q_std[q_std == 0.0] = 1.0
                feat_array = (feat_array - q_mean) / q_std

            normalized.append((
                qid,
                torch.tensor(feat_array, dtype=torch.float32),
                torch.tensor(labels,     dtype=torch.float32),
            ))
        self.queries = normalized

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


def _collate_fn(batch):
    """
    Collate variable-length query groups into parallel lists.
    Returns (qids, features_list, labels_list) where each element
    is a list with one entry per query in the mini-batch.
    """
    qids     = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels   = [item[2] for item in batch]
    return qids, features, labels


def load_fold(
    base_path: str = "/content/MSLR-WEB10K",
    fold_num: int = 1,
    batch_size: int = 4,
    normalize: bool = True,
):
    """
    Load Train, Validation, and Test DataLoaders for a given MSLR fold.

    Normalization is applied per-query independently across all splits.

    Parameters
    ----------
    base_path : str
        Path to the directory containing Fold1/ ... Fold5/ subdirectories.
    fold_num : int
        Which fold to load (1-5).
    batch_size : int
        Number of queries per mini-batch.

    Returns
    -------
    train_loader, val_loader, test_loader : torch.utils.data.DataLoader
    """
    fold_dir = os.path.join(base_path, f"Fold{fold_num}")

    # Step 1: Load training data
    print(f"  [Fold {fold_num}] Loading train split...")
    train_ds = MSLRQueryDataset(os.path.join(fold_dir, "train.txt"), normalize=normalize)

    print(f"  [Fold {fold_num}] Loading val split...")
    val_ds   = MSLRQueryDataset(os.path.join(fold_dir, "vali.txt"), normalize=normalize)

    print(f"  [Fold {fold_num}] Loading test split...")
    test_ds  = MSLRQueryDataset(os.path.join(fold_dir, "test.txt"), normalize=normalize)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    print(f"  Fold {fold_num}: {len(train_ds)} train | {len(val_ds)} val | {len(test_ds)} test queries")
    print(f"  Features: {NUM_FEATURES} (136 total − 25 dropped)")
    print(f"  Normalization: Per-query z-score")

    return train_loader, val_loader, test_loader
