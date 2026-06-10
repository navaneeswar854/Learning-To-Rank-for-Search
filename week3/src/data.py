"""
data.py
-------
Dataset and DataLoader utilities for the LETOR4 / MQ2008 benchmark.

The LETOR format (libsvm) stores one document per line:
    <relevance>  qid:<query_id>  1:<f1>  2:<f2>  ...  46:<f46>  #docid=...

  relevance → integer label  (0 = not relevant, 1 = relevant, 2 = highly relevant)
  qid       → query group id (all docs sharing a qid form one ranking list)
  1..46     → 46 pre-computed IR features (TF, IDF, BM25, PageRank, etc.)
  #docid    → trailing comment with document id (ignored during parsing)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LETORQueryDataset(Dataset):
    """
    Parses a LETOR libsvm file and groups documents by Query ID.

    Each item returned is a tuple:
        (qid: str, features: FloatTensor[num_docs, 46], labels: FloatTensor[num_docs])

    Only queries with more than 1 document are retained, since pairwise
    ranking requires at least one valid pair.
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.queries: list = []  # list of (qid, features_tensor, labels_tensor)
        self._load()

    def _load(self) -> None:
        current_qid = None
        current_features: list = []
        current_labels: list = []

        with open(self.filepath, "r") as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(self.filepath)}"):
                parts = line.strip().split()
                if not parts:
                    continue

                label = int(parts[0])
                qid = parts[1].split(":")[1]
                features = [float(x.split(":")[1]) for x in parts[2:48]]

                # When the query id changes, flush the previous query's buffer
                if current_qid is not None and qid != current_qid:
                    if len(current_labels) > 1:          # need ≥2 docs for pairs
                        self.queries.append((
                            current_qid,
                            torch.tensor(current_features, dtype=torch.float32),
                            torch.tensor(current_labels,  dtype=torch.float32),
                        ))
                    current_features = []
                    current_labels = []

                current_qid = qid
                current_features.append(features)
                current_labels.append(label)

        # Flush the very last query in the file
        if current_qid is not None and len(current_labels) > 1:
            self.queries.append((
                current_qid,
                torch.tensor(current_features, dtype=torch.float32),
                torch.tensor(current_labels,  dtype=torch.float32),
            ))

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int):
        return self.queries[idx]


def query_collate_fn(batch):
    """
    Custom collate function for variable-length query batches.

    Returns lists of tensors instead of stacking them into a single rigid
    tensor (queries differ in the number of candidate documents).
    """
    qids     = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels   = [item[2] for item in batch]
    return qids, features, labels


def get_dataloaders_for_fold(
    base_path: str = "/content/MQ2008",
    fold_num: int = 1,
    batch_size: int = 4,
):
    """
    Build Train, Validation, and Test DataLoaders for a specific MQ2008 fold.

    Parameters
    ----------
    base_path  : Path to the MQ2008 root directory.
    fold_num   : Fold index (1 – 5).
    batch_size : Number of queries per batch.

    Returns
    -------
    (train_loader, vali_loader, test_loader)
    """
    print(f"\nInitializing DataLoaders for Fold {fold_num}...")
    fold_dir = os.path.join(base_path, f"Fold{fold_num}")

    train_path = os.path.join(fold_dir, "train.txt")
    vali_path  = os.path.join(fold_dir, "vali.txt")
    test_path  = os.path.join(fold_dir, "test.txt")

    train_dataset = LETORQueryDataset(train_path)
    vali_dataset  = LETORQueryDataset(vali_path)
    test_dataset  = LETORQueryDataset(test_path)

    print(f"  Train queries : {len(train_dataset)}")
    print(f"  Vali  queries : {len(vali_dataset)}")
    print(f"  Test  queries : {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=query_collate_fn
    )
    vali_loader  = DataLoader(
        vali_dataset,  batch_size=batch_size, shuffle=False, collate_fn=query_collate_fn
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=query_collate_fn
    )

    return train_loader, vali_loader, test_loader
