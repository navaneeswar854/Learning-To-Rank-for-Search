"""
dataset.py
==========
PyTorch Dataset and DataLoader utilities for the LETOR4 / MQ2008 benchmark.

The dataset uses the standard libsvm format:
    <relevance>  qid:<query_id>  1:<f1>  2:<f2>  ...  46:<f46>  #docid=...

  relevance → integer label  (0 = not relevant, 1 = relevant, 2 = highly relevant)
  qid       → query group id (all docs sharing a qid form one ranking list)
  1..46     → 46 pre-computed IR features (TF, IDF, BM25, PageRank, etc.)
  #docid    → trailing comment with document id (ignored during parsing)

Extracted from: notebooks/Pointwise.ipynb and notebooks/LambdaRank.ipynb
"""

import os

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class LETORQueryDataset(Dataset):
    """
    Parses a LETOR libsvm file and groups documents by Query ID.

    Each item returned is a 3-tuple:
        (qid, features_tensor, labels_tensor)

    where:
        qid             – query identifier (str)
        features_tensor – shape (n_docs, 46), dtype float32
        labels_tensor   – shape (n_docs,),    dtype float32

    Only queries with **more than one** candidate document are retained,
    because pairwise comparisons require at least two documents.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to a LETOR libsvm file
        (e.g. ``MQ2008/Fold1/train.txt``).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.queries = []   # Will hold tuples of (qid, features, labels)
        self._load_and()

    def _load_and(self):
        current_qid = None
        current_features = []
        current_labels = []

        with open(self.filepath, 'r') as f:
            for line in tqdm(f, desc=f"Loading {os.path.basename(self.filepath)}"):
                parts = line.strip().split()
                if not parts:
                    continue

                label = int(parts[0])
                qid = parts[1].split(':')[1]
                features = [float(x.split(':')[1]) for x in parts[2:48]]

                # If we hit a new query, save the old one and start fresh
                if current_qid is not None and qid != current_qid:
                    # Only keep queries with >1 document (needed for pairwise comparison)
                    if len(current_labels) > 1:
                        self.queries.append((
                            current_qid,
                            torch.tensor(current_features, dtype=torch.float32),
                            torch.tensor(current_labels, dtype=torch.float32)
                        ))
                    current_features = []
                    current_labels = []

                current_qid = qid
                current_features.append(features)
                current_labels.append(label)

            # Don't forget to save the very last query in the file
            if current_qid is not None and len(current_labels) > 1:
                self.queries.append((
                    current_qid,
                    torch.tensor(current_features, dtype=torch.float32),
                    torch.tensor(current_labels, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]


def query_collate_fn(batch):
    """
    Handles batching of queries with varying numbers of documents.

    Returns lists of tensors instead of stacking them into a single rigid
    tensor (because each query can have a different number of candidate docs).

    Parameters
    ----------
    batch : list of (qid, features_tensor, labels_tensor)

    Returns
    -------
    qids : list[str]
    features : list[torch.Tensor]
    labels : list[torch.Tensor]
    """
    qids = [item[0] for item in batch]
    features = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return qids, features, labels


def get_dataloaders_for_fold(base_path="/content/MQ2008", fold_num=1, batch_size=4):
    """
    Dynamically loads Train, Validation, and Test DataLoaders for a specific fold.

    Designed for 5-Fold Cross-Validation on the MQ2008 dataset.

    Parameters
    ----------
    base_path : str
        Root directory of the MQ2008 dataset (default: ``/content/MQ2008``).
        Note: The notebooks were developed on Google Colab; adjust this path
        if running locally (e.g. ``./MQ2008``).
    fold_num : int
        Fold number in the range [1, 5] (default: 1).
    batch_size : int
        Number of queries per mini-batch (default: 4).

    Returns
    -------
    train_loader : DataLoader
    vali_loader  : DataLoader
    test_loader  : DataLoader
    """
    print(f"\nInitializing DataLoaders for Fold {fold_num}...")
    fold_dir = os.path.join(base_path, f"Fold{fold_num}")

    train_path = os.path.join(fold_dir, "train.txt")
    vali_path  = os.path.join(fold_dir, "vali.txt")
    test_path  = os.path.join(fold_dir, "test.txt")

    # Instantiate Datasets
    train_dataset = LETORQueryDataset(train_path)
    vali_dataset  = LETORQueryDataset(vali_path)
    test_dataset  = LETORQueryDataset(test_path)

    # Instantiate DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=query_collate_fn)
    vali_loader  = DataLoader(vali_dataset,  batch_size=batch_size, shuffle=False, collate_fn=query_collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, collate_fn=query_collate_fn)

    print(f"  Train queries : {len(train_dataset)}")
    print(f"  Vali queries  : {len(vali_dataset)}")
    print(f"  Test queries  : {len(test_dataset)}")

    return train_loader, vali_loader, test_loader
