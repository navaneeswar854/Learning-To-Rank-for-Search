"""
utils.py
--------
Reproducibility helpers and device selection utilities for RankNet.
"""

import os
import random
import torch
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device(verbose: bool = True) -> torch.device:
    """
    Selects the best available compute device (CUDA GPU or CPU).

    Returns
    -------
    torch.device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"CUDA available : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU name       : {torch.cuda.get_device_name(0)}")
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory     : {mem_gb:.2f} GB")
        print(f"\nActive device  : {device}")

    return device
