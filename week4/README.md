# Learning to Rank for Search

A clean implementation of **Learning-to-Rank (LTR)** algorithms on the **LETOR4 / MQ2008** benchmark dataset, covering both the Pointwise and Listwise training paradigms using PyTorch.

---

## 📂 Repository Structure

```
week4/
├── notebooks/
│   ├── Pointwise.ipynb        # RankNet — Pointwise neural approach (MSE loss)
│   └── LambdaRank.ipynb       # LambdaRank — Listwise/Pairwise gradient approach
├── src/
│   ├── dataset.py             # Data loading and preprocessing utilities
│   └── model.py               # Model architectures (MLP, RankNet)
├── .gitignore                 # Excludes dataset, checkpoints, and caches
└── README.md                  # This file
```

> **Note:** The `MQ2008/` dataset directory is excluded from version control via `.gitignore`. See [Dataset Setup](#-dataset-setup) below.

---

## 📓 Notebooks

### 1. `Pointwise.ipynb` — RankNet (Pointwise)

Implements **RankNet** as a pointwise scoring network that treats document ranking as a regression problem.

| Step | Description |
|------|-------------|
| 1–2  | Environment setup & dataset loading (LETOR4 / MQ2008) |
| 3    | Query-grouped PyTorch `Dataset` and `DataLoader` |
| 4–5  | Model architecture (MLP scorer) and MSE loss function |
| 6–8  | Training, overfitting diagnosis, and dropout regularization |
| 9    | Test-set NDCG evaluation |
| 10   | 5-Fold Cross-Validation |
| 11   | Ablation study across 4 architectures |

**Key details:**
- **Model:** Deep Pointwise Scoring Network (46 → 64 → 32 → 1)
- **Loss:** Mean Squared Error (MSE) on relevance labels
- **Metric:** NDCG@K
- **Regularization:** Dropout (p=0.2), Early stopping via best-val-loss checkpoint

---

### 2. `LambdaRank.ipynb` — LambdaRank (Listwise)

Implements **LambdaRank**, which directly optimizes the NDCG metric by computing gradient weights (lambdas) from pairwise document comparisons within a query group.

**Key details:**
- **Gradient:** Lambda gradients weighted by ΔNDCG
- **Metric:** NDCG@K
- **Approach:** Listwise optimization via pairwise document comparisons

---

## 🗄 Dataset Setup

This repository uses the **MQ2008** dataset from the [LETOR 4.0 benchmark](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/).

1. Download `MQ2008.zip` from the LETOR 4.0 release page.
2. Extract it so the directory structure looks like:

```
week4/
└── MQ2008/
    ├── Fold1/
    │   ├── train.txt
    │   ├── vali.txt
    │   └── test.txt
    ├── Fold2/ ... Fold5/
    ├── S1.txt ... S5.txt
    ├── Querylevelnorm.txt
    └── readme.txt
```

The notebooks reference the dataset at `/content/MQ2008/` (Google Colab paths). **Do not change these paths.**

---

## ⚙️ Environment

The notebooks were developed and tested on **Google Colab** with a T4 GPU.

| Library    | Version |
|------------|---------|
| Python     | 3.10+   |
| PyTorch    | 2.11.0  |
| NumPy      | 2.0.2   |
| Pandas     | 2.2.2   |
| scikit-learn | latest |
| matplotlib | latest  |
| seaborn    | latest  |
| tqdm       | latest  |

Install all dependencies via:

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm
```

---

## 📊 Dataset Format — libsvm

Each line in the dataset files follows the **libsvm** format:

```
<relevance>  qid:<query_id>  1:<f1>  2:<f2>  ...  46:<f46>  #docid=...
```

| Field       | Description |
|-------------|-------------|
| `relevance` | Integer label: 0 = not relevant, 1 = relevant, 2 = highly relevant |
| `qid`       | Query group ID — all docs with the same qid form one ranking list |
| `1..46`     | 46 pre-computed IR features (TF, IDF, BM25, PageRank, etc.) |
| `#docid`    | Trailing comment with document ID (ignored during parsing) |

---

## 📖 References

- **RankNet:** Burges et al., *Learning to Rank using Gradient Descent*, ICML 2005.
- **LambdaRank:** Burges et al., *Learning to Rank with Nonsmooth Cost Functions*, NeurIPS 2006.
- **LETOR 4.0:** Qin & Liu, *Introducing LETOR 4.0 Datasets*, arXiv 2013.
