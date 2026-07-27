# Phase 2 & 3 — Learning to Rank on MQ2008 and MSLR-WEB10K

A structured study of **pairwise and listwise Learning-to-Rank (LTR)** algorithms, evaluated on the [MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) (Phase 2) and [MSLR-WEB10K](https://www.microsoft.com/en-us/research/project/mslr/) (Phase 3) benchmark datasets.

Each notebook is a self-contained experiment. They share a common `src/` library so there is no repeated boilerplate.

See [**RESULTS.md**](RESULTS.md) for the final leaderboards, per-fold breakdowns, observations, and key takeaways.

---

## Datasets

**Phase 2 — MQ2008 (LETOR 4.0)**

| | |
|--|--|
| **Corpus** | Web documents from the Gov2 crawl |
| **Queries** | 784 queries |
| **Features** | 46 hand-crafted relevance features per query-document pair |
| **Evaluation** | 5-fold cross-validation (Fold1–Fold5) |
| **Relevance** | Graded judgements (0, 1, 2) |

**Phase 3 — MSLR-WEB10K**

| | |
|--|--|
| **Queries** | 10,000 queries |
| **Features** | 136 relevance features per query-document pair |
| **Evaluation** | 5-fold cross-validation |
| **Relevance** | Graded judgements (0–4) |

---

## Notebooks

### Phase 2 — MQ2008 (`notebooks_letor/`)

Run in order — each builds on the conceptual foundation of the previous one.

| # | Notebook | Algorithm | Key Idea |
|---|----------|-----------|----------|
| 1 | [`01_pointwise`](notebooks_letor/01_pointwise.ipynb) | Pointwise Regression | Treat ranking as regression; predict relevance score per document independently |
| 2 | [`02_ranknet`](notebooks_letor/02_ranknet.ipynb) | RankNet | Pairwise neural network; minimise cross-entropy loss over document pairs |
| 3 | [`03_lambdarank`](notebooks_letor/03_lambdarank.ipynb) | LambdaRank | Pairwise gradient trick; weight pair updates by the NDCG change they would cause |
| 4 | [`04_lambdamart`](notebooks_letor/04_lambdamart.ipynb) | LambdaMART | Gradient-boosted trees trained with LambdaRank gradients |
| 5 | [`comparison`](notebooks_letor/comparison.ipynb) | Full Comparison | Head-to-head benchmark of all four methods across all folds |

### Phase 3 — MSLR-WEB10K (`notebooks_mslr/`)

| # | Notebook | Algorithm |
|---|----------|-----------|
| 0 | [`MSLR`](notebooks_mslr/MSLR.ipynb) | EDA and feature analysis |
| 1 | [`01_ranknet`](notebooks_mslr/01_ranknet.ipynb) | RankNet |
| 2 | [`02_pointwise`](notebooks_mslr/02_pointwise.ipynb) | Pointwise Regression |
| 3 | [`03_lambdarank`](notebooks_mslr/03_lambdarank.ipynb) | LambdaRank |
| 4 | [`04_lambdamart`](notebooks_mslr/04_lambdamart.ipynb) | LambdaMART |

---

## Project Structure

```
Phase2_3_Learning_to_Rank/
│
├── README.md
├── RESULTS.md                      ← Results tables, observations, key takeaways
├── pyproject.toml                  ← Package config (pip install -e .)
│
├── src/                            ← Shared utility library
│   ├── __init__.py
│   ├── data.py                     ← LETOR data loader & fold splitter
│   ├── data_mslr.py                ← MSLR-WEB10K data loader
│   ├── evaluate.py                 ← Cross-fold evaluation helpers
│   ├── lambdamart.py               ← LambdaMART implementation (GB trees)
│   ├── losses.py                   ← Pairwise loss functions (RankNet, Lambda)
│   ├── metrics.py                  ← NDCG, DCG, ranking metrics
│   ├── models.py                   ← Neural scoring model definition
│   └── train.py                    ← Training loop with fold cross-validation
│
├── notebooks_letor/                ← Phase 2: MQ2008 experiments
│   ├── 01_pointwise.ipynb
│   ├── 02_ranknet.ipynb
│   ├── 03_lambdarank.ipynb
│   ├── 04_lambdamart.ipynb
│   └── comparison.ipynb
│
└── notebooks_mslr/                 ← Phase 3: MSLR-WEB10K experiments
    ├── MSLR.ipynb                  ← EDA and feature analysis
    ├── 01_ranknet.ipynb
    ├── 02_pointwise.ipynb
    ├── 03_lambdarank.ipynb
    └── 04_lambdamart.ipynb
```

---

## Getting Started

```bash
# 1. Install the src package in editable mode
pip install -e .

# 2. Place the MQ2008 dataset at:
#    MQ2008/Fold1/train.txt  test.txt  vali.txt
#    MQ2008/Fold2/ ... Fold5/
#    (or point DATA_PATH in the notebooks to your MQ2008 path)

# 3. Open any notebook and run all cells
jupyter notebook notebooks_letor/01_pointwise.ipynb
```
