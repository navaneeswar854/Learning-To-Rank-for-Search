# Pairwise Approaches — Learning to Rank on MQ2008

A structured study of **pairwise Learning-to-Rank (LTR)** algorithms, evaluated on the [MQ2008](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/) benchmark dataset.

Each notebook is a self-contained experiment. They share a common `src/ltr/` library so there is no repeated boilerplate. A final comparison notebook benchmarks all three methods head-to-head.

---

## The Dataset: MQ2008 (LETOR 4.0)

| | |
|--|--|
| **Corpus** | Web documents from the Gov2 crawl |
| **Queries** | 784 queries |
| **Features** | 46 hand-crafted relevance features per query-document pair |
| **Evaluation** | 5-fold cross-validation (Fold1–Fold5) |
| **Relevance** | Graded judgements (0, 1, 2) |

---

## Notebooks

Run in order — each builds on the conceptual foundation of the previous one.

| # | Notebook | Algorithm | Key Idea |
|---|----------|-----------|----------|
| 1 | [`01_pointwise`](notebooks/01_pointwise.ipynb) | Pointwise Regression | Treat ranking as regression; predict relevance score per document independently |
| 2 | [`02_ranknet`](notebooks/02_ranknet.ipynb) | RankNet | Pairwise neural network; minimise cross-entropy loss over document pairs |
| 3 | [`03_lambdarank`](notebooks/03_lambdarank.ipynb) | LambdaRank | Pairwise gradient trick; weight pair updates by the NDCG change they would cause |
| 4 | [`04_comparison`](notebooks/04_comparison.ipynb) | Full Comparison | Head-to-head benchmark of all three methods across all folds |

---

## Results — NDCG@k (5-fold cross-validation)

> All results are averaged over 5 folds × 3 random seeds. Metric: **NDCG@k** (higher is better).

| Model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|-------|:------:|:------:|:------:|:-------:|
| Pointwise | 0.3594 | 0.4016 | 0.4473 | 0.4949 |
| RankNet | 0.3690 | 0.4084 | 0.4526 | 0.4993 |
| **LambdaRank** | 0.3641 | 0.4043 | 0.4516 | **0.4974** |

**Best overall: RankNet** edges ahead at NDCG@5 and NDCG@10, with LambdaRank competitive at the top of the ranking (NDCG@1).

### Per-Fold Breakdown (NDCG@10)

| Fold | Pointwise | RankNet | LambdaRank |
|------|:---------:|:-------:|:----------:|
| 1 | 0.4771 | 0.4720 | 0.4726 |
| 2 | 0.4467 | 0.4506 | 0.4478 |
| 3 | 0.4805 | 0.4789 | 0.4887 |
| 4 | 0.5425 | 0.5508 | 0.5461 |
| 5 | 0.5280 | 0.5442 | 0.5319 |

---

## Project Structure

```
Pairwise Approaches/
│
├── README.md
├── pyproject.toml                  ← Package config (pip install -e .)
│
├── src/
│   └── ltr/                        ← Shared utility library
│       ├── __init__.py
│       ├── data.py                 ← LETOR data loader & fold splitter
│       ├── evaluate.py             ← NDCG@k evaluation helpers
│       ├── losses.py               ← Pairwise loss functions (RankNet, Lambda)
│       ├── metrics.py              ← NDCG, DCG, ranking metrics
│       ├── models.py               ← Neural scoring model definition
│       └── train.py                ← Training loop with fold cross-validation
│
├── notebooks/
│   ├── 01_pointwise.ipynb
│   ├── 02_ranknet.ipynb
│   ├── 03_lambdarank.ipynb
│   └── 04_comparison.ipynb
│
└── ltr_results/                    ← Saved evaluation results (JSON)
    ├── pointwise_results.json
    ├── ranknet_results.json
    └── lambdarank_results.json
```

---

## Getting Started

```bash
# 1. Install the ltr package in editable mode
pip install -e .

# 2. Place the MQ2008 dataset at:
#    MQ2008/Fold1/train.txt  test.txt  vali.txt
#    MQ2008/Fold2/ ... Fold5/
#    (or point DATA_DIR in the notebooks to your MQ2008 path)

# 3. Open any notebook and run all cells
jupyter notebook notebooks/01_pointwise.ipynb
```

---

## Key Takeaways

- **Pointwise is the simplest baseline** — treats ranking as regression with MSE loss; fast to train but ignores relative document ordering.
- **RankNet introduces pairwise learning** — comparing document pairs directly leads to consistent gains over pointwise regression across all NDCG cutoffs.
- **LambdaRank targets NDCG directly** — by weighting pairwise gradients by the NDCG change, it focuses capacity on the top of the ranked list (strong NDCG@1).
- **Pairwise > Pointwise** — both pairwise methods outperform pointwise regression at every NDCG cutoff, confirming that ranking-aware training objectives matter.
