# Learning to Rank for Search

An end-to-end study of **Learning to Rank (LTR)** algorithms, from classical IR baselines to state-of-the-art gradient-boosted trees, conducted as part of an internship at **WSAI, IIT Madras** (May–July 2025).

---

## Project Phases

This repository is split into two self-contained phases:

### Phase 1 — Classical IR on SciFact → [`Classic IR/`](Classic%20IR/)

Classical, non-trained information retrieval algorithms evaluated on the **SciFact** scientific claim retrieval benchmark.

| Model | NDCG@10 |
|-------|:-------:|
| TF-IDF (raw) | 0.320 |
| VSM + LSI (SVD-300) | 0.394 |
| Normal VSM (TF-IDF + cosine) | 0.575 |
| LMIR (Jelinek-Mercer, λ=0.5) | 0.627 |
| BM25 (tuned k₁=0.8, b=0.9) | 0.641 |
| **VSM + Transformer Embeddings** | **0.656** |

📂 See [`Classic IR/README.md`](Classic%20IR/README.md) for full details.

---

### Phase 2 & 3 — Learning to Rank on MQ2008 & MSLR-WEB10K → [`Learning to Rank/`](Learning%20to%20Rank/)

Supervised LTR models implemented from scratch and evaluated using 5-fold cross-validation.

**Phase 2 — MQ2008 (784 queries, 46 features):**

| Model | NDCG@10 |
|-------|:-------:|
| Pointwise | 0.495 |
| **RankNet** | **0.499** |
| LambdaRank | 0.497 |
| LambdaMART | 0.485 |

**Phase 3 — MSLR-WEB10K (10,000 queries, 136 features):**

| Model | NDCG@10 |
|-------|:-------:|
| Pointwise | 0.433 |
| RankNet | 0.444 |
| **LambdaRank** | **0.451** |
| LambdaMART | TBD |

📂 See [`Learning to Rank/README.md`](Learning%20to%20Rank/README.md) for full details.

---

## Key Findings

- **BM25 is an exceptionally strong classical baseline** — simple, fast, and nearly matches transformer embeddings on short queries.
- **Pairwise > Pointwise** — ranking-aware training objectives consistently outperform direct label regression.
- **Per-query normalisation is critical** for pairwise neural models on MSLR-WEB10K.
- **LambdaRank > RankNet on large data** — the NDCG-weighted gradient trick makes a measurable difference at scale (MSLR), but not on the small MQ2008 dataset.
- **Non-linearity is mandatory** — linear models fail significantly on MSLR's 136-feature web search signal.

---

## Repository Structure

```
Learning-To-Rank-for-Search/
│
├── README.md                   ← You are here
│
├── Classic IR/                 ← Phase 1: Classical IR on SciFact
│   ├── README.md
│   ├── requirements.txt
│   ├── Dataset/scifact/
│   ├── notebooks/
│   ├── src/
│   └── results/
│
└── Learning to Rank/           ← Phase 2 & 3: LTR on MQ2008 + MSLR-WEB10K
    ├── README.md
    ├── pyproject.toml
    ├── ltr/                    ← Shared Python library
    ├── notebooks/              ← Phase 2: MQ2008 notebooks
    ├── notebooks_mslr/         ← Phase 3: MSLR-WEB10K notebooks
    ├── ltr_results/            ← Phase 2 results (JSON)
    └── ltr_results_mslr/       ← Phase 3 results (JSON)
```

---

## Getting Started

Each phase has its own `README.md` with setup instructions. In general:

```bash
# Phase 1 — Classical IR
cd "Classic IR"
pip install -r requirements.txt
jupyter notebook notebooks/01_TF-IDFvsBM25.ipynb

# Phase 2 & 3 — Learning to Rank
cd "Learning to Rank"
pip install -e .
jupyter notebook notebooks/01_pointwise.ipynb
```

> **Dataset Note:** Large dataset files (MSLR-WEB10K, MQ2008) are not included in this repository due to size. Download links are provided in each phase's README.
