# Learning to Rank for Search

An end-to-end study of **Learning to Rank (LTR)** algorithms, from classical IR baselines to state-of-the-art gradient-boosted trees, conducted as part of an internship at **WSAI, IIT Madras** (May–July 2025).

---

## Project Phases

This repository is split into two self-contained phases:

### Phase 1 — Classical IR on SciFact → [`Phase1_Classical_IR/`](Phase1_Classical_IR/)

Classical, non-trained information retrieval algorithms evaluated on the **SciFact** scientific claim retrieval benchmark.

📂 See [`Phase1_Classical_IR/README.md`](Phase1_Classical_IR/README.md) for full details and [`Phase1_Classical_IR/RESULTS.md`](Phase1_Classical_IR/RESULTS.md) for the final leaderboard and key takeaways.

---

### Phase 2 & 3 — Learning to Rank on MQ2008 & MSLR-WEB10K → [`Phase2_3_Learning_to_Rank/`](Phase2_3_Learning_to_Rank/)

Supervised LTR models implemented from scratch and evaluated using 5-fold cross-validation.

📂 See [`Phase2_3_Learning_to_Rank/README.md`](Phase2_3_Learning_to_Rank/README.md) for full details and [`Phase2_3_Learning_to_Rank/RESULTS.md`](Phase2_3_Learning_to_Rank/RESULTS.md) for the final leaderboard and key takeaways.

---

## Repository Structure

```
Learning-To-Rank-for-Search/
│
├── README.md                   ← You are here
│
├── Phase1_Classical_IR/        ← Phase 1: Classical IR on SciFact
│   ├── README.md
│   ├── RESULTS.md
│   ├── requirements.txt
│   ├── Dataset/                    ← SciFact files (corpus, queries, qrels)
│   ├── notebooks/
│   ├── src/
│   └── results/
│
└── Phase2_3_Learning_to_Rank/  ← Phase 2 & 3: LTR on MQ2008 + MSLR-WEB10K
    ├── README.md
    ├── RESULTS.md
    ├── pyproject.toml
    ├── src/                    ← Shared Python library
    ├── notebooks_letor/        ← Phase 2: MQ2008 notebooks
    └── notebooks_mslr/         ← Phase 3: MSLR-WEB10K notebooks
```

---

## Getting Started

Each phase has its own `README.md` with setup instructions. In general:

```bash
# Phase 1 — Classical IR
cd Phase1_Classical_IR
pip install -r requirements.txt
jupyter notebook notebooks/01_TF-IDFvsBM25.ipynb

# Phase 2 & 3 — Learning to Rank
cd Phase2_3_Learning_to_Rank
pip install -e .
jupyter notebook notebooks_letor/01_pointwise.ipynb
```

> **Dataset Note:** Large dataset files (MSLR-WEB10K, MQ2008) are not included in this repository due to size. Download links are provided in each phase's README.
