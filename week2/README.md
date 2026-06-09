# Week 2 — Information Retrieval Algorithms on SciFact

A structured study of classical and neural **Information Retrieval (IR)** algorithms, applied to the [SciFact](https://huggingface.co/datasets/allenai/scifact) benchmark dataset.

Each notebook is a self-contained experiment. They share a common `src/` library so there is no repeated boilerplate.

---

## The Dataset: SciFact

| | |
|--|--|
| **Corpus** | ~5,183 scientific paper abstracts |
| **Queries** | 1,409 scientific claims |
| **Answer key** | Human-labelled relevance judgements (qrels) |

---

## Notebooks

Run in order — each builds on the conceptual foundation of the previous one.

| # | Notebook | Algorithm | Key Idea |
|---|----------|-----------|----------|
| 1 | [`01_TF-IDFvsBM25`](notebooks/01_TF-IDFvsBM25.ipynb) | TF-IDF vs BM25 | Lexical matching; BM25 adds saturation + length normalisation |
| 2 | [`02_LMIR`](notebooks/02_LMIR.ipynb) | Language Model IR | Treat each document as a probability distribution over words |
| 3 | [`03_VSM`](notebooks/03_VSM.ipynb) | VSM variants | From sparse TF-IDF vectors → LSI topics → Transformer embeddings |

---

## Results Leaderboard

| Model | Precision@10 | Recall@10 | NDCG@10 |
|-------|:---:|:---:|:---:|
| TF-IDF (raw) | 0.0551 | 0.4943 | 0.3200 |
| VSM + LSI (SVD-300) | 0.0610 | 0.5358 | 0.3944 |
| Normal VSM (TF-IDF + cosine) | 0.0814 | 0.7286 | 0.5746 |
| BM25 (default k1=1.5, b=0.75) | 0.0846 | 0.7576 | 0.6380 |
| BM25 (tuned k1=0.8, b=0.9) | — | — | **0.6406** |
| **VSM + Transformers (MiniLM)** | **0.0890** | **0.7850** | **0.6561** |

**Winner: Transformer embeddings** (`all-MiniLM-L6-v2`) — the only model that understands *meaning*, not just word overlap.  
**Best lexical method: BM25 (tuned)** — simple, fast, and surprisingly competitive.

See [`results/leaderboard.md`](results/leaderboard.md) for the full breakdown.

---

## Project Structure

```
week2/
│
├── README.md
├── requirements.txt
│
├── scifact/                        ← Dataset (place here before running)
│   ├── corpus.jsonl
│   ├── queries.jsonl
│   └── qrels/
│       ├── train.tsv
│       └── test.tsv
│
├── src/                            ← Shared utility library
│   ├── __init__.py
│   ├── data_loader.py              ← load_dataset(), tokenize_dataset()
│   ├── evaluation.py               ← evaluate_rankings(), print_results()
│   └── visualization.py            ← plot_dataset_overview()
│
├── notebooks/
│   ├── 01_TF-IDFvsBM25.ipynb
│   ├── 02_LMIR.ipynb
│   └── 03_VSM.ipynb
│
└── results/
    └── leaderboard.md
```

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the SciFact dataset at:
#    week2/scifact/corpus.jsonl
#    week2/scifact/queries.jsonl
#    week2/scifact/qrels/train.tsv
#    week2/scifact/qrels/test.tsv

# 3. Open any notebook and run all cells
jupyter notebook notebooks/01_TF-IDFvsBM25.ipynb
```

---

## Key Takeaways

- **BM25 >> raw TF-IDF** — Two simple changes (saturation + length normalisation) nearly double NDCG@10.
- **LSI can hurt** — Compressing to 300 latent topics discards the precise vocabulary that scientific retrieval depends on.
- **Transformer embeddings win** — Dense semantic representations generalise across paraphrasing and synonymy that keyword-based methods miss entirely.
