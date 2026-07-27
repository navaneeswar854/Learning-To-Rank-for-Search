# Phase 1 — Classical IR Algorithms on SciFact

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

## Results & Key Takeaways

See [**RESULTS.md**](RESULTS.md) for the final leaderboard, performance metrics, and key takeaways from these experiments.

---

## Project Structure

```
Phase1_Classical_IR/
│
├── README.md
├── RESULTS.md
├── requirements.txt
│
├── Dataset/                        ← SciFact dataset (place here before running)
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

# 2. Place the SciFact dataset files at:
#    Phase1_Classical_IR/Dataset/corpus.jsonl
#    Phase1_Classical_IR/Dataset/queries.jsonl
#    Phase1_Classical_IR/Dataset/qrels/train.tsv
#    Phase1_Classical_IR/Dataset/qrels/test.tsv

# 3. Open any notebook and run all cells
jupyter notebook notebooks/01_TF-IDFvsBM25.ipynb
```
