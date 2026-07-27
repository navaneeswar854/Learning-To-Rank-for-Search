# Phase 1 — Results and Key Takeaways

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

## Key Takeaways

- **BM25 >> raw TF-IDF** — Two simple changes (saturation + length normalisation) nearly double NDCG@10.
- **LSI can hurt** — Compressing to 300 latent topics discards the precise vocabulary that scientific retrieval depends on.
- **Transformer embeddings win** — Dense semantic representations generalise across paraphrasing and synonymy that keyword-based methods miss entirely.
