# Results Leaderboard

All models evaluated on **SciFact** using the combined train+test qrels.  
Metrics computed at cutoff k=10.

## Metrics

| Metric | What it measures |
|--------|-----------------|
| **Precision@10** | Of the 10 returned docs, what fraction are relevant? |
| **Recall@10** | Of all relevant docs, what fraction did we retrieve? |
| **NDCG@10** | Are the most relevant docs ranked highest? |

## Results

| Model | Notebook | Precision@10 | Recall@10 | NDCG@10 |
|-------|----------|:---:|:---:|:---:|
| TF-IDF (raw, additive) | [01_TF-IDFvsBM25](../notebooks/01_TF-IDFvsBM25.ipynb) | 0.0551 | 0.4943 | 0.3200 |
| VSM + LSI (SVD-300) | [03_VSM](../notebooks/03_VSM.ipynb) | 0.0610 | 0.5358 | 0.3944 |
| Normal VSM (TF-IDF + cosine) | [03_VSM](../notebooks/03_VSM.ipynb) | 0.0814 | 0.7286 | 0.5746 |
| BM25 (k1=1.5, b=0.75) | [01_TF-IDFvsBM25](../notebooks/01_TF-IDFvsBM25.ipynb) | 0.0846 | 0.7576 | 0.6380 |
| BM25 (k1=0.8, b=0.9) — tuned | [01_TF-IDFvsBM25](../notebooks/01_TF-IDFvsBM25.ipynb) | — | — | **0.6406** |
| LMIR JM (optimal λ) | [02_LMIR](../notebooks/02_LMIR.ipynb) | — | — | ~0.55* |
| **VSM + Transformers (MiniLM-L6)** | [03_VSM](../notebooks/03_VSM.ipynb) | **0.0890** | **0.7850** | **0.6561** |

> \* LMIR result varies with λ; see the λ-sweep plot in notebook 02.

## Key Observations

1. **BM25 is far better than raw TF-IDF** (NDCG: 0.638 vs 0.320). Two simple changes — term saturation and length normalisation — make a dramatic difference.
2. **LSI performs worse than plain TF-IDF VSM**. Compressing to 300 latent topics discards the precise scientific vocabulary that these queries rely on.
3. **Transformer embeddings are the clear winner** across all three metrics. Semantic representations close the gap between synonyms and paraphrases that keyword-based methods miss.
4. **BM25 is competitive with embeddings** and much faster (no GPU needed). In resource-constrained settings, well-tuned BM25 is a strong baseline.
