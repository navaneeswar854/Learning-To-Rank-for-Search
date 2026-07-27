# Phase 2 & 3 — Results and Key Takeaways

## Results — NDCG@k (5-fold cross-validation, MQ2008)

> All results are averaged over 5 folds × 3 random seeds. Metric: **NDCG@k** (higher is better).

| Model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 |
|-------|:------:|:------:|:------:|:-------:|
| Pointwise | 0.3594 | 0.4016 | 0.4473 | 0.4949 |
| **RankNet** | **0.3690** | **0.4084** | **0.4526** | **0.4993** |
| LambdaRank | 0.3641 | 0.4043 | 0.4516 | 0.4974 |
| LambdaMART | 0.3432 | 0.3898 | 0.4344 | 0.4846 |

**Best overall: RankNet** edges ahead at every NDCG cutoff, with LambdaRank close behind. LambdaMART, despite being a more powerful model class, underperforms on MQ2008 — likely due to dataset size (784 queries) and the near-absence of tie-breaking variation in some folds.

### Per-Fold Breakdown (NDCG@10)

| Fold | Pointwise | RankNet | LambdaRank | LambdaMART |
|------|:---------:|:-------:|:----------:|:----------:|
| 1 | 0.4771 | 0.4720 | 0.4726 | 0.4682 |
| 2 | 0.4467 | 0.4506 | 0.4478 | 0.4411 |
| 3 | 0.4805 | 0.4789 | 0.4887 | 0.4558 |
| 4 | 0.5425 | 0.5508 | 0.5461 | 0.5268 |
| 5 | 0.5280 | 0.5442 | 0.5319 | 0.5312 |

**Observations:**
- Folds 4 and 5 are noticeably better than 1, 2, and 3 across all models — NDCG@10 jumps from ~0.44–0.49 in the first three folds to ~0.52–0.55 in the last two.
- LambdaMART shows **near-zero variance for Fold 2** — the training data may not contain tie-breaking variation, so the seed has no effect on tree splits.
- The expected trend **LambdaRank ≥ RankNet ≥ Pointwise** does not hold — LambdaRank couldn't beat RankNet due to noise and fewer relevant documents. The difference remains minimal.
- LambdaMART saturates quickly (best validation NDCG often reached by the 3rd tree iteration), suggesting the dataset is small enough that gradient boosting overfits early.

---

## Key Takeaways

- **Pointwise is the simplest baseline** — treats ranking as regression with MSE loss; fast to train but ignores relative document ordering.
- **RankNet introduces pairwise learning** — comparing document pairs directly leads to consistent gains over pointwise regression across all NDCG cutoffs.
- **LambdaRank targets NDCG directly** — by weighting pairwise gradients by the NDCG change, it focuses capacity on the top of the ranked list, but the gains over RankNet are marginal on this dataset.
- **LambdaMART is powerful but data-hungry** — gradient-boosted trees with Lambda gradients are state-of-the-art on large LETOR benchmarks, but underperform neural methods here due to MQ2008's small scale and low label diversity in some folds.
- **Pairwise > Pointwise** — both pairwise neural methods outperform pointwise regression at every NDCG cutoff, confirming that ranking-aware training objectives matter.
