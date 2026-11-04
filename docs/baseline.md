# Item cosine baseline

The second course baseline is an item-item collaborative method. It complements
the global popularity baseline with recommendations conditioned on observed
history. It is a relevant alternative baseline, not a claimed reproduction of a
named neural baseline from the TIGER paper.

For each user, use the **set of training items** to build binary user-item
membership. Let n(i) be the number of training users who interacted with item i,
and n(i,j) the number who interacted with both. Similarity is
`n(i,j) / sqrt(n(i)*n(j))`; self-similarity is zero. Unobserved pairs and items
absent from training have similarity zero. Repeated events do not multiply
co-occurrence counts.

For a query, sum similarities from the distinct items among the latest 20 history
events. All neighbors are retained; there is no tuned neighborhood size. Rank
the full catalog by descending score, then descending original training-event
popularity, then ascending item ID. This gives deterministic full lists even
when all scores are zero. Previously seen items remain eligible. This tie policy
also means the baseline uses popularity as a fallback.

Validation history contains training events; test history additionally includes
the validation event, exactly as for TIGER. The similarity matrix and popularity
counts are fitted exclusively on training events. The target is not a prediction
argument. No hyperparameters were selected from validation or test performance.

Implementation: [ItemCosine](../src/baselines.py). Tests in
[test_item_cosine.py](../tests/test_item_cosine.py) check hand-calculated cosine,
repeated events, held-out-label independence, history truncation, and fallback.

Run from the repository root, using fresh directories:

```bash
uv run python -m scripts.evaluate_baselines --config configs/poc01.yaml --out reports/baselines_recheck --predictions-dir artifacts/baselines_recheck
```

The CLI fits once and evaluates both splits, saving the protocol before metric
computation. It validates data hashes against the original reports, reproduces
the original popularity scores, and copies frozen TIGER metrics with a source
hash. It never retrains TIGER or regenerates its test predictions. Original
`reports/poc01/` remains intact. Existing output directories are rejected.

| Test metric | Popularity | Item cosine | TIGER |
|---|---:|---:|---:|
| Recall@5 | 0.00250000 | 0.00500000 | 0.01250000 |
| Recall@10 | 0.00750000 | 0.01750000 | 0.02350000 |
| NDCG@5 | 0.00124614 | 0.00261997 | 0.00753377 |
| NDCG@10 | 0.00279747 | 0.00655030 | 0.01110530 |

All methods return ten unique items for all 2,000 test users. Top-ten hit counts
are 15, 35, and 47 respectively. TIGER exceeds item cosine by 0.60 percentage
points of Recall@10 (12 more hits), but its absolute hit rate remains 2.35%.
This is a single-seed subset comparison without significance testing.

Source results: [test](../reports/baselines/results_test.json),
[validation](../reports/baselines/results_val.json),
[fixed protocol](../reports/baselines/protocol.json).
