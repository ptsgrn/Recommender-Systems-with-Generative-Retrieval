# Amazon Beauty 2014 dataset sample

The public source is the Amazon Reviews 2014 release:

- [Dataset information](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)
- [Beauty 5-core reviews](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Beauty_5.json.gz)
- [Beauty product metadata](https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Beauty.json.gz)

`interactions.jsonl` contains **100 real retained interaction events**, selected
in file order from the project's chronological audit. Each row contains the
public dataset user ID, product ID, Unix timestamp, original source-row index,
and the event's role in the original train/validation/test split.

`items.jsonl` contains the matching cleaned product metadata, including title,
brand, category, price, and the deterministic text used for embedding. Join
these files on `item_id`. `manifest.json` records counts, selection policy, and
source/sample SHA-256 hashes.

This sample demonstrates data formats. It is not the 2,000-user experimental
subset and is not suitable for reproducing the reported scores: the last user's
history may be truncated at the 100-event boundary. Its split labels describe
the original full histories, not a newly calculated split of the sample.

With the local processed dataset present, regenerate it from the repository root:

```bash
uv run python -m scripts.make_dataset_sample
```

For full preprocessing from the public source, follow the repository README and
`scripts.prepare_data`. The final metrics are available in
[the three-method comparison](../reports/baselines/results_test.md).
