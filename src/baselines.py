from collections import Counter


def popularity_ranking(splits, catalog):
    counts = Counter(item for row in splits for item in row["train_items"])
    if set(counts) - set(catalog):
        raise ValueError("Training items missing from catalog")
    return sorted(catalog, key=lambda item: (-counts[item], item))
