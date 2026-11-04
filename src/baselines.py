from collections import Counter
from collections import defaultdict
from itertools import combinations
import math


def popularity_ranking(splits, catalog):
    counts = Counter(item for row in splits for item in row["train_items"])
    if set(counts) - set(catalog):
        raise ValueError("Training items missing from catalog")
    return sorted(catalog, key=lambda item: (-counts[item], item))


class ItemCosine:
    """Binary training-user co-occurrence cosine; all neighbors, no tuning.

    Sum similarities to distinct items in the latest 20 observed events.
    Self-similarity is zero. Ties use training-event popularity then item ID.
    Previously observed items remain eligible, matching the experiment policy.
    """

    def __init__(self, splits, catalog):
        self.catalog = sorted(catalog)
        self.popularity = Counter(i for row in splits for i in row['train_items'])
        if set(self.popularity) - set(self.catalog):
            raise ValueError('Training items missing from catalog')
        users = Counter()
        pairs = Counter()
        for row in splits:
            items = sorted(set(row['train_items']))
            users.update(items)
            pairs.update(combinations(items, 2))
        self.similarity = defaultdict(dict)
        for (left, right), count in pairs.items():
            value = count / math.sqrt(users[left] * users[right])
            self.similarity[left][right] = value
            self.similarity[right][left] = value

    def recommend(self, history, k=10):
        scores = Counter()
        for item in sorted(set(history[-20:])):
            scores.update(self.similarity.get(item, {}))
        return sorted(self.catalog, key=lambda i: (-scores[i], -self.popularity[i], i))[:k]
