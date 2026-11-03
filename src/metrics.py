"""Full-catalog, single-relevant-item Recall and NDCG."""
import math


def evaluate_rankings(targets, predictions, cutoffs=(5, 10)):
    if not targets:
        raise ValueError("No users to evaluate")
    if set(predictions) - set(targets):
        raise ValueError("Predictions contain unknown users")
    totals = {f"{metric}@{k}": 0.0 for k in cutoffs for metric in ("Recall", "NDCG")}
    totals.update({f"coverage@{k}": 0.0 for k in cutoffs})
    returned = 0
    for user, target in targets.items():
        ranked = list(dict.fromkeys(predictions.get(user, [])))
        returned += len(ranked[:max(cutoffs)])
        rank = ranked.index(target) + 1 if target in ranked else None
        for k in cutoffs:
            hit = rank is not None and rank <= k
            totals[f"Recall@{k}"] += int(hit)
            totals[f"NDCG@{k}"] += 1 / math.log2(rank + 1) if hit else 0.0
            totals[f"coverage@{k}"] += int(len(ranked) >= k)
    return {**{key: value / len(targets) for key, value in totals.items()},
            "evaluated_users": len(targets), "mean_returned_items": returned / len(targets)}
