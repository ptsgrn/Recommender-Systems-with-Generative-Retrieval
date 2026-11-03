import argparse
from pathlib import Path
import time

from src.baselines import popularity_ranking
from src.common import canonical_hash, file_hash, load_config, read_jsonl, write_json
from src.metrics import evaluate_rankings


def load_predictions(path, users, catalog):
    predictions = {}
    for row in read_jsonl(path):
        user, items = row["user_id"], row["item_ids"]
        if user not in users or user in predictions:
            raise ValueError("Unknown or duplicate prediction user")
        if not isinstance(items, list) or any(not isinstance(item, str) or item not in catalog for item in items):
            raise ValueError("Predictions contain malformed or unknown item IDs")
        predictions[user] = list(dict.fromkeys(items))
    return predictions


def evaluate(config, split="test", predictions_path=None):
    start = time.perf_counter()
    data_dir = Path(config["data_dir"])
    splits = read_jsonl(data_dir / "splits.jsonl")
    catalog = sorted(item["item_id"] for item in read_jsonl(data_dir / "items.jsonl"))
    targets = {row["user_id"]: row[split + "_item"] for row in splits}
    ranked = popularity_ranking(splits, catalog)[:10]
    results = {"split": split, "catalog_size": len(catalog), "catalog_hash": canonical_hash(catalog),
               "split_sha256": file_hash(data_dir / "splits.jsonl"), "seed": config["seed"],
               "candidate_policy": "full catalog, repeats allowed, no sampled negatives",
               "popularity": evaluate_rankings(targets, {user: ranked for user in targets}),
               "model": {"status": "pending Developer B predictions"}}
    if predictions_path:
        predictions = load_predictions(predictions_path, targets, set(catalog))
        results["model"] = {"status": "measured", **evaluate_rankings(targets, predictions),
                            "predictions_sha256": file_hash(predictions_path)}
    results["evaluation_seconds"] = time.perf_counter() - start
    output = Path(config["report_dir"])
    write_json(output / f"results_{split}.json", results)
    table = ["| Method | Recall@5 | Recall@10 | NDCG@5 | NDCG@10 |", "|---|---:|---:|---:|---:|"]
    for name in ("popularity", "model"):
        if "Recall@5" in results[name]:
            table.append("| " + name + " | " + " | ".join(f"{results[name][key]:.6f}" for key in ("Recall@5", "Recall@10", "NDCG@5", "NDCG@10")) + " |")
    output.mkdir(parents=True, exist_ok=True)
    (output / f"results_{split}.md").write_text("\n".join(table) + "\n")
    print(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="Full-catalog popularity and saved-prediction evaluation")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--predictions")
    args = parser.parse_args()
    evaluate(load_config(args.config), args.split, args.predictions)


if __name__ == "__main__":
    main()
