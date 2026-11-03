import argparse
from pathlib import Path
import urllib.request

from src.common import file_hash, load_config, write_json, write_jsonl
from src.data import load_clean_data, make_subset


def download(url, destination):
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "TIGER-course-project/0.1"})
    with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as output:
        while block := source.read(1024 * 1024):
            output.write(block)
    temporary.replace(destination)


def prepare(config, user_count=None):
    raw, output = Path(config["raw_dir"]), Path(config["data_dir"])
    paths = {name: raw / name for name in ("reviews_Beauty_5.json.gz", "meta_Beauty.json.gz")}
    for filename, key in (("reviews_Beauty_5.json.gz", "reviews_url"), ("meta_Beauty.json.gz", "metadata_url")):
        download(config["data"][key], paths[filename])
    items, users, stats = load_clean_data(paths["reviews_Beauty_5.json.gz"], paths["meta_Beauty.json.gz"],
                                          config["data"]["min_interactions"])
    catalog, splits, audit = make_subset(items, users, user_count or config["data"]["users"], config["seed"])
    if not splits:
        raise ValueError("No eligible users remain after cleaning")
    write_jsonl(output / "items.jsonl", catalog)
    write_jsonl(output / "splits.jsonl", splits)
    write_jsonl(output / "chronological.jsonl", audit)
    stats.update(selected_users=len(splits), selected_items=len(catalog),
                 selected_interactions=sum(len(a["events"]) for a in audit),
                 excluded_interactions=stats["interactions_total"] - sum(len(a["events"]) for a in audit),
                 training_interactions=sum(len(s["train_items"]) for s in splits),
                 training_examples=sum(len(s["train_items"]) - 1 for s in splits),
                 validation_examples=len(splits), test_examples=len(splits),
                 unsampled_eligible_users=len(users) - len(splits),
                 excluded_catalog_items=stats["source_items"] - len(catalog), seed=config["seed"],
                 exact_duplicate_policy="identical complete parsed review records only",
                 timestamp_tie_breaker="original zero-based source row",
                 sources={name: {"url": config["data"][key], "sha256": file_hash(paths[name])}
                          for name, key in (("reviews_Beauty_5.json.gz", "reviews_url"), ("meta_Beauty.json.gz", "metadata_url"))})
    for key in ("metadata_malformed", "metadata_duplicate_asins", "metadata_unusable",
                "interactions_malformed", "interactions_invalid_fields", "interactions_exact_duplicates",
                "interactions_missing_metadata"):
        stats.setdefault(key, 0)
    write_json(output / "data_stats.json", stats)
    print({key: stats[key] for key in ("selected_users", "selected_items", "training_examples")})
    for row in audit[:3]:
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Amazon Beauty 2014")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--users", type=int)
    args = parser.parse_args()
    if args.users is not None and args.users < 1:
        parser.error("--users must be positive")
    prepare(load_config(args.config), args.users)


if __name__ == "__main__":
    main()
