import gzip
import json
from src.common import write_jsonl
from src.data import clean_item, load_clean_data, make_subset, records


def test_cleaning_optional_fields():
    item = clean_item({"asin": "A", "title": " A &amp;  B ", "price": "NaN",
                       "categories": [["Beauty", "Skin"], ["Beauty", "Skin"]]})
    assert item["title"] == "A & B"
    assert item["price"] is None
    assert item["brand"] == "Unknown"
    assert item["category"] == "Beauty > Skin"
    assert "Price: Unknown" in item["text"]
    assert clean_item({"asin": "A", "title": " "}) is None


def test_legacy_parser_is_safe(tmp_path):
    path = tmp_path / "legacy.gz"
    with gzip.open(path, "wt") as stream:
        stream.write("{'asin': 'A', 'title': 'Product'}\n")
        stream.write("__import__('os').system('exit 99')\n")
    assert list(records(path)) == [(0, {"asin": "A", "title": "Product"})]


def test_splits_filter_duplicates_ties_and_keep_repeats(tmp_path):
    metadata = [{"asin": str(i), "title": f"Item {i}"} for i in range(7)]
    metadata += [{"asin": "0", "title": "Replacement"}, {"asin": "bad", "title": ""}]
    interactions = []
    # Intentionally out of order; the final repeated ASIN remains an event.
    for user in ("u1", "u2", "short"):
        for item, time in (("2", 2), ("0", 1), ("1", 1), ("3", 3), ("0", 4)):
            interactions.append({"reviewerID": user, "asin": item, "unixReviewTime": time})
            if user == "short":
                break
    interactions.append(interactions[0].copy())
    interactions.append({"reviewerID": "u1", "asin": "bad", "unixReviewTime": 0})
    meta, reviews = tmp_path / "meta.jsonl", tmp_path / "reviews.jsonl"
    write_jsonl(meta, metadata)
    write_jsonl(reviews, interactions)
    items, users, stats = load_clean_data(reviews, meta)
    catalog, splits, audit = make_subset(items, users, 20, 42)
    assert items["0"]["title"] == "Item 0"
    assert stats["metadata_duplicate_asins"] == 1
    assert stats["interactions_exact_duplicates"] == 1
    assert stats["interactions_missing_metadata"] == 1
    assert stats["excluded_users"] == 1
    assert {i["item_id"] for i in catalog} == {"0", "1", "2", "3"}
    for split, history in zip(splits, audit):
        assert split["train_items"] == ["0", "1", "2"]
        assert split["val_item"] == "3" and split["test_item"] == "0"
        assert [e["item_id"] for e in history["events"]] == ["0", "1", "2", "3", "0"]
    assert make_subset(items, users, 1, 42) == make_subset(items, dict(reversed(list(users.items()))), 1, 42)
