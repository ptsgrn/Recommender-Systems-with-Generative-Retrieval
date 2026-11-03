"""Writes a tiny hand-made catalog and splits to tests/fixtures/ (debugging only, not real Semantic IDs).

The 16 fixture items have hand-assigned three-code prefixes that share leading codes (to exercise
the trie) and include collision groups that differ only in the suffix. Each user walks the catalog
with a fixed successor rule (next = item index + 1 mod 16), so every target is predictable from the
history, which makes the tiny overfit check unambiguous. Maps and manifest are produced with A's
`finalize_ids`/`make_manifest`, so the fixture has exactly the real artifact format.

Usage: uv run python -m scripts.make_fixture [--out tests/fixtures]
"""
import argparse
from pathlib import Path

from src.common import write_json, write_jsonl
from src.semantic_ids import finalize_ids, make_manifest

FIXTURE_CODES = [
    (0, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 0),        # share first code 0; one 2-item collision
    (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 2, 3),        # 3-item collision group (1,0,0,*)
    (2, 5, 5), (2, 5, 6), (2, 6, 5), (3, 7, 7),
    (3, 7, 7), (4, 0, 9), (5, 1, 1), (255, 255, 255),  # last one checks range edges
]
NUM_USERS = 8
HISTORY_LEN = 6


def build(out_dir: Path):
    item_ids = [f"ITEM{i:02d}" for i in range(len(FIXTURE_CODES))]
    item_to_sid, sid_to_item = finalize_ids(item_ids, FIXTURE_CODES)
    items = [{"item_id": item, "title": f"Fixture product {i}", "brand": "Fixture", "category": "Test",
              "price": None, "text": f"Fixture product {i}"} for i, item in enumerate(item_ids)]
    splits = []
    for u in range(NUM_USERS):
        start = (2 * u) % len(item_ids)
        history = [item_ids[(start + t) % len(item_ids)] for t in range(HISTORY_LEN)]
        splits.append({"user_id": f"U{u}", "train_items": history[:-2],
                       "val_item": history[-2], "test_item": history[-1]})

    write_jsonl(out_dir / "items.jsonl", items)
    write_jsonl(out_dir / "splits.jsonl", splits)
    write_json(out_dir / "item_ids.json", item_ids)
    write_json(out_dir / "item_to_sid.json", item_to_sid)
    write_json(out_dir / "sid_to_item.json", sid_to_item)
    write_json(out_dir / "sid_manifest.json", make_manifest(item_ids, item_to_sid, seed=0))
    print(f"fixture written to {out_dir} ({len(item_ids)} items, {NUM_USERS} users)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/fixtures")
    build(Path(parser.parse_args().out))
