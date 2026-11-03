"""Deterministic metadata cleaning and user-level chronological splits."""
import ast
from collections import Counter, defaultdict
import gzip
import html
import json
import math
import random
import re

from src.common import canonical_hash


def records(path, stats=None, prefix="records"):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if stats is not None:
                stats[prefix + "_total"] += 1
            try:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    value = ast.literal_eval(line)
                if not isinstance(value, dict):
                    raise ValueError("Expected object")
                yield index, value
            except (ValueError, SyntaxError, TypeError):
                if stats is not None:
                    stats[prefix + "_malformed"] += 1


def clean_text(value):
    return re.sub(r"\s+", " ", html.unescape(value)).strip() if isinstance(value, str) else ""


def clean_item(raw):
    item_id, title = clean_text(raw.get("asin")), clean_text(raw.get("title"))
    if not item_id or not title:
        return None
    paths = raw.get("categories", [])
    paths = paths if isinstance(paths, list) else []
    categories = []
    for path in paths:
        parts = path if isinstance(path, list) else [path]
        label = " > ".join(filter(None, (clean_text(part) for part in parts)))
        if label:
            categories.append(label)
    category = " | ".join(sorted(set(categories))) or "Unknown"
    brand = clean_text(raw.get("brand")) or "Unknown"
    price = raw.get("price")
    try:
        price = float(price)
        if not math.isfinite(price) or price < 0:
            price = None
    except (ValueError, TypeError):
        price = None
    price_text = f"{price:.2f}" if price is not None else "Unknown"
    text = f"Title: {title}. Brand: {brand}. Category: {category}. Price: {price_text}."
    return dict(item_id=item_id, title=title, brand=brand, category=category, price=price, text=text)


def load_clean_data(reviews_path, metadata_path, min_interactions=5):
    stats = Counter()
    items, metadata_seen = {}, set()
    for _, raw in records(metadata_path, stats, "metadata"):
        asin = clean_text(raw.get("asin"))
        if asin in metadata_seen:
            stats["metadata_duplicate_asins"] += 1
        metadata_seen.add(asin)
        item = clean_item(raw)
        if item is None:
            stats["metadata_unusable"] += 1
        elif item["item_id"] not in items:
            items[item["item_id"]] = item
    users, seen, source_users, source_items = defaultdict(list), set(), set(), set()
    for row, raw in records(reviews_path, stats, "interactions"):
        user, asin, timestamp = raw.get("reviewerID"), raw.get("asin"), raw.get("unixReviewTime")
        if not isinstance(user, str) or not user or not isinstance(asin, str) or not isinstance(timestamp, int) or isinstance(timestamp, bool):
            stats["interactions_invalid_fields"] += 1
            continue
        source_users.add(user)
        source_items.add(asin)
        signature = canonical_hash(raw)
        if signature in seen:
            stats["interactions_exact_duplicates"] += 1
            continue
        seen.add(signature)
        if asin not in items:
            stats["interactions_missing_metadata"] += 1
            continue
        users[user].append(dict(item_id=asin, timestamp=timestamp, source_row=row))
    eligible = {u: sorted(events, key=lambda e: (e["timestamp"], e["source_row"]))
                for u, events in users.items() if len(events) >= min_interactions}
    stats.update(source_users=len(source_users), source_items=len(source_items),
                 usable_metadata_items=len(items), eligible_users=len(eligible),
                 excluded_users=len(source_users) - len(eligible))
    return items, eligible, dict(stats)


def select_users(users, count, seed):
    return sorted(random.Random(seed).sample(sorted(users), min(count, len(users))))


def make_subset(items, users, count, seed):
    selected = select_users(users, count, seed)
    splits, audit = [], []
    catalog = set()
    for user in selected:
        events = users[user]
        history = [e["item_id"] for e in events]
        catalog.update(history)
        splits.append(dict(user_id=user, train_items=history[:-2], val_item=history[-2], test_item=history[-1]))
        audit.append(dict(user_id=user, events=events))
    return [items[asin] for asin in sorted(catalog)], splits, audit
