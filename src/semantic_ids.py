"""Frozen raw-code maps and their shared vocabulary contract (not B's tokenizer)."""
from collections import defaultdict
import hashlib
import numbers
import numpy as np

from src.common import canonical_hash


def finalize_ids(item_ids, codes, suffix_size=256):
    codes = np.asarray(codes)
    if codes.shape != (len(item_ids), 3) or len(set(item_ids)) != len(item_ids) or not np.issubdtype(codes.dtype, np.integer) or (codes < 0).any():
        raise ValueError("Expected unique item IDs and nonnegative integer three-code rows")
    groups = defaultdict(list)
    for item, code in zip(item_ids, codes):
        groups[tuple(int(value) for value in code)].append(item)
    forward, inverse = {}, {}
    for prefix, items in sorted(groups.items()):
        if len(items) > suffix_size:
            raise ValueError(f"Collision group has {len(items)} items; inspect collapse before enlarging suffix vocabulary")
        for suffix, item in enumerate(sorted(items)):
            sid = [*prefix, suffix]
            forward[item] = sid
            inverse[",".join(map(str, sid))] = item
    return forward, inverse


def make_manifest(item_ids, forward, seed, codebook_size=256, suffix_size=256):
    sizes = [codebook_size] * 3 + [suffix_size]
    offsets, cursor = [], 2
    for size in sizes:
        offsets.append(cursor)
        cursor += size
    return dict(version=1, position_vocab_sizes=sizes, offsets=offsets,
                special_tokens={"PAD": 0, "EOS": 1, "decoder_start": 0},
                user_bucket_count=2000, user_offset=cursor, vocabulary_size=cursor + 2000,
                user_hash="SHA-256 UTF-8 user_id interpreted as unsigned big-endian integer modulo 2000",
                hash_method="SHA-256 of UTF-8 JSON: sort_keys=True, separators=(',', ':'), ensure_ascii=False",
                seed=seed, catalog_hash=canonical_hash(item_ids), mapping_hash=canonical_hash(forward))


def offset_codes(codes, manifest):
    if len(codes) != 4 or any(not isinstance(c, numbers.Integral) or isinstance(c, bool) or c < 0 or c >= size for c, size in zip(codes, manifest["position_vocab_sizes"])):
        raise ValueError("Invalid raw four-code Semantic ID")
    return [int(code) + offset for code, offset in zip(codes, manifest["offsets"])]


def user_token(user_id, manifest):
    bucket = int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest(), "big") % manifest["user_bucket_count"]
    return manifest["user_offset"] + bucket


def validate_maps(forward, inverse, manifest):
    if canonical_hash(forward) != manifest["mapping_hash"]:
        raise ValueError("Mapping hash mismatch")
    if len(forward) != len(inverse):
        raise ValueError("Maps differ in size")
    for item, codes in forward.items():
        offset_codes(codes, manifest)
        if inverse.get(",".join(map(str, codes))) != item:
            raise ValueError("Semantic ID maps are not exact inverses")
