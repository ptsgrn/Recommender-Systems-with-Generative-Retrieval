"""Developer B's production tokenizer on top of A's frozen manifest (`src/semantic_ids.py`).

Two representations are kept apart on purpose:
  * a *code tuple* (sid): raw Semantic ID, e.g. (7, 7, 130, 0), each value in [0, size_of_position)
  * *tokens*: integer vocabulary ids fed to T5, e.g. [9, 265, 644, 770]

Vocabulary layout with four positions of 256 codes and 2000 user buckets:
  0 PAD | 1 EOS | 2..257 pos0 | 258..513 pos1 | 514..769 pos2 | 770..1025 pos3 | 1026..3025 users
So code 7 at position 0 (token 9) and code 7 at position 1 (token 265) are different tokens.
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from src.common import read_json
from src.semantic_ids import validate_maps

PAD = 0
EOS = 1
NUM_SPECIAL = 2
IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss

Sid = Tuple[int, ...]


def user_bucket(user_id: str, num_buckets: int) -> int:
    """Stable across processes, unlike Python's salted hash(); same rule as A's `user_token`."""
    return int.from_bytes(hashlib.sha256(user_id.encode("utf-8")).digest(), "big") % num_buckets


def sid_key(sid: Sequence[int]) -> str:
    """Key format used by sid_to_item.json: "c0,c1,c2,c3"."""
    return ",".join(str(int(c)) for c in sid)


class SIDVocab:
    """Maps between code tuples, offset tokens and user tokens."""

    def __init__(self, position_sizes: Sequence[int], num_user_buckets: int):
        self.position_sizes = [int(s) for s in position_sizes]
        self.num_positions = len(self.position_sizes)
        self.num_user_buckets = int(num_user_buckets)
        self.position_offsets = []
        offset = NUM_SPECIAL
        for size in self.position_sizes:
            self.position_offsets.append(offset)
            offset += size
        self.user_offset = offset
        self.vocab_size = offset + self.num_user_buckets

    @classmethod
    def from_manifest(cls, manifest: dict) -> "SIDVocab":
        vocab = cls(manifest["position_vocab_sizes"], manifest["user_bucket_count"])
        # The manifest is the frozen source of truth; refuse to silently disagree with it.
        special = manifest["special_tokens"]
        assert special["PAD"] == PAD and special["EOS"] == EOS and special["decoder_start"] == PAD, special
        assert list(manifest["offsets"]) == vocab.position_offsets
        assert manifest["user_offset"] == vocab.user_offset
        assert manifest["vocabulary_size"] == vocab.vocab_size
        return vocab

    # --- code tuple <-> tokens -------------------------------------------------
    def code_to_token(self, position: int, code: int) -> int:
        if not 0 <= code < self.position_sizes[position]:
            raise ValueError(f"code {code} out of range for position {position}")
        return self.position_offsets[position] + code

    def token_to_code(self, token: int) -> Optional[Tuple[int, int]]:
        """Returns (position, code), or None for special/user/out-of-range tokens."""
        for position, (offset, size) in enumerate(zip(self.position_offsets, self.position_sizes)):
            if offset <= token < offset + size:
                return position, token - offset
        return None

    def sid_to_tokens(self, sid: Sequence[int]) -> List[int]:
        if len(sid) != self.num_positions:
            raise ValueError(f"expected {self.num_positions} codes, got {len(sid)}")
        return [self.code_to_token(p, c) for p, c in enumerate(sid)]

    def tokens_to_sid(self, tokens: Sequence[int]) -> Optional[Sid]:
        """Inverse of sid_to_tokens. Returns None for any malformed sequence
        (wrong length, wrong position order, special or user tokens)."""
        if len(tokens) != self.num_positions:
            return None
        sid = []
        for expected_position, token in enumerate(tokens):
            decoded = self.token_to_code(int(token))
            if decoded is None or decoded[0] != expected_position:
                return None
            sid.append(decoded[1])
        return tuple(sid)

    def user_token(self, user_id: str) -> int:
        return self.user_offset + user_bucket(user_id, self.num_user_buckets)

    def describe(self, token: int) -> str:
        """Human-readable name of a token, for printing batches."""
        if token == PAD:
            return "PAD"
        if token == EOS:
            return "EOS"
        if token == IGNORE_INDEX:
            return "IGN"
        decoded = self.token_to_code(token)
        if decoded is not None:
            return f"p{decoded[0]}:{decoded[1]}"
        if self.user_offset <= token < self.vocab_size:
            return f"user{token - self.user_offset}"
        return f"?{token}"


class HistoryEncoder:
    """Implements the `encode_history(user_id, item_ids)` contract."""

    def __init__(self, vocab: SIDVocab, item_to_sid: Dict[str, Sequence[int]], max_items: int = 20):
        self.vocab = vocab
        self.item_to_sid = item_to_sid
        self.max_items = max_items

    def encode_history(self, user_id: str, item_ids: Sequence[str]) -> List[int]:
        """User token + flattened four-token IDs of the latest `max_items` items + EOS."""
        recent = list(item_ids)[-self.max_items:] if self.max_items > 0 else []
        tokens = [self.vocab.user_token(user_id)]
        for item_id in recent:
            tokens.extend(self.vocab.sid_to_tokens(self.item_to_sid[item_id]))
        tokens.append(EOS)
        return tokens

    def encode_target(self, item_id: str) -> List[int]:
        """Four ID tokens + EOS: the decoder labels for one next-item prediction."""
        return self.vocab.sid_to_tokens(self.item_to_sid[item_id]) + [EOS]


def load_sid_artifacts(directory) -> Tuple[SIDVocab, Dict[str, List[int]], Dict[str, str], dict]:
    """Loads A's frozen maps; A's `validate_maps` checks the mapping hash, ranges and exact inverse."""
    directory = Path(directory)
    manifest = read_json(directory / "sid_manifest.json")
    item_to_sid = read_json(directory / "item_to_sid.json")
    sid_to_item = read_json(directory / "sid_to_item.json")
    validate_maps(item_to_sid, sid_to_item, manifest)
    return SIDVocab.from_manifest(manifest), item_to_sid, sid_to_item, manifest
