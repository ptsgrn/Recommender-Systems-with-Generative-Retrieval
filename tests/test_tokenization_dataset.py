"""B1 checks: vocabulary ranges, round-trips, stable user buckets, leakage-free examples."""
from pathlib import Path

import pytest

from src.common import read_json, read_jsonl, write_json
from src.dataset import SequenceDataset, build_eval_examples, build_train_examples, collate
from src.semantic_ids import user_token
from src.tokenization import EOS, IGNORE_INDEX, PAD, HistoryEncoder, SIDVocab, load_sid_artifacts

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def artifacts():
    return load_sid_artifacts(FIXTURES)


def test_vocab_matches_a_manifest(artifacts):
    vocab, _, _, manifest = artifacts
    assert vocab.position_offsets == manifest["offsets"] == [2, 258, 514, 770]
    assert vocab.user_offset == 1026 and vocab.vocab_size == manifest["vocabulary_size"] == 3026


def test_ranges_do_not_overlap():
    v = SIDVocab([256] * 4, 2000)
    ranges = [(PAD, PAD + 1), (EOS, EOS + 1)]
    ranges += [(o, o + s) for o, s in zip(v.position_offsets, v.position_sizes)]
    ranges.append((v.user_offset, v.vocab_size))
    owners = {}
    for r, (lo, hi) in enumerate(ranges):
        for t in range(lo, hi):
            assert t not in owners, f"token {t} in two ranges"
            owners[t] = r
    assert len(owners) == v.vocab_size


def test_same_code_different_position_is_different_token():
    v = SIDVocab([256] * 4, 2000)
    assert v.code_to_token(0, 7) == 9 and v.code_to_token(1, 7) == 265


def test_sid_round_trip_and_malformed(artifacts):
    vocab, item_to_sid, _, _ = artifacts
    for sid in item_to_sid.values():
        assert vocab.tokens_to_sid(vocab.sid_to_tokens(sid)) == tuple(sid)
    assert vocab.tokens_to_sid([258, 2, 514, 770]) is None   # positions swapped
    assert vocab.tokens_to_sid([2, 258, 514]) is None        # too short
    assert vocab.tokens_to_sid([2, 258, 514, EOS]) is None   # special token inside ID
    assert vocab.tokens_to_sid([2, 258, 514, 1500]) is None  # user token inside ID
    with pytest.raises(ValueError):
        vocab.code_to_token(0, 256)


def test_user_token_is_stable_and_matches_a(artifacts):
    vocab, _, _, manifest = artifacts
    # Fixed value: guards against switching to Python's per-process salted hash().
    assert vocab.user_token("U0") == 1026 + 227
    for i in range(100):
        assert vocab.user_token(f"user{i}") == user_token(f"user{i}", manifest)


def test_history_encoding_truncates_to_latest_items(artifacts):
    vocab, item_to_sid, _, _ = artifacts
    enc = HistoryEncoder(vocab, item_to_sid, max_items=2)
    tokens = enc.encode_history("U0", ["ITEM00", "ITEM01", "ITEM02"])
    assert tokens[0] == vocab.user_token("U0") and tokens[-1] == EOS
    assert tokens[1:-1] == vocab.sid_to_tokens(item_to_sid["ITEM01"]) + vocab.sid_to_tokens(item_to_sid["ITEM02"])
    assert len(HistoryEncoder(vocab, item_to_sid, 20).encode_history("U0", ["ITEM00"] * 30)) == 82


def test_held_out_items_never_become_training_targets():
    splits = read_jsonl(FIXTURES / "splits.jsonl")
    train = build_train_examples(splits)
    for s in splits:
        mine = [e for e in train if e.user_id == s["user_id"]]
        assert [e.target for e in mine] == s["train_items"][1:]
        for e in mine:
            assert e.history + [e.target] == s["train_items"][:len(e.history) + 1]
    val = build_eval_examples(splits, "val")
    test = build_eval_examples(splits, "test")
    for s, v, t in zip(splits, val, test):
        assert v.history == s["train_items"] and v.target == s["val_item"]
        assert t.history == s["train_items"] + [s["val_item"]] and t.target == s["test_item"]


def test_collate_shapes_and_padding(artifacts):
    vocab, item_to_sid, _, _ = artifacts
    ds = SequenceDataset(build_train_examples(read_jsonl(FIXTURES / "splits.jsonl")),
                         HistoryEncoder(vocab, item_to_sid))
    batch = collate([ds[0], ds[2]])  # different history lengths
    assert batch["labels"].shape == (2, 5)
    assert batch["input_ids"].shape[1] == max(len(ds.inputs[0]), len(ds.inputs[2]))
    assert (batch["input_ids"][batch["attention_mask"] == 0] == PAD).all()
    assert not (batch["labels"] == IGNORE_INDEX).any()


def test_artifact_loader_rejects_tampered_mapping(tmp_path):
    for name in ["sid_to_item.json", "sid_manifest.json"]:
        write_json(tmp_path / name, read_json(FIXTURES / name))
    tampered = read_json(FIXTURES / "item_to_sid.json")
    tampered["ITEM00"] = [9, 9, 9, 9]
    write_json(tmp_path / "item_to_sid.json", tampered)
    with pytest.raises(ValueError, match="hash"):
        load_sid_artifacts(tmp_path)
