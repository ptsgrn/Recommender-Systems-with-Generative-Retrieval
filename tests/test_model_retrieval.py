"""B2/B3 checks: teacher forcing, tiny overfit, checkpoint reload, trie decoding, beam-score sanity."""
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from src.common import load_config, read_jsonl, write_jsonl
from src.dataset import SequenceDataset, build_eval_examples, build_train_examples, collate
from src.recommender import build_model, load_checkpoint, save_checkpoint, train_step
from src.retrieval import CatalogTrie, Retriever
from src.tokenization import EOS, PAD, HistoryEncoder, load_sid_artifacts

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
REC_CFG = load_config(ROOT / "configs" / "small.yaml")["recommender"]


@pytest.fixture(scope="module")
def artifacts():
    return load_sid_artifacts(FIXTURES)


@pytest.fixture(scope="module")
def trained(artifacts):
    """Overfits the 24 fixture training examples with dropout disabled."""
    vocab, item_to_sid, _, _ = artifacts
    torch.manual_seed(0)
    ds = SequenceDataset(build_train_examples(read_jsonl(FIXTURES / "splits.jsonl")),
                         HistoryEncoder(vocab, item_to_sid))
    loader = DataLoader(ds, batch_size=24, shuffle=True, collate_fn=collate)
    model = build_model(REC_CFG, vocab.vocab_size, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = [train_step(model, b, opt, "cpu") for _ in range(150) for b in loader]
    model.eval()
    return model, ds, losses


def test_decoder_input_is_shifted_labels(artifacts):
    vocab, _, _, _ = artifacts
    model = build_model(REC_CFG, vocab.vocab_size)
    labels = torch.tensor([[2, 258, 514, 771, EOS]])
    # Decoder sees [start, c0, c1, c2, c3] while predicting [c0, c1, c2, c3, EOS]: never its own target.
    assert model._shift_right(labels).tolist() == [[PAD, 2, 258, 514, 771]]


def test_tiny_overfit(trained):
    _, _, losses = trained
    assert losses[0] > 7.0 and losses[-1] < 0.1, (losses[0], losses[-1])


def test_overfit_model_generates_targets(trained, artifacts):
    model, ds, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item, num_beams=20)
    recs = r.recommend_batch([e.user_id for e in ds.examples], [e.history for e in ds.examples], k=10)
    hits = sum(rec.items[0][0] == e.target for rec, e in zip(recs, ds.examples))
    assert hits >= 0.9 * len(ds.examples)


def test_checkpoint_reload_reproduces_predictions(trained, artifacts, tmp_path):
    model, ds, _ = trained
    _, _, _, manifest = artifacts
    batch = collate([ds[i] for i in range(8)])
    save_checkpoint(model, tmp_path / "ckpt", {"mapping_hash": manifest["mapping_hash"]})
    reloaded, _ = load_checkpoint(tmp_path / "ckpt", expected_mapping_hash=manifest["mapping_hash"])
    inputs = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
    with torch.no_grad():
        assert torch.allclose(model(**inputs).logits, reloaded(**inputs).logits, atol=1e-5)
    with pytest.raises(ValueError, match="mapping"):
        load_checkpoint(tmp_path / "ckpt", expected_mapping_hash="0" * 64)


def test_trie_allows_only_catalog_continuations(artifacts):
    vocab, item_to_sid, _, _ = artifacts
    trie = CatalogTrie(vocab, item_to_sid)
    assert trie.allowed([]) == sorted({vocab.code_to_token(0, s[0]) for s in item_to_sid.values()})
    assert trie.allowed([vocab.code_to_token(0, 1), 258, 514]) == [770, 771, 772]  # collision group 1,0,0,*
    assert trie.allowed([2, 258, 514, 770]) == [EOS]


def test_lookup_never_crashes_on_bad_ids(trained, artifacts):
    model, _, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item)
    assert r.lookup([2, 258, 514, 770, EOS]) == "ITEM00"
    assert r.lookup([2, 258, 514, 780, EOS]) is None     # valid tokens, tuple not in catalog
    assert r.lookup([258, 2, 514, 770, EOS]) is None     # wrong position order
    assert r.lookup([2, 258, 514, 770]) is None          # missing EOS
    assert r.lookup([]) is None


def test_constrained_output_is_unique_known_items(trained, artifacts):
    model, _, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item, num_beams=20, constrained=True)
    rec = r.recommend("U3", ["ITEM06", "ITEM07"], k=10)
    ids = [i for i, _ in rec.items]
    assert len(ids) == 10 and len(set(ids)) == 10 and all(i in item_to_sid for i in ids)
    assert rec.num_invalid == 0 and rec.shortfall == 0


def test_unconstrained_mode_runs_and_counts(trained, artifacts):
    model, _, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item, num_beams=20, constrained=False)
    rec = r.recommend("U3", ["ITEM06", "ITEM07"], k=10)
    assert rec.num_invalid + rec.num_duplicate + len(rec.items) == rec.num_beams_returned
    assert all(i in item_to_sid for i, _ in rec.items)


def test_beam_scores_match_exhaustive_enumeration(trained, artifacts):
    """With beams >= catalog size the constrained search keeps every valid prefix, so its scores must
    equal exact teacher-forced log-probabilities of every catalog item."""
    model, _, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item, num_beams=len(item_to_sid), constrained=True)
    ex = build_eval_examples(read_jsonl(FIXTURES / "splits.jsonl"), "val")[0]
    beam = r.recommend(ex.user_id, ex.history, k=len(item_to_sid))
    exact = dict(r.score_all_items(ex.user_id, ex.history))
    assert len(beam.items) == len(item_to_sid)
    for item, score in beam.items:
        assert score == pytest.approx(exact[item], abs=1e-3)


def test_predictions_pass_a_evaluator_validation(trained, artifacts, tmp_path):
    from scripts.evaluate import load_predictions
    model, _, _ = trained
    vocab, item_to_sid, sid_to_item, _ = artifacts
    r = Retriever(model, vocab, item_to_sid, sid_to_item)
    examples = build_eval_examples(read_jsonl(FIXTURES / "splits.jsonl"), "test")
    recs = r.recommend_batch([e.user_id for e in examples], [e.history for e in examples], k=10)
    rows = [{"user_id": e.user_id, "item_ids": [i for i, _ in rec.items], "scores": [s for _, s in rec.items]}
            for e, rec in zip(examples, recs)]
    write_jsonl(tmp_path / "pred.jsonl", rows)
    loaded = load_predictions(tmp_path / "pred.jsonl", {e.user_id for e in examples}, set(item_to_sid))
    assert all(len(v) == 10 for v in loaded.values())
