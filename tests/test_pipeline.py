"""Synthetic end-to-end proof; deliberately not reported as a real experiment."""
import copy
import numpy as np
import pytest

from scripts.build_semantic_ids import build
from scripts.evaluate import evaluate
from scripts.inspect_example import inspect
from scripts.train_rqvae import train
from src.common import file_hash, load_config, read_json, write_json, write_jsonl
from src.data import clean_item


def test_checkpoint_maps_evaluation_and_trace(tmp_path):
    config = copy.deepcopy(load_config())
    for key, folder in (("data_dir", "processed"), ("artifact_dir", "artifacts"), ("report_dir", "reports")):
        config[key] = str(tmp_path / folder)
    config["rqvae"].update(dimensions=[4, 8, 2], codebook_size=4, epochs=3, cpu_batch_size=4, warmup_epochs=1)
    config["embedding"]["dimension"] = 4
    config["device"] = "cpu"
    data, artifacts, reports = (tmp_path / name for name in ("processed", "artifacts", "reports"))
    ids = [f"item{i:02}" for i in range(12)]
    write_jsonl(data / "items.jsonl", [clean_item({"asin": item, "title": item}) for item in ids])
    write_jsonl(data / "splits.jsonl", [{"user_id": "u", "train_items": ids[:3], "val_item": ids[3], "test_item": ids[4]}])
    artifacts.mkdir()
    values = np.random.default_rng(42).normal(size=(12, 4)).astype(np.float32)
    np.save(artifacts / "item_embeddings.npy", values)
    write_json(artifacts / "item_ids.json", ids)
    write_json(artifacts / "embedding_manifest.json", {"identity": {"item_ids": ids}, "array_sha256": file_hash(artifacts / "item_embeddings.npy")})
    train(config)
    training_log = read_json(artifacts / "rqvae_training.json")
    assert len(training_log["warmup"]) == 1
    assert training_log["warmup"][0]["latent_std_mean"] > 0
    build(config)
    first = read_json(artifacts / "item_to_sid.json")
    build(config)
    assert read_json(artifacts / "item_to_sid.json") == first
    with pytest.raises(ValueError, match="frozen"):
        train(config)
    predictions_path = tmp_path / "predictions.jsonl"
    write_jsonl(predictions_path, [{"user_id": "u", "item_ids": [ids[4]]}])
    result = evaluate(config, predictions_path=predictions_path)
    assert result["model"]["Recall@5"] == 1
    inspect(config, predictions_path)
    trace = read_json(reports / "example_trace.json")
    assert trace["history_item_ids"] == ids[:4]
    assert trace["recommendation"]["item_id"] == ids[4]
    assert len(trace["history_tokens"]) == 18
    assert read_json(reports / "id_diagnostics.json")["mapping_hash"] == read_json(artifacts / "sid_manifest.json")["mapping_hash"]
