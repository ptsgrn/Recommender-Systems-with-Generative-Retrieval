import argparse
from collections import Counter, defaultdict
from pathlib import Path
import torch

from scripts.evaluate import load_predictions
from src.common import canonical_hash, file_hash, load_config, read_json, read_jsonl, seed_everything, write_json
from src.embeddings import load_embeddings
from src.rqvae import inspect_model, load_checkpoint
from src.semantic_ids import offset_codes, user_token, validate_maps


def inspect(config, predictions_path=None, split="test"):
    directory = Path(config["artifact_dir"])
    ids, embeddings, embedding_manifest = load_embeddings(directory, config["rqvae"]["dimensions"][0])
    forward = read_json(directory / "item_to_sid.json")
    inverse = read_json(directory / "sid_to_item.json")
    manifest = read_json(directory / "sid_manifest.json")
    validate_maps(forward, inverse, manifest)
    if manifest["catalog_hash"] != canonical_hash(ids) or manifest["checkpoint_sha256"] != file_hash(directory / "rqvae.pt") or manifest["embedding_sha256"] != embedding_manifest["array_sha256"]:
        raise ValueError("Diagnostics artifacts do not belong to the same frozen run")
    model, _ = load_checkpoint(directory / "rqvae.pt")
    metrics, actual_codes = inspect_model(model, torch.from_numpy(embeddings), config["rqvae"]["cpu_batch_size"])
    if any(forward[item][:3] != codes.tolist() for item, codes in zip(ids, actual_codes)):
        raise ValueError("Checkpoint codes differ from frozen maps")
    item_rows = {row["item_id"]: row for row in read_jsonl(Path(config["data_dir"]) / "items.jsonl")}
    splits = read_jsonl(Path(config["data_dir"]) / "splits.jsonl")
    if set(item_rows) != set(ids):
        raise ValueError("Current item catalog differs from frozen run")
    groups = defaultdict(list)
    for item in ids:
        groups[forward[item][0]].append({"item_id": item, "title": item_rows[item]["title"], "category": item_rows[item]["category"]})
    row = splits[0]
    history = row["train_items"] + ([row["val_item"]] if split == "test" else [])
    target = row[split + "_item"]
    vector = torch.from_numpy(embeddings[ids.index(target):ids.index(target) + 1])
    with torch.inference_mode():
        latent_shape = list(model.encoder(vector).shape)
    trace = dict(user_id=row["user_id"], split=split, metadata=item_rows[target],
                 embedding_shape=list(vector.shape), latent_shape=latent_shape,
                 learned_codes=forward[target][:3], collision_suffix=forward[target][3],
                 target_offset_tokens=offset_codes(forward[target], manifest) + [1],
                 history_item_ids=history[-20:], history_tokens=[user_token(row["user_id"], manifest)] +
                 [token for item in history[-20:] for token in offset_codes(forward[item], manifest)] + [1],
                 recommendation={"status": "pending Developer B predictions"})
    if predictions_path:
        predictions = load_predictions(predictions_path, {s["user_id"] for s in splits}, set(ids))
        ranked = predictions.get(row["user_id"], [])
        item = ranked[0] if ranked else None
        trace["recommendation"] = {"status": "from saved predictions", "item_id": item,
                                   "generated_sid": forward[item] if item else None,
                                   "title": item_rows[item]["title"] if item else None,
                                   "note": "SID resolved from ranked item; raw generation tokens belong in B's generation log"}
    report = {**metrics, "mapping_hash": manifest["mapping_hash"],
              "suffix_usage": dict(sorted(Counter(str(codes[3]) for codes in forward.values()).items())),
              "shared_first_code_examples": [{"code": key, "items": values[:5]} for key, values in sorted(groups.items()) if len(values) > 1][:5],
              "interpretation": "Inspect content similarity empirically; codes are not supervised category labels"}
    write_json(Path(config["report_dir"]) / "id_diagnostics.json", report)
    write_json(Path(config["report_dir"]) / "example_trace.json", trace)
    print(report)
    print(trace)


def main():
    parser = argparse.ArgumentParser(description="Inspect learned Semantic IDs and trace an evaluation example")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--predictions")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"], config["threads"])
    inspect(config, args.predictions, args.split)


if __name__ == "__main__":
    main()
