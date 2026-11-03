import argparse
from pathlib import Path
import torch

from src.common import canonical_hash, file_hash, load_config, seed_everything, write_json
from src.embeddings import load_embeddings
from src.rqvae import inspect_model, load_checkpoint
from src.semantic_ids import finalize_ids, make_manifest, validate_maps


def build(config):
    directory = Path(config["artifact_dir"])
    ids, embeddings, embedding_manifest = load_embeddings(directory, config["rqvae"]["dimensions"][0])
    model, checkpoint = load_checkpoint(directory / "rqvae.pt")
    if checkpoint["catalog_hash"] != canonical_hash(ids) or checkpoint["embedding_hash"] != embedding_manifest["array_sha256"]:
        raise ValueError("Checkpoint and embedding cache mismatch")
    metrics, codes = inspect_model(model, torch.from_numpy(embeddings), config["rqvae"]["cpu_batch_size"])
    if all(count <= 1 for count in metrics["active_codes"]) and len(ids) > 1:
        raise ValueError("All quantizer levels collapsed; repair training before exporting IDs")
    forward, inverse = finalize_ids(ids, codes)
    manifest = make_manifest(ids, forward, checkpoint["seed"], checkpoint["settings"]["codebook_size"])
    manifest.update(checkpoint_sha256=file_hash(directory / "rqvae.pt"), embedding_sha256=embedding_manifest["array_sha256"])
    validate_maps(forward, inverse, manifest)
    # An existing frozen mapping cannot be overwritten with a different one.
    if (directory / "sid_manifest.json").exists():
        from src.common import read_json
        if read_json(directory / "sid_manifest.json") != manifest:
            raise ValueError("Frozen Semantic IDs differ; use a new artifact directory")
    write_json(directory / "item_to_sid.json", forward)
    write_json(directory / "sid_to_item.json", inverse)
    write_json(directory / "sid_manifest.json", manifest)
    print({"items": len(forward), "mapping_hash": manifest["mapping_hash"], **metrics})


def main():
    parser = argparse.ArgumentParser(description="Export frozen unique four-code Semantic IDs")
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    seed_everything(config["seed"], config["threads"])
    build(config)


if __name__ == "__main__":
    main()
