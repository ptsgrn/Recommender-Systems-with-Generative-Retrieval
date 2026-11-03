"""Frozen content encoder and provenance-checked embedding cache."""
from pathlib import Path
import re
import time
import numpy as np

from src.common import canonical_hash, file_hash, read_json, read_jsonl, write_json

TEXT_TEMPLATE_VERSION = 1


def resolve_revision(settings):
    if re.fullmatch(r"[0-9a-f]{40}", settings["revision"]):
        return settings["revision"]
    from huggingface_hub import HfApi
    return HfApi().model_info(settings["model"], revision=settings["revision"]).sha


def load_encoder(config, device):
    from sentence_transformers import SentenceTransformer
    settings = config["embedding"]
    revision = resolve_revision(settings)
    model = SentenceTransformer(settings["model"], revision=revision, device=device,
                                cache_folder=".cache/sentence-transformers", trust_remote_code=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, revision


def encode(model, texts, batch_size):
    import torch
    with torch.inference_mode():
        values = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              show_progress_bar=True, normalize_embeddings=False)
    return np.asarray(values, dtype=np.float32)


def embedding_identity(items, config, revision):
    return {"item_ids": [item["item_id"] for item in items],
            "texts_hash": canonical_hash([item["text"] for item in items]),
            "template_version": TEXT_TEMPLATE_VERSION,
            "model": config["embedding"]["model"], "revision": revision,
            "dimension": config["embedding"]["dimension"]}


def validate_embeddings(ids, values, dimension=768):
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Embedding item IDs must be nonempty and unique")
    if values.shape != (len(ids), dimension) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Embedding cache must be aligned finite float32 [N, dimension]")


def load_embeddings(directory, dimension=768):
    directory = Path(directory)
    ids = read_json(directory / "item_ids.json")
    values = np.load(directory / "item_embeddings.npy", allow_pickle=False)
    validate_embeddings(ids, values, dimension)
    manifest = read_json(directory / "embedding_manifest.json")
    if manifest["identity"]["item_ids"] != ids or manifest["array_sha256"] != file_hash(directory / "item_embeddings.npy"):
        raise ValueError("Embedding cache order or array checksum mismatch")
    return ids, values, manifest


def embed_items(config, device):
    directory = Path(config["artifact_dir"])
    items = read_jsonl(Path(config["data_dir"]) / "items.jsonl")
    items.sort(key=lambda item: item["item_id"])
    # Resolve immutable revision even for cache reuse; changed remote main must not
    # silently select old embeddings. Set config revision to the saved SHA offline.
    revision = resolve_revision(config["embedding"])
    identity = embedding_identity(items, config, revision)
    if (directory / "embedding_manifest.json").exists():
        ids, values, manifest = load_embeddings(directory, config["embedding"]["dimension"])
        if manifest["identity"] != identity:
            raise ValueError("Stale embedding cache; use a fresh artifact directory")
        print("Reusing verified embedding cache", values.shape)
        return
    model, actual_revision = load_encoder(config, device)
    if actual_revision != revision:
        raise ValueError("Model revision changed during loading; pin a commit SHA")
    start = time.perf_counter()
    values = encode(model, [item["text"] for item in items], config["embedding"]["batch_size"])
    encoding_seconds = time.perf_counter() - start
    ids = identity["item_ids"]
    validate_embeddings(ids, values, config["embedding"]["dimension"])
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "item_embeddings.tmp.npy"
    np.save(temporary, values, allow_pickle=False)
    temporary.replace(directory / "item_embeddings.npy")
    write_json(directory / "item_ids.json", ids)
    write_json(directory / "embedding_manifest.json", {
        "identity": identity, "array_sha256": file_hash(directory / "item_embeddings.npy"),
        "encoding_seconds": encoding_seconds,
        "shape": list(values.shape), "normalization": "preserve model modules; no extra L2 normalization",
        "examples": [{"item_id": item["item_id"], "text": item["text"], "shape": [values.shape[1]]} for item in items[:5]]})
    print("Saved embeddings", values.shape)
