"""Measure local compute before choosing the production subset; never train a recommender."""
import argparse
import copy
import math
from pathlib import Path
import time
import numpy as np
import torch
import yaml

from src.common import device_for, load_config, read_jsonl, runtime_info, seed_everything, write_json
from src.data import load_clean_data, make_subset
from src.embeddings import encode, load_encoder
from src.rqvae import model_from_settings


def benchmark(config):
    seed_everything(config["seed"], config["threads"])
    device = device_for(config)
    settings = config["benchmark"]
    items = read_jsonl(Path(config["data_dir"]) / "items.jsonl")
    model, revision = load_encoder(config, device)
    encode(model, [items[0]["text"]], 1)  # warm-up excluded
    sample = items[:settings["items"]]
    start = time.perf_counter()
    sample_embeddings = encode(model, [item["text"] for item in sample], config["embedding"]["batch_size"])
    embedding_seconds = time.perf_counter() - start
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    # Timing-only synthetic vectors share the observed scale, not production IDs.
    count = max(config["rqvae"]["codebook_size"] * 2, len(sample_embeddings))
    rng = np.random.default_rng(config["seed"])
    timing_vectors = rng.normal(sample_embeddings.mean(0), sample_embeddings.std(0) + 1e-6,
                                size=(count, sample_embeddings.shape[1])).astype(np.float32)
    values = torch.from_numpy(timing_vectors).to(device)
    quantizer = model_from_settings(config["rqvae"]).to(device)
    start = time.perf_counter()
    quantizer.initialize_codebooks(values, config["seed"])
    initialization_seconds = time.perf_counter() - start
    optimizer = torch.optim.Adam(quantizer.parameters(), lr=config["rqvae"]["learning_rate"])
    batch_size = config["rqvae"]["cpu_batch_size"] if device == "cpu" else config["rqvae"]["batch_size"]
    batch = values[:batch_size]
    def step():
        optimizer.zero_grad(set_to_none=True)
        quantizer(batch)["loss"].backward()
        optimizer.step()
    for _ in range(5):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(settings["steps"]):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    seconds_per_step = (time.perf_counter() - start) / settings["steps"]
    raw = Path(config["raw_dir"])
    all_items, users, _ = load_clean_data(raw / "reviews_Beauty_5.json.gz", raw / "meta_Beauty.json.gz", config["data"]["min_interactions"])
    projections = []
    for candidate in settings["user_candidates"]:
        catalog, splits, _ = make_subset(all_items, users, candidate, config["seed"])
        n = len(catalog)
        embedding_estimate = embedding_seconds / len(sample) * n
        # Include a conservative second training-pass allowance for epoch diagnostics.
        training_estimate = seconds_per_step * math.ceil(n / batch_size) * config["rqvae"]["epochs"] * 2
        initialization_estimate = initialization_seconds * max(n / count, 1)
        projections.append(dict(requested_users=candidate, users=len(splits), items=n,
                                embedding_seconds=embedding_estimate, training_seconds=training_estimate,
                                initialization_seconds=initialization_estimate,
                                total_seconds=embedding_estimate + training_estimate + initialization_estimate))
    selected = next((p for p in projections if p["total_seconds"] <= settings["budget_seconds"]), None)
    report = dict(runtime=runtime_info(), device=device, embedding_sample_items=len(sample),
                  embedding_seconds=embedding_seconds, rqvae_seconds_per_step=seconds_per_step,
                  initialization_seconds=initialization_seconds, projections=projections, selected=selected,
                  caveat="Linear estimate, download excluded; training includes 2x allowance for evaluation",
                  model_revision=revision)
    write_json(Path(config["report_dir"]) / "benchmark.json", report)
    if selected is None:
        raise RuntimeError("No local subset fits the configured budget; see reports/benchmark.json")
    chosen = copy.deepcopy(config)
    chosen["data"]["users"] = selected["requested_users"]
    chosen["embedding"]["revision"] = revision
    path = Path(config["artifact_dir"]) / "run_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(chosen, sort_keys=False))
    print(report)
    print(f"Next: python -m scripts.prepare_data --config {path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark and choose a reproducible local run")
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    benchmark(load_config(args.config))


if __name__ == "__main__":
    main()
