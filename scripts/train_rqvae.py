import argparse
from pathlib import Path
import time
import torch

from src.common import canonical_hash, device_for, load_config, runtime_info, seed_everything, write_json
from src.embeddings import load_embeddings
from src.rqvae import inspect_model, model_from_settings


def train(config):
    seed_everything(config["seed"], config["threads"])
    settings, device = config["rqvae"], device_for(config)
    directory = Path(config["artifact_dir"])
    if (directory / "sid_manifest.json").exists():
        raise ValueError("This run has frozen Semantic IDs; train in a new artifact directory")
    ids, embeddings, manifest = load_embeddings(directory, settings["dimensions"][0])
    values = torch.from_numpy(embeddings).to(device)
    model = model_from_settings(settings).to(device)
    start = time.perf_counter()
    model.initialize_codebooks(values, config["seed"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    batch_size = settings["cpu_batch_size"] if device == "cpu" else settings["batch_size"]
    history, best, stale, steps = [], float("inf"), 0, 0
    initial, _ = inspect_model(model, values, batch_size)
    for epoch in range(settings["epochs"]):
        model.train()
        for indices in torch.randperm(len(values), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            output = model(values[indices])
            if not torch.isfinite(output["loss"]):
                raise ValueError("Nonfinite quantizer loss")
            output["loss"].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing or nonfinite quantizer gradient")
            optimizer.step()
            steps += 1
        metrics, _ = inspect_model(model, values, batch_size)
        metrics.update(epoch=epoch + 1, steps=steps, seconds=time.perf_counter() - start)
        history.append(metrics)
        print(metrics, flush=True)
        if metrics["reconstruction"] < best:
            best, stale = metrics["reconstruction"], 0
            checkpoint = dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                              settings=settings, seed=config["seed"], epoch=epoch + 1, steps=steps,
                              catalog_hash=canonical_hash(ids), embedding_hash=manifest["array_sha256"],
                              reconstruction=best)
            temporary = directory / "rqvae.tmp.pt"
            torch.save(checkpoint, temporary)
            temporary.replace(directory / "rqvae.pt")
        else:
            stale += 1
        write_json(directory / "rqvae_training.json", dict(initial=initial, epochs=history, config=config,
                   selection="lowest full-catalog reconstruction; unsupervised metadata only",
                   best_reconstruction=best, runtime=runtime_info(), device=device,
                   seconds=time.perf_counter() - start))
        if stale >= settings["patience"]:
            break


def main():
    parser = argparse.ArgumentParser(description="Train residual quantization on cached item embeddings")
    parser.add_argument("--config", default="configs/small.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
