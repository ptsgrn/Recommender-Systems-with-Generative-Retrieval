"""Trains the TIGER recommender on Semantic ID histories and selects a checkpoint on validation data.

Reads `data_dir/splits.jsonl` and the frozen maps in `artifact_dir`; writes `artifact_dir/recommender/`.
Selection uses validation NDCG@10 on a fixed validation subset; test data is never read here.

    uv run python -m scripts.train_recommender --config artifacts/run_config.yaml --benchmark-steps 100
    uv run python -m scripts.train_recommender --config artifacts/run_config.yaml

Tiny overfit diagnostic (dropout off, trains and checks on the fixture's training examples):
    uv run python -m scripts.train_recommender --data-dir tests/fixtures --sid-dir tests/fixtures \
        --out runs/overfit --overfit --max-steps 300 --eval-every 100 --batch-size 32
"""
import argparse
import importlib.metadata
import json
import math
import platform
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.common import device_for, load_config, read_jsonl, runtime_info, seed_everything, write_json
from src.dataset import SequenceDataset, build_eval_examples, build_train_examples, collate
from src.recommender import build_model, evaluate_loss, save_checkpoint, train_step
from src.retrieval import Retriever
from src.tokenization import HistoryEncoder, load_sid_artifacts


def b_sections(config):
    """B's settings; a run config written before B's sections existed falls back to the defaults."""
    defaults = load_config()
    return {key: config.get(key, defaults[key]) for key in ("sequence", "recommender", "retrieval")}


def ranking_metrics(retriever, examples, k=10, batch_size=32):
    """Recall@k / NDCG@k with one relevant item per user, used only for checkpoint selection.
    Users with short or empty outputs still count. Final numbers come from A's scripts.evaluate."""
    recall = ndcg = 0.0
    returned = invalid = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        recs = retriever.recommend_batch([e.user_id for e in chunk], [e.history for e in chunk], k)
        for e, rec in zip(chunk, recs):
            ranked = [item for item, _ in rec.items]
            returned += len(ranked)
            invalid += rec.num_invalid
            if e.target in ranked:
                recall += 1.0
                ndcg += 1.0 / math.log2(ranked.index(e.target) + 2)
    n = max(len(examples), 1)
    return {f"recall@{k}": recall / n, f"ndcg@{k}": ndcg / n, "mean_returned": returned / n,
            "invalid_beams": invalid, "num_users": len(examples)}


def main():
    p = argparse.ArgumentParser(description="Train the TIGER sequence model")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--data-dir", help="directory with splits.jsonl (default: config data_dir)")
    p.add_argument("--sid-dir", help="directory with frozen ID maps (default: config artifact_dir)")
    p.add_argument("--out", help="output directory (default: artifact_dir/recommender)")
    p.add_argument("--overfit", action="store_true", help="dropout 0; evaluate on training examples")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--val-subset", type=int)
    p.add_argument("--benchmark-steps", type=int, default=0,
                   help="only time this many steps (e.g. 100) and exit, to size the real run")
    args = p.parse_args()

    config = load_config(args.config)
    b = b_sections(config)
    rec_cfg, ret_cfg = b["recommender"], b["retrieval"]
    max_steps = args.max_steps or rec_cfg["max_steps"]
    eval_every = args.eval_every or rec_cfg["eval_every"]
    batch_size = args.batch_size or rec_cfg["batch_size"]
    lr = args.learning_rate or rec_cfg["learning_rate"]
    val_subset = args.val_subset if args.val_subset is not None else rec_cfg["val_subset"]
    data_dir = Path(args.data_dir or config["data_dir"])
    sid_dir = Path(args.sid_dir or config["artifact_dir"])
    out_dir = Path(args.out or Path(config["artifact_dir"]) / "recommender")

    seed_everything(config["seed"], config.get("threads", 2))
    device = device_for(config)
    vocab, item_to_sid, sid_to_item, manifest = load_sid_artifacts(sid_dir)
    encoder = HistoryEncoder(vocab, item_to_sid, b["sequence"]["max_history"])

    splits = read_jsonl(data_dir / "splits.jsonl")
    train_examples = build_train_examples(splits)
    if args.overfit:
        check_examples = train_examples
    else:
        # Fixed, seeded validation subset so every checkpoint is compared on the same users.
        val_examples = build_eval_examples(splits, "val")
        order = torch.randperm(len(val_examples), generator=torch.Generator().manual_seed(config["seed"])).tolist()
        check_examples = [val_examples[i] for i in order[:val_subset]] if val_subset else val_examples

    train_ds = SequenceDataset(train_examples, encoder)
    check_ds = SequenceDataset(check_examples, encoder)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(config["seed"]))
    check_loader = DataLoader(check_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model = build_model(rec_cfg, vocab.vocab_size, dropout=0.0 if args.overfit else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=rec_cfg["weight_decay"])
    retriever = Retriever(model, vocab, item_to_sid, sid_to_item, b["sequence"]["max_history"],
                          ret_cfg["num_beams"], ret_cfg["constrained"], device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device} params={n_params:,} vocab={vocab.vocab_size} train_examples={len(train_ds)} "
          f"check_examples={len(check_ds)} ({'train/overfit' if args.overfit else 'validation'}) "
          f"mapping={manifest['mapping_hash'][:12]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.jsonl", "w")
    best = {"ndcg@10": -1.0, "step": None}
    step, t0, window_t0 = 0, time.time(), time.time()
    selection = "train-set ndcg@10 (overfit diagnostic)" if args.overfit else \
        f"validation ndcg@10, fixed seeded subset of {len(check_examples)} users"

    while step < max_steps:
        for batch in loader:
            loss = train_step(model, batch, optimizer, device)
            step += 1
            if step % 100 == 0:
                print(f"step {step} loss {loss:.4f} ({time.time() - window_t0:.1f}s / 100 steps)")
                window_t0 = time.time()
            if args.benchmark_steps and step >= args.benchmark_steps:
                secs = time.time() - t0
                print(f"benchmark: {step} steps in {secs:.1f}s -> {secs / step:.3f}s/step on {device}; "
                      f"one epoch = {len(loader)} steps")
                return
            if step % eval_every == 0 or step == max_steps:
                check_loss = evaluate_loss(model, check_loader, device)
                metrics = ranking_metrics(retriever, check_examples, k=ret_cfg["top_k"])
                record = {"step": step, "train_loss": loss, "check_loss": check_loss,
                          "elapsed_s": round(time.time() - t0, 1), **metrics}
                print(json.dumps(record))
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()
                if metrics["ndcg@10"] > best["ndcg@10"]:
                    best = {"ndcg@10": metrics["ndcg@10"], "step": step}
                    save_checkpoint(model, out_dir / "best", {
                        "step": step, "selection": selection, "metrics": record,
                        "mapping_hash": manifest["mapping_hash"], "catalog_hash": manifest["catalog_hash"],
                        "sections": b, "seed": config["seed"], "learning_rate": lr, "batch_size": batch_size,
                        "device": str(device), "overfit_mode": args.overfit,
                        "decoding": "constrained-trie" if ret_cfg["constrained"] else "unconstrained"})
                    model.to(device)
            if step >= max_steps:
                break

    try:
        runtime = runtime_info()
    except importlib.metadata.PackageNotFoundError:  # B-only environment without A's packages
        runtime = {"python": platform.python_version(), "torch": torch.__version__}
    write_json(out_dir / "summary.json", {
        "steps": step, "runtime_s": round(time.time() - t0, 1), "best_step": best["step"],
        "best_ndcg@10": best["ndcg@10"], "selection": selection, "runtime": runtime})
    print(f"done: {step} steps, best step {best['step']} ({selection}) -> {out_dir / 'best'}")


if __name__ == "__main__":
    main()
