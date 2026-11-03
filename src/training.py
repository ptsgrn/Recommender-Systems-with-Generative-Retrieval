"""Seeded batch scheduling and recoverable sequence-model training."""
import copy
import importlib.metadata
import math
from pathlib import Path
import platform
import time

import torch
from torch.utils.data import DataLoader

from src.common import (canonical_hash, device_for, file_hash, read_jsonl,
                        runtime_info, seed_everything, write_json, write_jsonl)
from src.dataset import SequenceDataset, build_eval_examples, build_train_examples, collate
from src.recommender import build_model, evaluate_loss, train_step
from src.retrieval import Retriever
from src.tokenization import HistoryEncoder, load_sid_artifacts
from src.training_state import (StopRequest, atomic_save, capture_rng, cpu_weights,
                                export_best, load_recovery, restore_rng)


def ranking_metrics(retriever, examples, k=10, batch_size=32):
    recall = ndcg = returned = invalid = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        recs = retriever.recommend_batch([e.user_id for e in chunk], [e.history for e in chunk], k)
        for example, rec in zip(chunk, recs):
            ranked = [item for item, _ in rec.items]
            returned += len(ranked)
            invalid += rec.num_invalid
            if example.target in ranked:
                recall += 1
                ndcg += 1 / math.log2(ranked.index(example.target) + 2)
    n = max(len(examples), 1)
    return {f"recall@{k}": recall / n, f"ndcg@{k}": ndcg / n,
            "mean_returned": returned / n, "invalid_beams": invalid, "num_users": len(examples)}


def epoch_loader(dataset, batch_size, seed, epoch, next_batch):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    # Iterator creation must not consume global dropout RNG on restart.
    return DataLoader(dataset, batch_sampler=batches[next_batch:], collate_fn=collate,
                      generator=generator)


def run_training(config, args, sections, stop=None):
    stop = stop or StopRequest()
    sections = copy.deepcopy(sections)
    rec, ret = sections["recommender"], sections["retrieval"]
    for argument, key in (("max_steps", "max_steps"), ("eval_every", "eval_every"),
                          ("batch_size", "batch_size"), ("learning_rate", "learning_rate"),
                          ("val_subset", "val_subset"), ("save_every", "save_every")):
        value = getattr(args, argument, None)
        if value is not None:
            rec[key] = value
    rec.setdefault("save_every", 100)
    if any(rec[key] <= 0 for key in ("max_steps", "eval_every", "batch_size", "learning_rate", "save_every")) or rec["val_subset"] < 0:
        raise ValueError("Training budgets must be positive; val_subset must be nonnegative")
    if args.benchmark_steps < 0 or (args.resume and args.benchmark_steps):
        raise ValueError("Benchmark steps must be nonnegative and cannot be resumed")
    data_dir, sid_dir = Path(args.data_dir or config["data_dir"]), Path(args.sid_dir or config["artifact_dir"])
    out = Path(args.out or (Path(args.resume).parent if args.resume else Path(config["artifact_dir"]) / "recommender"))
    if args.resume:
        if Path(args.resume).resolve() != (out / "latest.pt").resolve():
            raise ValueError("Resume must use this run's latest.pt")
    elif out.exists() and any(out.iterdir()):
        raise ValueError("Output directory is nonempty; use --resume or a fresh run directory")

    seed, threads = config["seed"], config.get("threads", 2)
    seed_everything(seed, threads)
    device = device_for(config)
    vocab, forward, inverse, manifest = load_sid_artifacts(sid_dir)
    encoder = HistoryEncoder(vocab, forward, sections["sequence"]["max_history"])
    splits = read_jsonl(data_dir / "splits.jsonl")
    train_examples = build_train_examples(splits)
    check_examples = train_examples if args.overfit else build_eval_examples(splits, "val")
    if not args.overfit:
        order = torch.randperm(len(check_examples), generator=torch.Generator().manual_seed(seed)).tolist()
        check_examples = [check_examples[i] for i in (order[:rec["val_subset"]] if rec["val_subset"] else order)]
    if not train_examples or not check_examples:
        raise ValueError("Training and validation examples must be nonempty")
    train_ds, check_ds = SequenceDataset(train_examples, encoder), SequenceDataset(check_examples, encoder)
    check_loader = DataLoader(check_ds, batch_size=rec["batch_size"], collate_fn=collate,
                              generator=torch.Generator().manual_seed(seed))
    batches_per_epoch = math.ceil(len(train_ds) / rec["batch_size"])
    runtime = runtime_info()
    identity = {"sections": sections, "seed": seed, "overfit": args.overfit,
                "mapping_hash": manifest["mapping_hash"], "catalog_hash": manifest["catalog_hash"],
                "sid_manifest_sha256": file_hash(sid_dir / "sid_manifest.json"),
                "splits_sha256": file_hash(data_dir / "splits.jsonl"),
                "mapping_file_sha256": file_hash(sid_dir / "item_to_sid.json"),
                "validation_examples_hash": canonical_hash([vars(e) for e in check_examples]),
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "numpy": importlib.metadata.version("numpy"),
                                "transformers": importlib.metadata.version("transformers"),
                                "platform": platform.platform(), "device": str(device), "threads": threads}}
    model = build_model(rec, vocab.vocab_size, dropout=0.0 if args.overfit else None).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rec["learning_rate"], weight_decay=rec["weight_decay"])
    retriever = Retriever(model, vocab, forward, inverse, sections["sequence"]["max_history"],
                          ret["num_beams"], ret["constrained"], device)
    selection = "train-set ndcg@10 (overfit diagnostic)" if args.overfit else f"validation ndcg@10, fixed seeded subset of {len(check_examples)} users"
    step = epoch = next_batch = 0
    elapsed_before, best, records = 0.0, None, []
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        state = load_recovery(args.resume, identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, epoch, next_batch = state["step"], state["epoch"], state["next_batch"]
        if not (0 <= step <= rec["max_steps"] and epoch >= 0 and 0 <= next_batch < batches_per_epoch):
            raise ValueError("Invalid recovery cursor")
        elapsed_before, best, records = state["elapsed_seconds"], state["best"], state["records"]
        export_best(model, best, out / "best")
        write_jsonl(out / "train_log.jsonl", records)
        restore_rng(state["rng"])
    else:
        write_json(out / "run_manifest.json", {"identity": identity, "runtime": runtime,
                   "resolved_config": {**config, **sections}, "validation_users": [e.user_id for e in check_examples]})
        write_jsonl(out / "train_log.jsonl", [])
    start = time.perf_counter()

    def elapsed():
        return elapsed_before + time.perf_counter() - start

    def snapshot():
        atomic_save(out / "latest.pt", {"version": 1, "identity": identity, "model": cpu_weights(model),
                    "optimizer": optimizer.state_dict(), "step": step, "epoch": epoch,
                    "next_batch": next_batch, "rng": capture_rng(), "best": best,
                    "records": records, "elapsed_seconds": elapsed()})

    print(f"device={device} train_examples={len(train_ds)} validation_examples={len(check_ds)} "
          f"mapping={manifest['mapping_hash']} step={step}/{rec['max_steps']}", flush=True)
    if not args.resume and not args.benchmark_steps:
        snapshot()
    while step < rec["max_steps"] and not stop.signal_number:
        for batch in epoch_loader(train_ds, rec["batch_size"], seed, epoch, next_batch):
            loss = train_step(model, batch, optimizer, device)
            if not math.isfinite(loss) or any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite training loss or parameters; recovery snapshot retained")
            step += 1
            next_batch += 1
            if next_batch == batches_per_epoch:
                epoch, next_batch = epoch + 1, 0
            if step % 100 == 0:
                print(f"step {step} loss {loss:.6f} elapsed {elapsed():.1f}s", flush=True)
            if args.benchmark_steps and (step >= args.benchmark_steps or step == rec["max_steps"]):
                result = {"status": "benchmark", "steps": step, "runtime_s": elapsed(),
                          "seconds_per_step": elapsed() / step, "steps_per_epoch": batches_per_epoch,
                          "runtime": runtime, "identity": identity}
                write_json(out / "benchmark.json", result)
                print(result, flush=True)
                return result
            best_changed = False
            if not args.benchmark_steps and (step % rec["eval_every"] == 0 or step == rec["max_steps"]):
                check_loss = evaluate_loss(model, check_loader, device)
                metrics = ranking_metrics(retriever, check_examples, k=10)
                if not math.isfinite(check_loss) or not all(math.isfinite(x) for x in metrics.values()):
                    raise ValueError("Nonfinite validation values; recovery snapshot retained")
                record = {"step": step, "train_loss": loss, "check_loss": check_loss,
                          "elapsed_s": elapsed(), **metrics}
                records.append(record)
                write_jsonl(out / "train_log.jsonl", records)
                print(record, flush=True)
                if best is None or metrics["ndcg@10"] > best["meta"]["metrics"]["ndcg@10"]:
                    best = {"weights": cpu_weights(model), "meta": {
                        "step": step, "selection": selection, "metrics": record,
                        "mapping_hash": manifest["mapping_hash"], "catalog_hash": manifest["catalog_hash"],
                        "sections": sections, "seed": seed, "learning_rate": rec["learning_rate"],
                        "batch_size": rec["batch_size"], "device": str(device), "overfit_mode": args.overfit,
                        "decoding": "constrained-trie" if ret["constrained"] else "unconstrained"}}
                    best_changed = True
            if not args.benchmark_steps and (step % rec["save_every"] == 0 or best_changed or stop.signal_number or step == rec["max_steps"]):
                snapshot()
            if best_changed:
                export_best(model, best, out / "best")
            if step >= rec["max_steps"] or stop.signal_number:
                break
    if not args.benchmark_steps:
        snapshot()
    result = {"status": "completed" if step == rec["max_steps"] else "interrupted", "steps": step,
              "runtime_s": elapsed(), "best_step": best["meta"]["step"] if best else None,
              "best_ndcg@10": best["meta"]["metrics"]["ndcg@10"] if best else None,
              "selection": selection, "runtime": runtime}
    write_json(out / "summary.json", result)
    print(result, flush=True)
    return result
