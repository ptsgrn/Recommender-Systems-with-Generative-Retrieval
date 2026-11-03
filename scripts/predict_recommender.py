"""Generates ranked predictions from the frozen selected checkpoint, in the JSONL format read by
A's `scripts.evaluate`, plus a generation log with raw beam tokens for the oral-exam trace.

    uv run python -m scripts.predict_recommender --config artifacts/run_config.yaml --split val
    uv run python -m scripts.predict_recommender --config artifacts/run_config.yaml --split test
    uv run python -m scripts.evaluate --config artifacts/run_config.yaml --split test \
        --predictions artifacts/recommender/predictions_test.jsonl
"""
import argparse
import time
from pathlib import Path

from src.common import device_for, load_config, read_jsonl, seed_everything, write_json, write_jsonl
from src.dataset import build_eval_examples
from src.recommender import load_checkpoint
from src.retrieval import Retriever
from src.tokenization import load_sid_artifacts


def main():
    p = argparse.ArgumentParser(description="Write TIGER predictions for A's evaluator")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--split", choices=("val", "test"), required=True)
    p.add_argument("--data-dir")
    p.add_argument("--sid-dir")
    p.add_argument("--checkpoint", help="default: artifact_dir/recommender/best")
    p.add_argument("--out-dir", help="default: artifact_dir/recommender")
    p.add_argument("--unconstrained", action="store_true", help="diagnostic: disable the catalog trie")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    config = load_config(args.config)
    seed_everything(config["seed"], config.get("threads", 2))
    device = device_for(config)
    data_dir = Path(args.data_dir or config["data_dir"])
    rec_dir = Path(config["artifact_dir"]) / "recommender"
    out_dir = Path(args.out_dir or rec_dir)

    vocab, item_to_sid, sid_to_item, manifest = load_sid_artifacts(args.sid_dir or config["artifact_dir"])
    model, meta = load_checkpoint(args.checkpoint or rec_dir / "best", manifest["mapping_hash"], device)
    sections = meta["sections"]
    constrained = sections["retrieval"]["constrained"] and not args.unconstrained
    k = sections["retrieval"]["top_k"]
    retriever = Retriever(model, vocab, item_to_sid, sid_to_item, sections["sequence"]["max_history"],
                          sections["retrieval"]["num_beams"], constrained, device)

    examples = build_eval_examples(read_jsonl(data_dir / "splits.jsonl"), args.split)
    predictions, log = [], []
    start = time.perf_counter()
    for i in range(0, len(examples), args.batch_size):
        chunk = examples[i:i + args.batch_size]
        # Only user ids and histories go in; targets stay out of generation.
        recs = retriever.recommend_batch([e.user_id for e in chunk], [e.history for e in chunk], k)
        for e, rec in zip(chunk, recs):
            predictions.append({"user_id": e.user_id, "item_ids": [item for item, _ in rec.items],
                                "scores": [score for _, score in rec.items]})
            log.append({"user_id": e.user_id, "beams": [{"tokens": t, "score": s} for t, s in rec.beams],
                        "num_invalid": rec.num_invalid, "num_duplicate": rec.num_duplicate,
                        "shortfall": rec.shortfall})
    seconds = time.perf_counter() - start

    suffix = "" if constrained else "_unconstrained"
    write_jsonl(out_dir / f"predictions_{args.split}{suffix}.jsonl", predictions)
    write_jsonl(out_dir / f"generation_log_{args.split}{suffix}.jsonl", log)
    summary = {"split": args.split, "users": len(predictions), "checkpoint_step": meta["step"],
               "mapping_hash": manifest["mapping_hash"], "decoding": "constrained-trie" if constrained else "unconstrained",
               "num_beams": retriever.num_beams, "k": k, "generation_seconds": round(seconds, 1),
               "mean_returned": sum(len(r["item_ids"]) for r in predictions) / max(len(predictions), 1),
               "fraction_full_k": sum(len(r["item_ids"]) == k for r in predictions) / max(len(predictions), 1),
               "invalid_beams": sum(r["num_invalid"] for r in log)}
    write_json(out_dir / f"prediction_summary_{args.split}{suffix}.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
