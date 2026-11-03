"""J1 integration smoke test: load the frozen Semantic ID maps, print a readable batch, run one
forward/backward pass and decode the labels back to the original target items.

    uv run python -m scripts.smoke_test --data-dir tests/fixtures --sid-dir tests/fixtures   # fixture
    uv run python -m scripts.smoke_test --config artifacts/run_config.yaml                   # real IDs
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.common import load_config, read_jsonl, seed_everything
from src.dataset import SequenceDataset, build_train_examples, collate, format_example
from src.recommender import build_model
from src.retrieval import Retriever
from src.tokenization import HistoryEncoder, load_sid_artifacts


def main():
    p = argparse.ArgumentParser(description="Real-ID batch forward/backward and label round trip")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--data-dir")
    p.add_argument("--sid-dir")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    config = load_config(args.config)
    defaults = load_config()
    seed_everything(config["seed"], config.get("threads", 2))
    vocab, item_to_sid, sid_to_item, manifest = load_sid_artifacts(args.sid_dir or config["artifact_dir"])
    print(f"mapping_hash={manifest['mapping_hash']}  items={len(item_to_sid)}  vocab={vocab.vocab_size}")
    print(f"offsets={vocab.position_offsets} user_offset={vocab.user_offset}")

    max_history = config.get("sequence", defaults["sequence"])["max_history"]
    encoder = HistoryEncoder(vocab, item_to_sid, max_history)
    splits = read_jsonl(Path(args.data_dir or config["data_dir"]) / "splits.jsonl")
    ds = SequenceDataset(build_train_examples(splits), encoder)
    batch = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)))

    for row in range(min(3, args.batch_size)):
        i = batch["index"][row].item()
        print(format_example(ds.examples[i], ds.inputs[i], ds.labels[i], vocab))
    print("shapes: input_ids", tuple(batch["input_ids"].shape), "labels", tuple(batch["labels"].shape))

    model = build_model(config.get("recommender", defaults["recommender"]), vocab.vocab_size)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
    out.loss.backward()
    grads_ok = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"logits {tuple(out.logits.shape)} loss {out.loss.item():.4f} finite_grads={grads_ok}")
    assert out.logits.shape[-1] == vocab.vocab_size and grads_ok

    retriever = Retriever(model, vocab, item_to_sid, sid_to_item)
    for row in range(batch["labels"].shape[0]):
        decoded = retriever.lookup(batch["labels"][row].tolist())
        expected = ds.examples[batch["index"][row].item()].target
        assert decoded == expected, (decoded, expected)
    print(f"all {batch['labels'].shape[0]} labels decode back to their target items: OK")


if __name__ == "__main__":
    main()
