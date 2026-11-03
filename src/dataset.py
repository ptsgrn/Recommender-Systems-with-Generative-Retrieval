"""Turns chronological splits into (history -> next item) examples and padded batches.

For a user history [A, B, C, D, E] the split file holds
    train_items = [A, B, C], val_item = D, test_item = E
and this module builds
    training:   [A] -> B, [A, B] -> C           (rolling prefixes of train_items only)
    validation: [A, B, C] -> D
    test:       [A, B, C, D] -> E
D and E are never training targets for this user.
"""
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from src.tokenization import IGNORE_INDEX, PAD, HistoryEncoder


@dataclass
class Example:
    user_id: str
    history: List[str]   # raw item ids, oldest first
    target: str          # raw item id


def build_train_examples(splits: List[dict]) -> List[Example]:
    examples = []
    for split in splits:
        train_items = split["train_items"]
        for t in range(1, len(train_items)):
            examples.append(Example(split["user_id"], train_items[:t], train_items[t]))
    return examples


def build_eval_examples(splits: List[dict], which: str) -> List[Example]:
    if which == "val":
        return [Example(s["user_id"], list(s["train_items"]), s["val_item"]) for s in splits]
    if which == "test":
        return [Example(s["user_id"], list(s["train_items"]) + [s["val_item"]], s["test_item"])
                for s in splits]
    raise ValueError(f"unknown split {which!r}")


class SequenceDataset(Dataset):
    """Pre-encodes every example into encoder input tokens and decoder labels."""

    def __init__(self, examples: List[Example], encoder: HistoryEncoder):
        self.examples = examples
        self.inputs = [encoder.encode_history(e.user_id, e.history) for e in examples]
        self.labels = [encoder.encode_target(e.target) for e in examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, List[int]]:
        return {"input_ids": self.inputs[idx], "labels": self.labels[idx], "index": idx}


def collate(batch: List[dict]) -> Dict[str, torch.Tensor]:
    """input_ids/attention_mask: [batch, <=82]; labels: [batch, 5]. Inputs are right-padded with PAD,
    labels with -100 so padded positions never contribute to the loss."""
    max_in = max(len(b["input_ids"]) for b in batch)
    max_lab = max(len(b["labels"]) for b in batch)
    input_ids = torch.full((len(batch), max_in), PAD, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_in), dtype=torch.long)
    labels = torch.full((len(batch), max_lab), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        input_ids[i, :n] = torch.tensor(b["input_ids"])
        attention_mask[i, :n] = 1
        labels[i, :len(b["labels"])] = torch.tensor(b["labels"])
    index = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "index": index}


def format_example(example: Example, input_ids: List[int], labels: List[int], vocab) -> str:
    """Readable dump of one example: raw items, integer tokens and token names."""
    lines = [
        f"user={example.user_id} history={example.history} -> target={example.target}",
        f"  input_ids = {input_ids}",
        f"  as names  = {[vocab.describe(t) for t in input_ids]}",
        f"  labels    = {labels}  ({[vocab.describe(t) for t in labels]})",
    ]
    return "\n".join(lines)
