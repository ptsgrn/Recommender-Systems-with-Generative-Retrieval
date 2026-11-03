"""Small T5 encoder-decoder trained from scratch on the custom Semantic ID vocabulary.

Training uses teacher forcing: given labels [c0, c1, c2, c3, EOS], T5 builds
    decoder_input_ids = [PAD(start), c0, c1, c2, c3]
internally (shift right), so at step t the decoder sees only the true tokens before t.
Loss = mean token cross-entropy over the 5 label positions (L_rec in the plan, up to averaging).
"""
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.common import read_json, write_json
from src.tokenization import EOS, PAD


def build_model(rec_cfg: dict, vocab_size: int, dropout: Optional[float] = None) -> T5ForConditionalGeneration:
    """Randomly initialised T5 from configuration only; no pretrained weights, no text tokenizer."""
    config = T5Config(
        vocab_size=vocab_size,
        d_model=rec_cfg["d_model"],
        d_kv=rec_cfg["d_kv"],
        d_ff=rec_cfg["d_ff"],
        num_layers=rec_cfg["num_layers"],
        num_decoder_layers=rec_cfg["num_decoder_layers"],
        num_heads=rec_cfg["num_heads"],
        dropout_rate=rec_cfg["dropout"] if dropout is None else dropout,
        feed_forward_proj="relu",
        pad_token_id=PAD,
        eos_token_id=EOS,
        decoder_start_token_id=PAD,
        tie_word_embeddings=True,
    )
    return T5ForConditionalGeneration(config)


def batch_to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}


def train_step(model, batch, optimizer, device) -> float:
    model.train()
    out = model(**batch_to_device(batch, device))
    optimizer.zero_grad()
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return out.loss.item()


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    """Token-weighted mean cross-entropy over a whole loader."""
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(**batch)
        n = (batch["labels"] != -100).sum().item()
        total += out.loss.item() * n
        count += n
    return total / max(count, 1)


def save_checkpoint(model, path, meta: dict):
    """Saves weights + config + run metadata (including the Semantic ID mapping hash)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    write_json(path / "run_meta.json", dict(meta, saved_at=time.strftime("%Y-%m-%d %H:%M:%S")))


def load_checkpoint(path, expected_mapping_hash: Optional[str] = None, device=None):
    """Reloads a checkpoint and refuses to pair it with a different item->ID mapping."""
    path = Path(path)
    meta = read_json(path / "run_meta.json")
    if expected_mapping_hash is not None and meta.get("mapping_hash") != expected_mapping_hash:
        raise ValueError(f"checkpoint was trained with mapping {meta.get('mapping_hash', '?')[:12]}, "
                         f"current mapping is {expected_mapping_hash[:12]}")
    model = T5ForConditionalGeneration.from_pretrained(path)
    if device is not None:
        model.to(device)
    model.eval()
    return model, meta
