"""Generative retrieval: beam-search the next item's Semantic ID, then look it up in the catalog.

No nearest-neighbour search is involved: the model emits 4 ID tokens + EOS, the tokens are mapped
back to a code tuple, and `sid_to_item` turns the tuple into an item id.

Two decoding modes (keep the same one across all compared runs and report it):
  * constrained=True  - project adaptation: a catalog prefix trie only allows continuations that
                        exist in the known catalog, so every finished beam is a real item.
  * constrained=False - paper-style diagnostic: free generation, invalid IDs are counted and filtered.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers.modeling_outputs import BaseModelOutput

from src.tokenization import EOS, PAD, HistoryEncoder, SIDVocab, sid_key

DEAD_SCORE = -1e8  # beams that only survived through masked (-inf) tokens score below this


class CatalogTrie:
    """Prefix tree over the catalog's offset-token IDs. Built from the catalog only, never from targets."""

    def __init__(self, vocab: SIDVocab, item_to_sid: Dict[str, Sequence[int]]):
        self.depth = vocab.num_positions
        nexts: Dict[Tuple[int, ...], set] = {}
        for sid in item_to_sid.values():
            tokens = vocab.sid_to_tokens(sid)
            for t in range(self.depth):
                nexts.setdefault(tuple(tokens[:t]), set()).add(tokens[t])
        self.children = {prefix: sorted(s) for prefix, s in nexts.items()}

    def allowed(self, generated: Sequence[int]) -> List[int]:
        """Tokens allowed after `generated` (decoder output without the start token)."""
        n = len(generated)
        if n < self.depth:
            # An unknown prefix can only be reached through a -inf beam; end it immediately.
            return self.children.get(tuple(generated), [EOS])
        if n == self.depth:
            return [EOS]          # EOS only after exactly four ID tokens
        return [PAD]              # beam already finished; HF pads it


def make_prefix_allowed_tokens_fn(trie: CatalogTrie):
    def prefix_allowed_tokens_fn(batch_id: int, decoder_input_ids: torch.Tensor) -> List[int]:
        # decoder_input_ids[0] is the decoder start token (PAD); it is not part of the ID.
        return trie.allowed(decoder_input_ids[1:].tolist())
    return prefix_allowed_tokens_fn


@dataclass
class Recommendation:
    items: List[Tuple[str, float]] = field(default_factory=list)  # (item_id, sum of token log-probs)
    beams: List[Tuple[List[int], float]] = field(default_factory=list)  # raw generated tokens + score
    num_beams_returned: int = 0
    num_invalid: int = 0      # malformed token sequence or tuple absent from the catalog
    num_duplicate: int = 0
    shortfall: int = 0        # k - len(items)


class Retriever:
    """Implements the `recommend(user_id, history, k)` contract on top of a trained model."""

    def __init__(self, model, vocab: SIDVocab, item_to_sid: Dict[str, Sequence[int]],
                 sid_to_item: Dict[str, str], max_history: int = 20, num_beams: int = 20,
                 constrained: bool = True, device=None):
        self.model = model
        self.vocab = vocab
        self.encoder = HistoryEncoder(vocab, item_to_sid, max_history)
        self.sid_to_item = sid_to_item
        self.num_beams = num_beams
        self.constrained = constrained
        self.device = device or next(model.parameters()).device
        self.trie = CatalogTrie(vocab, item_to_sid)
        self.prefix_fn = make_prefix_allowed_tokens_fn(self.trie)

    def lookup(self, generated: Sequence[int]) -> Optional[str]:
        """Generated tokens (without start token) -> item id, or None if invalid. Never raises."""
        tokens = list(generated)
        depth = self.vocab.num_positions
        if len(tokens) < depth + 1 or tokens[depth] != EOS:
            return None
        sid = self.vocab.tokens_to_sid(tokens[:depth])
        if sid is None:
            return None
        return self.sid_to_item.get(sid_key(sid))

    @torch.no_grad()
    def recommend_batch(self, user_ids: Sequence[str], histories: Sequence[Sequence[str]], k: int = 10
                        ) -> List[Recommendation]:
        self.model.eval()
        inputs = [self.encoder.encode_history(u, h) for u, h in zip(user_ids, histories)]
        max_len = max(len(x) for x in inputs)
        input_ids = torch.full((len(inputs), max_len), PAD, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for i, x in enumerate(inputs):
            input_ids[i, :len(x)] = torch.tensor(x)
            attention_mask[i, :len(x)] = 1

        out = self.model.generate(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            num_beams=self.num_beams,
            num_return_sequences=self.num_beams,
            max_new_tokens=self.vocab.num_positions + 1,
            length_penalty=0.0 if self.num_beams > 1 else 1.0,
            early_stopping=self.num_beams > 1,
            do_sample=False,
            prefix_allowed_tokens_fn=self.prefix_fn if self.constrained else None,
            output_scores=True,
            return_dict_in_generate=True,
        )
        sequences = out.sequences[:, 1:].tolist()   # drop decoder start token
        if self.num_beams == 1:
            # Greedy output has no sequences_scores; its processed scores are
            # masked raw logits. Score generated tokens against the full model
            # vocabulary, without renormalizing probabilities over trie choices.
            generated = out.sequences[:, 1:]
            eos = generated.eq(EOS)
            active = (eos.long().cumsum(dim=1) - eos.long()).eq(0)
            logits = self.model(input_ids=input_ids.to(self.device),
                                attention_mask=attention_mask.to(self.device),
                                labels=generated.masked_fill(~active, -100)).logits
            token_logp = logits.log_softmax(dim=-1).gather(-1, generated.unsqueeze(-1)).squeeze(-1)
            scores = (token_logp * active).sum(dim=1).tolist()
        else:
            scores = out.sequences_scores.tolist()

        results = []
        for b in range(len(inputs)):
            rec = Recommendation()
            seen = set()
            rows = range(b * self.num_beams, (b + 1) * self.num_beams)
            for seq, score in sorted(((sequences[r], scores[r]) for r in rows), key=lambda x: -x[1]):
                if score < DEAD_SCORE:
                    continue
                rec.num_beams_returned += 1
                rec.beams.append((seq, score))
                item = self.lookup(seq)
                if item is None:
                    rec.num_invalid += 1
                elif item in seen:
                    rec.num_duplicate += 1
                elif len(rec.items) < k:
                    seen.add(item)
                    rec.items.append((item, score))
            rec.shortfall = k - len(rec.items)
            results.append(rec)
        return results

    def recommend(self, user_id: str, history: Sequence[str], k: int = 10) -> Recommendation:
        """Ranked unique items for one user. The target item is never an argument."""
        return self.recommend_batch([user_id], [history], k)[0]

    @torch.no_grad()
    def score_all_items(self, user_id: str, history: Sequence[str]) -> List[Tuple[str, float]]:
        """Exact log P(ID + EOS | history) for every catalog item via teacher forcing.
        Only feasible for tiny catalogs; used to sanity-check beam-search scores."""
        self.model.eval()
        x = torch.tensor([self.encoder.encode_history(user_id, history)], device=self.device)
        item_ids = list(self.sid_to_item.values())
        labels = torch.tensor([self.encoder.encode_target(i) for i in item_ids], device=self.device)
        enc = self.model.get_encoder()(input_ids=x).last_hidden_state.expand(len(item_ids), -1, -1)
        logits = self.model(encoder_outputs=BaseModelOutput(last_hidden_state=enc), labels=labels).logits
        logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).sum(-1)
        return sorted(zip(item_ids, logp.tolist()), key=lambda x: -x[1])
