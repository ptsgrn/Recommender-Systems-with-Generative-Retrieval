# Explainable TIGER: Developer A pipeline

This repository implements item preprocessing, frozen Sentence-T5 embeddings,
residual quantization, unique Semantic IDs, full-catalog metrics, and a
training-only popularity baseline. The scope and task ownership are in
[todo.md](todo.md). Developer B's sequence model and retrieval are separate work.
See [HANDOFF.md](HANDOFF.md) for the completed real-data run and B integration commands.
This is a small teaching implementation, not a reproduction of published scores.

## Setup and verification

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --locked
uv run pytest -q
```

Python 3.12 is selected by `.python-version`. `uv.lock` pins the environment.
The default PyTorch wheel is CPU-only to avoid downloading CUDA libraries on a
CPU machine. On a GPU host, install a matching CUDA PyTorch build in a separate
environment and record that environment; this is a dependency deviation.
All commands below support `--config PATH`. Data and large artifacts are ignored
by Git; copy them separately for the handoff and oral exam.

## Real-data run

```bash
uv run python -m scripts.prepare_data
uv run python -m scripts.benchmark
uv run python -m scripts.prepare_data --config artifacts/run_config.yaml
uv run python -m scripts.embed_items --config artifacts/run_config.yaml
uv run python -m scripts.train_rqvae --config artifacts/run_config.yaml
uv run python -m scripts.build_semantic_ids --config artifacts/run_config.yaml
uv run python -m scripts.evaluate --config artifacts/run_config.yaml --split test
uv run python -m scripts.inspect_example --config artifacts/run_config.yaml
```

The initial preparation selects 2,000 eligible users. The benchmark measures up
to 100 real texts and 100 optimizer steps, estimates embedding and 50-epoch
quantizer costs for 2,000/500/200 users, and selects the largest subset fitting a
two-hour compute budget. Downloads are excluded. Diagnostics receive a 2x
training-time allowance. Estimates are approximate; the measured training log is
the authoritative runtime. The benchmark writes a config with an immutable model
revision. A failed budget check leaves its projections for review and does not
start a full embedding/training run.

Raw inputs come from the [Amazon Reviews 2014 release](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html):
Beauty **5-core** reviews and Beauty metadata, not the 2018/2023 releases. The
download URLs and SHA-256 checksums are saved in `data_stats.json`. Existing raw
files are reused and parsed; remove a damaged file before retrying a download.
JSON and legacy literal records are parsed without executing source code.

Usable items have a nonempty cleaned title. HTML entities and whitespace are
normalized. Missing brand/category/price values are explicit. The first usable
metadata record wins for duplicate ASINs. Exact duplicate complete review records
are removed; other repeat purchases remain. After metadata filtering, users need
at least five retained events. Seeded sampling uses sorted user IDs and retains
complete surviving histories. Timestamp ties follow original source-row order.
The last event is test, the preceding event validation; earlier events train.
`chronological.jsonl` preserves timestamps and source rows for auditing.

Item text contains title, brand, category paths, and price, without review text.
The frozen [Sentence-T5 base model](https://huggingface.co/sentence-transformers/sentence-t5-base)
provides 768-dimensional vectors. Model pooling/normalization is preserved; no
extra L2 normalization is applied. The cache checks item order, text identity,
revision, dimensions, dtype, finiteness, and array checksum. Set the revision to
the saved SHA to reuse a cache offline. Use fresh data and artifact directories
when changing the subset or encoder.

Embedding progress is checkpointed atomically every 128 items in
`artifacts/embedding_progress.npz`. Rerun the same command after interruption to
resume completed chunks. A changed catalog, text, or model revision rejects that
partial cache. The checkpoint is removed after the final cache is saved.

## Quantization and frozen artifacts

The encoder is 768→512→256→128→32 with a mirrored decoder and ReLU hidden layers.
Three independent codebooks have 256 vectors each. A ten-epoch continuous
autoencoder warm-up at learning rate 0.0001 precedes sequential k-means
initialization on the learned latents and their residuals. Quantized training
uses Adam at learning rate 0.0001 and
batch 256 (16 on CPU), at most 50 epochs, and patience five. Selection uses lowest
full-catalog reconstruction loss; no held-out interaction labels enter this stage.
The original 0.001/no-warm-up run collapsed the final encoder ReLU layer and
assigned every item the same tuple. Its checkpoint/logs are retained under
`artifacts/failed_collapse/`; warm-up and the lower rate are measured project
adaptations, not settings attributed to the paper.

Squared losses are means over batch and feature coordinates; auxiliary losses
are summed across levels, with commitment weight 0.25. The straight-through path
sends reconstruction gradients to the encoder. Codebook losses update selected
embeddings; commitment losses update the encoder. Earlier selected embeddings
are detached from later residual losses. There is no Gaussian sampler or KL loss.
All component losses, code utilization, collisions, steps and runtime are logged.

`rqvae.pt` contains the selected parameters, optimizer state, settings and input
hashes; `rqvae_training.json` contains the run configuration and runtime versions.
Export reloads that checkpoint. Items sharing three learned codes receive suffix
0,1,… in sorted-ASIN order; singletons receive 0. A collision group over 256 fails
instead of wrapping suffixes. Total collapse also fails export. Inspect unusually
low utilization rather than assuming every code is a human category.

The handoff artifacts are:

| Artifact | Meaning |
|---|---|
| `data/processed/items.jsonl` | Clean catalog with raw ASINs and deterministic text |
| `data/processed/splits.jsonl` | User train items, validation item, test item |
| `data/processed/chronological.jsonl`, `data_stats.json` | Split audit, provenance and counts |
| `artifacts/item_ids.json`, `item_embeddings.npy` | Aligned item order and float32 `[N,768]` cache |
| `artifacts/item_to_sid.json`, `sid_to_item.json` | Exact inverse maps for raw four-code IDs |
| `artifacts/sid_manifest.json` | Position ranges, special tokens and frozen hashes |
| `artifacts/rqvae.pt`, `embedding_manifest.json`, `run_config.yaml` | Reloadable quantizer and reproducibility metadata |

Default token offsets are `[2,258,514,770]`, PAD=0, EOS=1, decoder start=PAD,
user offset=1026, and vocabulary size=3026. User buckets are the SHA-256 UTF-8
digest interpreted as an unsigned big-endian integer modulo 2,000. Manifest hashes
use UTF-8 JSON with sorted keys, compact separators, and `ensure_ascii=False`.
Raw SID codes and offset vocabulary tokens are different representations.
Frozen mappings cannot be replaced with different contents, and training refuses
to overwrite a run with frozen IDs. B must store and verify the mapping hash with
every model checkpoint. The diagnostic trace illustrates tokens; B owns the
production tokenizer and sequence dataset.

## Evaluation and B integration

The popularity baseline counts each original training event once, including
retained repeated events. Ties use ascending ASIN; items with zero count remain in
the catalog. Both methods use the complete catalog, allow repeat recommendations,
and retain the target in the candidate pool. Metrics average over every split
user, including users with empty predictions. Coverage@K is the fraction receiving
K unique items; mean returned count is capped at ten.

B supplies JSONL with one record per user:

```json
{"user_id":"USER_ASIN", "item_ids":["RANK_1_ASIN","RANK_2_ASIN"], "scores":[-1.2,-2.3]}
```

Scores are optional and do not change the supplied order. Duplicate item IDs are
removed preserving first occurrence; missing users receive empty lists. Unknown
users/items, malformed rankings and duplicate user records fail validation.

```bash
uv run python -m scripts.evaluate --config artifacts/run_config.yaml --split val --predictions predictions_val.jsonl
uv run python -m scripts.evaluate --config artifacts/run_config.yaml --split test --predictions predictions_test.jsonl
uv run python -m scripts.inspect_example --config artifacts/run_config.yaml --split test --predictions predictions_test.jsonl
```

Use validation for B's checkpoint selection, then freeze the model before final
test evaluation. Reports are `reports/results_{val,test}.{json,md}`,
`reports/id_diagnostics.json`, and `reports/example_trace.json`. Without B's
predictions, model metrics and recommendation output are explicitly pending.
With predictions, the trace resolves the first ranked item back to its SID/title;
it does not claim to prove generation. B's generation log must supply raw generated
tokens and scores to complete that part of the oral-exam trace.

## Paper correspondence and limitations

| Paper area | Source |
|---|---|
| §3.1, Figure 3: residual quantization | `src/rqvae.py` |
| §3.1: unique Semantic IDs | `src/semantic_ids.py` |
| Appendix C: chronological evaluation | `src/data.py`, `src/metrics.py` |
| §3.2: sequence model and retrieval | Pending Developer B |

See [docs/deviations.md](docs/deviations.md) for adaptations. The reference is
[Rajput et al., Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065).
Passing synthetic tests proves pipeline mechanics, not real-data model quality.
Matching paper scores, a random-ID ablation, a web UI, and B's recommender are
outside this implementation. The live demo should reload saved artifacts rather
than retrain. Back up data, maps, checkpoints and reports before the exam. Small
reports are eligible for Git; the data, model cache and large artifacts are ignored.
