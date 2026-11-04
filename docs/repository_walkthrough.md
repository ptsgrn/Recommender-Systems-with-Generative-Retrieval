# Repository walkthrough for the oral exam

This guide follows one item and one user history through the recommendation
pipeline. Work through the stops in order to understand what each stage consumes,
what it produces, and which result supports the project's conclusions. The main
route takes roughly 90 minutes of reading and practice, excluding installation.

For a 15-minute overview, read the pipeline below, stops 1, 6, and 7, then explain
the trace in stop 8. Use [HANDOFF.md](../HANDOFF.md) afterward to prepare slides.

## The pipeline you should be able to explain

The system predicts the next product a user will interact with. It first learns
compact identifiers from product content, then learns to generate those
identifiers from interaction histories.

```text
Amazon Beauty reviews + metadata
  │
  ├─ clean metadata and split user events chronologically
  │    ├─ items.jsonl: product content
  │    └─ splits.jsonl: train history, validation item, test item
  │
  ├─ product content → frozen Sentence-T5 → 768-dimensional embedding
  │    → residual-quantized autoencoder → 3 learned codes + unique suffix
  │    → frozen item_to_sid / sid_to_item maps
  │
  └─ user history → position-specific ID tokens → small T5 recommender
       → generate 4 tokens + EOS → catalog lookup → ranked items
       → compare with held-out target → Recall and NDCG
```

Sentence-T5 is pretrained and frozen. The recommendation T5 is trained from
scratch. The recommendation output is a product ID, with its title retrieved
after lookup. The two models have different jobs despite both using the T5 name.

The final course comparison also includes item cosine: Recall@10 0.0175 and
NDCG@10 0.00655030. Read [the baseline guide](baseline.md) for the algorithm,
training-only fit, and new CLI. Its reports are separate from frozen `poc01`.

## Find your way around

| Location | What to use it for |
|---|---|
| [README.md](../README.md) | Setup, artifact formats, and command reference |
| [configs/poc01.yaml](../configs/poc01.yaml) | Settings for the completed experiment |
| [configs/small.yaml](../configs/small.yaml) | General defaults; some budgets differ from the completed run |
| [src](../src) | Reusable data, model, retrieval, and evaluation logic |
| [scripts](../scripts) | Command-line entry points that connect the modules |
| [tests](../tests) | Small examples of expected behavior and regressions |
| [reports/poc01](../reports/poc01) | Final experiment results and verified trace |
| [docs/experiment.md](experiment.md) | Training, selection, recovery, and test protocol |
| [docs/deviations.md](deviations.md) | Project adaptations |
| [HANDOFF.md](../HANDOFF.md) | Presentation brief with data, findings, and limitations |
| [todo.md](../todo.md) | Original plan and task ownership; not the completion record |

`data/processed/` contains the local dataset. `artifacts/` contains embeddings,
ID maps, weights, and training logs. Both are ignored by Git, so a fresh clone
may lack them. You can follow the saved-report route without those files.

## Prepare your environment

Run commands from the repository root. Python 3.12 and the locked dependencies
are configured in [pyproject.toml](../pyproject.toml) and `uv.lock`.

```bash
uv sync --locked
```

The exercises below read saved files or run isolated checks. Keep the completed
experiment frozen while learning. In particular, `scripts.evaluate` and
`scripts.inspect_example` write reports; they are not read-only viewers.

## Stop 1 Read the result before the implementation

**Allow 5 minutes.** Open the [test table](../reports/poc01/results_test.md),
[experiment record](../reports/poc01/experiment.json), and
[configuration](../configs/poc01.yaml).

Locate `training.steps`, `selection.selected_step`, `selection.selection`, and
`results.test` in the experiment record. They establish which model was tested
and how it was chosen.

| Fact | Completed run |
|---|---:|
| Users / catalog items | 2,000 / 7,154 |
| Rolling training examples | 11,264 |
| Training steps / selected step | 5,000 / 4,500 |
| Selection users | Fixed seeded 500-user validation subset |
| Model Recall@10 / popularity Recall@10 | 0.0235 / 0.0075 |
| Model NDCG@10 / popularity NDCG@10 | 0.01110530 / 0.00279747 |

The model had a top-ten hit for 47 users, compared with 15 for popularity. It
outperformed this baseline in this run, but missed 1,953 of 2,000 test targets.

**Check your understanding:** Why was step 4,500 evaluated instead of automatically
using the final step? Because checkpoint selection used validation NDCG@10.
Test targets did not choose the checkpoint.

## Stop 2 Follow the data into chronological examples

**Allow 10 minutes.** Read [src/data.py](../src/data.py), focusing on
`clean_item`, `load_clean_data`, and `make_subset`. Then read
[scripts/prepare_data.py](../scripts/prepare_data.py), especially `prepare`.

`clean_item` turns metadata into deterministic text using title, brand,
categories, and price. Review text is not an embedding input. The loader removes
events without usable metadata, preserves repeated events except identical
complete records, and requires at least five retained events per user. Sampling
uses seed 42 after filtering. Timestamp ties use source-row order.

The resulting local files have separate purposes:

| File | Contents |
|---|---|
| `data/processed/items.jsonl` | One product per line, including its metadata and embedding text |
| `data/processed/splits.jsonl` | One user per line: `train_items`, `val_item`, `test_item` |
| `data/processed/chronological.jsonl` | Timestamps and source rows for auditing event order |
| `data/processed/data_stats.json` | Counts, filtering statistics, source URLs and checksums |

Read [src/dataset.py](../src/dataset.py): `build_train_examples` and
`build_eval_examples` turn each split into prediction tasks.

```text
Chronological events: A, B, C, D, E
Stored split:         train=[A,B,C], validation=D, test=E
Training examples:   [A]→B and [A,B]→C
Validation example:  [A,B,C]→D
Test example:        [A,B,C,D]→E
```

There are 13,264 original training events. Each user's first event has no
nonempty preceding history, leaving 13,264 − 2,000 = 11,264 training examples.
This is a per-user chronological split, not a global calendar-time split.

If the local data exists, inspect a real row:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from src.dataset import build_train_examples, build_eval_examples

with Path('data/processed/splits.jsonl').open() as stream:
    row = json.loads(next(stream))
print('Stored split:', row)
print('First training example:', build_train_examples([row])[0])
print('Validation:', build_eval_examples([row], 'val')[0])
print('Test:', build_eval_examples([row], 'test')[0])
PY
```

**Check your understanding:** Why can validation appear in test history? It is
an earlier event that is observed by test prediction time, while remaining
excluded from this user's training targets.

## Stop 3 Follow product text into an embedding

**Allow 8 minutes.** Read [src/embeddings.py](../src/embeddings.py): `encode`,
`embedding_identity`, `load_embeddings`, and `encode_checkpointed`. The CLI is
[scripts/embed_items.py](../scripts/embed_items.py).

Sentence-T5 turns each metadata text into 768 floating-point values. The final
matrix has shape `[7154,768]`. Its rows align with `artifacts/item_ids.json`;
row order is part of the artifact contract, not an incidental detail.

The cache checks the catalog/text/model identity and array checksum.
`artifacts/embedding_manifest.json` records the encoder revision and other
provenance. Partial embedding progress is saved every 128 items so matching work
can resume after interruption.

**Check your understanding:** Why is a matrix with the correct shape insufficient?
Its rows might refer to different items, different text, or a different model
revision. Those changes would silently alter the meanings of downstream IDs.

## Stop 4 Follow embeddings into Semantic IDs

**Allow 12 minutes.** Read [src/rqvae.py](../src/rqvae.py) in this order:
`RQVAE.__init__`, `quantize`, `forward`, and `initialize_codebooks`. Then read
`train` in [scripts/train_rqvae.py](../scripts/train_rqvae.py).

The encoder compresses 768 dimensions through `512 → 256 → 128 → 32`.
Each of three codebooks contains 256 vectors of size 32. The first codebook
approximates the latent, the second approximates the remaining residual, and the
third approximates what is still left. The decoder reconstructs the original
content embedding from the sum of the three selected vectors.

In `forward`, distinguish three losses: reconstruction teaches the autoencoder
to preserve content; codebook loss moves selected vectors toward encoder
residuals; commitment loss encourages encoder residuals to stay near selected
vectors. The straight-through expression sends reconstruction gradients through
the discrete selection to the encoder. There is no Gaussian sampler or KL term.

The initial run collapsed to identical codes. The repair added ten continuous
warm-up epochs, residual k-means initialization, and learning rate 0.0001.
The selected trained quantizer was epoch 2, step 896. Quantized fine-tuning did
not beat the reconstruction MSE immediately after warm-up and initialization.
Read [the diagnostics](../reports/poc01/id_diagnostics.json) and the quantizer
measurements in [reports/run_summary.json](../reports/run_summary.json).

Next, read `finalize_ids`, `make_manifest`, and `validate_maps` in
[src/semantic_ids.py](../src/semantic_ids.py), then `build` in
[scripts/build_semantic_ids.py](../scripts/build_semantic_ids.py).

Three learned codes are not always unique. Items with identical triples receive
suffixes in sorted item-ID order. Thus `[236,227,111,0]` has three learned
content codes and one deterministic disambiguator. The maps give all 7,154 items
unique four-code IDs. Each learned codebook uses 249 of its 256 entries.

**Check your understanding:** Why freeze these maps? A recommender checkpoint
learns the meaning of each token sequence. Changing the maps changes its labels.
Code utilization and uniqueness do not prove better recommendation quality.

## Stop 5 Follow IDs into a training batch

**Allow 10 minutes.** Read `SIDVocab`, `HistoryEncoder`, and `load_sid_artifacts`
in [src/tokenization.py](../src/tokenization.py), then `SequenceDataset` and
`collate` in [src/dataset.py](../src/dataset.py).

The raw Semantic ID and the integer tokens passed to T5 are different:

```text
Raw ID:             [236,227,111,0]
Position offsets:   [  2,258,514,770]
Model item tokens:  [238,485,625,770]
Target labels:      [238,485,625,770,1]   # 1 is EOS
```

Position-specific ranges distinguish the same code number at different levels.
The complete vocabulary includes PAD=0, EOS=1, four 256-token ID ranges, and
2,000 user buckets: 3,026 tokens total. A user bucket is SHA-256 modulo 2,000,
so different users can share a bucket.

The encoder input is a user bucket token, up to 20 item IDs flattened into
tokens, and EOS: at most `1 + 20×4 + 1 = 82` tokens. `collate` pads input tokens
with 0, creates an attention mask, and pads labels with −100 so label padding is
ignored by cross-entropy.

Run this fixture check after setup; the fixture maps are already in Git:

```bash
uv run python -m scripts.smoke_test --data-dir tests/fixtures --sid-dir tests/fixtures
```

It prints histories, readable token names, batch shapes, a finite loss/gradient
check, and label-to-item round trips. It builds a fresh random model and does
not save a checkpoint. Its loss is a mechanics check, not a quality result.

**Check your understanding:** Why is EOS not the fourth item code? There are
four item tokens, including the collision suffix; EOS is the fifth target token.

## Stop 6 Follow one optimizer step and checkpoint selection

**Allow 10 minutes.** Read `build_model` and `train_step` in
[src/recommender.py](../src/recommender.py). Then follow `run_training` in
[src/training.py](../src/training.py), using
[scripts/train_recommender.py](../scripts/train_recommender.py) as the CLI map.

The recommender uses 2 encoder layers, 2 decoder layers, width 128, 4 attention
heads, feed-forward width 512, and dropout 0.1. AdamW updates its parameters
using mean token cross-entropy and gradient clipping at norm 1.0.

Teacher forcing shifts the target to form decoder input:

```text
Labels:         [c0,  c1, c2, c3, EOS]
Decoder input:  [PAD, c0, c1, c2, c3 ]
```

At each position, the decoder receives previous true tokens during training.
At inference it must use its own previously generated tokens.

Within `run_training`, locate the train/check datasets, fixed seeded validation
subset, optimizer loop, call to `ranking_metrics(..., k=10)`, best-score comparison,
and `snapshot`. Selection evaluates the same 500 validation users every 500
steps. Training lasts 5,000 steps; the selected checkpoint is step 4,500.

Two local artifacts serve different purposes:

| Artifact | Purpose |
|---|---|
| `artifacts/recommender/poc01/best/` | Selected weights, model config, and metadata for inference |
| `artifacts/recommender/poc01/latest.pt` | Recovery snapshot with optimizer, RNG states, batch cursor, best state, and logs |

Read [src/training_state.py](../src/training_state.py) after understanding the
main loop. Saving random state and the next batch permits exact continuation
in the tested CPU environment; weights alone would not reproduce the trajectory.

**Check your understanding:** Why does the lowest training loss not automatically
choose the checkpoint? The objective optimizes token predictions; the selection
criterion measures validation ranking quality.

## Stop 7 Follow generation into a ranked list and metrics

**Allow 10 minutes.** Read [src/retrieval.py](../src/retrieval.py):
`CatalogTrie`, `make_prefix_allowed_tokens_fn`, `Retriever.recommend_batch`,
and `lookup`. Then read [scripts/predict_recommender.py](../scripts/predict_recommender.py).

The trie stores valid catalog token prefixes. Its callback skips the decoder's
start token, permits only catalog continuations, and permits EOS after four ID
tokens. Beam search keeps 20 candidate paths and returns up to 10 unique items,
ranked by summed token log-probabilities. It is approximate search over token
sequences, not exhaustive scoring of the entire catalog.

`recommend(user_id, history, k)` has no target argument. The prediction script
passes only IDs and histories, then writes two complementary JSONL files:
`predictions_test.jsonl` contains ranked product IDs and scores;
`generation_log_test.jsonl` contains raw generated tokens and beam diagnostics.
For the completed run, these live under `artifacts/recommender/poc01/`.

Finally read `popularity_ranking` in [src/baselines.py](../src/baselines.py),
`evaluate_rankings` in [src/metrics.py](../src/metrics.py), and
`load_predictions` / `evaluate` in [scripts/evaluate.py](../scripts/evaluate.py).

Popularity counts original training events once. Both methods use the full
7,154-item catalog, allow previously seen items, and use no sampled negatives.
For one relevant item, Recall@K is a hit indicator and NDCG@K is
`1/log2(rank+1)` for a hit within K. A hit at rank 3 contributes Recall=1 and
NDCG=0.5. A missing user's prediction counts as empty, rather than removing that
user from the average.

**Check your understanding:** Why can valid recommendations still be poor?
The trie enforces membership in the catalog, not relevance. This repository's
`coverage@10 = 1.0` means every user received ten unique items; it does not
measure the fraction of the catalog recommended.

## Stop 8 Explain an actual prediction and its limits

**Allow 10 minutes.** Open [the saved trace](../reports/poc01/example_trace.json)
and read `verify_generation` in [scripts/inspect_example.py](../scripts/inspect_example.py).

Follow these fields in order:

1. `observed_item_encoding`: an EltaMD sunscreen in the user's history becomes
   a `[1,768]` embedding, a `[1,32]` latent, codes `[236,227,111]`, suffix 0,
   and item tokens `[238,485,625,770]`.
2. `history_item_ids` and `history_tokens`: five observed items become the
   recommender input, with a user token and EOS.
3. `raw_generation.tokens_without_decoder_start`: `[215,310,561,770,1]`.
4. `raw_generation.decoded_sid`: subtract offsets from the first four tokens
   to obtain `[213,52,47,0]`.
5. `recommendation`: the inverse map resolves this to an Olay cleansing system.
6. `metadata`: the held-out target is a Pureology shampoo/conditioner set.
   This metadata is inspection-only for the recommender.

The top recommendation differs from the target. This trace verifies generation
and lookup; it does not establish a top-ten hit or miss by itself. Metadata for
the entire catalog, including held-out items, was available during ID learning.

### Inspect the saved evidence without writing reports

This exercise needs only the tracked reports and Python's standard library:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path('reports/poc01')
result = json.loads((root / 'results_test.json').read_text())
trace = json.loads((root / 'example_trace.json').read_text())
for method in ('popularity', 'model'):
    scores = result[method]
    print(method, {k: scores[k] for k in ('Recall@10', 'NDCG@10')})
print('Generated tokens:', trace['raw_generation']['tokens_without_decoder_start'])
print('Decoded ID:', trace['raw_generation']['decoded_sid'])
print('Recommended:', trace['recommendation']['title'])
print('Held-out target:', trace['metadata']['title'])
PY
```

### Generate one recommendation from the selected checkpoint

If the local processed data, maps, and selected weights are present, this loads
the trained model and generates for the first test history. It does not write
prediction files, update weights, or compute a new aggregate test score.

```bash
uv run python - <<'PY'
from pathlib import Path
from src.common import load_config, read_jsonl, seed_everything
from src.dataset import build_eval_examples
from src.recommender import load_checkpoint
from src.retrieval import Retriever
from src.tokenization import load_sid_artifacts

config = load_config('configs/poc01.yaml')
seed_everything(config['seed'], config['threads'])
vocab, forward, inverse, manifest = load_sid_artifacts(config['artifact_dir'])
model, meta = load_checkpoint(
    'artifacts/recommender/poc01/best', manifest['mapping_hash'], device='cpu'
)
sections = meta['sections']
retriever = Retriever(
    model, vocab, forward, inverse,
    max_history=sections['sequence']['max_history'],
    num_beams=sections['retrieval']['num_beams'],
    constrained=sections['retrieval']['constrained'], device='cpu'
)
example = build_eval_examples(
    read_jsonl(Path(config['data_dir']) / 'splits.jsonl'), 'test'
)[0]
titles = {row['item_id']: row['title'] for row in
          read_jsonl(Path(config['data_dir']) / 'items.jsonl')}
recommendation = retriever.recommend(example.user_id, example.history, k=10)
print('Selected step:', meta['step'])
for rank, (item, score) in enumerate(recommendation.items, 1):
    print(rank, item, titles[item], round(score, 4))
print('Held-out target, displayed after generation:', titles[example.target])
PY
```

For the oral exam, separate what this demonstrates from what remains unknown:

- The model exceeded popularity in one seed on a small Beauty subset. This
  does not establish significance, general superiority, or paper reproduction.
- No random-ID ablation isolates the contribution of semantic structure.
- The [supplementary data profile](../reports/presentation_data_profile.json)
  shows 728 test targets (36.4%) absent from all training events. Their metadata
  is in the known catalog, and no separate subgroup accuracy was reported.
  This alone does not demonstrate cold-start success.
- Shared-code examples provide qualitative content patterns, not category labels
  or explanations of why a recommendation suits a user.

## Use tests as worked examples

When a stage feels unclear, read the corresponding test before the full runner:

| Topic | Test to read |
|---|---|
| Cleaning, timestamp ties, repeats, splitting | [test_data.py](../tests/test_data.py) |
| Residual selection and gradient flow | [test_quantizer.py](../tests/test_quantizer.py) |
| Cache alignment, suffixes, hashes | [test_artifacts.py](../tests/test_artifacts.py) |
| Embedding interruption and reuse | [test_embedding_resume.py](../tests/test_embedding_resume.py) |
| Tokens, histories, padding | [test_tokenization_dataset.py](../tests/test_tokenization_dataset.py) |
| Teacher forcing, tiny overfit, trie, beam scores | [test_model_retrieval.py](../tests/test_model_retrieval.py) |
| Selection remains NDCG@10 when serving top five | [test_training_cli.py](../tests/test_training_cli.py) |
| Recovery, optimizer/RNG state, signals | [test_training_resume.py](../tests/test_training_resume.py) |
| Hand-calculated metrics and prediction validation | [test_metrics.py](../tests/test_metrics.py) |
| Raw-token evidence and score consistency | [test_generation_trace.py](../tests/test_generation_trace.py) |
| Synthetic pipeline integration | [test_pipeline.py](../tests/test_pipeline.py) |

Run the full suite with `uv run pytest -q`. The completed experiment records
57 passing tests. Fixture overfitting and passing tests establish mechanics;
use the real held-out scores to discuss recommendation quality.

## Choose the right commands and evidence

Most CLIs default to `configs/small.yaml`. For the completed run, explicitly use
`configs/poc01.yaml` and the selected checkpoint at
`artifacts/recommender/poc01/best/`. Prediction defaults point instead to
`artifacts/recommender/best/`; the configuration's report directory does not
automatically change that default checkpoint path.

| Task | Where to start |
|---|---|
| Understand or rehearse | Read the reports and run the read-only exercises above |
| Trace a batch without local real data | Use the fixture smoke-test command |
| Check implementation behavior | Run the relevant tests or full suite |
| Reproduce data/ID creation | Follow README stages with fresh data/artifact/report directories |
| Train another recommender on existing frozen IDs | Use `scripts.train_recommender` with a fresh `--out` directory and a separate experiment record |
| Recover interrupted, incomplete training | Follow the resume protocol in [docs/experiment.md](experiment.md) |

Root-level `reports/results_*` and `reports/run_summary.json` describe earlier
stages. The latter still contains historical “integration pending” wording.
Fixture and 20-step verification results also predate the completed run. Use
`reports/poc01/` for final recommendation results, and the older summary only
for its item-ID-stage measurements.

If local artifacts are missing, use saved reports or restore the checksummed
`artifacts/poc01_handoff.tar.gz` backup. A mapping mismatch means the checkpoint
and IDs belong to different runs; load the matching pair. A missing temporary
Python interpreter can be repaired by recreating the environment with `uv sync
--locked`, as described in the [engineering handoff](engineering_handoff.md).

## Rehearse the complete explanation

Close the code and give a two-minute explanation in this order:

1. State the next-item task and the 2,000-user, 7,154-item dataset.
2. Explain chronological holdout and rolling-prefix training examples.
3. Trace metadata through embeddings, residual codes, and collision suffixes.
4. Explain position-specific tokens and teacher forcing.
5. Describe trie-constrained beam generation and item lookup.
6. State 47 versus 15 top-ten hits, then explain the low absolute accuracy and
   why a random-ID ablation and repeated runs are needed.

For each stage, be ready to name its input, output, key source file, and one
failure its checks catch. That connects the implementation to the measured
findings without relying on memorized slide text.
