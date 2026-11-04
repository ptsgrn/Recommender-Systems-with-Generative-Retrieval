# Explainable TIGER: Semantic IDs and generative retrieval

This repository implements item preprocessing, frozen Sentence-T5 embeddings,
residual quantization, unique Semantic IDs, full-catalog metrics, and a
training-only popularity and item-cosine baselines, plus a T5 sequence model and generative retrieval.
The original implementation plan is in
[todo.md](todo.md); current course readiness is tracked below. Developer B's sequence model and retrieval are described in
[Developer B: sequence model and retrieval](#developer-b-sequence-model-and-retrieval).
See [HANDOFF.md](HANDOFF.md) for a self-contained ChatGPT presentation prompt with
the measured data, findings, and limitations, and the
[engineering handoff](docs/engineering_handoff.md) for run and integration details.
This is a small teaching implementation, not a reproduction of published scores.
Start with the [repository walkthrough](docs/repository_walkthrough.md) for a
guided reading order, hands-on exercises, and oral-exam checkpoints.
The bounded training, recovery, evaluation and trace protocol is in
[docs/experiment.md](docs/experiment.md).

The completed 5,000-step POC selected checkpoint step 4,500 using validation data.
On 2,000 test users, Recall@10 is **0.0235** and NDCG@10 is **0.011105**;
training-only popularity scores **0.0075** and **0.002797**. See the
[test comparison](reports/poc01/results_test.md),
[experiment record](reports/poc01/experiment.json), and
[verified prediction trace](reports/poc01/example_trace.json).
These are one-run results on a small subset, not paper benchmark reproduction.

## Course requirements and submission status

The implementation and two-baseline comparison are complete. The submission
package and presentation are prepared as described below; individual rehearsal, and the MyCourseVille upload still require the group.
Status updated **2026-11-04**. See the [three-method comparison](reports/baselines/results_test.md),
[baseline protocol](docs/baseline.md), and [oral-exam slides](docs/oral_exam_slides.md) and [editable PPTX](docs/oral_exam.pptx) / [PDF](docs/oral_exam.pdf).

| Requirement | Status | Evidence or remaining work |
|---|---|---|
| Implement the assigned paper's method | Core method implemented | Content embeddings, residual quantization, unique Semantic IDs, autoregressive T5, and item lookup; see the [paper correspondence](#paper-correspondence-and-limitations) |
| Report one dataset | Implemented with a disclosed reduction | Amazon Beauty 2014 subset: 2,000 users, 7,154 items, and 11,264 rolling training examples |
| Compare with two relevant baselines | Complete | Popularity and item cosine, fitted on training events and evaluated under the same protocol; [comparison](reports/baselines/results_test.md) |
| Report at least one evaluation metric | Complete | Recall and NDCG at 5 and 10 are measured on all 2,000 test users |
| Disclose configuration changes in the presentation | Prepared | Dedicated modifications slide in [oral-exam slides](docs/oral_exam_slides.md), with source settings below |
| Every member understands the full code | Requires individual rehearsal | Each member must explain data preparation, modeling, prediction ranking, and evaluation, regardless of presentation ownership |
| Explain contributions among three members | Complete | Names, student IDs, and actual work recorded in [contributions](docs/contributions.md) and the presentation |
| Submit all source code as a ZIP to MyCourseVille before class | ZIP prepared; upload left to user | `submission/5_term project.zip` includes source, a 100-event dataset sample and source links, presentation files, and reports; see [submission instructions](docs/submission.md) |
| Confirm the group number | Complete | Group 5, confirmed by the group; filename: `5_term project.zip` |

Verification on the current checkout: `.venv/bin/python -m pytest -q` passed
**59 tests** after adding the second baseline. All four selected-checkpoint files matched the SHA-256 values in
[reports/poc01/experiment.json](reports/poc01/experiment.json). This supports
implementation integrity, not a guarantee of oral-exam performance.

### Paper settings and project modifications

The course permits computational reductions when clearly stated in the
presentation. The project preserves the two-stage method while using the
following settings. Paper values come from
[Section 4 of the original paper](https://arxiv.org/html/2305.05065#S4);
project values come from [configs/poc01.yaml](configs/poc01.yaml) and the saved
experiment records.

| Setting | Original paper | Completed project run |
|---|---|---|
| Encoder / decoder layers | 4 / 4 | 2 / 2 |
| Attention heads × head dimension | 6 × 64 | 4 × 32 |
| Feed-forward width | 1,024 | 512 |
| Beauty recommender training | 200,000 steps | 5,000 steps; step 4,500 selected on validation |
| Recommender batch size | 256 | 64 |
| Recommender learning rate | 0.01 for the first 10,000 steps, then inverse-square-root decay | Constant 0.001 with AdamW |
| Quantizer optimizer / learning rate | Adagrad / 0.4 | Adam / 0.0001 |
| Quantizer batch size | 1,024 | 16 on CPU |
| Quantizer training | 20,000 epochs | 10 continuous warm-up + 7 quantized epochs completed; quantized epoch 2 selected |
| Invalid generated IDs | Beam generation followed by filtering | Catalog-prefix-constrained beam search |

Also disclose the 2,000-user sample, the exact frozen Sentence-T5 checkpoint,
residual k-means initialization after warm-up, and checkpoint selection on a
fixed 500-user validation subset. The warm-up/lower-rate repair addressed an
initial quantizer collapse. Quantized fine-tuning did not improve reconstruction
over the post-warm-up initialization, so usable IDs should not be presented as
proof that every training phase improved quality.

The paper's smallest original dataset by users and items is Toys and Games;
this project instead uses a reduced Beauty subset. Explain this as a
computational adaptation under the course's smaller-dataset allowance. The
[paper's dataset statistics and splitting protocol](https://arxiv.org/html/2305.05065#A3)
provide the original comparison.

### Completion checklist

- [x] Implement and measure a second relevant baseline with the same split,
  full catalog, repeat-item policy, and metrics. Item cosine uses binary
  training-user co-occurrence and has no tuned hyperparameters.
- [x] Store the new comparison in `reports/baselines/`, reusing frozen TIGER
  measurements without retraining or rerunning model test generation.
- [x] Prepare presentation slides covering paper-versus-project modifications,
  measured results, limitations, and code questions.
- [x] Record all three members' names and reported actual contributions in
  [docs/contributions.md](docs/contributions.md). Individual cross-module rehearsal remains open.
- [x] Prepare and inspect `submission/5_term project.zip`, with all source,
  tests, configs, documentation, reports, lockfile, presentation files, public
  dataset download links, and 100 real events with matching item metadata.
  The ZIP includes a SHA-256 manifest; its checksum is provided alongside it.
  Full weights/data remain in the separate local demo backup.
- [ ] Group action: one representative submits `5_term project.zip` to MyCourseVille before class on **November 5, 2026**.
  Automated upload is skipped at the user’s request.
- [ ] Rehearse individual questions across the entire pipeline using the
  [repository walkthrough](docs/repository_walkthrough.md).

The current model's Recall@10 is 2.35% versus popularity's 0.75%: **47 versus 15
hits among 2,000 users**. Item cosine achieves 1.75%, or **35 hits**. It still misses 1,953 targets. The supplied course
requirements do not require beating baselines, reproducing published scores, or
running a random-ID ablation. They do require the baseline comparison, honest
analysis, and full understanding of the implementation.

### Ten minute introduction and three presenter plan

The oral examination allocates about **60 minutes per group**: at most
**10 minutes** to introduce brief results and task allocation, followed by about
**50 minutes of questions**. The following is a proposed presentation split,
not a record of who implemented each component.

| Presenter | Topics | Time |
|---|---|---:|
| นายพัสกร ยืนยง | Problem, chronological data split, embeddings, and Semantic IDs | 3 minutes |
| นายภาณุพงศ์ จันทร์ลาม | Tokenization, Transformer operation, teacher forcing, and checkpoint selection | 3 minutes |
| นายธิปก ปัญญาวุธ | Beam search, evaluation, baseline comparison, findings, and limitations | 3 minutes |
| All three | Actual contributions and how members reviewed each other's code | 1 minute |

Keep detailed code excerpts, tensor shapes, equations, gradient routing, and
recovery behavior in backup slides for questions. Every member should be ready
to explain the full pipeline, including why the decoder cannot see future
target tokens, how beam scores become rankings, why mappings are frozen, and
how validation selection differs from final test evaluation. Use saved
checkpoints and outputs for demonstrations rather than retraining live.

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

## Developer B: sequence model and retrieval

| File | Role |
|---|---|
| `src/tokenization.py` | `SIDVocab` (code tuple ↔ offset token), `HistoryEncoder.encode_history`, loads and validates A's maps |
| `src/dataset.py` | Rolling training prefixes, val/test examples, padded batches (`labels` padding = -100) |
| `src/recommender.py` | T5 from `T5Config` with the 3,026-token vocabulary; train step; checkpoints store the mapping hash |
| `src/retrieval.py` | Beam search, catalog prefix trie, lookup, `Retriever.recommend(user_id, history, k)` |
| `scripts/make_fixture.py` | 16-item hand-made fixture in A's exact artifact format (debugging only) |
| `scripts/smoke_test.py` | J1: real-ID batch forward/backward and label → item round trip |
| `scripts/train_recommender.py` | Training, validation NDCG@10 checkpoint selection, `--benchmark-steps`, `--overfit` |
| `scripts/predict_recommender.py` | Predictions JSONL for `scripts.evaluate` plus a raw-token generation log |

Fixture checks (no real data needed):

```bash
uv run python -m scripts.make_fixture
uv run python -m scripts.smoke_test --data-dir tests/fixtures --sid-dir tests/fixtures
uv run python -m scripts.train_recommender --data-dir tests/fixtures --sid-dir tests/fixtures --out runs/overfit --overfit --max-steps 300 --eval-every 100 --batch-size 32
```

Real run, after A's maps are frozen in `artifacts/`:

```bash
uv run python -m scripts.smoke_test --config artifacts/run_config.yaml
uv run python -m scripts.train_recommender --config artifacts/run_config.yaml --benchmark-steps 100
uv run python -m scripts.train_recommender --config artifacts/run_config.yaml --max-steps N
uv run python -m scripts.predict_recommender --config artifacts/run_config.yaml --split val
uv run python -m scripts.predict_recommender --config artifacts/run_config.yaml --split test
uv run python -m scripts.evaluate --config artifacts/run_config.yaml --split test --predictions artifacts/recommender/predictions_test.jsonl
```

The selected checkpoint is `artifacts/recommender/best/`, chosen by validation
NDCG@10 on a fixed seeded subset; `run_meta.json` records the step, metrics,
mapping hash and decoding mode, and loading refuses a different mapping hash.
Selection stays at NDCG@10 even when `retrieval.top_k` changes the number of served
recommendations. Single-beam generation is supported; its generated sequence is
scored against the full vocabulary in a separate forward pass, without using a
ground-truth target or renormalizing over trie choices.
`generation_log_{split}.jsonl` holds the raw generated tokens and beam scores
for the oral-exam trace. `--unconstrained` writes a separate diagnostic run that
counts invalid IDs instead of using the catalog trie.

Merged-pipeline verification is recorded in `reports/merge_verification.json`.
The fixture and 20-step real-data verification checkpoints are local under
`artifacts/verification/`; they establish mechanics, not final model quality.

Intel Macs cannot install the locked `torch>=2.5`. B's code and tests were also
verified there in a plain virtual environment with Python 3.9/3.12, torch 2.2.2
and transformers 4.44.2/4.57.6 (`PYTHONPATH=. python -m pytest tests/test_tokenization_dataset.py tests/test_model_retrieval.py`).

## Paper correspondence and limitations

| Paper area | Source |
|---|---|
| §3.1, Figure 3: residual quantization | `src/rqvae.py` |
| §3.1: unique Semantic IDs | `src/semantic_ids.py` |
| Appendix C: chronological evaluation | `src/data.py`, `src/metrics.py` |
| §3.2, Figure 2: sequence-to-sequence model | `src/tokenization.py`, `src/dataset.py`, `src/recommender.py` |
| §3.2, §4.5: beam search and invalid IDs | `src/retrieval.py` |

See [docs/deviations.md](docs/deviations.md) for adaptations. The reference is
[Rajput et al., Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065).
Passing synthetic tests proves pipeline mechanics, not real-data model quality.
Matching paper scores, a random-ID ablation, and a web UI are outside this
implementation. The live demo should reload saved artifacts rather
than retrain. Back up data, maps, checkpoints and reports before the exam. Small
reports are eligible for Git; the data, model cache and large artifacts are ignored.
