# Developer A handoff — paused 2026-11-03

Work stopped at the user's request. Resume from this file and `todo.md`.

## Completed

- Implemented A1–A6 code, setup, pinned `uv.lock`, documentation and tests.
- Downloaded and cleaned Amazon Beauty 2014 reviews and matching metadata.
- Prepared 2,000 users, 7,154 items, 17,264 retained interactions and 11,264 rolling
  training examples. Splits, chronological audits, source checksums and exclusion
  counts are in ignored `data/processed/`.
- Measured the local CPU benchmark and saved `reports/benchmark.json`. It selected
  2,000 users under the two-hour projected compute budget. Actual full inference
  was slower than the initial estimate, but remained within that budget.
- Pinned Sentence-T5 revision `fc5d4628481afbbaaacd7af6bb07cf9d3865f781` in
  ignored `artifacts/run_config.yaml`; downloaded its model cache locally.
- Measured popularity on validation and test, using only original training events.
  Test Recall@10 = 0.0075; NDCG@10 = 0.0027974684. See `reports/results_test.json`.
- Last full test run: **22 passed**. Includes synthetic end-to-end quantizer
  training, checkpoint reload, stable map export, evaluation and trace generation.
  Synthetic tests do not establish real-data Semantic ID quality.

## Exact stopping point

Interrupted the full embedding process with Ctrl-C at **109/448 batches** (about
1,744 items processed), after approximately **14 minutes 28 seconds** of encoding.
Confirmed exit code 130; no embedding or training process was left running.

**Partial embeddings were only in memory and were not saved.** Restart the full
embedding command. There is no `item_embeddings.npy`, real RQ-VAE checkpoint,
frozen real Semantic ID mapping, or real ID diagnostics yet. Do not report A2–A4
as completed experiments. The inference-time logging addition in the source was
made after that process started; a fresh run will save `encoding_seconds`.

The implementation was already committed as `c7604ff` when the handoff began;
this handoff is a separate documentation commit. No Developer B model is present.

## Resume on this machine

Run from the repository root. Large raw files, caches and artifacts are ignored
by Git and remain on this machine. A different machine needs those copied or
downloaded again. Small measured reports are tracked.

```bash
uv sync --locked
uv run pytest -q
HF_HOME=.cache/huggingface HF_HUB_OFFLINE=1 uv run python -m scripts.embed_items --config artifacts/run_config.yaml
uv run python -m scripts.train_rqvae --config artifacts/run_config.yaml
uv run python -m scripts.build_semantic_ids --config artifacts/run_config.yaml
uv run python -m scripts.inspect_example --config artifacts/run_config.yaml
```

The environment used Python 3.12.14 installed under `/tmp/tiger-python`, with
`UV_CACHE_DIR=/tmp/tiger-uv`. If that temporary interpreter has been removed,
`uv sync` must install/recreate Python and the environment. Network access was
needed outside the restricted sandbox for initial downloads. The embedding
command above reuses the existing cache offline.

For a fresh checkout without local data/config/cache, follow the full sequence
in `README.md`: prepare data, benchmark, prepare with the selected run config,
then embed. Do not run data preparation and embedding concurrently.

## Remaining acceptance checks

1. Finish real embeddings: verify `[7154,768]`, float32, finite values, aligned
   item IDs, exact model revision and cache checksums.
2. Train the real RQ-VAE; inspect loss reduction, finite gradients, active codes
   and collision rates. Investigate collapse before exporting maps. Current
   defaults are Adam 0.001, CPU batch 16, at most 50 epochs, patience five.
3. Reload the best reconstruction checkpoint and export real IDs. Re-export and
   compare mapping hashes. Collision suffix overflow must fail rather than wrap.
4. Generate `reports/id_diagnostics.json` and `reports/example_trace.json`.
   Record measured runtime and limitations. Back up the ignored real artifacts.
5. Give B the splits, item catalog, checkpoint, forward/inverse maps and manifest
   with its mapping hash. Keep B's production tokenizer/recommender out of A's scope.
6. Once B supplies saved ranked predictions, run `scripts.evaluate` and
   `scripts.inspect_example` with `--predictions`. Model metrics and actual
   generated-token evidence remain pending until then. A lookup of a predicted
   item's known SID is not evidence of autoregressive generation.

Use validation for model selection, freeze the checkpoint, then evaluate test.
Preserve full-catalog candidates and allow repeat recommendations. No score
target, random-ID ablation, external GPU provisioning or B implementation is required.
