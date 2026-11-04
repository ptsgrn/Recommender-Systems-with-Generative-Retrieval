# Project handoff — POC experiment complete, 2026-11-04

The real-data TIGER POC is complete: frozen Semantic IDs, trained recommender,
validation-selected checkpoint, held-out test comparison, and verified raw-token
trace. The earlier verification runs below remain as provenance.

## Completed experiment: poc01

- Trained 5,000 steps with seed 42, CPU/two threads, batch 64, AdamW 0.001.
  Active optimization/validation runtime: 1,877.3 seconds (31.3 minutes).
- Selected step 4,500 by NDCG@10 on the fixed seeded 500-user validation subset
  (0.01795087). The final checkpoint's own step 5,000 was not selected.
- Final test covered all 2,000 users and 7,154 catalog items. Model Recall@10
  **0.0235**, NDCG@10 **0.01110530**; popularity **0.0075**, **0.00279747**.
- All users received ten unique valid items. Raw beam IDs and scores match all
  exported validation/test recommendations. Test was evaluated once after freeze.
- All 57 tests pass. Exact CPU resume is tested with dropout; an actual SIGINT
  stopped this run at step 288 and resumed from its full optimizer/RNG/batch state.
- Inference checkpoint: `artifacts/recommender/poc01/best/`; recovery state:
  `artifacts/recommender/poc01/latest.pt`. Completed runs should remain frozen;
  use prediction/inspection commands for the demo rather than resuming training.
- Results, protocol, checkpoint hashes, training records and verified trace:
  `reports/poc01/`. Full summary: `reports/poc01/experiment.json`.
- Complete local backup: `artifacts/poc01_handoff.tar.gz` with SHA-256 alongside
  it. It includes the selected/recovery checkpoints and frozen ID artifacts.

This is one seed on a small real subset. Higher scores than popularity in this
run do not establish statistical significance, the effect of semantic versus
random IDs, or reproduction of the paper's benchmark results.

## Merged pipeline verification

- All 44 tests pass, including B's fixture overfit, checkpoint reload, beam-score
  enumeration, and regressions for top-5 serving and single-beam generation.
- The real-ID smoke test passes forward/backward, finite gradients and all eight
  label round-trips with the frozen 7,154-item catalog.
- A 300-step fixture CLI run reaches training NDCG@10=1.0; reload, prediction
  export and A's evaluator work together.
- A separate 20-step real-data verification run exercises training, checkpoint
  selection/reload and validation prediction export. Its artifacts are under
  `artifacts/verification/real/`, with an isolated report directory. This short
  run is a mechanical check, not the final experiment. It never evaluates test.
- See `reports/merge_verification.json` for measured verification results. The
  published A-stage scores and frozen artifacts remain unchanged.

## Measured run

- 2,000 users, 7,154 items, 17,264 retained events, 11,264 rolling training examples.
- Finite float32 embeddings `[7154,768]`; encoding took 2,557.7 seconds (42.6 min).
- Sentence-T5 revision: `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`.
- Ten continuous warm-up epochs plus seven quantized epochs took 85.5 seconds.
- Best trained checkpoint: quantized epoch 2, step 896; reconstruction MSE
  0.0001396825436. Selection compares quantized-training epochs only.
- Active codes: 249/249/249 of 256; largest three-code collision group: two items.
- Suffix usage: 7,101 zeros and 53 ones. All 7,154 four-code IDs are unique and
  round-trip through the maps. Repeat export was byte-identical.
- Test popularity Recall@10: 0.0075; NDCG@10: 0.0027974684. These are baseline scores.
- The A-stage checks include interruption/resume, gradients, metrics, collision
  handling, and a synthetic end-to-end run. Real cache reuse/export/diagnostics pass.

Mapping hash: `688ea781e09ddfdb8ea1875a08858dc131f4c305f33b0e35753ec1693a8ab260`.

## Training repair and limitations

The initial Adam 0.001/no-warm-up run killed the final encoder ReLU: latent
standard deviations were zero and every item shared one tuple. Its checkpoint
and logs remain in `artifacts/failed_collapse/`. The repaired run retains the
architecture/codebook sizes, warms up the continuous autoencoder for ten epochs,
then initializes residual codebooks with k-means. Both phases use rate 0.0001.
This is a recorded project adaptation, not a setting attributed to the paper.

Warm-up MSE decreased from 0.00020559 after epoch 1 to 0.00013807 after epoch 10.
Quantized fine-tuning did **not** improve post-warm-up initialization MSE
(0.00013895). The chosen checkpoint is the best trained epoch, not evidence that
every phase improved reconstruction or that these IDs improve recommendation.

## Artifact handoff

- `data/processed/`: catalog, splits, chronological audit, statistics and source hashes.
- `artifacts/`: aligned item IDs/embeddings, embedding manifest, `rqvae.pt`, full
  training log/config, forward/inverse SID maps and frozen vocabulary manifest.
- `reports/`: measured baseline scores, ID diagnostics, example trace and run summary.
- `artifacts/developer_a_handoff.tar.gz`: packaged backup with SHA-256 alongside it.
  Includes processed data, frozen artifacts, reports, docs and failed-run evidence;
  excludes raw downloads and the large pretrained encoder cache.

Large data, model caches and artifacts are ignored by Git and remain local.
Copy the archive to another machine; Git alone does not contain model weights.
Small measured reports are tracked. See README for full formats and commands.

## Demo and reproducibility

```bash
uv sync --locked
uv run pytest -q
uv run python -m scripts.inspect_example --config configs/poc01.yaml --split test --predictions artifacts/recommender/poc01/predictions_test.jsonl --generation-log artifacts/recommender/poc01/generation_log_test.jsonl
```

B must verify the mapping hash with each checkpoint. Offsets are `[2,258,514,770]`,
PAD=0, EOS=1, user offset=1026, vocabulary size=3026. Keep held-out targets out of
training, use the full catalog, and allow repeated items. Select on validation,
freeze the model, then evaluate test. Missing users receive empty predictions.

The historical A-stage trace and isolated 20-step verification reports are kept
unchanged. Use `reports/poc01/example_trace.json` for the actual experiment: it
includes an observed history item's content encoding and verifies raw generated
tokens against the returned item/title. Ground-truth metadata is marked as
inspection-only and was not given to generation. Do not mix fixture or 20-step
verification scores into the poc01 result table.

## Re-running A

Use a fresh artifact directory for a new experiment; frozen mappings prevent
overwriting this run. Embedding progress now checkpoints every 128 items and
resumes matching catalog/text/model identity automatically. The completed run
needs no partial checkpoint. Its timing excludes the discarded first interrupted pass.

Python 3.12.14 was installed under `/tmp/tiger-python`. Rebuild with `uv sync` if
that temporary interpreter is removed. Back up the ignored artifacts before the exam.
