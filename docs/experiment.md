# POC experiment protocol

`configs/poc01.yaml` resolves all model, optimizer, data, and decoding settings.
Use the existing frozen IDs and content cache; do not retrain the quantizer.

Run from the repository root:

```bash
uv run python -m scripts.train_recommender --config configs/poc01.yaml --benchmark-steps 100 --out artifacts/benchmark/poc01
uv run python -m scripts.train_recommender --config configs/poc01.yaml --out artifacts/recommender/poc01
```

The benchmark is separate from the experiment. Train 5,000 total steps with
seed 42, CPU/two threads, batch 64, AdamW 0.001/weight decay 0.01. Checkpoint
selection compares NDCG@10 every 500 steps on the same 500 seeded validation users;
ties retain the earlier checkpoint. Decoding uses catalog constraints, beam 20,
top 10, full-catalog candidates and repeat recommendations.

## Recovery

`best/` remains the inference checkpoint. `latest.pt` is an atomic recovery
snapshot saved initially, every 100 steps, on improvement, at completion, and on
a graceful stop. It includes model/optimizer weights, random states, epoch/batch
cursor, best weights/metadata, validation records and cumulative active runtime.

Ctrl-C/SIGTERM requests a stop after the current safe training boundary; the
process saves recovery state and exits with 130/143. A hard kill can require
replaying up to 100 steps. Restart with:

```bash
uv run python -m scripts.train_recommender --config configs/poc01.yaml --resume artifacts/recommender/poc01/latest.pt
```

The target remains 5,000 total steps, not 5,000 additional steps. Fresh runs refuse
nonempty output directories. Resume verifies data/mapping/config/environment hashes,
reconciles validation logs and best exports with the authoritative snapshot, then
continues the saved epoch permutation at the next batch. Permutations use seed +
epoch and independent loader generators, so iterator creation does not consume
dropout RNG. This batching policy is an experiment implementation choice.

Exact continuation is tested on the current CPU environment with dropout enabled,
including epoch/validation boundaries, Python/NumPy/Torch random draws, optimizer
moments, weights and losses. Do not claim bitwise portability across environments.

## Validation, test and trace

After completion, record the selected checkpoint hash, then run:

```bash
uv run python -m scripts.predict_recommender --config configs/poc01.yaml --split val --checkpoint artifacts/recommender/poc01/best --out-dir artifacts/recommender/poc01
uv run python -m scripts.evaluate --config configs/poc01.yaml --split val --predictions artifacts/recommender/poc01/predictions_val.jsonl
uv run python -m scripts.predict_recommender --config configs/poc01.yaml --split test --checkpoint artifacts/recommender/poc01/best --out-dir artifacts/recommender/poc01
uv run python -m scripts.evaluate --config configs/poc01.yaml --split test --predictions artifacts/recommender/poc01/predictions_test.jsonl
uv run python -m scripts.inspect_example --config configs/poc01.yaml --split test --predictions artifacts/recommender/poc01/predictions_test.jsonl --generation-log artifacts/recommender/poc01/generation_log_test.jsonl
```

Reports are isolated under `reports/poc01/`. The trace verifies raw beam tokens,
EOS, decoded SID, exported score (when present), lookup and title, and includes an
observed history item's metadata-to-token transformation. Target metadata is
labeled as inspection-only, not a model input. `--generation-log` requires
predictions; missing or inconsistent evidence fails.

This is one predeclared experiment. Evaluate the frozen selected checkpoint on
test once; retain weak scores without tuning on test. Compare Recall/NDCG@5/10
against training-only popularity and report coverage, users and actual runtimes.
The raw generation log and prediction hashes bind the trace to the experiment.
Large artifacts belong in the local checksummed archive, while source, config,
small reports and documentation are tracked in Git.

## Measured outcome

The completed run used 5,000 steps and selected **step 4,500**. Active training and
validation took **1,877.3 seconds**; full validation/test generation took **27.2 /
22.4 seconds**. Both splits contain 2,000 users; every user received ten unique
known items and both generation passes had zero invalid beams.

| Test metric | Popularity | Recommender |
|---|---:|---:|
| Recall@5 | 0.002500 | 0.012500 |
| Recall@10 | 0.007500 | 0.023500 |
| NDCG@5 | 0.001246 | 0.007534 |
| NDCG@10 | 0.002797 | 0.011105 |

The selected checkpoint scored 0.01795087 NDCG@10 on the selection subset and
0.01529662 on all validation users. It was frozen before the single final test
assessment. No settings were changed using test results. Losses in the training
log are last-batch training losses and token-weighted validation means.

All 57 tests passed. A controlled real-run SIGINT stopped at step 288 and resumed
from that step; the separate fixture comparisons prove exact continuation with
dropout, optimizer moments and batch order. See `reports/poc01/experiment.json`
for hashes, selection, metrics, runtime and integrity checks.

The archive `artifacts/poc01_handoff.tar.gz` contains the paired inference and
recovery checkpoints, frozen ID pipeline, data, logs, reports and source, with a
SHA-256 sidecar. Preserve the completed run: `--resume` is for incomplete training;
use the cached checkpoint and generation log for the demo.

This single-seed subset result supports the POC demonstration. It does not
establish statistical significance or isolate the benefit of Semantic IDs; no
random-ID ablation, cold-start evaluation or paper benchmark reproduction was run.
