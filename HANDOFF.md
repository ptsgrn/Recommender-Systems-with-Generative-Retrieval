# Oral-exam presentation handoff — 2026-11-04

Copy this entire document into ChatGPT Web, or attach it with the message:
“Create the oral-exam presentation specified in this handoff.” It contains the
facts needed to draft slides without access to this repository. File paths
identify evidence; they are not files ChatGPT must be able to open.

The earlier implementation and recovery handoff is preserved in
`docs/engineering_handoff.md` for teammates.

---

## Updated submission and presentation evidence

Use `docs/oral_exam_slides.md` for the three-presenter, 10-minute introduction
and backup questions. The completed second baseline is item cosine, fitted only
on training-user co-occurrence, with no tuned hyperparameters. The current
comparison is `reports/baselines/results_test.json`; the original `poc01` records
remain frozen. Test Recall@10 / NDCG@10: popularity 0.0075 / 0.0027974684;
item cosine 0.0175 / 0.0065503020; TIGER 0.0235 / 0.0111052976. These correspond
to 15, 35, and 47 top-ten hits among 2,000 users. Current suite: 59 passing tests.
The original experiment's 57-test count below is historical. Confirmed member names and contributions are in `docs/contributions.md`;
use those names in the deck.

## Instructions to the presentation creator

Create an English oral-exam presentation about **Recommender Systems with
Generative Retrieval: a TIGER-inspired proof of concept**. The audience is a
computer-science examiner who will ask about implementation, data, experimental
findings, and limitations. Assume a **10-minute talk with 10 main slides**,
followed by questions. Use placeholders for presenter names and course details;
do not invent them. No clarification is needed to produce the first draft.

Deliver:

1. Ten numbered slides with a clear title, at most four short bullets each,
   and a suggested diagram or chart where useful.
2. Speaker notes for every slide, explaining what each finding means, with
   timing allocations totaling 10 minutes. Put detail in notes or backup slides.
3. Exact chart data for the test comparison from the table below.
4. Five backup slides: quantization and gradients; attention and tensor shapes;
   baseline worked example; ranking and metrics; checkpoint recovery.
5. Ten likely examiner questions with concise, defensible answers.
6. A short opening and closing script. End with the supported conclusion:
   “The pipeline works end to end and outperformed popularity in this run,
   but absolute accuracy is low and the contribution of Semantic IDs is unproven.”

Use a readable academic design, consistent colors for the three methods, and large
labels. Favor a pipeline diagram, chronological split example, compact result
table, two separate metric charts, and one honest prediction trace. If file
creation is available, also produce an editable 16:9 PPTX with speaker notes;
otherwise provide the complete slide content and notes directly.

Treat the evidence below as authoritative for this project. Distinguish
**measured results**, **interpretations**, and **future work**. Do not invent
experiments, error bars, significance tests, learning curves, paper benchmark
numbers, causal explanations, or results for untested baselines. Keep source
labels such as [E1] in speaker notes. Fixture and integration checks are not final
model accuracy. Valid output does not mean accurate recommendation, and
content-derived IDs do not explain a user's preference.

## Suggested slide sequence

| Slide | Presenter | Topic | Time |
|---|---|---|---:|
| 1 | พัสกร ยืนยง | Generative retrieval for next-item recommendation | 0:30 |
| 2 | พัสกร ยืนยง | Data and chronological holdout | 1:00 |
| 3 | พัสกร ยืนยง | From product content to unique Semantic IDs | 1:30 |
| 4 | ภาณุพงศ์ จันทร์ลาม | Constructing model inputs and labels | 1:00 |
| 5 | ภาณุพงศ์ จันทร์ลาม | Training and selecting the recommender | 1:00 |
| 6 | ภาณุพงศ์ จันทร์ลาม | Disclosed changes from the paper | 1:00 |
| 7 | ธิปก ปัญญาวุธ | Generating and ranking catalog items | 1:00 |
| 8 | ธิปก ปัญญาวุธ | Two baselines and held-out results | 1:00 |
| 9 | ธิปก ปัญญาวุธ | What the results support and leave open | 1:00 |
| 10 | All three | Actual contributions and shared preparation | 1:00 |

## Project purpose and implementation [E1, E2, E5]

The task is next-item recommendation from a user's chronological interaction
history. This repository implements a small TIGER-inspired teaching proof of
concept, not a reproduction of published benchmark scores.

```text
OFFLINE ITEM REPRESENTATION
Product title + brand + categories + price
  → frozen pretrained Sentence-T5 → 768-dimensional embedding
  → trained residual-quantized autoencoder → 32-dimensional latent
  → 3 learned codes + deterministic collision suffix → frozen item ID maps

SEQUENTIAL RECOMMENDATION
User bucket token + latest 20 history items expressed as ID tokens
  → small T5 encoder–decoder trained from scratch
  → generate 4 item-ID tokens + EOS with catalog-constrained beam search
  → inverse-map lookup → ranked catalog items and titles
```

The pretrained component is `sentence-transformers/sentence-t5-base`. The
recommendation Transformer starts from random weights with a custom integer
vocabulary. It generates item identifiers, not product-name text. Retrieval uses
generation followed by lookup, without nearest-neighbor search at serving.

Implemented project logic includes deterministic preprocessing/splits, residual
quantization, collision handling, frozen maps, rolling-prefix datasets,
tokenization, training/checkpoint selection, retrieval, evaluation, and a raw-token
trace. PyTorch, Hugging Face Transformers, Sentence Transformers, and scikit-learn
supply neural-network and k-means tools.

## Data and preprocessing [E3, E4, E5]

Source: **Amazon Reviews 2014, Beauty 5-core reviews plus Beauty metadata**.
Do not substitute the 2018 or 2023 release. Reviews are treated as interaction
events; the prediction target is the next item, not its rating or review text.

| Quantity | Value |
|---|---:|
| Source review records | 198,502 |
| Source review users / items | 22,363 / 12,101 |
| Metadata records / unusable metadata records | 259,204 / 444 |
| Review events missing usable metadata | 131 |
| Eligible users after filtering | 22,334 |
| Sampled users, seed 42 | 2,000 |
| Selected catalog items | 7,154 |
| Retained interaction events | 17,264 |
| Original training events | 13,264 |
| Rolling-prefix training examples | 11,264 |
| Validation / test targets | 2,000 / 2,000 |
| Retained events per user: min / median / mean / max | 5 / 6 / 8.632 / 149 |
| Distinct items appearing in training events | 5,999 |
| Catalog items absent from all training events | 1,155 |
| Test targets absent from all training events | 728 of 2,000 (36.4%) |

The final four rows are descriptive statistics computed from the existing split
file for this presentation [E4], not additional recommendation experiments. The
catalog includes items from retained training, validation, and test events. The
metadata file is broader than the review catalog; its total is not the number of
items recommended in this run.

Cleaning requires a nonempty item ID and cleaned title, normalizes HTML entities
and whitespace, and marks missing brand/category/price explicitly. Only identical
complete review records are deduplicated; other repeated interactions are kept.
The measured source has zero exact duplicate review records. Require at least
five retained events per user, then sample 2,000 users deterministically and keep
their complete surviving histories. The filtered subset is not guaranteed to
retain the original dataset's item-level 5-core property.

Item text uses title, brand, category paths, and price; it excludes review text.
All selected-catalog metadata is available to embedding and quantization,
including metadata for held-out items. This is a **transductive catalog
assumption**, not an evaluation of unseen future inventory.

## Splitting, examples, and leakage boundary [E2, E5]

Sort each user's events by timestamp; break ties by original source-row order.
The last event is test, the preceding event validation, and earlier events train.

For `[A,B,C,D,E]`:

- Training examples: `[A] → B`, `[A,B] → C`.
- Validation: `[A,B,C] → D`.
- Test: `[A,B,C,D] → E`; the validation event is observed history at test time.

The first training event cannot be a rolling-prefix target, explaining
13,264 training events minus 2,000 users = 11,264 training examples. Input encoding
keeps the latest 20 items. Held-out events for a user are never that user's
training targets; the same product can occur in other users' training histories.
This is a per-user chronological split, not a global calendar-time split.

Popularity counts each original training event once, rather than duplicated
appearances across rolling prefixes. It uses no validation or test events.
Quantization uses catalog metadata without interaction targets.

## Semantic IDs and quantizer findings [E5, E6]

The autoencoder has encoder dimensions `768 → 512 → 256 → 128 → 32` and a
mirrored decoder. Three residual codebooks each contain 256 vectors of size 32.
At each level, choose the nearest code vector and subtract it from the residual.
Reconstruction uses their sum. The objective combines reconstruction, codebook,
and commitment losses with commitment weight 0.25. A straight-through estimator
lets reconstruction gradients reach the encoder. No Gaussian sampling or
KL-divergence term is used.

The initial learning-rate-0.001 run without warm-up collapsed: latent variance
was zero and all items received the same learned tuple. The recorded repair was
ten continuous-autoencoder warm-up epochs at 0.0001, followed by residual k-means
initialization and quantized training at 0.0001. This is a project adaptation.

| Quantizer observation | Measured value |
|---|---:|
| Warm-up / quantized epochs completed | 10 / 7 |
| Selected quantized epoch / step | 2 / 896 |
| Selected reconstruction MSE | 0.0001396825436 |
| Post-warm-up, k-means-initialized reconstruction MSE | 0.0001389477627 |
| Active entries in each learned codebook | 249 / 256 at all three levels |
| Largest three-code collision group | 2 items |
| Suffix 0 / suffix 1 usage | 7,101 / 53 items |
| Unique final four-code IDs | 7,154 / 7,154 items |

Selection compares trained quantized epochs by lowest full-catalog reconstruction
MSE. **Quantized fine-tuning did not improve post-warm-up initialization MSE**;
do not present every phase as an improvement. Lower total training loss does not
imply lower reconstruction error or better recommendation.

Items sharing three learned codes receive suffixes 0, 1, … in sorted item-ID
order. The fourth code is a collision disambiguator, not a fourth learned
semantic level. Freeze the forward/inverse maps before recommender training.

Some inspected shared-prefix groups look related (mascara or hair care), while
others are mixed. This is qualitative evidence, not measured category purity.
High utilization and uniqueness establish usable IDs, not recommendation benefit
or explanations of user preferences.

## Recommender and experimental protocol [E1, E2, E5]

| Setting | Actual `poc01` configuration |
|---|---|
| Seed / device | 42 / CPU, two computation threads |
| Architecture | T5, 2 encoder + 2 decoder layers |
| Model width / feed-forward / attention heads | 128 / 512 / 4 |
| Dropout | 0.1 |
| Vocabulary / max encoder length | 3,026 tokens / 82 tokens |
| Input | User bucket token + up to 20 four-token item IDs + EOS |
| Output | Four item-ID tokens + EOS |
| Objective | Mean token cross-entropy with teacher forcing |
| Optimizer | AdamW, learning rate 0.001, weight decay 0.01 |
| Batch size / gradient norm clip | 64 / 1.0 |
| Training budget | 5,000 optimizer steps |
| Checkpoint selection | Highest NDCG@10 every 500 steps on the same seeded 500 validation users; earliest wins ties |
| Selected checkpoint | Step 4,500, rather than final step 5,000 |
| Decoding | Catalog prefix trie, beam width 20, return top 10 |
| Ranking score | Sum of generated-token log-probabilities; length penalty 0 |
| Final evaluation | All 2,000 users per split; test evaluated once after freeze |

Teacher forcing supplies previous true target tokens during training; inference
supplies previous generated tokens. T5 shifts labels right internally: labels
`[c0,c1,c2,c3,EOS]` correspond to decoder inputs `[PAD,c0,c1,c2,c3]`.

The trie allows only catalog-valid continuations and EOS after four ID tokens.
This ensures catalog validity under the declared constraints, not relevance.
Beam search is approximate, not exhaustive ranking of every candidate.
Constrained decoding is a documented adaptation; no unconstrained accuracy
comparison was run.

Both methods use all **7,154 catalog items**, allow previously seen items, and
use **no sampled negatives**. All 2,000 split users count in averages; missing
predictions would be empty lists. Duplicate recommendations are removed.

## Metrics and measured results [E1, E7]

There is one relevant held-out item per user:

- **Recall@K:** 1 if the target occurs in the first K recommendations, otherwise
  0, averaged over users. Here it is also a hit rate.
- **NDCG@K:** `1 / log2(rank + 1)` for a hit within K, otherwise 0, averaged over
  users. Higher-ranked hits contribute more; ideal DCG is 1.
- **coverage@K in this repository:** fraction of users receiving at least K
  unique items. This is list completeness, not catalog diversity.

### Final held-out test: headline result

These values are proportions/scores on the **same 2,000 test users**.

| Test metric | Training-only popularity | Item cosine | Generative recommender |
|---|---:|---:|---:|
| Recall@5 | 0.002500 | 0.005000 | 0.012500 |
| Recall@10 | 0.007500 | 0.017500 | 0.023500 |
| NDCG@5 | 0.0012461414 | 0.0026199736 | 0.0075337710 |
| NDCG@10 | 0.0027974684 | 0.0065503020 | 0.0111052976 |
| Users receiving ten unique valid items | 2,000 / 2,000 | 2,000 / 2,000 | 2,000 / 2,000 |
| Mean returned items | 10.0 | 10.0 | 10.0 |

Interpretation, with arithmetic derived from measured scores:

- Recall@10 is **2.35% versus 0.75%**: **47 versus 15** users have their target
  in the top ten. The difference is **1.60 percentage points**, approximately
  **3.13×** the baseline rate, not a 3.13-percentage-point difference.
- NDCG@10 is approximately **3.97×** the popularity score. Keep it as a decimal
  score on its own chart, separate from Recall percentages.
- Absolute performance remains low: **1,953 of 2,000 users (97.65%)** have no
  top-ten hit. This run does not establish statistical significance.
- Zero invalid beams and complete lists demonstrate retrieval correctness under
  constraints, not accuracy, diversity, or success for unseen items.

For slide 8, use two small bar charts with zero-based axes. Recall@10 data:
`Popularity = 0.75%`, `Item cosine = 1.75%`, `Recommender = 2.35%`. NDCG@10 data:
`Popularity = 0.0027974684`, `Item cosine = 0.0065503020`, `Recommender = 0.0111052976`. Include the absolute
Recall difference and hit counts; do not show relative gains alone.

### Validation: distinguish selection and full-split results

| Scope | Popularity Recall@10 / NDCG@10 | Recommender Recall@10 / NDCG@10 |
|---|---|---|
| Fixed 500-user checkpoint-selection subset | Not reported here | 0.032000 / 0.0179508664 |
| All 2,000 validation users after freeze | 0.012000 / 0.0050889479 | 0.033000 / 0.0152966184 |

The selection-subset score is not the all-validation or test score. Do not infer
a cause for validation-to-test differences from this one run.

### Runtime and verification

- Content embedding: **2,557.7 seconds (42.6 minutes)**.
- Quantizer training including warm-up: **85.5 seconds**.
- Recommender optimization and periodic validation: **1,877.3 seconds
  (31.3 minutes)**.
- Full validation/test generation: **27.2 / 22.4 seconds** for 2,000 users each.

These are separate stage measurements, not an end-to-end wall-clock benchmark.
They exclude some setup/download and discarded interrupted work; test-generation
time is not a production request-latency measurement.

The completed experiment record reports **57 passing tests**, exact CPU resume
fixture checks with dropout, and an actual SIGINT stop/resume at step 288. All
exported IDs match raw beam tokens, and the selected checkpoint and maps were
unchanged after evaluation. Engineering checks are separate from evidence about
recommendation quality.

## Actual trace for retrieval and backup questions [E8]

This verifies representation, generation, and lookup. It is **not a successful
next-item prediction**. Do not replace the mismatch with a fabricated hit.

One observed item in the user's five-item history:

```text
“Eltamd UV Clear SPF 46, 48-Gram 1.7 oz.” (B002MSN3QQ)
Title + EltaMD brand + skin-protection category + price 22.67
  → embedding [1,768] → latent [1,32]
  → learned codes [236,227,111] + collision suffix 0
  → position-offset tokens [238,485,625,770]
```

The encoder supplies the user bucket token, five encoded items, and EOS.
The recommender's top-ranked output is:

```text
Generated tokens, excluding decoder start: [215,310,561,770,1]
Subtract offsets [2,258,514,770]; final token 1 is EOS
  → raw Semantic ID [213,52,47,0]
  → catalog lookup B0043OYFKU
  → “Olay Pro-X Advanced Cleansing System 0.68 Fl Oz, 1-Count”
Sum of token log-probabilities: -4.744565963745117
```

The held-out target is **“Pureology Hydrate Shampoo 33.8 oz & Condition 33.8 oz
Duo Set” (B0030MNNFO)**. The first recommendation differs from the target; this
trace alone does not state whether the target appeared elsewhere in the top ten.
Target metadata displayed by the inspector is inspection-only, not input to
recommender generation. The full-catalog metadata assumption still applies to
the earlier ID-learning stage.

## Limitations and next work [E1–E8]

| Limitation / observation | What can be concluded | Next work (not run) |
|---|---|---|
| One seed, one small subset, one product domain | Describes this run; uncertainty and generalization are unknown | Multiple seeds/subsets and paired uncertainty estimates |
| Two simple baselines measured: popularity and item cosine | Exceeded both in this run | Add a stronger sequential baseline under identical evaluation |
| No random-ID ablation | Cannot attribute gains to semantic structure | Retrain using seeded random unique IDs with matched data/model/budget |
| Recall@10 is only 2.35% | The POC works mechanically but misses most targets | Use validation to compare training budgets and diagnose errors |
| 36.4% of test targets lack training events | Many targets have no training-event support; subgroup accuracy is unreported | Measure training-seen/unseen target groups; design a separate cold-start protocol |
| Held-out-item metadata participates in ID learning | Assumes a known catalog, including held-out-item content | Test new-item insertion and temporal metadata availability |
| Quantized fine-tuning did not beat initialization MSE | Usable codes do not establish optimal representations | Compare initialized and trained codes on downstream validation metrics |
| Small CPU model, bounded training, latest-20 histories | Possible constraints, not proven causes of errors | Controlled capacity/budget/history comparisons |
| Trie decoding and warm-up are adaptations | These results do not reproduce the paper exactly | Compare decoding modes under matched budgets |
| No semantic-purity or user-explanation evaluation | Content-derived codes have no guaranteed human meaning | Measure prefix/category association and inspect mixed groups |
| No diversity, fairness, online, or production-scale evaluation | Ranking scores and complete lists do not establish deployment quality | Measure diversity, subgroup behavior, scalability, and online outcomes |

Prioritize random-ID ablation, stronger matched baselines, and repeated-seed
evaluation before claiming why the model beats popularity. Training-unseen test
items alone do **not** establish cold-start success. Keep test as a final
assessment rather than a tuning signal.

## Backup material and examiner questions

### Equations and tokenization [E5]

```text
r0 = E(x)
c_l = argmin_j ||r_l - e_(l,j)||²
r_(l+1) = r_l - e_(l,c_l)
z_quantized = sum_l e_(l,c_l), for l = 0,1,2
L_RQ = reconstruction MSE + sum_l codebook MSE_l
       + 0.25 × sum_l commitment MSE_l
L_rec = mean_t [-log P(y_t | history, y_<t)], for 4 ID tokens + EOS
```

Squared losses average over batch and feature dimensions; auxiliary losses sum
across levels. Stop-gradient separates codebook and commitment updates; earlier
code vectors are detached in later residual losses. The straight-through value
is `z + (z_quantized - z).detach()`.

Position offsets are `[2,258,514,770]`; PAD/decoder start is 0 and EOS is 1.
Four position-specific 256-token ranges plus 2,000 user buckets starting at 1,026
give 3,026 tokens. User buckets use SHA-256 modulo 2,000 and can collide; they
are not guaranteed unique user identifiers. Raw code 7 at positions 0 and 1
corresponds to different tokens (9 and 265).

### Questions to rehearse

| Examiner question | Defensible answer |
|---|---|
| What exactly is generated? | Four custom item-ID tokens and EOS, then catalog lookup; product names are displayed after lookup. |
| What is pretrained? | Only the frozen Sentence-T5 content encoder; the recommender T5 starts from random weights. |
| Why call the IDs semantic? | They reconstruct content embeddings. This describes their source, not proof of better ranking or human-readable categories. |
| Why three learned codes and a suffix? | Successive codebooks encode residual information; the deterministic suffix makes colliding tuples unique. |
| How do you avoid target leakage? | Train on earlier per-user events, count popularity from training events, select on validation, then freeze before test. Separately disclose held-out-item metadata in catalog ID learning. |
| Why is validation in test history? | It precedes test and is observed at prediction time; it is not this user's training target. |
| What does the comparison prove? | This checkpoint beat popularity on the same full-catalog protocol; one seed proves neither significance nor the specific benefit of Semantic IDs. |
| Does zero invalid output mean good recommendations? | It shows constraints and lookup work; only 47 of 2,000 targets were in the model's top ten. |
| What did the quantizer repair accomplish? | It restored usable codes after collapse, but quantized fine-tuning did not beat initialized reconstruction MSE. |
| What would you test next? | Random IDs and stronger matched baselines, multiple seeds, then training-unseen target diagnostics and a separate cold-start protocol. |

## Evidence index and reproducibility

These labels refer to repository evidence, not external web citations. All
presentation-critical numbers are included above.

| Label | Repository source | What it supports |
|---|---|---|
| E1 | `reports/poc01/experiment.json` | Completed run, selected checkpoint, results, runtime, integrity checks |
| E2 | `configs/poc01.yaml`, `reports/poc01/protocol.json`, `docs/experiment.md` | Settings, selection subset, freeze/test policy |
| E3 | `data/processed/data_stats.json` | Source/filter/subset counts and source checksums; local ignored data |
| E4 | `reports/presentation_data_profile.json` | Supplementary history lengths and training-unseen item counts; source split SHA-256 |
| E5 | `README.md`, `docs/deviations.md`, `src/data.py`, `src/dataset.py`, `src/rqvae.py`, `src/tokenization.py`, `src/recommender.py`, `src/retrieval.py`, `src/baselines.py`, `src/metrics.py` | Implementation, metric definitions, adaptations |
| E6 | `reports/poc01/id_diagnostics.json`, `reports/run_summary.json`, `artifacts/rqvae_training.json` | Frozen quantizer measurements and repair; last file is local ignored data |
| E7 | `reports/poc01/results_test.json`, `reports/poc01/results_val.json` | Final model/popularity tables |
| E8 | `reports/poc01/example_trace.json` | Observed encoding, generated tokens, lookup, inspection-only target |

`reports/run_summary.json`, root-level `reports/results_*`, fixture scores, and
the 20-step integration verification predate the completed recommendation run.
Use the older summary only for frozen item-ID-stage measurements; its
“integration pending” text is historical. Final recommendation results come
from `reports/poc01/`. Do not mix these scores or claim the recommender is pending.

Frozen mapping hash:
`688ea781e09ddfdb8ea1875a08858dc131f4c305f33b0e35753ec1693a8ab260`.
Sentence-T5 revision: `fc5d4628481afbbaaacd7af6bb07cf9d3865f781`.
Inference checkpoint: `artifacts/recommender/poc01/best/`.
Recovery state: `artifacts/recommender/poc01/latest.pt`.

For rehearsal, use saved outputs and the selected checkpoint. Do not resume or
retrain the completed experiment. From the repository root:

```bash
uv sync --locked
uv run pytest -q
uv run python -m scripts.inspect_example --config configs/poc01.yaml --split test --predictions artifacts/recommender/poc01/predictions_test.jsonl --generation-log artifacts/recommender/poc01/generation_log_test.jsonl
```

The checksummed local archive `artifacts/poc01_handoff.tar.gz` contains the
experiment's checkpoints, frozen IDs, data, logs, and source snapshot. Large data,
pretrained caches, and model artifacts are ignored by Git; the repository alone
does not transfer a runnable experiment. The archive predates this presentation
handoff and supplementary E4 report; transfer these updated tracked files too.
The earlier A-stage archive is `artifacts/developer_a_handoff.tar.gz`. See
`docs/engineering_handoff.md` for historical verification and recovery details.
