Build a small, explainable implementation of TIGER from Rajput et al., “Recommender Systems with Generative Retrieval” for a code review and oral exam. The goal is to preserve the central algorithm, run it end to end on real data, and explain every transformation from an item description to a recommendation.

This plan assumes **two active developers, eight working hours each per day, basic Python skills, and preferably access to one GPU**. It budgets 28 person-hours of planned work plus four person-hours of contingency. Training can run unattended, but the deadline does not depend on an overnight run succeeding. Matching the paper’s benchmark numbers is outside this two-day scope. If neither developer has used PyTorch, expect to use the smaller fallback dataset.

Developer A owns item data and Semantic IDs. Developer B owns sequence modeling and retrieval. The third teammate, if available, can independently rehearse questions and check the report; no required task depends on their availability.

# What to build

The paper calls its method **TIGER**, short for Transformer Index for GEnerative Recommenders. It has two separately trained stages (§3, Figures 2–3).

| Stage                   | Input                                         | Operation                                                               | Output                         |
| ----------------------- | --------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------ |
| Content embedding       | Item title, brand, category and price         | Frozen pretrained sentence encoder                                      | One continuous vector per item |
| Semantic ID learning    | Item vectors                                  | Train an RQ-VAE: encoder, residual quantizer and reconstruction decoder | Three discrete codes per item  |
| ID finalization         | Three-code IDs                                | Add a fourth collision suffix and freeze both lookup maps               | Unique four-token item IDs     |
| Recommendation training | Chronological user histories converted to IDs | Train an encoder–decoder Transformer to predict the next item’s ID      | A next-token prediction model  |
| Retrieval               | A user’s observed history                     | Autoregressively generate candidate IDs and look them up                | Ranked item recommendations    |

**RQ-VAE** means residual-quantized variational autoencoder. Here its job is to compress a content vector into discrete codes. A **codebook** is a trainable table of vectors; a **residual** is the part of the vector not yet represented by selected codebook vectors. A **Semantic ID** is the tuple of selected indices. The fourth code resolves collisions; it is not a fourth learned semantic quantizer.

The sentence encoder is pretrained and frozen. The RQ-VAE is trained first and then frozen. The recommendation Transformer is trained afterward on the resulting IDs. Do not optimize these three components jointly in this implementation.

# Scope and starting configuration

These are project decisions for the deadline. Keep a record of every deviation in `docs/deviations.md`.

| Component | Two-day implementation |
| - | - |
| Data | Amazon Reviews **2014 Beauty** 5-core interactions plus matching product metadata. Start with a reproducible sample of 2,000 users and retain their complete histories. |
| Item catalog | Union of items in the selected users’ histories, after metadata cleaning. Report the resulting size. |
| Evaluation split | Per user: final interaction is test, penultimate is validation, earlier interactions are training. Use at most 20 historical items per example. |
| Content encoder | Frozen `sentence-transformers/sentence-t5-base`; cache its 768-dimensional embeddings. |
| RQ-VAE | Encoder 768→512→256→128→32; mirrored decoder; three separate codebooks of 256 vectors of dimension 32; commitment coefficient 0.25. |
| RQ-VAE training | K-means initialization at each residual level; start with Adam at 0.001, batch 256. This optimizer and short training budget differ from the paper. |
| Recommendation model | PyTorch and Hugging Face `T5ForConditionalGeneration(T5Config(...))`, initialized from configuration with a custom integer vocabulary. Use 2 encoder and 2 decoder layers, model width 128, 4 heads, head dimension 32, feed-forward width 512, dropout 0.1. |
| Recommendation training | Start with AdamW at 0.001, batch 64; reduce batch size for memory. Select a checkpoint using validation data. These are starting choices, not tuned settings. |
| Retrieval | Beam width 20, return top 10 unique items. Start with a catalog prefix constraint for reliable valid output; also expose an unconstrained diagnostic mode. |
| Metrics | Recall\@5, Recall\@10, NDCG\@5, NDCG\@10, plus prediction coverage and actual runtimes. |
| Baseline | Popularity computed only from training interactions. |
| Optional experiment | Retrain the same recommender using random unique four-code IDs; same split, vocabulary capacity and compute budget. |

The paper uses three datasets, a larger recommendation model, and much longer training (§4). This project preserves RQ-VAE and autoregressive ID prediction while reducing data and training.

**Decoding deviation:** the paper generates IDs with beam search and describes increasing beam size and filtering invalid IDs (§4.5). A prefix constraint allows only continuations present in the known item catalog. It is a practical adaptation for this short run, not an implementation detail to attribute to the original experiments. Keep the chosen decoding mode fixed across comparisons and report it.

**Do not schedule:** all published baselines, cold-start experiments, diversity experiments, a web interface, distributed training, or reproducing Table 1 scores. Do not replace RQ-VAE with random IDs and present that as the completed TIGER method.

If only a CPU is available, start with 200–500 users, batch 16 and a one-layer recommendation model. Keep Sentence-T5 embeddings cached and retain the same core stages. If downloads or GPU access fail, a synthetic fixture can establish correctness, but synthetic-only output is a partial demonstration and must be labeled that way.

# First hour

Both developers read **Figure 2 and §3.2** together to understand what the model predicts. Developer A then reads **§3.1, Figure 3 and the RQ-VAE details in §4**. Developer B reads **the sequence-to-sequence details in §4, §4.5 and Appendix C**. Both read the evaluation protocol in Appendix C. Defer related work.

- [ ] Confirm whether the course allows a library Transformer and pretrained content encoder. Until a stricter course rule is known, reuse those components and implement the paper-specific pipeline yourselves.

- [ ] Choose the training machine; verify a PyTorch forward/backward pass and available GPU memory.

- [ ] Create one repository, `configs/small.yaml`, a pinned environment file and two feature branches.

- [ ] Agree on the contracts below and create a tiny fixture immediately.

- [ ] Start the Beauty review/metadata downloads and pretrained encoder download.

- [ ] Record dataset URL/version, model identifier/revision, seed, package versions and hardware.

**Checkpoint:** both developers can read the same fixture and agree on the exact meaning of every token. Do not wait for a real trained RQ-VAE before Developer B starts.

# Shared data and token contracts

Choose these formats once in the first hour. Freeze the mapping before real recommendation training.

| Artifact or function | Contract |
| - | - |
| `items.jsonl` | One record per item: `item_id`, `title`, `brand`, `category`, `price`, `text`. Preserve the raw ASIN as `item_id`. |
| `splits.jsonl` | One record per user: `user_id`, `train_items`, `val_item`, `test_item`; retain a separate chronological record for split auditing. |
| `item_ids.json` and `item_embeddings.npy` | Ordered item ID list and float32 array `[N,768]`; row i always represents item_ids[i]. |
| `item_to_sid.json` | `item_id -> [c0,c1,c2,c3]`; the first three values are learned codes, the last is the collision suffix. |
| `sid_to_item.json` | `"c0,c1,c2,c3" -> item_id`; exact inverse of the forward map. |
| `sid_manifest.json` | Per-position vocabulary sizes, offsets, special tokens, hash method, seed, catalog hash and mapping hash. |
| `encode_history(user_id, item_ids)` | User token + flattened four-token IDs for the latest 20 items + EOS. |
| `recommend(user_id, history, k)` | Ranked unique `item_id` values with generation scores; never given the target item. |

Use separate token ranges for each code position. With four positions of size 256: PAD=0, EOS=1, position 0 starts at 2, position 1 at 258, position 2 at 514, position 3 at 770, and the 2,000 user buckets start at 1026. Total vocabulary: 3026. Thus code 7 at the first position and code 7 at the second position are different tokens.

Use a stable hash such as SHA-256 modulo 2,000 for user buckets. Python’s process-dependent `hash()` is unsuitable for a saved mapping. Set T5’s decoder start token to PAD. For each target, labels are four ID tokens plus EOS; the training decoder sees the start token followed by the shifted labels. Label padding, if present, is -100.

Encoder tensors have shape `[batch, at_most_82]`; labels have shape `[batch,5]`; logits have shape `[batch,5,vocabulary_size]`. Raw code tuples and offset vocabulary tokens are different representations: name them clearly.

For collisions, sort item IDs within each identical three-code group and assign suffixes 0,1,2,…; singleton groups receive 0. If a group exceeds 256 items, stop and inspect codebook collapse. If a larger suffix vocabulary is still required, enlarge that range and regenerate the manifest **before** training the recommender. Never truncate or wrap suffixes.

For a user history `[A,B,C,D,E]`, training interactions are `[A,B,C]`, validation target is D with history `[A,B,C]`, and test target is E with history `[A,B,C,D]`. A concrete training-example choice is the rolling prefixes `[A]→B` and `[A,B]→C`. This example construction is our explicit implementation choice; never include D or E as training targets for this user.

The catalog may include metadata for held-out items under the declared assumption that product metadata is already available. This does not permit held-out interaction labels in recommendation training. It is not the paper’s separate cold-start protocol.

# Two-day schedule

Hours are elapsed working hours within each day; take breaks outside these blocks. Time estimates include implementation and verification, with short model runs inside the relevant blocks.

| Time | Developer A | Developer B |
| - | - | - |
| Day 1, 0–1 | T0: reading, setup and contracts together | T0 |
| Day 1, 1–2.5 | A1: clean data and split histories | B1: fixture, vocabulary and batch construction |
| Day 1, 2.5–3.5 | A2: cache content embeddings | B2: model and training loop |
| Day 1, 3.5–6.5 | A3: implement and train RQ-VAE | B2: finish loop and overfit tiny examples |
| Day 1, 6.5–7.5 | A4: freeze unique IDs and export maps | B3: start generation and lookup |
| Day 1, 7.5–8 | J1: integrate actual IDs together | J1 |
| Day 2, 0–1 | A5: metrics | B3: finish retrieval |
| Day 2, 1–2 | A5: popularity baseline | B4: launch real-data training |
| Day 2, 2–3 | A6: diagnostics and trace | B4: validation and checkpoint selection |
| Day 2, 3–4 | J2: final evaluation and result audit | J2 |
| Day 2, 4–6 | Contingency and repairs; optional ablation only if all required checks pass | Same |
| Day 2, 6–8 | J3: package, swap code walkthroughs and rehearse | J3 |

**End of Day 1:** actual item→Semantic ID mappings are frozen, the Transformer has learned a tiny fixture, and a batch using real IDs completes a forward/backward pass.

**By Day 2 hour 4:** one real-data model and the popularity baseline have measured results. Freeze features and use the remaining time for reliability and explanation.

# Developer A task cards

## A1 Prepare item metadata and chronological splits

**Estimate:** 1.5 hours. **Depends on:** T0. **Files:** `src/data.py`, `scripts/prepare_data.py`.

Read Beauty interactions and matching metadata; join on ASIN. Retain only items with usable content, then recheck that retained users have at least five interactions. Sample users deterministically and keep complete histories. Sort by timestamp with a documented tie-breaker. Split each user before creating training examples. Record any deduplication policy.

**Done when:** `items.jsonl`, `splits.jsonl` and `data_stats.json` exist; all referenced items have metadata; three printed user histories show the correct chronological split. Counts include excluded records, users, items and examples.

**Exam question:** “Show exactly where you prevent the future target interaction from entering the model’s history.”

## A2 Generate and cache content embeddings

**Estimate:** 1 hour plus bounded model/download time. **Depends on:** A1. **File:** `scripts/embed_items.py`.

Create a deterministic text template using title, brand, category and price, with explicit missing-field handling. Exclude user review text and interaction-derived signals. Batch the frozen sentence encoder and save vectors with the matching item order. Encode only the selected catalog; do not embed the entire metadata archive.

**Done when:** the cache has shape `[N,768]`, contains finite values and reloads with identical item ordering. Save five item texts and vector shapes as an inspection example.

**Exam question:** “Where does the initial vector come from, and which parameters are frozen?”

## A3 Implement residual quantization and its training objective

**Estimate:** 3 hours. **Depends on:** A2 for real training; code can start from random vectors. **Files:** `src/rqvae.py`, `scripts/train_rqvae.py`.

Build the encoder, three codebooks and decoder. At each level, select the nearest codebook vector by Euclidean distance and subtract it to form the next residual. Initialize the first codebook from latent vectors and later codebooks from the corresponding residuals using k-means on a sufficiently large batch.

Implement the reconstruction, codebook and commitment losses from §3.1. Use stop-gradient/detach deliberately; route reconstruction gradients to the encoder with a straight-through estimator. Freeze the sentence encoder. Log each loss, code usage per level and collision rate. Save the best reconstruction checkpoint and the configuration.

**Done when:** a hand-built nearest-vector example selects the expected code; encoder, decoder and codebooks receive finite gradients; a tiny embedding batch can be reconstructed better after training; code use is measured. If almost every item collapses to one code, investigate initialization and gradients before continuing. Do not require the paper’s reported utilization after a short run.

**Exam question:** “Why quantize the residual, and how can training work when argmin is not differentiable?”

## A4 Export stable unique Semantic IDs

**Estimate:** 1 hour. **Depends on:** A3. **File:** `scripts/build_semantic_ids.py`.

Generate three codes for every catalog item. Append deterministic collision suffixes, export both lookup maps and the manifest, then freeze them. Save the quantizer checkpoint alongside the maps.

**Done when:** every item has exactly one four-code ID; all full IDs are unique; every item round-trips through both maps; a forced two-item collision is resolved. Send Developer B the artifact paths and mapping hash.

**Exam question:** “If two items get the same three codes, how do you retrieve one specific item?”

## A5 Implement evaluation and a popularity baseline

**Estimate:** 2 hours. **Depends on:** A1; use fixture predictions until B3 is ready. **Files:** `src/metrics.py`, `src/baselines.py`, `scripts/evaluate.py`.

Implement Recall and NDCG at 5 and 10. Count item popularity once from original training interactions, not duplicated rolling-prefix examples. Rank the same candidate catalog for both methods. For this project, allow repeat recommendations and document that rule; do not silently remove ground-truth targets from the candidate pool.

**Done when:** a target ranked first has Recall=1 and NDCG=1; ranked third it has Recall\@5=1 and NDCG\@5=0.5; absent it scores zero. Average over every evaluated user, including users with empty or short model outputs. There is no sampled-negative test set.

**Exam question:** “Why can Recall stay unchanged when NDCG improves?”

## A6 Produce ID diagnostics and one full trace

**Estimate:** 1 hour. **Depends on:** A4; coordinate with B4 for the recommendation output. **Files:** `scripts/inspect_example.py`, `reports/id_diagnostics.json`.

Report active codes per level, largest collision group and suffix usage. Print a few groups sharing the first code and inspect whether their item content looks related. Avoid claiming that every code is a human category.

**Done when:** one trace shows metadata→768-dimensional vector→32-dimensional latent→three learned codes→collision suffix→offset tokens→history→generated ID→item title. Shape summaries are sufficient; printing hundreds of vector coordinates is unnecessary.

**Exam question:** “What makes these IDs semantic, and what evidence do we actually have in this run?”

# Developer B task cards

## B1 Build the fixture and sequence batches

**Estimate:** 1.5 hours. **Depends on:** T0; no dependency on real Semantic IDs. **Files:** `tests/fixtures/`, `src/tokenization.py`, `src/dataset.py`.

Create a small catalog with manually assigned valid four-code IDs and simple histories. Build the position-specific vocabulary, stable user buckets, rolling training examples and padded batches. These fixture IDs are only for debugging.

**Done when:** a batch prints readable histories, integer input tokens and target labels; special/user/item ranges do not overlap; decoding tokens reconstructs the original code tuple; held-out interactions never become training targets.

**Exam question:** “Why is code 7 at position zero different from code 7 at position one?”

## B2 Implement the recommender and overfit a tiny batch

**Estimate:** 4 hours. **Depends on:** B1. **Files:** `src/recommender.py`, `scripts/train_recommender.py`.

Instantiate a small T5 encoder–decoder from configuration with the custom vocabulary. Use integer tokens directly rather than passing printed numbers through a natural-language tokenizer. Train with teacher forcing and token cross-entropy; add optimizer steps, validation, deterministic seeds, checkpoint saving and loading.

Overfit 16–32 unambiguous fixture examples with dropout disabled for this diagnostic. Inspect the shifted decoder inputs to ensure the model cannot see the token it is being asked to predict. Save a loss trace.

**Done when:** loss clearly falls, nearly all fixture targets can be generated exactly, and reloading the saved model reproduces predictions in evaluation mode. If this fails, debug here rather than starting a large training run.

**Exam question:** “What does the decoder receive during training, and how does that change during inference?”

## B3 Implement ID generation and item lookup

**Estimate:** 2 hours, split across both days. **Depends on:** B2 for learned predictions, A4 for the real catalog. **File:** `src/retrieval.py`.

Generate four ID tokens followed by EOS, then invert the token offsets and look up the item. Use beam search and rank by the sum of token log-probabilities; keep sequence length handling explicit. Remove duplicate items.

Build a prefix dictionary/trie from catalog IDs and use `prefix_allowed_tokens_fn` to allow only catalog-valid continuations. The trie must not depend on the current target. Handle the initial decoder-start token separately and permit EOS only after four ID tokens. Preserve an unconstrained mode to inspect invalid outputs and demonstrate the paper’s filtering approach.

**Done when:** fixture and real-catalog output contain unique known items; invalid or malformed IDs never crash lookup; top-K shortfalls are reported. On a tiny catalog, enumerate all candidate ID probabilities to sanity-check beam scores, recognizing that a narrow beam need not find the exact global top K.

**Exam question:** “How do you retrieve items without nearest-neighbor search, and what did we change to handle invalid IDs?”

## B4 Train the real-data model and select a checkpoint

**Estimate:** 2 hours of active work with training running within the available window. **Depends on:** J1 and B2; validation generation needs B3. **Output:** checkpoint, run configuration and training log.

Replace the fixture mapping with A4’s frozen maps. Verify their hash before loading a checkpoint. Measure the time for 100 steps and estimate the affordable budget; do not start a run that consumes the final rehearsal window. Validate periodically and select by validation NDCG\@10. Use a fixed validation subset during development if decoding is slow, and disclose it.

**Done when:** a real-data checkpoint exists; the training log records actual steps, runtime and selection criterion; the selected checkpoint is fixed before final test evaluation. A weak score is a result to analyze, not a reason to tune on the test set.

**Exam question:** “Which checkpoint did you choose, and why is that choice independent of the test targets?”

# Joint task cards and completion criteria

## J1 Integrate the real mapping

**Estimate:** 30 minutes each, end of Day 1. **Depends on:** A4 and B2.

Use the same manifest and item order. Run one real-data batch through model forward, loss and backward; decode its labels back to the original target item. Save a smoke-test command and the mapping hash.

**Done when:** there is no manual file editing or remapping between the two workstreams.

## J2 Run and audit the experiment

**Estimate:** 1 hour each, Day 2. **Depends on:** A5, B3 and B4.

Evaluate the frozen selected checkpoint and popularity baseline under the same split, catalog and repeat-item policy. Report Recall\@5/10 and NDCG\@5/10, evaluated user count, catalog size, seed, training steps and decoding mode. Record mean number of returned items and the fraction of users receiving K results.

**Done when:** `reports/results.json` and a human-readable table contain actual measured results, with no paper numbers mixed into project results. Test data is used for final assessment, not model selection.

## J3 Package and rehearse

**Estimate:** 2 hours each, final block. **Depends on:** all required tasks.

The README must explain setup, data provenance, preprocessing, artifact order, training, evaluation and the demo. Include a paper-section→source-file table, deviations, limitations and commands that another teammate can execute. Save the real checkpoints, maps and logs before the oral exam. Use a cached checkpoint for the live demo; do not rely on retraining live.

Swap modules: A explains the Transformer and retrieval code; B explains RQ-VAE and splitting. Each person traces one example without reading a prepared script.

**Definition of done**

- [ ] The pipeline runs on a documented real dataset subset.

- [ ] RQ-VAE has learned parameters and unique stable Semantic IDs.

- [ ] The recommendation model consumes histories and generates next-item IDs.

- [ ] A tiny-example overfit check passes.

- [ ] Recall/NDCG and a training-only popularity baseline are measured correctly.

- [ ] Every prediction resolves to an item, or its failure is explicitly counted.

- [ ] The selected checkpoint and mappings reload together.

- [ ] The README distinguishes paper choices from project adaptations.

- [ ] Both active developers can explain both major stages.

A functioning toy pipeline without real data is partial completion. Matching Table 1 or beating popularity is not a required claim for this short implementation.

# Equations and exam answers to understand

For the quantizer, z is the learned latent vector, r is the residual, and e is a codebook vector. Each level has its own codebook.

```math
r_0=E(x),\qquad
c_\ell=\arg\min_j\|r_\ell-e_{\ell,j}\|_2^2,\qquad
r_{\ell+1}=r_\ell-e_{\ell,c_\ell},\qquad
\hat z=\sum_{\ell=0}^{2}e_{\ell,c_\ell}.
```

The loss combines reconstruction with codebook and commitment terms; sg means stop-gradient. Match the reduction conventions in code to the logged loss scale.

```math
\mathcal L_{RQ}=\|x-D(\hat z)\|_2^2+
\sum_{\ell=0}^{2}\left(
\|\mathrm{sg}(r_\ell)-e_{\ell,c_\ell}\|_2^2+
\beta\|r_\ell-\mathrm{sg}(e_{\ell,c_\ell})\|_2^2
\right).
```

One common implementation feeds `z + (z_quantized - z).detach()` to the reconstruction decoder: the forward value is quantized while the reconstruction gradient reaches the encoder. The explicit codebook and commitment terms train their intended parameters. Do not detach the entire encoder path accidentally. The paper’s stated objective does not require adding a Gaussian latent sampler or a KL term merely because the name includes “VAE.”

The recommender models a distribution over the next ID token conditioned on the history and preceding target tokens. Here y includes the four item codes and EOS.

```math
\mathcal L_{rec}=-\sum_{t=1}^{5}\log P_\theta(y_t\mid H,y_{<t}).
```

With one held-out relevant item per user, Recall\@K is 1 if that item occurs in the first K recommendations, otherwise 0. NDCG\@K is 1/log2(rank+1) for a hit and 0 for a miss; average over users.

| Likely question | What the answer must establish |
| - | - |
| Is this an LLM generating product names? | The recommendation output is a sequence of custom discrete ID tokens, which are looked up in a catalog. |
| Which model is pretrained? | The content sentence encoder; the recommendation Transformer is trained on the custom vocabulary. |
| Why several residual codebooks? | Later levels encode information left after earlier selections; this supports a coarse-to-fine representation. |
| Why freeze the item mapping? | Changing IDs would change the meaning of the recommender’s training labels and checkpoints. |
| Is each code a category label? | No. Codes are learned from content reconstruction; category-like organization is an empirical pattern, not a supervised label guarantee. |
| What is teacher forcing? | Training supplies previous true target tokens; inference supplies previous generated tokens. |
| Where are the negative samples? | This implementation uses token cross-entropy over the vocabulary; it does not require a sampled-negative recommendation objective. |
| Why beam search? | It retains multiple promising partial IDs; greedy decoding yields only one path. |
| Why might the model generate a nonexistent item? | Valid vocabulary tokens can form a tuple absent from the catalog. |
| Are we reproducing cold start? | No. The paper has a separate unseen-item protocol and candidate expansion in §4.3. |
| What if our score is worse than popularity? | Report it; investigate training budget, split correctness, ID quality and sequence learning. A short run does not establish the paper’s accuracy claims. |
| What did we reuse? | State precisely which neural-network/library operations are reused and which paper-specific stages we implemented. |

Suggested 10-minute demo: 1 minute for the problem and pipeline; 2 for data and splitting; 2 for RQ-VAE and collision handling; 2 for one training example and teacher forcing; 2 for generation and metrics; 1 for differences from the paper.

# Decision rules when the schedule slips

| Trigger | Required action |
| - | - |
| Data or metadata unavailable after the first hour | Continue both workstreams using the fixture. Time-box obtaining a matching real interaction/metadata source; record any dataset change. |
| No usable GPU | Reduce users, batch size and model depth immediately; benchmark 100 steps before committing to a run. |
| Tiny fixture cannot be overfit by Day 1 hour 6.5 | Prioritize token offsets, masks, shifted labels, optimizer updates and checkpoint loading. Do not add features. |
| Quantizer collapses | Inspect initialization, gradients and active codes. Reduce codebook size only with a recorded configuration change and regenerated downstream maps. |
| Real IDs are not ready at the Day 1 integration checkpoint | Use the contingency budget to repair A3/A4 while B completes retrieval on the fixture. Real RQ-VAE IDs remain required. |
| Unconstrained decoding returns mostly invalid IDs | Use the declared prefix-constrained mode and report it. Do not substitute popularity results while labeling them TIGER. |
| Training is slower than projected | Reduce the user subset or step budget, preserve the checkpoint and report actual runtime; do not sacrifice the final rehearsal. |
| All required checks pass early | Use spare time for the random-ID ablation, not more infrastructure. |

For a random-ID ablation, assign each item a seeded random unique four-code tuple, rebuild its inverse map and catalog trie, then retrain the recommender from scratch. Keep all other settings and budgets fixed. If the run cannot finish, mark it “not run”; do not invent a result.

# Sources and implementation references

- [Rajput et al. 2023 uploaded paper](https://chatgpt.com/library?file_id=libfile_53afa67d12748191a95d199cf9903f36\&account_id=58bedd70-8cc2-4fbf-b911-2a52d7eed103): §3.1 and Figure 3 for RQ-VAE; §3.2 and Figure 2 for generative retrieval; §4 for experimental model settings; §4.5 for invalid IDs; Appendix C for splits and history length; Appendix E for frozen lookup maps.

- [Amazon Reviews 2014 dataset page](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html): select Beauty **5-core** reviews in the small-subsets table and Beauty **metadata** in the per-category table. Do not accidentally select the 2018 or 2023 release. The downloads themselves must be checked in T0.

- [Sentence-T5 base model card](https://huggingface.co/sentence-transformers/sentence-t5-base): practical pretrained 768-dimensional content encoder. Record the exact checkpoint/revision chosen; the paper does not identify this specific Hugging Face revision.

- [Hugging Face T5 documentation](https://huggingface.co/docs/transformers/en/model_doc/t5): configurable encoder–decoder implementation and model inputs.

- [Hugging Face generation documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation): beam generation and prefix-allowed-token callback.

The task names, estimates, smaller configuration, optimizer choices and constrained decoding policy are project recommendations. The actual implementation, downloads and training remain the team’s work; this plan does not represent completed experiments.
