# Project adaptations and run record

- Use Amazon Beauty 2014 5-core, deterministically sampled at user level. Metadata
  filtering may remove events and alter the original k-core properties; recheck
  minimum user history length, without claiming the result remains item 5-core.
- Assume metadata for the complete selected catalog, including held-out targets,
  is available. No held-out interaction label affects the quantizer or baseline.
- Use frozen Hugging Face `sentence-transformers/sentence-t5-base`; pin the resolved
  revision in each run. This exact checkpoint is a project choice.
- Use three 256×32 residual codebooks, Adam at 0.0001, at most 50 epochs, and
  reconstruction-based early stopping rather than the paper's full training run.
- The first real run at the TODO's starting rate 0.001 collapsed to one code per
  level. Its final encoder ReLU had zero active outputs and all latent standard
  deviations were zero. Preserve that checkpoint/log under
  `artifacts/failed_collapse/`. To retain informative latents, add ten continuous
  autoencoder warm-up epochs at 0.0001 before residual k-means initialization,
  then train the unchanged RQ architecture at 0.0001. Warm-up loss and latent
  standard deviation are logged separately. These are empirical project repairs.
- Default to a CPU dependency environment and batch 16 on CPU. Select among
  2,000/500/200 users using measured timing and a two-hour compute budget. Record
  actual selection, package versions and hardware in the ignored run artifacts.
- Squared losses use elementwise means and sum auxiliary terms across levels.
  Later residual losses detach earlier selected code vectors.
- Preserve repeated events except identical complete review records. Order tied
  timestamps by original source row. Sample after filtering; retain complete
  surviving histories. Allow previously seen items during full-catalog evaluation.
- Developer B owns rolling examples, production tokenization, sequence training,
  beam search and generation logs (section below). Saved ranked predictions allow
  evaluation through A's `scripts.evaluate` without coupling A to B's code.
- Model results remain pending until predictions from a selected real checkpoint
  exist. The diagnostic can resolve a ranked item to its known SID, but raw token
  generation evidence must come from B. Do not label lookup output as a verified
  autoregressive generation trace.

## Developer B: sequence model and retrieval

- Recommender: randomly initialised Hugging Face T5 from configuration, 2+2 layers,
  width 128, 4 heads × 32, feed-forward 512, dropout 0.1 (~1.3M parameters), far
  smaller and shorter-trained than the paper's model. AdamW 0.001, weight decay
  0.01, batch 64, gradient-norm clip 1.0: starting choices, not tuned settings.
- Training examples are rolling prefixes of each user's training items only
  (`[A]→B`, `[A,B]→C`); validation/test items are never training targets.
  Histories keep the latest 20 items.
- Loss is the mean (not sum) token cross-entropy over four ID tokens plus EOS.
- Decoding: beam 20 with a catalog prefix trie through `prefix_allowed_tokens_fn`.
  The paper filters invalid IDs (§4.5); the trie is a project adaptation for a
  short run. The unconstrained mode is kept as a diagnostic and counts invalid IDs.
- Beam score is the plain sum of token log-probabilities (`length_penalty=0`). The
  trie masks tokens after log-softmax, so scores are the model's probabilities,
  not renormalised over allowed tokens.
- Checkpoint selection uses validation NDCG@10 on a fixed seeded subset of
  validation users (`recommender.val_subset`), because decoding all users at
  every evaluation is slow. Test data is used only after the checkpoint is fixed.

Real measurements belong in `reports/`, and provenance in `data_stats.json`,
`embedding_manifest.json`, and `rqvae_training.json`. Do not fill missing results
with paper numbers or synthetic fixture results.
