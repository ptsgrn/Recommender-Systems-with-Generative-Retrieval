# Project adaptations and run record

- Use Amazon Beauty 2014 5-core, deterministically sampled at user level. Metadata
  filtering may remove events and alter the original k-core properties; recheck
  minimum user history length, without claiming the result remains item 5-core.
- Assume metadata for the complete selected catalog, including held-out targets,
  is available. No held-out interaction label affects the quantizer or baseline.
- Use frozen Hugging Face `sentence-transformers/sentence-t5-base`; pin the resolved
  revision in each run. This exact checkpoint is a project choice.
- Use three 256×32 residual codebooks, Adam at 0.001, at most 50 epochs, and
  reconstruction-based early stopping rather than the paper's full training run.
- Default to a CPU dependency environment and batch 16 on CPU. Select among
  2,000/500/200 users using measured timing and a two-hour compute budget. Record
  actual selection, package versions and hardware in the ignored run artifacts.
- Squared losses use elementwise means and sum auxiliary terms across levels.
  Later residual losses detach earlier selected code vectors.
- Preserve repeated events except identical complete review records. Order tied
  timestamps by original source row. Sample after filtering; retain complete
  surviving histories. Allow previously seen items during full-catalog evaluation.
- This work implements Developer A only. Developer B owns rolling examples,
  production tokenization, sequence training, beam search and generation logs.
  Saved ranked predictions allow later evaluation without coupling A to B's code.
- Model results remain pending until predictions from a selected real checkpoint
  exist. The diagnostic can resolve a ranked item to its known SID, but raw token
  generation evidence must come from B. Do not label lookup output as a verified
  autoregressive generation trace.

Real measurements belong in `reports/`, and provenance in `data_stats.json`,
`embedding_manifest.json`, and `rqvae_training.json`. Do not fill missing results
with paper numbers or synthetic fixture results.
