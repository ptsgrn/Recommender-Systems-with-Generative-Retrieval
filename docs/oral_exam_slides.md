# Oral exam slides and speaker notes

Ten-minute introduction; backup slides are for the approximately fifty-minute Q&A.
English slide content with confirmed Thai presenter names. Timings total 600 seconds.

## 1 Generative retrieval for next-item recommendation

Presenter: พัสกร ยืนยง. Time: 30 seconds.

- Encode product content as Semantic IDs
- Learn to generate the next item ID from a user history
- One real-data CPU experiment; two simple baselines

Speaker notes: Open with the task: given past events, predict the next product. The content encoder is pretrained; the recommendation Transformer starts from random weights. This is a teaching implementation with disclosed computational adaptations.

Code and evidence: README.md; configs/poc01.yaml

## 2 Data and chronological holdout

Presenter: พัสกร ยืนยง. Time: 60 seconds.

- Amazon Beauty 2014: 2,000 users and 7,154 items
- 17,264 events → 13,264 train events → 11,264 examples
- [A,B,C] train; D validation; E test
- Test history includes D; training targets exclude D and E

Speaker notes: Explain metadata cleanup and the minimum five retained events per user. Each first training event has no nonempty prefix, giving 13,264 minus 2,000 training examples. All catalog metadata is available for ID learning, including held-out products. This is per-user chronology rather than a global temporal split.

Code and evidence: src/data.py: make_subset; src/dataset.py: build_train_examples

## 3 From product content to unique Semantic IDs

Presenter: พัสกร ยืนยง. Time: 90 seconds.

- Title, brand, category, price → Sentence-T5 vector [768]
- Encoder 768 → 512 → 256 → 128 → 32
- Three residual codebooks: 256 vectors each
- Append collision suffix; freeze 7,154 unique IDs

Speaker notes: At each level choose the nearest code vector, subtract it, and quantize the residual. Decode the sum to reconstruct the embedding. Reconstruction, codebook, and commitment losses train different paths. Three codes are learned; the fourth disambiguates collisions. The initial quantizer collapsed; warm-up and a lower learning rate restored usable codes, but quantized fine-tuning did not improve initialized reconstruction. Handoff: these fixed maps define the next presenter’s tokens.

Code and evidence: src/rqvae.py; src/semantic_ids.py; reports/poc01/id_diagnostics.json

## 4 Constructing model inputs and labels

Presenter: ภาณุพงศ์ จันทร์ลาม. Time: 60 seconds.

- Raw ID [236,227,111,0] → tokens [238,485,625,770]
- Input: user bucket + latest 20 item IDs + EOS
- Input length ≤82; target length =5; vocabulary =3,026
- PAD=0; EOS=1; ignored label padding =−100

Speaker notes: Offsets [2,258,514,770] distinguish code positions. User tokens use SHA-256 modulo 2,000 and can collide. The collator creates an attention mask. Walk through one example; distinguish integer ID tokens from the earlier continuous content embeddings.

Code and evidence: src/tokenization.py: HistoryEncoder; src/dataset.py: collate

## 5 Training and selecting the recommender

Presenter: ภาณุพงศ์ จันทร์ลาม. Time: 60 seconds.

- Small T5 trained from scratch with teacher forcing
- Labels [c0,c1,c2,c3,EOS]; input [PAD,c0,c1,c2,c3]
- AdamW and mean token cross-entropy; 5,000 steps
- Validation NDCG@10 selects step 4,500

Speaker notes: The encoder attends to observed history. Decoder causal attention prevents access to later target tokens; cross-attention uses the encoded history. At inference previous generated tokens replace true prefixes. Select every 500 steps on a fixed 500-user validation subset. Freeze before final test. best/ is inference; latest.pt additionally stores optimizer, RNG, and batching state for recovery.

Code and evidence: src/recommender.py; src/training.py; reports/poc01/experiment.json

## 6 Disclosed changes from the paper

Presenter: ภาณุพงศ์ จันทร์ลาม. Time: 60 seconds.

- Encoder/decoder: 4+4 → 2+2 layers; heads: 6×64 → 4×32
- Feed-forward: 1,024 → 512; batch: 256 → 64
- Beauty training: 200,000 → 5,000 steps; smaller subset
- Quantizer: Adam + warm-up; retrieval: catalog prefix constraints

Speaker notes: Paper RQ training uses Adagrad 0.4, batch 1,024, 20,000 epochs. Ours uses Adam 0.0001, CPU batch 16, ten warm-up plus seven completed quantized epochs. Recommender learning rate changes from 0.01 then inverse-square-root decay to constant 0.001. Quantizer selection uses reconstruction and recommender selection uses validation ranking. Explain practical compute and stability reasons, without claiming identical paper reproduction.

Code and evidence: README.md: settings table; docs/deviations.md; arxiv.org/html/2305.05065#S4

## 7 Generating and ranking catalog items

Presenter: ธิปก ปัญญาวุธ. Time: 60 seconds.

- Beam width 20; generate four ID tokens and EOS
- Trie permits catalog-valid continuations
- Rank by summed token log-probabilities; return ten unique items
- Generated tokens → raw SID → product lookup

Speaker notes: The target is never an argument to recommend. Strip the decoder start token before lookup. Constraints ensure validity, not relevance. Beam search is approximate. A narrow beam can discard a prefix leading to a stronger complete item. The saved trace generates an Olay cleansing system while the held-out target is Pureology shampoo; describe that mismatch honestly.

Code and evidence: src/retrieval.py: CatalogTrie, Retriever; reports/poc01/example_trace.json

## 8 Two baselines and held-out results

Presenter: ธิปก ปัญญาวุธ. Time: 60 seconds.

- Popularity: Recall@10 0.75%; NDCG@10 0.00279747
- Item cosine: Recall@10 1.75%; NDCG@10 0.00655030
- TIGER: Recall@10 2.35%; NDCG@10 0.01110530
- Top-ten hits: 15 / 35 / 47 out of 2,000 users

Speaker notes: Popularity counts original training events. Item cosine uses binary training-user co-occurrence, then sums similarities to observed history; ties use popularity and item ID. Both use the full catalog and allow repeated items. Recall is one held-out target hit; NDCG discounts its rank. Item cosine has no tuned hyperparameters. TIGER metrics are copied from the frozen original report, not rerun.

Code and evidence: src/baselines.py; src/metrics.py; reports/baselines/results_test.json

## 9 What the results support and leave open

Presenter: ธิปก ปัญญาวุธ. Time: 60 seconds.

- Pipeline verified; TIGER exceeds two simple baselines in this run
- Absolute accuracy remains low: 1,953 top-ten misses
- One seed; no random-ID ablation or significance test
- 36.4% of test targets lack training events; subgroup accuracy unknown

Speaker notes: No causal claim that semantic structure produces the gain. All selected-catalog metadata is available, so do not claim an unseen-inventory cold-start evaluation. Full result-list coverage is not catalog diversity. Future work: matched random-ID ablation, stronger sequential baselines, repeated seeds, and training-seen/unseen subgroup analysis. Tests establish mechanics; ranking metrics establish this run’s quality.

Code and evidence: reports/presentation_data_profile.json; reports/baselines/results_test.json

## 10 Actual contributions and shared preparation

Presenter: All three. Time: 60 seconds.

- 6734446323 พัสกร ยืนยง: initialization and Developer A code
- 6734453723 ภาณุพงศ์ จันทร์ลาม: Developer B code
- 6634424423 ธิปก ปัญญาวุธ: code review and dataset cleanup
- Every member prepares to explain all modules

Speaker notes: State the reported contributions without implying exclusive understanding. Each person takes about twenty seconds. Disclose library reuse and LLM assistance when asked; the supplemental baseline and preparation materials were LLM-assisted. Invite code questions after the ten-minute introduction. Individual rehearsal remains the group’s responsibility.

Code and evidence: docs/contributions.md

## 11 Backup: quantization and gradient routing

Backup slide. Lead: พัสกร ยืนยง.

- c = argmin ||residual − codebook[j]||²
- Residual subtracts the selected vector at each level
- Forward: z + (quantized − z).detach() equals quantized
- Backward: reconstruction reaches encoder; auxiliary losses train codebooks

Speaker notes: The argmin is discrete. Straight-through supplies a surrogate identity gradient to the encoder. Codebook loss detaches residuals; commitment loss detaches code vectors. Later residual losses detach earlier selected vectors. Reconstruction alone does not train codebooks through this path. Use test_straight_through_and_auxiliary_gradient_routing as evidence.

Code and evidence: src/rqvae.py: quantize, forward; tests/test_quantizer.py

## 12 Backup: attention and tensor shapes

Backup slide. Lead: ภาณุพงศ์ จันทร์ลาม.

- Input IDs and attention mask: [batch,L], L≤82
- Labels: [batch,5]; logits: [batch,5,3026]
- Encoder self-attention uses observed history
- Decoder: causal self-attention plus encoder cross-attention

Speaker notes: Explain Q, K and V as learned projections, with attention weights combining values based on query/key compatibility. The history is already observed, so encoder self-attention can use all its nonpadding positions. Decoder masking limits access to its prefix. Padding masks and causal masks have different jobs. T5 is supplied by Hugging Face; project code configures it and constructs the inputs.

Code and evidence: src/recommender.py: build_model; src/dataset.py: collate

## 13 Backup: baseline worked example

Backup slide. Lead: ธิปก ปัญญาวุธ.

- Training users: U1={a,b}; U2={a,c}
- n(a)=2; n(b)=1; shared(a,b)=1
- sim(a,b)=1/√2; sim(b,c)=0
- Scores sum similarities; zero-score ties use training popularity

Speaker notes: Repeated events do not multiply binary co-occurrence counts. Self-similarity is zero. Test-time validation history is allowed as observed input, but the matrix is fitted only from train_items. Explain unseen items and empty-history fallback. All candidates remain eligible; no sampled negatives.

Code and evidence: src/baselines.py: ItemCosine; tests/test_item_cosine.py

## 14 Backup: ranking scores and evaluation

Backup slide. Lead: ธิปก ปัญญาวุธ.

- Beam score = Σ log P(token | history, prefix)
- Constraint masking does not prove globally optimal top K
- Target at rank 3: Recall@5=1; NDCG@5=0.5
- Empty predictions remain in averages; duplicates removed

Speaker notes: Full vocabulary probabilities underpin beam scores, rather than renormalizing over allowed trie tokens. The fixture test checks beam scores against exact item scoring with sufficiently wide beams. Distinguish full-catalog candidate eligibility from exhaustively scoring every item. coverage@10 means ten unique returned items per user.

Code and evidence: tests/test_model_retrieval.py; src/metrics.py

## 15 Backup: frozen checkpoints and recovery

Backup slide. Lead: ภาณุพงศ์ จันทร์ลาม.

- best/ stores validation-selected inference weights
- latest.pt stores model, optimizer, RNG, epoch, and batch cursor
- Mapping and input hashes guard against incompatible restores
- 59 tests pass; original experiment recorded 57

Speaker notes: Dropout and shuffling require restored RNG state for exact continuation. Optimizer moments affect the next update. Restoring weights alone is insufficient. Exact resume is tested in the CPU environment, not promised across arbitrary systems. Use saved outputs if the conference demo cannot load dependencies.

Code and evidence: src/training_state.py; tests/test_training_resume.py
