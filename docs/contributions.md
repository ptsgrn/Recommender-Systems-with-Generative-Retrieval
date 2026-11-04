# Group contributions and rehearsal

Contributions confirmed by the group:

| Student ID | Member | Actual contribution | Presentation assignment |
|---|---|---|---|
| 6734446323 | นายพัสกร ยืนยง | Project initialization and Developer A implementation | Data, embeddings, quantization, Semantic IDs |
| 6734453723 | นายภาณุพงศ์ จันทร์ลาม | Developer B implementation | Tokenization, sequence model, training |
| 6634424423 | นายธิปก ปัญญาวุธ | Code review and dataset cleanup | Retrieval, evaluation, findings and limitations |

These are the reported contributions. Each member must also review the modules
outside their original role before the oral examination.

Record library reuse and LLM assistance honestly. The implementation uses
PyTorch, Hugging Face T5, Sentence Transformers, and scikit-learn. This session
added an LLM-assisted item-cosine baseline, tests, reports, and preparation
materials. Every member must still understand the complete implementation.

## Rehearsal procedure

Each member gives the complete two-minute pipeline explanation from the
[walkthrough](repository_walkthrough.md). Rotate examiner roles and ask each
person to demonstrate these without reading a prepared answer:

1. Build train, validation, and test examples for `[A,B,C,D,E]`.
2. Convert raw SID `[236,227,111,0]` to `[238,485,625,770]` and explain EOS.
3. Explain each quantizer loss and what `detach()` changes in the backward pass.
4. Show shifted decoder inputs and explain causal and padding masks.
5. Explain why twenty beams do not guarantee the exact global top ten.
6. Calculate Recall and NDCG for a target at rank three.
7. Calculate an item-cosine similarity from two tiny user histories.
8. Explain why checkpoint selection uses validation and why known catalog
   metadata differs from using held-out interaction labels.
9. Locate the relevant source function and a test that catches a failure.
10. State the two-baseline result and the limits of its interpretation.

Use the backup slides for answers, then repeat with them closed. Mark rehearsal
complete only after all three can explain every stage. This cannot be certified
by automated tests or performed on the group's behalf.
