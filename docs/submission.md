# Source submission and exam demonstration

Run `uv run python -m scripts.package_submission` from the repository root to
build `submission/tiger_submission.zip`. The packager includes all project
Python source and tests, configs, docs, reports, dependency files, processed
data, root ID/embedding artifacts, the selected and recovery checkpoints,
generation logs, and second-baseline predictions. It excludes raw downloads,
pretrained encoder caches, virtual environments, Git internals, old archives,
and temporary verification runs.

The ZIP contains `MANIFEST.sha256` for every bundled file. A ZIP SHA-256 sidecar
and verification report are saved beside it. The packager checks CRCs and each
file's content hash after writing. Regenerate the ZIP after any final edits to
member names, contributions, or slides.

## On the examiner machine

Extract into a directory and run commands from the extracted project root:

```bash
uv sync --locked
uv run pytest -q
uv run python -m scripts.smoke_test --data-dir tests/fixtures --sid-dir tests/fixtures
```

Dependencies require installation/network access on a fresh machine. The saved
checkpoint demo in [the walkthrough](repository_walkthrough.md) uses packaged
processed data and weights and does not require a pretrained Sentence-T5
download. Recreating embeddings from raw metadata does require the pinned
pretrained encoder and source downloads. The package does not promise bitwise
training continuation across different environments.

Use [the comparison](../reports/baselines/results_test.md) and
[the trace](../reports/poc01/example_trace.json) as offline demonstration fallbacks.
The [editable presentation](oral_exam.pptx) and [PDF copy](oral_exam.pdf) includes ten timed slides and five backup slides, with speaker notes. Content is also available in [Markdown](oral_exam_slides.md). Rebuild it with `python -m scripts.build_slides` in an environment with `python-pptx` installed; this optional dependency is not needed to train or evaluate models.

## Remaining actions requiring the group

- Confirm the assignment table; [member contributions](contributions.md) are recorded.
- Rehearse all modules individually; each person must understand the full code.
- Upload is **skipped at the user’s request**. The group will submit the ZIP to
  MyCourseVille before class on **November 5, 2026**. No submission is claimed.
- Deliver at most ten minutes of introduction, with approximately fifty minutes
  reserved for questions. Use the contribution slide to describe actual work.
