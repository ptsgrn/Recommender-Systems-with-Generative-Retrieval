# Source submission and exam demonstration

The group is **5**. Submit exactly one file named **`5_term project.zip`**.
Only one representative submits for the group, before class on **November 5,
2026**. Upload remains skipped at the user's request.

Build the course package from the repository root:

```bash
uv run python -m scripts.package_submission --group-number 5
```

The result is `submission/5_term project.zip`, containing:

| Course deliverable | Included files |
|---|---|
| Source code | `src/`, `scripts/`, tests, configs, README, dependency manifests, and lockfile |
| Dataset source and sample | `samples/README.md` with direct download links, 100 real events in `samples/interactions.jsonl`, and matching metadata in `samples/items.jsonl` |
| Presentation | `docs/oral_exam.pptx`, `docs/oral_exam.pdf`, and speaker notes |
| Optional supplementary material | Experiment reports, baseline protocol, walkthrough, and contribution records |

The ZIP contains `MANIFEST.sha256` for each bundled file. A ZIP SHA-256 sidecar
and verification report are saved beside it. The packager checks CRCs and every
file's content hash. The sidecar is for local verification; the course requires
only the one ZIP. Regenerate it after final edits.

The course ZIP omits full processed data, trained weights, caches, raw downloads,
and virtual environments because the stated checklist allows public dataset
links and a small sample. For the live saved-checkpoint demo, retain the earlier
`submission/tiger_submission.zip` or create a separate local bundle:

```bash
uv run python -m scripts.package_submission --group-number 5 --include-artifacts
```

That optional command creates `submission/5_term project_with_artifacts.zip`.
It is a local demo backup; the course submission filename remains
`5_term project.zip`.

## On the examiner machine

Extract into a directory and run commands from the extracted project root:

```bash
uv sync --locked
uv run pytest -q
uv run python -m scripts.smoke_test --data-dir tests/fixtures --sid-dir tests/fixtures
```

Dependencies require installation/network access on a fresh machine. The saved
checkpoint demo in [the walkthrough](repository_walkthrough.md) requires the
separate local demo bundle containing processed data and weights. The smaller
course ZIP supports fixture checks and inspection of saved reports; it does not
contain the trained checkpoint. Recreating embeddings from raw metadata does require the pinned
pretrained encoder and source downloads. The package does not promise bitwise
training continuation across different environments.

Use [the comparison](../reports/baselines/results_test.md) and
[the trace](../reports/poc01/example_trace.json) as offline demonstration fallbacks.
The [editable presentation](oral_exam.pptx) and [PDF copy](oral_exam.pdf) includes ten timed slides and five backup slides, with speaker notes. Content is also available in [Markdown](oral_exam_slides.md). Rebuild it with `python -m scripts.build_slides` in an environment with `python-pptx` installed; this optional dependency is not needed to train or evaluate models.

## Remaining actions requiring the group

- Group 5 and [member contributions](contributions.md) are confirmed by the group.
- Rehearse all modules individually; each person must understand the full code.
- Upload is **skipped at the user’s request**. One representative will submit `5_term project.zip` to
  MyCourseVille before class on **November 5, 2026**. No submission is claimed.
- Deliver at most ten minutes of introduction, with approximately fifty minutes
  reserved for questions. Use the contribution slide to describe actual work.
