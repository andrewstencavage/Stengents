---
name: benchmark-review
description: Run and interpret the Workout Review benchmark — the `stengents review-benchmark` corpus run, its passing-threshold gate, and the baseline tune loop. Use when running the benchmark, reading a gate result, deciding whether a run passed, or recording/diffing a baseline. NOT for the coding-fixture path (`stengents run <fixture>`).
---

# Workout Review benchmark

The benchmark scores the **Workout Review** capability against a frozen 12-case
corpus and gates the aggregate against fixed floors. It is a distinct workflow
from the coding-fixture `stengents run <fixture>` path — don't conflate them.

> Vocabulary trap: `CONTEXT.md` lists *benchmark* as an _Avoid_ term **for the
> coding fixture**. The workout-review corpus here *is* the benchmark; that
> naming rule is about not calling a coding fixture a benchmark.

## The command

```bash
stengents review-benchmark [--model <name>] [--write-baseline]
```

- Model is chosen per run via `--model`, falling back to `STENGENTS_MODEL_NAME`,
  then the capability's `DEFAULT_MODEL_NAME`. A model is required or preflight
  exits `2`.
- Only the model call inside `review_workout` is live; corpus, evaluation, and
  gating run offline against each case's frozen `input.json`.

Source of truth: [`_review_benchmark_command`](src/stengents/cli.py:65) wires
[`benchmark_runner.py`](src/stengents/workout_review/benchmark_runner.py):
`load_corpus` → `run_benchmark` → `gate_benchmark` → `build_artifact` →
`write_artifact` (→ `write_baseline` if `--write-baseline`).

## Preflight (same as the run path)

The command calls `connection.preflight()`, which needs the `gym` model reachable.

1. Load the connection (sets base URL + key only, no model pin):
   ```bash
   eval "$(scripts/stengents-env.sh)"
   ```
2. In a second terminal, keep the tunnel up:
   ```bash
   scripts/gym-tunnel
   ```
   If it reports port `11434` already in use, a tunnel is already running —
   leave it. A cold local model may need one retry to pass its tool-call
   preflight.

Preflight failure prints `preflight failed: ...` to stderr and exits `2`; **no
artifact is written**.

## Reading the result

The command prints three JSON lines: a startup line (`run_id`, `corpus_hash`,
`case_count`, `model`, `capability_version`), then the result line with
`record_path`, `aggregate`, and `gate.passed`.

A run **passes only when every floor in
[`Threshold`](src/stengents/workout_review/threshold.py:44) holds** — floors are
over the *aggregate*, not a count of passing cases (`cases_passing` is recorded
but is deliberately **not** a floor). The five floors:

| Metric | Kind | Floor | Default |
| --- | --- | --- | --- |
| `evidence_validity_rate` | safety | `>= 1.0` | 1.0 |
| `unsupported_claim_rate` | safety | `<= 0.0` | 0.0 |
| `schema_valid_rate` | structural | `>= 1.0` | 1.0 |
| `required_detail_recall` | quality | `>= 0.60` | 0.60 |
| `required_limitation_recall` | quality | `>= 0.33` | 0.33 |

**Safety floors get a confirm-before-failing retest.** When a safety floor fails,
each offending case is re-generated up to `SAFETY_RETEST_N = 3` times to tell a
real defect from a flake; if all offenders turn out to be flakes the floor
`passed` via `retest_cleared` even though `raw_passed` is false. Quality and
structural floors get **no** retest — they fail on the metric alone. So when
diagnosing a fail, check each floor's `raw_passed` vs `passed` and `kind`.

## Artifacts vs. baselines

- **Artifact** (transient, gitignored): one file per run under
  `.stengents/benchmark/`, carrying the aggregate, per-case results, the
  generated reviews, the gate report, and the stamps (`corpus_hash`, model
  record, `capability_version`) that make a score reproducible.
- **Baseline** (tracked, committed): `--write-baseline` writes one
  `baseline-<capVersion>-<model>-<corpusHash>.json` under
  [`baselines/`](src/stengents/workout_review/baselines), trimmed of the
  model-nondeterministic reviews so it reads as a stable reference.

## The tune loop

1. Run the benchmark; read `gate.passed` and the per-floor report in the artifact.
2. Diff the run's aggregate against the current baseline for the same
   `(capability_version, model, corpus_hash)` triple — a differing `corpus_hash`
   or `capability_version` means it's not comparable, not a regression.
3. Make a capability change, re-run, confirm the intended floor moved and no
   safety/structural floor regressed.
4. Record a new baseline with `--write-baseline` **only** once a run is the new
   intended reference (e.g. a floor ratcheted up). Baselines are per
   (capability, model, corpus).

**Discard interrupted runs.** A run whose connection to the endpoint is broken
mid-flight — laptop suspend dropping the SSH tunnel, a network drop, a partial
response — truncates its generations and scores a spuriously low
`required_detail_recall` (observed as low as ~0.15 vs. a stable ~0.66), which can
read gate=FAIL for a reason unrelated to the capability. Re-run on a stable, quiet
connection before trusting a low score, and never record a baseline from a run you
had to interrupt. Controlled testing found no reproducible endpoint-load-state dip
otherwise — including two genuinely concurrent runs, which both passed
([#42](https://github.com/andrewstencavage/Stengents/issues/42), closed as
inconsistent).

## Corpus

12 cases under [`benchmark/`](src/stengents/workout_review/benchmark), each a dir
with a frozen `input.json` (raw Kiln Session JSON) and `expectations.json`
(structured `required_evidence` / `required_limitations` / `forbidden_observations`).
Matching is on structured fields only; free-text claim prose is deferred to a
later LLM rubric. See
[`coverage.md`](src/stengents/workout_review/benchmark/coverage.md) for the
case→scenario map.

## Verify offline

`run_benchmark` accepts an injected fake `complete`, so the whole pipeline is
unit-testable without the tunnel:

```bash
.venv/bin/python -m pytest
```
