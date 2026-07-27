---
name: tune-capability
description: The development discipline for changing the Workout Review capability — when to bump CAPABILITY_VERSION and how to write its changelog, running the A/B, and landing a new baseline. Use when editing the generator/prompt/decode, tuning a floor, or recording a baseline. Pairs with the benchmark-review skill, which covers *operating* the benchmark.
---

# Tuning the Workout Review capability

This is the discipline behind the tune loop (PRs #26→#41): change the
capability, prove the change on the benchmark, and land a versioned baseline.
For the mechanics of *running* the benchmark and reading its gate, use the
**benchmark-review** skill — this one covers making the change correctly.

## The loop

1. **Change** the generator / prompt / decode in `src/stengents/workout_review/`.
2. **Bump `CAPABILITY_VERSION`** and write its changelog entry (below).
3. **Run the benchmark** (see benchmark-review) as an A/B against the prior version.
4. **Confirm** the intended floor moved and no safety/structural floor regressed.
5. **Address code review**, then **record the baseline** with `--write-baseline`
   once the run is the new intended reference.

## When to bump `CAPABILITY_VERSION`

`CAPABILITY_VERSION` lives in
[`workout_review/__init__.py`](src/stengents/workout_review/__init__.py) and is
stamped onto every artifact and baseline so a score is tied to the capability
that produced it. It is **distinct from the model and from the output contract**.

Bump it on any **meaningful capability change** — a prompt revision, a change to
the evidence-candidate set, decode behaviour, or a loop landing. A pure refactor
that leaves generation byte-identical does **not** bump it.

**A version bump makes the run a distinct baseline, not a comparable diff.** You
can't read 0.3.0-vs-0.2.0 as a regression/improvement of the same thing —
they're different capabilities. That's exactly why the bump exists.

## The changelog entry

Every bump adds a comment block above the constant, in the house format the
existing entries follow. Match it:

```
# X.Y.Z — <one-line what-changed>: <the concrete mechanism> (#ticket).
#   <baseline-comparability note — "prompt-only change, so a distinct baseline
#   from W's" / "changes the candidate set, so ...">. On a <warm|clean>
#   <model> A/B (corpus <hash8>) it lifts <metric> A -> B (which cases), with
#   the safety floors held, taking the gate <FAIL -> PASS if it flipped>.
#   <what remains unrecalled and why, with ticket refs>.
```

The parts that matter and keep getting dropped:
- **Ticket ref** for the change.
- **Comparability note**: say explicitly whether it's a distinct baseline and why.
- **A/B deltas**: name the metric, the before→after, *which cases* moved, and the
  model + corpus hash the A/B ran on. State that safety floors held.
- **What's still unrecalled**: name the cases the change did *not* fix and point
  at the follow-up ticket. (e.g. 0.3.0 left c05/c11; #41 became 0.4.0.)

## Running a clean A/B

- Run the same corpus/model before and after; a differing `corpus_hash` or model
  makes the two runs incomparable (see benchmark-review).
- **Keep clean-case prompts byte-identical** when the change should only touch a
  subset. 0.4.0 injects its DATA QUALITY NOTES section *only* for sessions with
  dropped sets, so every clean-case prompt stays identical to 0.3.0 — no
  collateral perturbation muddying the A/B. Prefer this pattern: gate new prompt
  material behind the exact condition it targets.
- Prefer a **warm** model run for the recorded A/B; a cold model can need a retry
  before its tool-call preflight settles.

## Recording the baseline

`--write-baseline` writes one **tracked, trimmed** file per
`(capability_version, model, corpus_hash)` under
[`baselines/`](src/stengents/workout_review/baselines), named
`baseline-<version>-<model-slug>-<hash8>.json`, overwritten on re-record. It's
trimmed of the model-nondeterministic `reviews` (keys: `run_id`, `generated_at`,
`capability_version`, `model`, `corpus`, `aggregate`, `gate`, `cases`) so it
reads as a stable reference.

Record a baseline **only when the run is the intended new reference** — a version
landed, or a floor ratcheted up. A one-off exploratory run does not get a
committed baseline; just read its artifact.

## Don't touch here

The **output contract** (`contract.py`, the #17 types) is locked and versioned
separately — changing it is a contract change, not a capability tune, and is out
of scope for this loop.
