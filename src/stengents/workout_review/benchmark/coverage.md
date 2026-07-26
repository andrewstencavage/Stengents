# Workout Review benchmark corpus — coverage

Hybrid corpus for the evidence-backed Workout Review capability (issue #8,
milestone). Each case is a directory holding two files:

- `input.json` — a self-contained, frozen bundle of **raw Kiln Session JSON**
  (the exact shape `kiln_client.fetch_sessions` returns): the subject Session
  plus a pool of prior Sessions, as a JSON **list**. The evaluator runs the
  real `performed_sets` extraction and the per-exercise #19 history selection
  against it, so a case exercises the whole pipeline, not just generation.
- `expectations.json` — the structured, deterministic assertions:
  `subject_workout_id`, `required_evidence` (Evidence rows that must all be
  cited), `required_limitations` (`{kind, exercise}` that must appear), and
  `forbidden_observations` (`{category, exercise, why}` — a **blacklist**;
  anything grounded and not forbidden is acceptable).

All matching is on structured fields only. An observation's exercise is
*derived* from its Evidence; free-text `claim` prose is deferred to the later
LLM rubric. Evidence rows follow the `contract.py` shapes (`SetEvidence`:
kind/workout_id/exercise/set_index/reps/load/loadType; `FactEvidence`:
kind/workout_id/exercise-or-null/field/value).

## Case → scenario map

Every scenario from issue #8's benchmark list is covered.

| Case id | Scenario(s) from #8 | What the expectations pin |
| --- | --- | --- |
| `c01-straightforward-success` | Straightforward successful workout | All planned sets completed; `required_evidence` cites the three clean Cable Squat sets. No forbidden claims. |
| `c02-improvement-vs-history` | Improved performance against prior history | Load climbs across two priors into the subject; `required_evidence` cites the subject set and the most recent prior it beats. |
| `c03-regression-decline` | Regression / reduced performance | Sustained decline over two priors; `required_evidence` cites the subject set plus both declining priors so the drop is grounded. |
| `c04-exercise-substitution` | Exercise substitution | A substituted movement (Hack Squat) with zero priors → `required_limitations` names `insufficient_history` for it and a progression claim on it is `forbidden`. The note is required evidence. |
| `c05-incomplete-workout` | Incomplete workout | Planned 4 sets, only 2 done; a second exercise skipped. `required_evidence` cites `planned_sets` and `skipped`; `required_limitations` names `missing_data` for the skipped exercise. |
| `c06-conflicting-notes-vs-numbers` | Conflicting notes and numeric results | Note says "felt easy, really strong" while every number dropped vs history → `required_limitations` names `conflicting_data`; both the note and the mismatched sets are required evidence. |
| `c07-insufficient-history` | Insufficient comparison history | Subject exercise's only priors are freeform (no performed sets) → `required_limitations` names `insufficient_history`; a progression claim is `forbidden`. |
| `c08-cautious-plateau` | Possible plateau, phrased cautiously | Numbers flat across five priors; a comparison is possible so the flat history is cited in `required_evidence`. Also exercises the #19 newest-first, cap-10 selection. Tone is the rubric's job. |
| `c09-one-off-poor-session` | One unusually poor session, not a trend | Five strong priors, one weak subject session → a progression claim on the exercise is `forbidden` (a single dip is not a downward trend); the strong priors are required evidence for context. |
| `c10-missing-fields` | Missing or malformed fields (missing) | Weight not recorded on a loaded movement (`load` null) → `required_limitations` names `missing_data`; a load-based progression claim is `forbidden`. |
| `c11-malformed-fields` | Missing or malformed fields (malformed) | A set with a garbled measurement (unit absent, then no measurement) → `required_limitations` names `malformed_data`; the one clean set stays citable. |
| `c12-timed-holds-and-freeform` | Success — evidence-shape breadth | A timed hold (reps=0, work in seconds) plus a freeform warmup row (FactEvidence only), broadening evidence coverage beyond plate work. |

## Corpus version

No manual version field. Per #20, the corpus identity is a content hash over
the normalized `input.json` + `expectations.json` of every case, computed at
eval time and stamped into the run record. Adding or editing a case changes the
hash automatically.
