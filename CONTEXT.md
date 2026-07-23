# Stengents

Stengents is an engineering platform for building and improving agent systems through reproducible, measurable evidence.

## Language

**Vertical-slice MVP**:
The first end-to-end system slice: one Python ADK agent completing a deterministic coding fixture with structured tracing and verification.
_Avoid_: prototype, demo

**Coding fixture**:
An isolated, tiny Python repository containing one tightly scoped defect and a failing test that defines the required fix.
_Avoid_: benchmark, sample project

**Deterministic evaluation**:
Fixture-defined checks—such as a test command and artifact validation—that decide a run's outcome without model judgment.
_Avoid_: LLM evaluation, subjective review

**Development-time model source**:
The model endpoint used while building the system; for the MVP, the tunnelled `gym` Ollama or LM Studio service, selected through a portable provider adapter.
_Avoid_: production model, hard-coded provider

**Run record**:
A local JSON trace of one execution, including fixture and harness identities, model/provider, tool events, timing, artifacts, and deterministic verification outcome.
_Avoid_: log file, analytics database

**Coding agent**:
The sole execution agent in the MVP, responsible for attempting the coding fixture within the versioned harness.
_Avoid_: planner, worker pool, reviewer

**Reproducibility boundary**:
The constrained development harness around a coding fixture: an ephemeral workspace, fixed verification command, controlled environment, and recorded tool activity. It is not a security sandbox for hostile code.
_Avoid_: hardened sandbox, security boundary

**Named harness action**:
A narrowly defined operation the coding agent may invoke, such as reading a file, writing a file, or running the fixture's verification command. The MVP exposes these rather than a general command runner.
_Avoid_: shell tool, arbitrary command

**Fixture source surface**:
The allowlisted source files a coding agent may change in a coding fixture. Tests and project configuration are immutable evaluation inputs in the MVP.
_Avoid_: editable fixture, writable test suite

**Run budget**:
The fixed cap on a coding-agent run's named actions and elapsed time. Exhausting either budget is a deterministic failed outcome recorded in the run record.
_Avoid_: unlimited retry, open-ended run

**Run invocation**:
The CLI request `stengents run <fixture-id>` that starts exactly one coding-fixture execution using versioned fixture and budget configuration plus environment-only model connection settings.
_Avoid_: ad hoc script, multi-run batch

**Run ID**:
A generated unique identifier for one run invocation, echoed at startup and used as the JSON record filename under `.stengents/runs/`.
_Avoid_: fixture ID, log filename

**Run outcome**:
The process-level result of a run invocation: `0` for deterministic verification pass, `1` for a completed deterministic failure, and a default distinct nonzero status when the harness cannot complete a valid run.
_Avoid_: model response, tool outcome

**Run announcement**:
The credential-free machine-readable startup line that identifies a run, its fixture, resolved model/provider, fixed budgets, and intended run-record path.
_Avoid_: console log, endpoint configuration
