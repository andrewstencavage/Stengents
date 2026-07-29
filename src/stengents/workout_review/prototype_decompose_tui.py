"""PROTOTYPE — throwaway TUI for Wayfinder ticket #58 (map #56). Not shipped.

Drives prototype_decompose_logic by hand against a REAL Kiln session and the
REAL local model, so the question ("does per-exercise extraction + one
synthesis call actually work, and what does it cost in latency?") gets
answered against real data and real timings, not a fake.

Run:
    source .venv/bin/activate
    export STENGENTS_MODEL_BASE_URL=http://127.0.0.1:11434 STENGENTS_MODEL_API_KEY=local
    export KILN_BASE_URL=http://192.168.40.161:4173
    python src/stengents/workout_review/prototype_decompose_tui.py [workout_id]

Defaults to the real session ticket #57 was tested against
(ac28771d-fe25-44a0-a4ed-6ab6f9639dd7) if no workout_id is given.
"""

from __future__ import annotations

import sys
import time

from farm_system.kiln_coach import kiln_client
from stengents.utilities.model_source import resolve_model

from .review import _candidate_evidence, review_grounding, select_comparison_history
from .prototype_decompose_logic import (
    Partition,
    extract,
    partition_session,
    synthesize,
)

DEFAULT_WORKOUT_ID = "ac28771d-fe25-44a0-a4ed-6ab6f9639dd7"


def _complete_with_model(model, prompt: str) -> str:
    from litellm import completion

    response = completion(
        model=f"openai/{model.name}",
        api_base=f"{model.base_url.rstrip('/')}/v1",
        api_key=model.api_key,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return response["choices"][0]["message"]["content"]


class State:
    def __init__(self, workout_id: str) -> None:
        self.workout_id = workout_id
        self.session: dict = {}
        self.candidates = []
        self.partitions: list[Partition] = []
        self.findings_by_partition: dict[str, list] = {}
        self.timings: dict[str, float] = {}
        self.review = None
        self.model = resolve_model("qwen2.5:7b-8k")

    def load(self) -> None:
        self.session = kiln_client.fetch_workout(self.workout_id)
        pool = kiln_client.finished_sessions(kiln_client.fetch_sessions())
        self.selected = select_comparison_history(self.session, pool)
        self.candidates = _candidate_evidence(self.session, self.selected)
        self.grounding = review_grounding(self.session, self.selected)
        self.partitions = partition_session(self.session, self.selected, self.candidates)


def render(state: State) -> None:
    print("\033[2J\033[H", end="")
    print(f"\x1b[1mPROTOTYPE — decompose review_workout\x1b[0m  (#58, map #56)")
    print(f"\x1b[2mworkout_id={state.workout_id}  model={state.model.name}\x1b[0m")
    print()
    print(f"\x1b[1mCandidate pool:\x1b[0m {len(state.candidates)} slots (subject + grouped history)")
    print()
    print("\x1b[1mPartitions:\x1b[0m")
    total_time = 0.0
    for partition in state.partitions:
        findings = state.findings_by_partition.get(partition.key)
        elapsed = state.timings.get(partition.key)
        if elapsed is not None:
            total_time += elapsed
        status = (
            f"\x1b[2m(not run)\x1b[0m"
            if findings is None
            else f"{len(findings)} finding(s), {elapsed:.1f}s"
        )
        print(f"  [{partition.key:20s}] {len(partition.candidate_ids):3d} ids  {status}")
        if findings:
            for f in findings:
                if f.kind == "observation":
                    print(f"      \x1b[2mobs\x1b[0m  {f.category:12s} | {f.claim}")
                else:
                    print(f"      \x1b[2mlim\x1b[0m  {f.limitation_kind:16s} | {f.limitation_detail}")
    print()
    if state.review is not None:
        synth_time = state.timings.get("__synthesis__", 0.0)
        total_time += synth_time
        print(f"\x1b[1mSYNTHESIZED REVIEW\x1b[0m  (synthesis call: {synth_time:.1f}s)")
        print(f"  summary: {state.review.summary}")
        for o in state.review.observations:
            print(f"  obs [{o.category}] {o.claim}")
            for e in o.evidence:
                print(f"      cited: {e.model_dump()}")
        for l in state.review.limitations:
            print(f"  lim [{l.kind}] {l.detail}")
        print()
    print(f"\x1b[1mTotal model time so far: {total_time:.1f}s\x1b[0m  \x1b[2m(today's single one-shot call: ~10-25s)\x1b[0m")
    print()
    print(
        "\x1b[1m[n]\x1b[0m next partition  \x1b[1m[a]\x1b[0m all remaining partitions  "
        "\x1b[1m[y]\x1b[0m synthesize  \x1b[1m[q]\x1b[0m quit"
    )


def run_partition(state: State, partition: Partition) -> None:
    if partition.key in state.findings_by_partition:
        return
    start = time.time()
    findings = extract(partition, state.candidates, lambda prompt: _complete_with_model(state.model, prompt))
    state.timings[partition.key] = time.time() - start
    state.findings_by_partition[partition.key] = findings


def main() -> None:
    workout_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_WORKOUT_ID
    state = State(workout_id)
    print("Loading session...")
    state.load()

    while True:
        render(state)
        try:
            key = input("> ").strip().lower()
        except EOFError:
            break
        if key == "q":
            break
        elif key == "n":
            remaining = [p for p in state.partitions if p.key not in state.findings_by_partition]
            if remaining:
                run_partition(state, remaining[0])
        elif key == "a":
            for p in state.partitions:
                run_partition(state, p)
        elif key == "y":
            all_findings = [f for findings in state.findings_by_partition.values() for f in findings]
            start = time.time()
            state.review = synthesize(
                state.session,
                all_findings,
                state.candidates,
                state.grounding,
                lambda prompt: _complete_with_model(state.model, prompt),
            )
            state.timings["__synthesis__"] = time.time() - start


if __name__ == "__main__":
    main()
