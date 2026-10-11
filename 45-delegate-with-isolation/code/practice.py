"""Phase 14 - Lesson 45: Delegate Agent Work with Isolation and Merge Contracts - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The change being delegated is the one lessons 43 and 44 framed and planned: lesson 16's Runner.run
returns an empty string when it runs out of hops, and three practice files assert that it does.
"""

from __future__ import annotations

from pathlib import Path

from main import WorkUnit, conflicts, delegation_plan, paths_overlap, waves

REPO = Path(__file__).resolve().parents[2]
RUNNER = "16-openai-agents-sdk/code/main.py"
LOCAL = "16-openai-agents-sdk/code/practice.py"
BENCHMARK = "19-benchmarks-swebench-gaia/code/practice.py"
EVALS = "30-eval-driven-agent-development/code/practice.py"


def receipt(path: str, text: str) -> str:
    """'path:line' for the line of that file that contains the text, or '' when no line does."""
    file = REPO / path
    if file.is_file():
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if text in line:
                return f"{path}:{number}"
    return ""


# ---------------------------------------------------------------------------
# Exercise 1 - a real change as two independent work units and an integrator
#
# The fix to the runner goes first; it is a dependency, not one of the two.
# What can then be split is the migration of the checks that pin the old
# behaviour:
#   local       lesson 16's own practice file
#   downstream  the two other lessons that reuse lesson 16's runner
# They are independent in the three ways that matter: no path in common,
# neither depends on the other, and each returns its own proof. The
# integrator owns no file. It collects the two proofs and runs the suite; it
# does not redo either unit's work.
# ---------------------------------------------------------------------------

def units() -> list[WorkUnit]:
    return [
        WorkUnit("runner", "worker-runner", (RUNNER,), (), "python 43-frame-the-task-before-code/code/practice.py --acceptance"),
        WorkUnit("local", "worker-a", (LOCAL,), ("runner",), f"python {LOCAL}"),
        WorkUnit("downstream", "worker-b", (BENCHMARK, EVALS), ("runner",), f"python {BENCHMARK} && python {EVALS}"),
        WorkUnit("integration", "integrator", (), ("local", "downstream"),
                 "python 30-eval-driven-agent-development/code/practice.py --run-suite"),
    ]


def ex1_two_units_and_an_integrator() -> None:
    plan = delegation_plan(units())
    by_id = {unit.id: unit for unit in units()}
    print(f"  status {plan['status']}, waves {plan['waves']}, conflicts {plan['conflicts']}")
    for name in plan["waves"][1]:
        print(f"  {name:<11} owns {list(by_id[name].paths)}")
    assert plan["status"] == "ready" and plan["waves"] == [["runner"], ["downstream", "local"], ["integration"]]
    assert not set(by_id["local"].depends_on) & {"downstream"} and not set(by_id["downstream"].depends_on) & {"local"}
    assert by_id["integration"].paths == () and all((REPO / path).is_file() for unit in units() for path in unit.paths)


# ---------------------------------------------------------------------------
# Exercise 2 - a parallel split that only looks independent
#
# "Fix the runner" and "write the check for it" have no path in common, so
# main.py's conflict check passes them as parallel work. They share a
# decision: what a used-up run gives back, down to the wording, because the
# check asserts on it.
#
# This repository has the proof. Lessons 19 and 30 each fixed this bug in
# their own practice file, separately, and each wrote its own check. Each fix
# passes its own check and fails the other's: one says "max hops ... exceeded"
# and is tested for "max hops", the other says "handoff limit reached" and is
# tested for "limit". Two workers, no shared file, one undecided contract.
# The receipts below are the four lines.
# ---------------------------------------------------------------------------

RELIES_ON = {"runner": {"what a used-up run gives back"}, "check": {"what a used-up run gives back"}}


def undecided_between(parallel: list[WorkUnit], relies_on: dict[str, set[str]], decided: set[str]) -> dict[str, list[str]]:
    """Decisions that two or more units of the same wave rely on and nobody has made yet."""
    shared: dict[str, list[str]] = {}
    for wave in waves(parallel):
        for decision in sorted(set().union(*(relies_on.get(unit, set()) for unit in wave)) - decided):
            users = [unit for unit in wave if decision in relies_on.get(unit, set())]
            if len(users) > 1:
                shared[decision] = users
    return shared


def ex2_looks_independent() -> None:
    split = [WorkUnit("runner", "worker-runner", (RUNNER,), (), "the acceptance command"),
             WorkUnit("check", "worker-a", (LOCAL,), (), f"python {LOCAL}")]
    shared = undecided_between(split, RELIES_ON, decided=set())
    print(f"  main.py: status {delegation_plan(split)['status']}, conflicts {conflicts(split)}, waves {waves(split)}")
    print(f"  decisions both rely on that nobody has made: {shared}")
    assert delegation_plan(split)["status"] == "ready" and waves(split) == [["check", "runner"]]
    assert shared == {"what a used-up run gives back": ["check", "runner"]}
    assert undecided_between(split, RELIES_ON, decided={"what a used-up run gives back"}) == {}

    fixes = {"lesson 19": "error: max hops (4) exceeded", "lesson 30": "handoff limit reached after 4 hops; escalating to a human"}
    checks = {"lesson 19": lambda text: "max hops" in text, "lesson 30": lambda text: "limit" in text and bool(text.strip())}
    lines = [receipt(BENCHMARK, 'error: max hops ({self.max_hops}) exceeded'), receipt(BENCHMARK, 'assert "max hops" in'),
             receipt(EVALS, 'answer = "handoff limit reached after 4 hops'), receipt(EVALS, 'if "limit" in candidate')]
    verdicts = {(fix, check): checks[check](fixes[fix]) for fix in fixes for check in checks}
    for (fix, check), passed in verdicts.items():
        print(f"  {fix}'s fix against {check}'s check: {'pass' if passed else 'FAIL'}")
    print(f"  where those four lines are: {lines}")
    assert all(lines) and [passed for passed in verdicts.values()] == [True, False, False, True]


# ---------------------------------------------------------------------------
# Exercise 3 - a read-only research worker whose output is a fact table
#
# The worker's contract has no paths, so it owns nothing and may change
# nothing (the merge gate of exercise 4 enforces that). What it returns is a
# table: one claim per row, each with the line that supports it. It runs in
# the first wave, and its table is what makes the shared decision of
# exercise 2 visible before two workers discover it the expensive way.
# ---------------------------------------------------------------------------

RESEARCH = WorkUnit("research", "worker-research", (), (), "every row of the fact table has a receipt that resolves")


def fact_table() -> list[tuple[str, str]]:
    return [
        ("the runner starts from an empty final_output", receipt(RUNNER, 'final_output = ""')),
        ("the loop is bounded by max_hops", receipt(RUNNER, "for hop in range(self.max_hops):")),
        ("max_hops defaults to 3", receipt(RUNNER, "max_hops: int = 3")),
        ("lesson 16's practice file asserts the empty string", receipt(LOCAL, 'assert silent == ""')),
        ("lesson 19 words the fix 'max hops ... exceeded'", receipt(BENCHMARK, "error: max hops ({self.max_hops}) exceeded")),
        ("lesson 30 words the fix 'handoff limit reached'", receipt(EVALS, 'answer = "handoff limit reached after 4 hops')),
    ]


def with_research(plan: list[WorkUnit]) -> list[WorkUnit]:
    return [RESEARCH, *[unit if unit.depends_on else WorkUnit(unit.id, unit.owner, unit.paths, ("research",), unit.proof)
                        for unit in plan]]


def ex3_research_worker() -> None:
    table = fact_table()
    for claim, evidence in table:
        print(f"  {claim:<52} {evidence}")
    plan = delegation_plan(with_research(units()))
    print(f"  status {plan['status']}, waves {plan['waves']}")
    assert all(evidence for _, evidence in table) and len(table) == 6
    assert plan["status"] == "ready" and plan["waves"][0] == ["research"] and plan["waves"][1] == ["runner"]
    assert RESEARCH.paths == () and conflicts(with_research(units())) == []


# ---------------------------------------------------------------------------
# Exercise 4 - a merge gate over the final changed-file set
#
# main.py checks the contracts against each other before the work starts. The
# gate checks the work against the contracts after it: every changed file has
# to fall inside the contract of the unit that changed it. That catches three
# things the plan cannot: a worker editing a file another unit owns, a
# read-only worker changing anything, and a file nobody's contract covers.
# ---------------------------------------------------------------------------

def merge_gate(plan: list[WorkUnit], changed: dict[str, list[str]]) -> list[str]:
    """changed: unit id -> files in that unit's diff. Returns the reasons to refuse the merge."""
    by_id = {unit.id: unit for unit in plan}
    refusals = []
    for unit_id, files in changed.items():
        for name in files:
            if any(paths_overlap(name, owned) for owned in by_id[unit_id].paths):
                continue
            owners = [other.id for other in plan if any(paths_overlap(name, owned) for owned in other.paths)]
            refusals.append(f"{unit_id} changed {name}, " + (f"which {owners[0]} owns" if owners else "which no unit's contract covers"))
    return refusals


def ex4_merge_gate() -> None:
    plan = with_research(units())
    diffs = {
        "every worker inside its contract": {"runner": [RUNNER], "local": [LOCAL], "downstream": [BENCHMARK, EVALS], "research": []},
        "local also edits lesson 19": {"runner": [RUNNER], "local": [LOCAL, BENCHMARK], "downstream": [EVALS]},
        "the research worker edits the runner": {"research": [RUNNER], "runner": [RUNNER]},
        "the integrator touches a file nobody owns": {"integration": ["README.md"]},
    }
    results = {label: merge_gate(plan, changed) for label, changed in diffs.items()}
    for label, refusals in results.items():
        print(f"  {label:<42} {'merge' if not refusals else 'REFUSE: ' + '; '.join(refusals)}")
    assert [len(refusals) for refusals in results.values()] == [0, 1, 1, 1]
    assert results["local also edits lesson 19"] == [f"local changed {BENCHMARK}, which downstream owns"]
    assert results["the integrator touches a file nobody owns"] == ["integration changed README.md, which no unit's contract covers"]


# ---------------------------------------------------------------------------
# Exercise 5 - a cancellation rule
#
# The rule: when a unit becomes invalid (its proof stops passing, or the
# decision it was built on is reversed), every unit that depends on it,
# directly or through others, is no longer building on something true.
#   - one that has not merged is cancelled, and restarts when the dependency
#     is valid again;
#   - one that has merged is not undone, but its proof has to be run again;
#   - everything else carries on.
# Cancelling early is the cheap direction: a worker left running finishes
# work that then has to be thrown away, and returns a proof that proves
# nothing.
# ---------------------------------------------------------------------------

def on_invalid(plan: list[WorkUnit], invalid: str, merged: set[str]) -> dict[str, list[str]]:
    needs = {unit.id: set(unit.depends_on) for unit in plan}
    affected: set[str] = set()
    changed = True
    while changed:
        newly = {unit for unit, deps in needs.items() if deps & (affected | {invalid})} - affected
        affected |= newly
        changed = bool(newly)
    return {"cancel": sorted(affected - merged), "reverify": sorted(affected & merged),
            "continue": sorted(set(needs) - affected - {invalid})}


def ex5_cancellation_rule() -> None:
    plan = with_research(units())
    cases = {"the runner's proof fails, nothing merged yet": ("runner", set()),
             "the runner's proof fails after local merged": ("runner", {"local"}),
             "local's proof fails": ("local", set()),
             "a row of the fact table turns out wrong": ("research", {"runner"})}
    outcomes = {label: on_invalid(plan, *case) for label, case in cases.items()}
    for label, outcome in outcomes.items():
        print(f"  {label:<45} cancel {outcome['cancel']}  re-verify {outcome['reverify']}  continue {outcome['continue']}")
    assert outcomes["the runner's proof fails, nothing merged yet"] == {
        "cancel": ["downstream", "integration", "local"], "reverify": [], "continue": ["research"]}
    assert outcomes["the runner's proof fails after local merged"]["reverify"] == ["local"]
    assert outcomes["local's proof fails"] == {"cancel": ["integration"], "reverify": [], "continue": ["downstream", "research", "runner"]}
    assert outcomes["a row of the fact table turns out wrong"]["continue"] == []      # everything was built on it


if __name__ == "__main__":
    print("Phase 14 - Lesson 45: Delegate Agent Work with Isolation and Merge Contracts - exercises")
    for exercise in (ex1_two_units_and_an_integrator, ex2_looks_independent, ex3_research_worker, ex4_merge_gate,
                     ex5_cancellation_rule):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
