"""Phase 14 - Lesson 31: Agent Workbench - Why Capable Models Still Fail - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from main import WORKBENCH_SURFACES, RepoTask, RunResult, stub_agent

REPO = Path(__file__).resolve().parents[2]
TASK = RepoTask(
    description="add input validation to /signup and a passing test",
    allowed_files=["app.py", "test_app.py"],
    forbidden_files=["README.md", "scripts/release.sh"],
    acceptance=["test_app.py::test_signup_rejects_short_password passes"],
)


# ---------------------------------------------------------------------------
# Exercise 1 - score the seven surfaces of a repo where an agent already runs
#
# The repo is this one: an agent wrote every practice file in it. Each score
# is decided by what is on disk, not by opinion, so it changes when the repo
# does. 0 is missing, 1 is present but nothing enforces it, 2 is healthy.
# ---------------------------------------------------------------------------

def audit(root: Path) -> dict[str, tuple[int, str]]:
    def exists(*names: str) -> bool:
        return any((root / name).exists() for name in names)

    practice_files = len(list(root.glob("[0-9][0-9]-*/code/practice.py")))
    try:
        last_commit = subprocess.run(["git", "log", "-1", "--format=%b"], cwd=root, capture_output=True, text=True, timeout=5).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        last_commit = ""
    return {
        "instructions": (2, "AGENTS.md or CLAUDE.md at the root") if exists("AGENTS.md", "CLAUDE.md")
        else (0, "no AGENTS.md or CLAUDE.md"),
        "state": (2, "agent_state.json at the root") if exists("agent_state.json") else (0, "no state file; each session starts from the chat"),
        "scope": (2, "a scope contract or feature list") if exists("scope_contract.json", "feature_list.json")
        else (0, "no contract says which files a task may touch"),
        "feedback": (1, f"{practice_files} runnable checks, but no log captures what they printed") if practice_files
        else (0, "nothing to run"),
        "verification": (2, "checks run in CI") if exists(".github/workflows")
        else (1, "checks exist and exit non-zero on failure, but nothing runs them before a push") if practice_files
        else (0, "no checks"),
        "review": (2, "review reports kept") if exists("outputs/review", "CODEOWNERS") else (0, "nobody but the builder looks before main"),
        "handoff": (2, "handoff packets kept") if exists("handoff.md", "outputs/handoff")
        else (1, "commit messages say what changed; nothing says what is left") if last_commit.strip()
        else (0, "no record of what changed"),
    }


def ex1_score_the_surfaces() -> None:
    scores = audit(REPO)
    for surface in WORKBENCH_SURFACES:
        score, evidence = scores[surface]
        print(f"  {surface:<13} {score}/2  {evidence}")
    lowest = min(score for score, _ in scores.values())
    weakest = [surface for surface in WORKBENCH_SURFACES if scores[surface][0] == lowest]
    print(f"  total {sum(score for score, _ in scores.values())}/14; weakest: {', '.join(weakest)}")
    assert set(scores) == set(WORKBENCH_SURFACES) and all(0 <= score <= 2 for score, _ in scores.values())


# ---------------------------------------------------------------------------
# Exercise 2 - a fake success claim, and the gate that catches it
#
# The prompt-only run already ends with declared_success=True on work that
# does not pass. Here the claim is made explicit, as the sentence a user would
# read, and a verification gate is put in front of it. The gate does not
# listen to the claim. It asks for the evidence: were the tests run, did the
# acceptance check pass, did the writes stay inside the task.
# ---------------------------------------------------------------------------

def success_claim(result: RunResult) -> str:
    return "Done. Validation added and all tests pass." if result.declared_success else "Not finished."


def verification_gate(result: RunResult, task: RepoTask) -> list[str]:
    """Reasons to refuse the claim. Empty means it may stand."""
    refusals = []
    if not result.tests_run:
        refusals.append("no test run on record")
    if not result.actually_passing:
        refusals.append("acceptance check has not passed")
    stray = [path for path in result.files_touched if path not in task.allowed_files]
    if stray:
        refusals.append(f"writes outside the task: {stray}")
    return refusals


def ex2_fake_success_claim() -> None:
    prompt_only, workbench = stub_agent(TASK, surfaces=[]), stub_agent(TASK, surfaces=WORKBENCH_SURFACES)
    for run in (prompt_only, workbench):
        refusals = verification_gate(run, TASK)
        print(f"  {run.label:<11} claims: {success_claim(run)!r}")
        print(f"  {'':<11} gate  : {'refused - ' + '; '.join(refusals) if refusals else 'claim stands'}")
    assert success_claim(prompt_only) == success_claim(workbench)           # the two claims read the same
    assert len(verification_gate(prompt_only, TASK)) == 3 and verification_gate(workbench, TASK) == []


# ---------------------------------------------------------------------------
# Exercise 3 - an eighth surface: telemetry
#
# Each of the seven surfaces looks at one task or one session. None of them
# looks across runs, and some failures only exist there: every run passes,
# and the runs are getting slower, or the same warning comes back each night.
#
# Why it does not collapse into an existing surface:
#   feedback      is this run's command output, kept for the next turn
#   verification  is pass or fail for one task
#   review        is one diff
#   handoff       is one session to the next
#   state         is where the work stands now, not how it has been trending
# Lesson 37 draws the same line between feedback and telemetry.
# ---------------------------------------------------------------------------

def telemetry_flags(runs: list[dict[str, object]]) -> list[str]:
    flags = []
    steps = [int(run["steps"]) for run in runs]
    if steps[-1] >= 2 * steps[0]:
        flags.append(f"steps per task went from {steps[0]} to {steps[-1]} over {len(runs)} runs")
    notes = [note for run in runs for note in run["warnings"]]          # type: ignore[union-attr]
    for note in sorted(set(notes)):
        if notes.count(note) > len(runs) / 2:
            flags.append(f"'{note}' recurred in {notes.count(note)} of {len(runs)} runs")
    return flags


def ex3_eighth_surface() -> None:
    runs = []
    for night in range(10):
        result = stub_agent(TASK, surfaces=WORKBENCH_SURFACES)
        assert verification_gate(result, TASK) == []                    # every single run is clean
        runs.append({"steps": 6 + night, "warnings": ["retried flaky test"] if night >= 3 else []})
    flags = telemetry_flags(runs)
    print("  ten runs, each one passes all seven surfaces")
    for flag in flags:
        print(f"  telemetry: {flag}")
    assert len(flags) == 2


# ---------------------------------------------------------------------------
# Exercise 4 - an agent that makes an extra file write
#
# Which surface catches it first? Scope, at the moment of the write. Take
# scope away and the next surface to notice is verification, at close-out,
# then review, then handoff, where a human may or may not read the list of
# changed files. Instructions, state and feedback never notice: none of them
# compares what was written with what was allowed.
# ---------------------------------------------------------------------------

LOOP_ORDER = ["scope", "verification", "review", "handoff"]
WHEN = {"scope": "as the file is written", "verification": "at task close-out",
        "review": "in the second-pass review", "handoff": "only if a human reads the changed-files list"}


def overreaching_agent(task: RepoTask, surfaces: list[str]) -> RunResult:
    result = stub_agent(task, surfaces)
    result.files_touched = [*task.allowed_files, "config/settings.py"]      # nobody asked for this one
    return result


def first_catch(result: RunResult, task: RepoTask) -> str | None:
    if all(path in task.allowed_files for path in result.files_touched):
        return None
    return next((surface for surface in LOOP_ORDER if surface in result.surfaces_present), None)


def ex4_extra_file_write() -> None:
    present = list(WORKBENCH_SURFACES)
    order = []
    while True:
        caught_by = first_catch(overreaching_agent(TASK, present), TASK)
        order.append(caught_by)
        print(f"  with {len(present)} surfaces: " + (f"caught by {caught_by}, {WHEN[caught_by]}" if caught_by else "not caught"))
        if caught_by is None:
            break
        present.remove(caught_by)
    assert order == ["scope", "verification", "review", "handoff", None]
    assert first_catch(stub_agent(TASK, WORKBENCH_SURFACES), TASK) is None      # a clean run trips nothing


# ---------------------------------------------------------------------------
# Exercise 5 - the failure modes of lesson 26 mapped onto the surfaces
#
# Each mode is listed with the surface built to absorb it and one that backs
# it up. Review and handoff absorb two that lesson 26's detector list adds to
# the five: an agent grading its own work, and context lost between sessions
# rather than within one.
#
# The course's capstone pack (lesson 42) ships a reliability policy that maps
# the same five modes, and the two are compared below. They agree on the
# first line of defence for three of the five. The two differences are about
# when a mode is caught, not whether.
#   hallucinated actions  the pack names the rule set and the gate. Here the
#                         feedback record comes first: a call to a tool that
#                         does not exist fails in the loop, turns before any
#                         gate runs.
#   tool misuse           the pack leaves it to the reviewer. Here the rules
#                         come first, because a rule is checked every turn
#                         and a review happens once, at the end.
# ---------------------------------------------------------------------------

PACK_POLICY = REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack" / "docs" / "reliability-policy.md"
PACK_WORDING = {"rule set": "instructions", "verification gate": "verification", "scope contract": "scope",
                "feedback": "feedback", "repo memory": "state", "reviewer": "review"}


def pack_mapping() -> list[set[str]]:
    """The surfaces the pack's policy names for each of its five modes, in the order it lists them."""
    numbered = [line for line in PACK_POLICY.read_text(encoding="utf-8").splitlines() if re.match(r"\d\. ", line)]
    return [{surface for wording, surface in PACK_WORDING.items() if wording in line} for line in numbered]


ABSORBED_BY = {     # failure mode -> (primary surface, backup, why)
    "hallucinated actions": ("feedback", "verification", "a tool that does not exist returns a real error, in the loop"),
    "scope creep": ("scope", "review", "the contract names the files; anything else is a finding"),
    "cascading errors": ("feedback", "verification", "a non-zero exit is on record before the next step builds on it"),
    "context loss": ("state", "instructions", "constraints and the next action are re-read from a file every turn"),
    "tool misuse": ("instructions", "feedback", "rules are checks on each call, and bad arguments come back as errors"),
    "success hallucination": ("verification", "review", "done is decided by evidence, then by someone other than the builder"),
    "context loss across sessions": ("handoff", "state", "what changed and what is left is written down for the next session"),
}


def ex5_modes_onto_surfaces() -> None:
    for mode, (primary, backup, why) in ABSORBED_BY.items():
        print(f"  {mode:<29} -> {primary:<13} (then {backup}): {why}")
    used = {surface for primary, backup, _ in ABSORBED_BY.values() for surface in (primary, backup)}
    primaries = {primary for primary, _, _ in ABSORBED_BY.values()}
    print(f"  every surface is the first line for at least one mode: {sorted(primaries) == sorted(set(WORKBENCH_SURFACES) - {'review'})} "
          "(review is always the second look, by design)")
    assert used == set(WORKBENCH_SURFACES)
    assert primaries == set(WORKBENCH_SURFACES) - {"review"}

    pack = pack_mapping()
    same_first_line = [mode for (mode, (primary, _, _)), named in zip(ABSORBED_BY.items(), pack) if primary in named]
    differs = {mode: sorted(named) for (mode, (primary, _, _)), named in zip(ABSORBED_BY.items(), pack) if primary not in named}
    print(f"  the capstone pack's policy names the same first line for {len(same_first_line)} of its {len(pack)} modes; "
          f"it differs on {differs}")
    assert len(pack) == 5 and all(pack) and len(same_first_line) == 3
    assert differs == {"hallucinated actions": ["instructions", "verification"], "tool misuse": ["review"]}


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder, so nothing is written here.
#   exits zero                              yes
#   the log shows the two runs              yes, one after the other
#   failure_modes.json lists every missed   partly: all seven missed surfaces
#   surface with the matching symptom       are listed, with four symptoms
#                                           kept as loose notes
# failure_modes() pairs each of the seven with its symptom, read from the
# "failure when missing" column of the lesson's own table. The one-line
# verdict for the workbench run is the gate of exercise 2.
# ---------------------------------------------------------------------------

def run_main(times: int = 1) -> list[tuple[int, str, dict[str, str]]]:
    """Run this folder's main.py in a temp copy. One (exit code, stdout, files left behind) per run."""
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "code"
        code.mkdir()
        (Path(tmp) / "outputs").mkdir()             # every lesson folder in the repo has one
        shutil.copy(Path(__file__).with_name("main.py"), code)
        for _ in range(times):
            done = subprocess.run([sys.executable, "main.py"], cwd=code, capture_output=True, text=True, timeout=120)
            files = {path.relative_to(tmp).as_posix(): path.read_text(errors="replace") for path in sorted(Path(tmp).rglob("*"), key=str)
                     if path.is_file() and path.name != "main.py" and "__pycache__" not in path.parts}
            runs.append((done.returncode, done.stdout, files))
    return runs


def failure_modes(result: RunResult) -> dict[str, str]:
    """Every surface the run was missing, with the symptom the lesson's table gives for it."""
    table = (REPO / "31-agent-workbench-why-models-fail" / "docs" / "en.md").read_text(encoding="utf-8")
    symptoms = {surface.lower(): symptom.strip() for surface, symptom in re.findall(r"^\| (\w+) \| [^|]+ \| ([^|]+) \|$", table, re.M)}
    return {surface: symptoms[surface] for surface in result.missing_surfaces()}


def mission_acceptance() -> None:
    (exit_code, stdout, files), = run_main()
    written = json.loads(files["outputs/failure_modes.json"])
    paired = failure_modes(stub_agent(TASK, surfaces=[]))
    verdict = verification_gate(stub_agent(TASK, surfaces=WORKBENCH_SURFACES), TASK)
    print(f"  main.py exits {exit_code}; failure_modes.json has {len(written['missing_surfaces'])} missed surfaces and "
          f"{len(written['notes'])} symptoms")
    print(f"  paired here: {len(paired)} of {len(WORKBENCH_SURFACES)} surfaces with a symptom, for example scope -> {paired['scope']!r}")
    print(f"  workbench run, one line: {'claim stands' if not verdict else 'claim refused'}")
    assert exit_code == 0 and stdout.index("=== prompt only ===") < stdout.index("=== workbench ===")
    assert written["missing_surfaces"] == WORKBENCH_SURFACES and len(written["notes"]) == 4
    assert list(paired) == WORKBENCH_SURFACES and all(paired.values()) and verdict == []
    assert failure_modes(stub_agent(TASK, surfaces=WORKBENCH_SURFACES)) == {}


if __name__ == "__main__":
    print("Phase 14 - Lesson 31: Why Capable Models Still Fail - exercises")
    for exercise in (ex1_score_the_surfaces, ex2_fake_success_claim, ex3_eighth_surface, ex4_extra_file_write,
                     ex5_modes_onto_surfaces, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
