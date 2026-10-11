"""Phase 14 - Lesson 41: The Workbench on a Real Repo - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py          (every exercise asserts its own result)
           python practice.py --live   (exercise 4 with a real model: needs the anthropic
                                        package and Anthropic credentials, and spends tokens)
Exercise 2 needs git and this repository's history.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from main import ALLOWED, FORBIDDEN, SAMPLE_APP_PY, SAMPLE_TEST_PY, TaskOutcome, run_prompt_only, run_workbench

REPO = Path(__file__).resolve().parents[2]


def load_lesson(folder: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / folder / "code" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


scope_lesson = load_lesson("36-scope-contracts", "lesson36")
gate_lesson = load_lesson("38-verification-gates", "lesson38")
review_lesson = load_lesson("39-reviewer-agent", "lesson39")
handoff_lesson = load_lesson("40-multi-session-handoff", "lesson40")


# ---------------------------------------------------------------------------
# The harness all five exercises share
#
# main.py scripts both pipelines as constants: the outcomes are written into
# the code. Here they are measured. The task is really done on the sample app,
# the tests really run, and the surfaces are the ones the earlier lessons
# built: lesson 36's scope check, lesson 38's gate, lesson 39's reviewer and
# lesson 40's handoff generator.
#
# The agent is still scripted (SCRIPTED: a Policy says what it does when left
# alone), and the same policy drives both pipelines. Its first patch returns
# 400 with a bare string where the task asks for 422 and a typed envelope, and
# it wants to touch the README and the release script. What differs between
# the two runs is only what stands around the agent.
#
# Step times are SIMULATED, in seconds, and only the first-edit exercise uses
# them.
# ---------------------------------------------------------------------------

APP, TEST = "sample_app/app.py", "sample_app/test_app.py"
GOAL = "add input validation to /signup"
TASK = ("Add input validation to /signup: reject passwords shorter than 8 characters, return 422 with a typed "
        "error envelope. Add a test that proves the new behavior.")
ACCEPTANCE = "run every test in sample_app/test_app.py"
STORE_LINE = "    USERS[email] = password\n"
STEP_SECONDS = {"init": 5, "read scope": 3, "read state": 2, "read code": 15, "edit": 6, "test": 4,
                "gate": 2, "review": 6, "handoff": 3}
WORKBENCH_ONLY_STEPS = ("init", "read scope", "read state", "gate", "review", "handoff")


def patched_app(status: int, error: str) -> str:
    check = "    if len(password) < 8:\n        return {'status': %d, 'error': %s}\n" % (status, error)
    return SAMPLE_APP_PY.replace(STORE_LINE, check + STORE_LINE)


RIGHT_APP = patched_app(422, "{'type': 'password_too_short', 'message': 'use at least 8 characters'}")
WRONG_APP = patched_app(400, "'password too short'")
NEW_TEST = SAMPLE_TEST_PY + '''

def test_signup_rejects_short_password():
    out = signup("a@b.co", "short")
    assert out["status"] == 422
    assert out["error"]["type"] == "password_too_short"
'''
ORIGINAL = {APP: SAMPLE_APP_PY, TEST: SAMPLE_TEST_PY}
CONTRACT = scope_lesson.ScopeContract(task_id="T-001", goal=GOAL, allowed_files=sorted(ALLOWED),
                                      forbidden_files=sorted(FORBIDDEN), acceptance_criteria=[ACCEPTANCE],
                                      rollback_plan="revert the commit")


@functools.lru_cache(maxsize=None)
def run_tests(app_source: str, test_source: str) -> int:
    """Write the two files into a temp sample_app and run every test in it. Returns the exit code."""
    runner = "import sample_app.test_app as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]"
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "sample_app"
        package.mkdir()
        (package / "app.py").write_text(app_source)
        (package / "test_app.py").write_text(test_source)
        return subprocess.run([sys.executable, "-c", runner], cwd=tmp, capture_output=True, timeout=30).returncode


@dataclass(frozen=True)
class Policy:
    """What the agent does when left to itself."""
    first_app: str = WRONG_APP
    second_app: str = RIGHT_APP
    stray_writes: tuple[str, ...] = ("README.md", "sample_app/scripts/release.sh")
    runs_tests_unprompted: bool = False
    pace: float = 1.0


@dataclass
class Measured(TaskOutcome):
    gate_passed: bool = False
    time_to_first_meaningful_edit_s: float | None = None


class Run:
    """One pipeline run in progress: the files it changed, the commands it ran, and when."""

    def __init__(self, pace: float = 1.0) -> None:
        self.files: dict[str, str] = {}
        self.feedback: list[dict[str, object]] = []
        self.events: list[tuple[float, str, str]] = []
        self.clock, self.pace = 0.0, pace

    def step(self, kind: str, path: str = "") -> None:
        self.clock += STEP_SECONDS[kind] * self.pace
        self.events.append((self.clock, kind, path))

    def edit(self, path: str, content: str) -> None:
        self.step("edit", path)
        self.files[path] = content

    def test(self) -> int:
        self.step("test")
        self.feedback.append({"command": ACCEPTANCE, "exit_code": run_tests(self.files[APP], self.files[TEST])})
        return int(self.feedback[-1]["exit_code"])      # type: ignore[call-overload]


def judge(contract: Any, touched: list[str], feedback: list[dict[str, object]],
          state: dict[str, object]) -> tuple[list[str], Any, Any, Any]:
    """Lessons 36, 38, 39 and 40 over one run's artifacts: (files outside scope, gate, review, handoff packet)."""
    scope = scope_lesson.scope_check(contract, scope_lesson.RunSummary(
        touched_files=touched, commands_run=[str(record["command"]) for record in feedback]))
    outside = scope.forbidden_writes + scope.off_scope_writes + scope.soft_off_scope_writes
    latest = list({str(record["command"]): record for record in feedback}.values())     # lesson 38, exercise 5
    gate = gate_lesson.verify(gate_lesson.Artifacts(
        task_id=contract.task_id, acceptance_commands=contract.acceptance_criteria, feedback=latest, rule_report=[],
        scope_report={"forbidden_writes": scope.forbidden_writes,
                      "off_scope_writes": scope.off_scope_writes + scope.soft_off_scope_writes}))
    verdict = {"passed": gate.passed, "findings": [asdict(finding) for finding in gate.findings]}
    review = review_lesson.review(review_lesson.ReviewerInputs(
        contract.task_id, contract.goal, {"touched": touched}, state, feedback, verdict))
    _, packet = handoff_lesson.generate_handoff(handoff_lesson.WorkbenchSnapshot(
        contract.task_id, state, verdict, {"verdict": review.verdict, "total": review.total}, feedback, {"touched": touched}))
    return outside, gate, review, packet


def handoff_quality(packet: Any) -> str:
    has_next = not packet.next_action.startswith("no next_action")
    return "full packet" if has_next and packet.commands_run else "next action only" if has_next else "missing"


def first_meaningful_edit(events: list[tuple[float, str, str]], files: dict[str, str]) -> float | None:
    """Seconds from the start of the task to the first edit of an in-scope file whose change is still there at the end."""
    return next((at for at, kind, path in events
                 if kind == "edit" and path in ALLOWED and files.get(path) != ORIGINAL[path]), None)


def measure(pipeline: str, run: Run, state: dict[str, object]) -> Measured:
    outside, gate, review, packet = judge(CONTRACT, sorted(run.files), run.feedback, state)
    return Measured(pipeline=pipeline, tests_actually_run=bool(run.feedback),
                    acceptance_met=run_tests(run.files[APP], run.files[TEST]) == 0,       # checked by the harness, not the agent
                    files_outside_scope=outside, handoff_quality=handoff_quality(packet), reviewer_total=review.total,
                    gate_passed=gate.passed, time_to_first_meaningful_edit_s=first_meaningful_edit(run.events, run.files))


def prompt_only(policy: Policy = Policy()) -> Measured:
    """Read the code, edit, claim done. Nothing stands around the agent."""
    run = Run(policy.pace)
    run.step("read code")
    for path in policy.stray_writes[:1]:
        run.edit(path, "edited")
    run.edit(APP, policy.first_app)
    run.edit(TEST, NEW_TEST)
    for path in policy.stray_writes[1:]:
        run.edit(path, "edited")
    if policy.runs_tests_unprompted:
        run.test()                              # it looks at the result and claims done either way
    return measure("prompt-only", run, state={})


def workbench(policy: Policy = Policy()) -> Measured:
    """The same agent inside the surfaces: scope is read before writing, acceptance runs before done."""
    run = Run(policy.pace)
    for kind in ("init", "read scope", "read state", "read code"):
        run.step(kind)
    for path in policy.stray_writes:            # a write the contract does not allow is refused before it happens
        if scope_lesson.matches_any(path, CONTRACT.allowed_files):
            run.edit(path, "edited")
    run.edit(APP, policy.first_app)
    run.edit(TEST, NEW_TEST)
    if run.test() != 0:                         # the failure is in the feedback log, so the agent gets one repair
        run.edit(APP, policy.second_app)
        run.test()
    done = run.feedback[-1]["exit_code"] == 0
    for kind in ("gate", "review", "handoff"):
        run.step(kind)
    state = {"active_task_id": None if done else "T-001",
             "assumptions": ["a typed error envelope is a dict with a type and a message"],
             "next_action": "open a pull request with the diff" if done else "fix the failing acceptance test"}
    return measure("workbench-guided", run, state)


FIVE_OUTCOMES = ("tests_actually_run", "acceptance_met", "files_outside_scope", "handoff_quality", "reviewer_total")
SIX_OUTCOMES = (*FIVE_OUTCOMES, "time_to_first_meaningful_edit_s")


# ---------------------------------------------------------------------------
# Exercise 1 - a sixth outcome: time to first meaningful edit
#
# How to measure it cleanly:
#   - start the clock when the task is handed over, not when a process starts;
#   - take the times from the log of writes, not from what the agent says;
#   - define "meaningful" from artifacts: the first edit of an in-scope file
#     whose change is still there at close-out. An edit to the README does not
#     count, and neither does an edit that was later undone;
#   - never report it alone. A fast first edit on a task that fails is not a
#     win, so it is read next to acceptance_met.
# It is the one outcome where the workbench is expected to look worse: it
# reads its scope and state before it writes anything.
# ---------------------------------------------------------------------------

def ex1_time_to_first_meaningful_edit() -> None:
    bare, guided = prompt_only(), workbench()
    for outcome in (bare, guided):
        print(f"  {outcome.pipeline:<16} first meaningful edit at {outcome.time_to_first_meaningful_edit_s:.0f} s, "
              f"acceptance met: {outcome.acceptance_met}")
    assert (bare.time_to_first_meaningful_edit_s, guided.time_to_first_meaningful_edit_s) == (27.0, 31.0)
    assert guided.acceptance_met and not bare.acceptance_met            # four seconds later, and right

    wandering = Run()
    wandering.edit("README.md", "edited")
    undone = Run()
    undone.edit(APP, WRONG_APP)
    undone.edit(TEST, NEW_TEST)
    undone.edit(APP, SAMPLE_APP_PY)                                       # the first edit is taken back
    print(f"  only off-scope edits: {first_meaningful_edit(wandering.events, wandering.files)}; "
          f"first edit undone: counted from {first_meaningful_edit(undone.events, undone.files):.0f} s, the test file")
    assert first_meaningful_edit(wandering.events, wandering.files) is None
    assert first_meaningful_edit(undone.events, undone.files) == 12.0


# ---------------------------------------------------------------------------
# Exercise 2 - the comparison on a real second-day task
#
# The task is commit 563dbed of this repository, "Add exercise solutions for
# lessons 11-30": the second day of the work, picked up from the first. It
# was run once, not twice, so there is no A/B. What can be done honestly is to
# put what the repository recorded about that task through the same four
# surfaces and set it beside the sample app's workbench run.
#
# Where the numbers slip: everywhere, and for one reason. tests_actually_run,
# the gate, the handoff and the reviewer's total all read records, and the
# repository kept none: no feedback log, no state file. The work was checked
# at the time; nothing in the repository can show it. The one number that
# reflects an event is scope: six files outside the task as it was given.
# So on a real repo the first thing the workbench has to earn is not a better
# score. It is having the records to be scored on.
# ---------------------------------------------------------------------------

REAL_COMMIT = "563dbed"


def ex2_real_second_day_task() -> str | None:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True, timeout=10).stdout

    try:
        touched, goal = git("show", "--name-only", "--format=", REAL_COMMIT).split(), git("show", "-s", "--format=%s", REAL_COMMIT).strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("  needs git and this repository's history")
        return "git"
    contract = scope_lesson.ScopeContract(
        task_id="T-11-30", goal=goal, rollback_plan=f"git revert {REAL_COMMIT}",
        allowed_files=["1[1-9]-*/code/practice.py", "2[0-9]-*/code/practice.py", "30-*/code/practice.py"],
        forbidden_files=["*/code/main.py", "*/docs/*", "*/quiz.json"], acceptance_criteria=["every practice file exits zero"])
    outside, gate, review, packet = judge(contract, touched, feedback=[], state={})
    sample = workbench()
    rows = {"tests_actually_run": (sample.tests_actually_run, False), "gate passed": (sample.gate_passed, gate.passed),
            "files_outside_scope": (len(sample.files_outside_scope), len(outside)),
            "handoff_quality": (sample.handoff_quality, handoff_quality(packet)),
            "reviewer_total": (sample.reviewer_total, review.total)}
    print(f"  {'':<20} {'sample app, workbench':<22} real task, as recorded")
    for name, (on_sample, on_record) in rows.items():
        print(f"  {name:<20} {on_sample!s:<22} {on_record}")
    assert rows == {"tests_actually_run": (True, False), "gate passed": (True, False), "files_outside_scope": (0, 6),
                    "handoff_quality": ("full packet", "missing"), "reviewer_total": (7, 3)}
    return None


# ---------------------------------------------------------------------------
# Exercise 3 - false negatives: tasks where prompt-only is faster
#
# MODELLED: the tasks, their failure rates and their rework times are
# assumptions, set out in the table below. The workbench's overhead is the sum
# of its own steps from the harness above, 21 seconds.
#
# A task costs its work, plus the rework when it fails silently. The
# workbench adds its overhead and catches most silent failures. Three of the
# six tasks are false negatives: nothing can go wrong, so the overhead is
# pure cost.
#
# Keeping the workbench anyway. On a false negative the most it can lose is
# its fixed overhead; here that is 62 seconds across all three. What
# prompt-only can lose has no ceiling, and across the other three it is close
# to an hour. And which kind of task it was is only known afterwards: "fix a
# typo" is a false negative until the typo was in the release script. So the
# rule is to keep the workbench on and shrink its fixed cost (lesson 35's
# lock skips the probes; a task with no diff needs no review), not to choose
# per task.
# ---------------------------------------------------------------------------

OVERHEAD_S = sum(STEP_SECONDS[step] for step in WORKBENCH_ONLY_STEPS)
CATCH_RATE = 0.9        # ASSUMPTION: share of silent failures the workbench surfaces before close-out
TASKS = [               # ASSUMPTION: (task, seconds of work, chance of a silent failure, seconds of rework)
    ("run the formatter", 10, 0.0, 0),
    ("answer: which port does the dev server use", 15, 0.0, 0),
    ("fix a typo in a comment", 20, 0.01, 60),
    ("add /signup validation", 300, 0.40, 1800),
    ("bump a dependency", 60, 0.15, 3600),
    ("change the retry policy of the payment client", 600, 0.35, 7200),
]


def expected_seconds(work: float, failure: float, rework: float, with_workbench: bool) -> float:
    return work + (OVERHEAD_S + failure * (1 - CATCH_RATE) * rework if with_workbench else failure * rework)


def ex3_false_negatives() -> None:
    costs = {name: (expected_seconds(*shape, False), expected_seconds(*shape, True)) for name, *shape in TASKS}
    for name, (bare, guided) in costs.items():
        print(f"  {name:<46} prompt-only {bare:>6.0f} s   workbench {guided:>6.0f} s"
              f"{'   <- false negative' if guided > bare else ''}")
    always_bare, always_guided = (sum(cost[i] for cost in costs.values()) for i in (0, 1))
    best_per_task = sum(min(cost) for cost in costs.values())
    print(f"  always prompt-only {always_bare:.0f} s, always workbench {always_guided:.0f} s, the cheaper one each time {best_per_task:.0f} s")
    print(f"  the workbench's false negatives cost {always_guided - best_per_task:.0f} s; "
          f"prompt-only's silent failures cost {always_bare - best_per_task:.0f} s")
    assert [name for name, (bare, guided) in costs.items() if guided > bare] == [name for name, *_ in TASKS[:3]]
    assert OVERHEAD_S == 21 and always_guided - best_per_task < 3 * OVERHEAD_S
    assert always_bare - best_per_task > 3000


# ---------------------------------------------------------------------------
# Exercise 4 - a real model in place of the scripted agent
#
# Which outcomes get noisier: the ones that depend on the model's work, and
# none of the ones a surface enforces.
#
# SIMULATED here: 200 agents whose behaviour is drawn at random (a right or a
# wrong first patch, stray writes or none, running the tests unprompted or
# not, a faster or slower pace), each put through both pipelines on the real
# harness. Prompt-only, every outcome moves but the handoff, which is always
# missing. Inside the workbench three stop moving: tests_actually_run,
# files_outside_scope and handoff_quality are decided by the gate, the scope
# check and the generator, whatever the agent does. What still moves is
# acceptance_met, the reviewer's total and the time to the first edit.
#
# `python practice.py --live` swaps the scripted patch for one a model writes
# (model_app_source below) and runs it through the same harness. It needs the
# anthropic package and Anthropic credentials, and every trial is two model
# calls. It runs model-written code in a temp directory, as any coding agent
# does when it runs tests.
# ---------------------------------------------------------------------------

def drawn_policy(rng: random.Random) -> Policy:
    return Policy(first_app=RIGHT_APP if rng.random() < 0.6 else WRONG_APP,
                  second_app=RIGHT_APP if rng.random() < 0.8 else WRONG_APP,
                  stray_writes=Policy().stray_writes if rng.random() < 0.5 else (),
                  runs_tests_unprompted=rng.random() < 0.3, pace=rng.uniform(0.5, 1.5))


def spread(runs: list[Measured]) -> dict[str, int]:
    """How many different values each outcome took across the runs. 1 means it did not move."""
    return {name: len({json.dumps(getattr(run, name)) for run in runs}) for name in SIX_OUTCOMES}


def ex4_noisier_outcomes() -> None:
    rng = random.Random(0)
    policies = [drawn_policy(rng) for _ in range(200)]
    bare, guided = [prompt_only(policy) for policy in policies], [workbench(policy) for policy in policies]
    moved = {"prompt-only": spread(bare), "workbench": spread(guided)}
    for name in SIX_OUTCOMES:
        print(f"  {name:<32} prompt-only {moved['prompt-only'][name]:>3} values   workbench {moved['workbench'][name]:>3} values"
              f"{'   pinned by a surface' if moved['workbench'][name] == 1 else ''}")
    print(f"  acceptance met: prompt-only {sum(run.acceptance_met for run in bare) / 200:.0%}, "
          f"workbench {sum(run.acceptance_met for run in guided) / 200:.0%}")
    pinned = [name for name in SIX_OUTCOMES if moved["workbench"][name] == 1]
    assert pinned == ["tests_actually_run", "files_outside_scope", "handoff_quality"]
    assert [name for name in SIX_OUTCOMES if moved["prompt-only"][name] == 1] == ["handoff_quality"]
    assert sum(run.acceptance_met for run in guided) > sum(run.acceptance_met for run in bare)


def model_app_source() -> str:
    """Ask a model for the patched app.py. One call; spends tokens."""
    import anthropic

    prompt = (f"{TASK}\n\nThe current sample_app/app.py:\n\n{SAMPLE_APP_PY}\nThe tests that must pass:\n\n{NEW_TEST}\n"
              "Reply with the complete new app.py in one python code block.")
    response = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5", max_tokens=16000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        messages=[{"role": "user", "content": prompt}])
    if response.stop_reason == "refusal":
        return SAMPLE_APP_PY                    # no patch came back: the task stays undone
    text = "".join(block.text for block in response.content if block.type == "text")
    fenced = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return fenced.group(1) if fenced else text


def live(trials: int = 3) -> int:
    try:
        import anthropic
    except ImportError:
        print("needs the anthropic package: pip install anthropic")
        return 1
    try:
        policies = [Policy(first_app=model_app_source(), second_app=model_app_source(), stray_writes=())
                    for _ in range(trials)]
    except anthropic.AuthenticationError:
        print("the credentials were rejected; set ANTHROPIC_API_KEY or run `ant auth login`")
        return 1
    except anthropic.APIConnectionError:
        print("could not reach the API")
        return 1
    except anthropic.AnthropicError as error:
        print(f"needs Anthropic credentials (ANTHROPIC_API_KEY or `ant auth login`): {error}")
        return 1
    runs = [workbench(policy) for policy in policies]
    for number, run in enumerate(runs, 1):
        print(f"trial {number}: acceptance met {run.acceptance_met}, reviewer {run.reviewer_total}/10, gate passed {run.gate_passed}")
    print(f"values per outcome across {trials} trials: {spread(runs)}")
    return 0


# ---------------------------------------------------------------------------
# Exercise 5 - one page for a non-engineer
#
# What survives the cut: four plain facts with their numbers, the one cost,
# a recommendation and a caution. What does not: the reviewer's score out of
# ten (it means nothing without the rubric), the names of the surfaces, and
# how any check works. The page is generated from the measured outcomes, so
# it cannot say more than the run showed.
# ---------------------------------------------------------------------------

JARGON = ("scope contract", "verification gate", "harness", "handoff", "feedback runner", "json", "rubric", "acceptance",
          "reviewer_total", "pipeline")


def summary_for_a_non_engineer(bare: Measured, guided: Measured) -> str:
    def yes_no(value: bool) -> str:
        return "yes" if value else "no"

    later = guided.time_to_first_meaningful_edit_s - bare.time_to_first_meaningful_edit_s
    return "\n".join([
        "What we tested",
        "We gave an AI coding assistant the same small job twice: add a password check to a sign-up page. Once it had "
        "only the written request. Once it also had working rules: which files it may change, a check it must run "
        "before saying it is finished, and notes it must leave when it stops.",
        "",
        "What happened",
        f"- Did it check its own work before saying it was finished? Without the rules: {yes_no(bare.tests_actually_run)}. "
        f"With them: {yes_no(guided.tests_actually_run)}.",
        f"- Was the job actually done correctly? Without the rules: {yes_no(bare.acceptance_met)}. "
        f"With them: {yes_no(guided.acceptance_met)}.",
        f"- Files changed that nobody asked it to change. Without the rules: {len(bare.files_outside_scope)}, one of them "
        f"the script that ships the product. With them: {len(guided.files_outside_scope)}.",
        f"- Could someone else pick the work up tomorrow from what it left behind? Without the rules: "
        f"{yes_no(bare.handoff_quality != 'missing')}. With them: {yes_no(guided.handoff_quality != 'missing')}.",
        "",
        "What it costs",
        f"With the rules, the assistant started on the real change {later:.0f} seconds later, because it reads the rules first.",
        "",
        "What we recommend",
        "Keep the rules on for any job that changes how the product behaves. Without them the assistant said it was "
        "finished when the job was not done, and nothing in what it left behind would have told us.",
        "",
        "One caution",
        "This was one small job, and the assistant's behaviour was scripted so that both runs started from the same "
        "mistake. It shows what the rules catch. It does not show how often an assistant makes that mistake.",
    ])


def ex5_one_page_summary() -> None:
    bare, guided = prompt_only(), workbench()
    page = summary_for_a_non_engineer(bare, guided)
    for line in page.splitlines():
        print(f"  | {line}")
    words = len(page.split())
    print(f"  {words} words; jargon terms used: {[term for term in JARGON if term in page.lower()]}")
    assert words < 300 and not any(term in page.lower() for term in JARGON)
    assert "Without the rules: 2" in page and "With them: 0" in page and "4 seconds later" in page
    assert str(guided.reviewer_total) + "/10" not in page


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder, so nothing is written here.
#   exits zero                               yes
#   the report measures all five outcomes    yes
#   the workbench beats prompt-only on at    yes, on all five. Those are
#   least four of the five                   main.py's scripted constants.
#                                            Measured on the harness above
#                                            the workbench also wins all
#                                            five, with a reviewer total of
#                                            7 against 2 where main.py
#                                            writes 9 against 3.
# The 7 is lesson 39's keyword reviewer giving zero for problem fit: no word
# of the goal appears in "sample_app/app.py".
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


def wins(bare: Any, guided: Any) -> list[str]:
    """The outcomes on which the workbench run is better than the prompt-only run."""
    better = {"tests_actually_run": lambda b, g: g and not b, "acceptance_met": lambda b, g: g and not b,
              "files_outside_scope": lambda b, g: len(g) < len(b), "handoff_quality": lambda b, g: g != "missing" == b,
              "reviewer_total": lambda b, g: g > b}
    return [name for name in FIVE_OUTCOMES if better[name](bare[name], guided[name])]


def mission_acceptance() -> None:
    (exit_code, _, files), = run_main()
    report, comparison = files["code/before-after-report.md"], json.loads(files["code/comparison.json"])
    scripted = wins(comparison["prompt_only"], comparison["workbench"])
    measured = wins(asdict(prompt_only()), asdict(workbench()))
    print(f"  main.py exits {exit_code}; the report has a row for {sum(name in report for name in FIVE_OUTCOMES)} of the 5 outcomes")
    print(f"  workbench better on: main.py's constants {len(scripted)}/5, measured here {len(measured)}/5")
    print(f"  reviewer totals: main.py {run_prompt_only().reviewer_total} and {run_workbench().reviewer_total}, "
          f"measured {prompt_only().reviewer_total} and {workbench().reviewer_total}")
    assert exit_code == 0 and all(name in report for name in FIVE_OUTCOMES)
    assert len(scripted) == 5 and len(measured) == 5
    assert (prompt_only().reviewer_total, workbench().reviewer_total) == (2, 7)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--live"]:
        sys.exit(live())
    print("Phase 14 - Lesson 41: The Workbench on a Real Repo - exercises")
    outcomes = []
    for exercise in (ex1_time_to_first_meaningful_edit, ex2_real_second_day_task, ex3_false_negatives,
                     ex4_noisier_outcomes, ex5_one_page_summary, mission_acceptance):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
