"""Phase 14 - Lesson 39: Reviewer Agent and Rubric Scoring - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

main.py's reviewer is deterministic, with no model behind it, and nothing here calls one.
Part of exercise 4 needs git and this repository's history.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

from main import SCORERS, DimensionScore, ReviewerInputs, ReviewReport, review

REPO = Path(__file__).resolve().parents[2]


def closeout(**changes: Any) -> ReviewerInputs:
    """A task closed out properly, with the given fields changed."""
    clean = dict(task_id="T-001", goal="add input validation to signup",
                 diff_summary={"touched": ["app/signup.py", "tests/test_signup.py"]},
                 state={"active_task_id": None, "assumptions": ["users sign up with email and password only"],
                        "next_action": "pick next task from board"},
                 feedback=[{"command": "pytest tests/test_signup.py", "exit_code": 0}],
                 verdict={"passed": True, "findings": []})
    return ReviewerInputs(**{**clean, **changes})


# ---------------------------------------------------------------------------
# Exercise 1 - a sixth dimension: claims backed by evidence
#
# The product is this repository. Each practice file is there to show that an
# exercise was done, so the failure that matters most is a close-out that
# says more than the run showed.
# ASSUMPTION: the builder lists its claims in state["claims"], each with the
# command that backs it.
#
# Why the existing five do not absorb it:
#   verification_quality  reads exit codes. It never reads what the builder
#                         said about them: a green lint run and the sentence
#                         "the integration tests pass" score 2.
#   problem_fit           is about which files were touched.
#   assumptions           is what was guessed before the work, not what is
#                         asserted after it.
#   scope and handoff     do not look at claims at all.
#
# Adding a dimension has a side effect worth knowing. main.py's thresholds
# are 5 and 7 points, written for a total of 10. Append a sixth scorer and
# the total is 12 while the thresholds stay put, so the reviewer gets more
# lenient: a task that was a soft fail passes on a middling sixth score.
# review_with keeps the thresholds at half and seven tenths of the total.
# ---------------------------------------------------------------------------

def score_claims(inputs: ReviewerInputs) -> DimensionScore:
    claims = inputs.state.get("claims") or []
    green = {str(record.get("command")) for record in inputs.feedback if record.get("exit_code") == 0}
    unbacked = [claim["claim"] for claim in claims if claim.get("command") not in green]       # type: ignore[union-attr]
    if unbacked:
        return DimensionScore("claims_backed", 0, f"claimed without a passing run: {unbacked}")
    if not claims:
        return DimensionScore("claims_backed", 1, "no claims listed; nothing to check them against")
    return DimensionScore("claims_backed", 2, f"{len(claims)} claims, each backed by a passing run")


def review_with(inputs: ReviewerInputs, scorers: list[Callable[[ReviewerInputs], DimensionScore]]) -> ReviewReport:
    dimensions = [scorer(inputs) for scorer in scorers]
    total, top = sum(dimension.score for dimension in dimensions), 2 * len(dimensions)
    if any(dimension.score == 0 for dimension in dimensions) or total < 0.5 * top:
        verdict = "hard_fail"
    else:
        verdict = "pass" if total >= 0.7 * top else "soft_fail"
    return ReviewReport(task_id=inputs.task_id, total=total, verdict=verdict, dimensions=dimensions)


def ex1_sixth_dimension() -> None:
    six = [*SCORERS, score_claims]
    backed = {"claim": "signup rejects short passwords", "command": "pytest tests/test_signup.py"}
    unbacked = {"claim": "the integration tests pass", "command": "pytest tests/integration"}
    cases = {"claim backed by the run": [backed], "no claims listed": [], "claims a run that never happened": [backed, unbacked]}
    verdicts = {}
    for label, claims in cases.items():
        inputs = closeout(state={**closeout().state, "claims": claims})
        five, sixth = review(inputs), review_with(inputs, six)
        verdicts[label] = (five.verdict, sixth.verdict)
        print(f"  {label:<33} five: {five.verdict} {five.total}/10   six: {sixth.verdict} {sixth.total}/12  ({sixth.dimensions[-1].note})")
    assert verdicts == {"claim backed by the run": ("pass", "pass"), "no claims listed": ("pass", "pass"),
                        "claims a run that never happened": ("pass", "hard_fail")}
    assert all(review_with(inputs, SCORERS).verdict == review(inputs).verdict for _, _, inputs in calibration_set())

    sloppy = closeout(state={"active_task_id": "T-001", "assumptions": [], "next_action": ""},
                      verdict={"passed": True, "findings": [{"code": "scope.off_scope", "severity": "warn"}]})
    assert review(sloppy).verdict == "soft_fail" and review(sloppy).total == 6
    appended, rescaled = sum(scorer(sloppy).score for scorer in six), review_with(sloppy, six)
    print(f"  a soft fail at 6/10 plus a sixth score of 1: {appended}/12 clears main.py's 7-point pass mark; "
          f"with thresholds rescaled: {rescaled.verdict}")
    assert appended == 7 and rescaled.verdict == "soft_fail"


# ---------------------------------------------------------------------------
# Exercise 2 - two system prompts, terse and verbose
#
# The same close-out went to the reviewer twice, once under each prompt
# below: task T-002, asked to add input validation to signup, touched only
# docs/api.md, left the task open, and came with the scores main.py computes.
# RECORDED: both reports were written by Claude (Opus 5.5) on 2026-10-08 and
# are kept here as text. Running this file measures them; it does not call a
# model. The checks below tie each report to main.py's scores, so a report
# that drifted from the code would fail.
#
# The terse one is the report a person is more likely to read. Measured: it
# is about a seventh of the length, and the verdict is its first words, where
# the verbose report reaches the verdict in its last sentence. The verbose
# report is not padding: its reasoning is what a builder needs to fix the
# work. So terse is what gets sent, and the reasoning is what gets linked.
# Terse still carries the evidence for each lost point; the dimensions at
# full marks are the part that can go.
# ASSUMPTION: 240 words a minute, and a reviewer who may stop after two lines.
# ---------------------------------------------------------------------------

TERSE_PROMPT = ("You are a code reviewer. First line: the verdict, the total and the task. Then one line for each "
                "dimension that lost points: name, score, the evidence. Nothing else.")
VERBOSE_PROMPT = ("You are a thorough code reviewer. For each dimension, restate what it measures, describe what you "
                  "examined, explain your reasoning, then give the score. Finish with an overall assessment and the verdict.")

TERSE_REPORT = """\
HARD FAIL 5/10 T-002
problem_fit 0/2: the goal is signup input validation; the only file touched is docs/api.md
scope_discipline 1/2: the gate reports one off-scope write
assumptions 1/2: none recorded
handoff_readiness 1/2: T-002 is still the active task and there is no next action
"""

VERBOSE_REPORT = """\
Problem fit. This dimension asks whether the change does what the task asked for. The task was to add input \
validation to signup. I looked at the list of touched files and found a single entry, docs/api.md. No handler, no \
validator and no test was changed, so nothing in the diff can reject a bad signup. Documentation alone does not \
implement validation. Score: 0 out of 2.

Scope discipline. This dimension asks whether the builder stayed inside the files the task allowed. I read the \
findings of the verification gate. The gate passed, and it carries one off-scope warning and no forbidden write. One \
stray write is tolerable, but it is not clean. Score: 1 out of 2.

Assumptions. This dimension asks whether the builder wrote down what it took for granted. The state file lists no \
assumptions. For a change this small that may be honest, but I cannot tell an obvious task from an undocumented one. \
Score: 1 out of 2.

Verification quality. This dimension asks whether the commands that were run show that the change works. The \
feedback log has one command, pytest on the signup tests, and it exited with code 0. The log is complete and green. \
It shows the existing tests still pass, which is all a documentation change can show. Score: 2 out of 2.

Handoff readiness. This dimension asks whether the next session could pick the work up. The state still shows T-002 \
as the active task and gives no next action. A new session would have to work out for itself whether the task is \
finished. Score: 1 out of 2.

Overall assessment. The total is 5 out of 10, and one dimension scored zero. The work that was asked for was not \
done, so the verdict is hard fail.
"""


def ex2_terse_and_verbose() -> None:
    report = review(closeout(task_id="T-002", diff_summary={"touched": ["docs/api.md"]},
                             state={"active_task_id": "T-002", "assumptions": [], "next_action": ""},
                             verdict={"passed": True, "findings": [{"code": "scope.off_scope", "severity": "warn"}]}))
    scores = [dimension.score for dimension in report.dimensions]
    assert (report.verdict, report.total, scores) == ("hard_fail", 5, [0, 1, 1, 2, 1])
    assert TERSE_REPORT.startswith("HARD FAIL 5/10 T-002")                              # the recorded reports say what main.py scored
    assert [f"{dimension.name} {dimension.score}/2" for dimension in report.dimensions if dimension.score < 2] == [
        line.split(":")[0] for line in TERSE_REPORT.splitlines()[1:]]
    assert [int(score) for score in re.findall(r"Score: (\d) out of 2", VERBOSE_REPORT)] == scores
    assert "5 out of 10" in VERBOSE_REPORT

    measured = {}
    for name, text in (("terse", TERSE_REPORT), ("verbose", VERBOSE_REPORT)):
        words = len(text.split())
        before_verdict = len(text[:text.lower().index("hard fail")].split())
        measured[name] = (words, before_verdict)
        print(f"  {name:<8} {words:>3} words, {before_verdict:>3} words before the verdict, {words / 240 * 60:>2.0f} s to read")
    for line in TERSE_REPORT.splitlines():
        print(f"    | {line}")
    assert measured["terse"][1] == 0 and measured["verbose"][1] > measured["verbose"][0] - 4
    assert measured["verbose"][0] > 5 * measured["terse"][0]


# ---------------------------------------------------------------------------
# Exercise 3 - a confidence on each dimension
#
# The confidence says how much evidence the score rests on. Each dimension
# has two levels, set by hand: the evidence was there, or it was absent.
# main.py's scorers treat absent evidence as good news in places: with no
# gate report at all, scope_discipline scores 2.
#
# The rule as the exercise states it looks at the lowest-scoring dimension
# only. Measured: that is not enough. A close-out with no gate report ships
# as a pass, because the dimension with nothing behind it is the one that
# scored highest. Checking every dimension refuses it.
# ---------------------------------------------------------------------------

CONFIDENCE_FLOOR = 0.6


@dataclass
class ConfidentScore(DimensionScore):
    confidence: float = 0.0


class LowConfidence(RuntimeError):
    pass


def confidence(inputs: ReviewerInputs) -> dict[str, float]:
    keywords = [word for word in inputs.goal.lower().split() if len(word) > 4]
    hit = any(keyword in path.lower() for keyword in keywords for path in inputs.diff_summary.get("touched", []))
    return {"problem_fit": 0.8 if hit else 0.4,                 # a goal word missing from the file names proves little
            "scope_discipline": 0.9 if "findings" in inputs.verdict else 0.3,
            "assumptions": 0.9 if "assumptions" in inputs.state else 0.4,
            "verification_quality": 0.9 if inputs.feedback else 0.3,
            "handoff_readiness": 0.9 if "next_action" in inputs.state else 0.4}


def ship(inputs: ReviewerInputs, every_dimension: bool = False) -> ReviewReport:
    report, levels = review(inputs), confidence(inputs)
    report.dimensions = [ConfidentScore(**asdict(dimension), confidence=levels[dimension.name]) for dimension in report.dimensions]
    lowest = min(report.dimensions, key=lambda dimension: (dimension.score, dimension.confidence))
    weak = [dimension for dimension in (report.dimensions if every_dimension else [lowest]) if dimension.confidence < CONFIDENCE_FLOOR]
    if weak:
        raise LowConfidence(f"would have said {report.verdict}; confidence in {weak[0].name} is {weak[0].confidence}")
    return report


def ex3_confidence() -> None:
    def outcome(inputs: ReviewerInputs, every_dimension: bool = False) -> str:
        try:
            return f"ships {ship(inputs, every_dimension).verdict}"
        except LowConfidence as refusal:
            return f"refused ({refusal})"

    cases = {"clean close-out": closeout(),
             "forbidden write, gate report present": closeout(verdict={"passed": False, "findings": [{"code": "scope.forbidden"}]}),
             "right files, named unlike the goal": closeout(goal="fix login redirect loop",
                                                           diff_summary={"touched": ["auth/session.py", "tests/test_session.py"]}),
             "no gate report at all": closeout(verdict={})}
    outcomes = {label: outcome(inputs) for label, inputs in cases.items()}
    for label, result in outcomes.items():
        print(f"  {label:<40} {result}")
    strict = outcome(cases["no gate report at all"], every_dimension=True)
    print(f"  {'no gate report, every dimension checked':<40} {strict}")
    assert [result.split()[0] for result in outcomes.values()] == ["ships", "ships", "refused", "ships"]
    assert outcomes["no gate report at all"] == "ships pass" and strict.startswith("refused") and "scope_discipline" in strict


# ---------------------------------------------------------------------------
# Exercise 4 - a calibration set of ten close-outs
#
# Part one, the real record. The first batch of this repository, commit
# 90bedf6, closed out ten tasks: the practice files for lessons 01 to 10.
# The owner read them, and the next commit changed five and kept five. That
# is ten close-outs with a known verdict each: changed at the owner's request
# is a soft fail, kept as committed is a pass. The reviewer is given what the
# repository recorded: the commit's subject as the goal, the file, a compile
# check as the one command, and no state file or gate report, because the
# repository had neither.
#
# Measured: it agrees on 0 of 10. It hard-fails all ten on two zeros that say
# nothing about the work: problem_fit, because no word of the commit subject
# appears in a file path, and handoff_readiness, because there was no state
# file to read a next action from. With those two left out it passes all
# ten, which is 5 of 10, and it still cannot tell the five the owner changed
# from the five that were kept. What the owner asked for was a change of
# wording inside the files, and no dimension reads the files.
#
# So on a real record the reviewer's verdict is decided by what the workbench
# failed to record, and calibration cannot start until it records it.
#
# Part two, shapes the real record does not hold. SYNTHETIC: ten close-outs
# written by hand, each with the verdict a careful person would give. The
# agreement rate describes this set.
#
# Measured: the reviewer agrees on 5 of 10, well under the 80% the lesson
# asks for before a reviewer ships. Where it disagrees:
#   passes work that should fail (3)
#     - the gate blocked the task and a test failed: the reviewer never reads
#       the gate's "passed" flag, and one failing run only costs a point
#     - nothing was run: an empty feedback log scores 1, "mixed exit codes"
#     - the only file touched is a notes file named after the goal: the
#       keyword match gives it full marks for problem fit
#   fails work that should pass (2)
#     - the right files, named unlike the goal (login -> auth/session.py)
#     - a goal made of short words: no keywords, so problem fit is 0
# Two guards, the gate's flag and the empty log, bring it to 7 of 10. The
# three left are all problem_fit, and file names cannot settle them: that
# dimension needs the scope contract's file list, or judgement.
# ---------------------------------------------------------------------------

FIRST_BATCH, NEXT_BATCH = "90bedf6", "563dbed"


def real_closeouts() -> list[tuple[str, str, ReviewerInputs]] | None:
    """Lessons 01 to 10 as first committed: (lesson, verdict on record, what the reviewer is given)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, encoding="utf-8", check=True,
                              timeout=10).stdout

    try:
        goal = git("show", "-s", "--format=%s", FIRST_BATCH).strip()
        files = [path for path in git("show", "--name-only", "--format=", FIRST_BATCH).split() if path.endswith("practice.py")]
        revised = set(git("diff", "--name-only", FIRST_BATCH, NEXT_BATCH, "--", *files).split())
        sources = {path: git("show", f"{FIRST_BATCH}:{path}") for path in files}
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    cases = []
    for path in files:
        try:
            compile(sources[path], path, "exec")
            exit_code = 0
        except SyntaxError:
            exit_code = 1
        inputs = ReviewerInputs(task_id=path[:2], goal=goal, diff_summary={"touched": [path]}, state={},
                                feedback=[{"command": f"compile {path}", "exit_code": exit_code}], verdict={})
        cases.append((path.split("/")[0], "soft_fail" if path in revised else "pass", inputs))
    return cases


def calibration_set() -> list[tuple[str, str, ReviewerInputs]]:
    rate_limit = dict(goal="rate-limit the signup endpoint")
    return [
        ("clean close-out", "pass", closeout()),
        ("edited the release script", "hard_fail",
         closeout(diff_summary={"touched": ["app/signup.py", "scripts/release.sh"]},
                  verdict={"passed": False, "findings": [{"code": "scope.forbidden", "severity": "block"}]})),
        ("test run timed out", "hard_fail",
         closeout(feedback=[{"command": "pytest tests/test_signup.py", "exit_code": None}],
                  verdict={"passed": False, "findings": [{"code": "feedback.null_exit", "severity": "block"}]})),
        ("test failed, gate blocked", "hard_fail",
         closeout(feedback=[{"command": "pytest tests/test_signup.py", "exit_code": 1}],
                  verdict={"passed": False, "findings": [{"code": "acceptance.failed", "severity": "block"}]})),
        ("right files, named unlike the goal", "pass",
         closeout(goal="fix login redirect loop", diff_summary={"touched": ["auth/session.py", "tests/test_session.py"]},
                  feedback=[{"command": "pytest tests/test_session.py", "exit_code": 0}])),
        ("only a notes file, named after the goal", "hard_fail",
         closeout(diff_summary={"touched": ["notes/signup_validation_ideas.md"]}, feedback=[{"command": "pytest", "exit_code": 0}])),
        ("nothing was run, no gate report", "hard_fail", closeout(feedback=[], verdict={})),
        ("second clean close-out", "pass",
         closeout(**rate_limit, diff_summary={"touched": ["app/signup.py", "app/ratelimit.py", "tests/test_ratelimit.py"]},
                  feedback=[{"command": "pytest tests/test_ratelimit.py", "exit_code": 0}])),
        ("works, but sloppy", "soft_fail",
         closeout(**rate_limit, diff_summary={"touched": ["app/signup.py", "CHANGELOG.md"]},
                  state={"active_task_id": None, "assumptions": [], "next_action": "pick next task from board"},
                  feedback=[{"command": "pytest", "exit_code": 0}, {"command": "ruff check .", "exit_code": 1}],
                  verdict={"passed": True, "findings": [{"code": "scope.off_scope", "severity": "warn"}]})),
        ("goal made of short words", "pass",
         closeout(goal="fix auth bug", diff_summary={"touched": ["auth/token.py", "tests/test_token.py"]},
                  feedback=[{"command": "pytest tests/test_token.py", "exit_code": 0}])),
    ]


def review_guarded(inputs: ReviewerInputs) -> ReviewReport:
    report = review(inputs)
    if inputs.verdict.get("passed") is False or not inputs.feedback:
        report.verdict = "hard_fail"
    return report


def ex4_calibration_set() -> str | None:
    recorded = real_closeouts()
    if recorded is None:
        print("  needs git and this repository's history")
    else:
        said = {lesson: review(inputs) for lesson, _, inputs in recorded}
        for lesson, on_record, _ in recorded:
            zeros = [dimension.name for dimension in said[lesson].dimensions if dimension.score == 0]
            print(f"  {lesson:<36} on record {on_record:<9} reviewer {said[lesson].verdict:<9} zero on {zeros}")
        agree = sum(said[lesson].verdict == on_record for lesson, on_record, _ in recorded)
        about_the_work = [scorer for scorer in SCORERS if scorer.__name__ not in ("score_problem_fit", "score_handoff")]
        trimmed = [review_with(inputs, about_the_work).verdict for _, _, inputs in recorded]
        agree_trimmed = sum(verdict == on_record for verdict, (_, on_record, _) in zip(trimmed, recorded))
        revised = [lesson for lesson, on_record, _ in recorded if on_record == "soft_fail"]
        print(f"  the real record: agreement {agree}/10; without problem_fit and handoff_readiness {agree_trimmed}/10, "
              f"every verdict {sorted(set(trimmed))}")
        assert len(recorded) == 10 and len(revised) == 5 and agree == 0 and all(r.verdict == "hard_fail" for r in said.values())
        assert set(trimmed) == {"pass"} and agree_trimmed == 5

    print("  shapes the record does not hold, written by hand:")
    cases = calibration_set()
    wrong = {label: (correct, review(inputs).verdict) for label, correct, inputs in cases if review(inputs).verdict != correct}
    still_wrong = [label for label, correct, inputs in cases if review_guarded(inputs).verdict != correct]
    for label, correct, inputs in cases:
        said = review(inputs).verdict
        print(f"  {label:<40} correct {correct:<9} reviewer {said:<9} {'' if said == correct else 'DISAGREES'}")
    print(f"  agreement {len(cases) - len(wrong)}/{len(cases)}; with the two guards {len(cases) - len(still_wrong)}/{len(cases)}")
    assert len(cases) == 10 and len(wrong) == 5 and len(still_wrong) == 3
    assert sorted(said for _, said in wrong.values()) == ["hard_fail", "hard_fail", "pass", "pass", "pass"]
    assert still_wrong == ["right files, named unlike the goal", "only a notes file, named after the goal", "goal made of short words"]
    return None if recorded else "git"


# ---------------------------------------------------------------------------
# Exercise 5 - "request more evidence", and the back-off that stops a loop
#
# Before scoring, the reviewer may ask the builder for one specific test run:
# a named command, for a test file in the diff that has no completed run on
# record. The builder answers with a feedback record, and the review goes on.
#
# The right back-off is a budget, not a delay. Waiting longer between asks
# does nothing for a builder that cannot produce the run. Three rules bound
# the exchange:
#   - a request names one command, so "more tests" is never a request;
#   - the same command is never asked for twice;
#   - at most two requests per review.
# When the rules run out and evidence is still missing, the reviewer does not
# score. It returns needs_human, which is neither a pass nor a fail.
# ---------------------------------------------------------------------------

MAX_REQUESTS = 2


def wanted_evidence(inputs: ReviewerInputs) -> str | None:
    """The first test command for the diff that has no completed run on record."""
    completed = {str(record.get("command")) for record in inputs.feedback if record.get("exit_code") is not None}
    tests = [path for path in inputs.diff_summary.get("touched", []) if Path(path).name.startswith("test_")]
    return next((command for command in [f"pytest {path}" for path in tests] or ["pytest"] if command not in completed), None)


def review_with_evidence(inputs: ReviewerInputs, builder: Callable[[str], dict[str, object] | None]) -> tuple[str, list[str]]:
    asked: list[str] = []
    while (command := wanted_evidence(inputs)) and command not in asked and len(asked) < MAX_REQUESTS:
        asked.append(command)
        record = builder(command)
        if record is None:
            break
        inputs = replace(inputs, feedback=[*inputs.feedback, record])
    return ("needs_human" if wanted_evidence(inputs) else review(inputs).verdict), asked


def ex5_request_more_evidence() -> None:
    unverified = closeout(feedback=[])
    three_tests = closeout(feedback=[], diff_summary={"touched": ["app/signup.py", "tests/test_signup.py", "tests/test_login.py",
                                                                  "tests/test_reset.py"]})
    builders: dict[str, Callable[[str], dict[str, object] | None]] = {      # SCRIPTED builders
        "runs what is asked": lambda command: {"command": command, "exit_code": 0},
        "the run times out": lambda command: {"command": command, "exit_code": None},
        "answers with a different run": lambda command: {"command": "pytest -k smoke", "exit_code": 0},
        "does not answer": lambda command: None,
    }
    results = {label: review_with_evidence(unverified, builder) for label, builder in builders.items()}
    results["three test files, runs what is asked"] = review_with_evidence(three_tests, builders["runs what is asked"])
    for label, (verdict, asked) in results.items():
        print(f"  {label:<37} {verdict:<11} after {len(asked)} request(s) {asked}")
    assert results["runs what is asked"] == ("pass", ["pytest tests/test_signup.py"])
    assert all(results[label] == ("needs_human", ["pytest tests/test_signup.py"]) for label in list(builders)[1:])
    assert results["three test files, runs what is asked"][0] == "needs_human" and len(results["three test files, runs what is asked"][1]) == 2
    assert review_with_evidence(closeout(), builders["does not answer"]) == ("pass", [])       # enough evidence: nothing asked


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder.
#   exits zero                               yes
#   the clean change scores at least 7,      yes: 9, pass
#   verdict pass
#   the wrong-problem change drops below 5   hard_fail, yes. Its total is
#   and the verdict flips to hard_fail       exactly 5; what flips the verdict
#                                            is the zero on problem_fit
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


def mission_acceptance() -> None:
    (exit_code, _, files), = run_main()
    clean, wrong = (json.loads(files[f"code/review_report_T-00{number}.json"]) for number in (1, 2))
    zeros = [dimension["name"] for dimension in wrong["dimensions"] if dimension["score"] == 0]
    print(f"  main.py exits {exit_code}; clean change {clean['total']}/10 {clean['verdict']}; "
          f"wrong-problem change {wrong['total']}/10 {wrong['verdict']}, zero on {zeros}")
    assert exit_code == 0 and (clean["total"], clean["verdict"]) == (9, "pass")
    assert (wrong["total"], wrong["verdict"], zeros) == (5, "hard_fail", ["problem_fit"])


if __name__ == "__main__":
    print("Phase 14 - Lesson 39: Reviewer Agent and Rubric Scoring - exercises")
    missing = []
    for exercise in (ex1_sixth_dimension, ex2_terse_and_verbose, ex3_confidence, ex4_calibration_set, ex5_request_more_evidence, mission_acceptance):
        print(f"\n{exercise.__name__}")
        missing.append(exercise())
    missing = [name for name in missing if name]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
