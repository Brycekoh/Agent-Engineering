"""Phase 14 - Lesson 52: Design Success Metrics Before the Result Exists - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The goal is lesson 47's outcome for the practice files. The plan and the decision rule are
written first, at the top of this file; the values are then measured on this repository
(needs git on the PATH).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import median

from main import MeasurementPlan, Metric, report, validate

REPO = Path(__file__).resolve().parents[2]
SAMPLE = ("01", "11", "21", "31", "41")       # every tenth lesson: the files timed for the run-time metric
SWITCH = r'os\.environ\.get\("[A-Z_]+(_KEY|_ENDPOINT)"'


@dataclass(frozen=True)
class Scoped(Metric):
    population: str = ""
    question: int = 0             # which of the plan's questions it answers


# ---------------------------------------------------------------------------
# The plan and the rule, fixed before anything below is measured.
# ---------------------------------------------------------------------------

GOAL = "A reader can confirm that a lesson's exercises were solved, without the check costing them setup, money or trust"
QUESTIONS = ["Does one run account for every exercise the lesson asks?",
             "Does that run finish in the time a reader has, with nothing to set up first?",
             "Does a run stay harmless: nothing sent anywhere, nothing left behind?"]
EVERY_FILE, EVERY_LESSON = "every practice file in the working tree", "lessons that have a practice file"
NAMED = Scoped("exercises_named_share", "at-least", 1.0, EVERY_FILE,
               "count of exN_ functions against the numbered exercises in docs/en.md", "outcome", EVERY_LESSON, 0)
SECONDS = Scoped("median_plain_run_seconds", "at-most", 60, "one run each of lessons " + ", ".join(SAMPLE),
                 "wall clock around python -S -E practice.py", "outcome", "a reader with a plain Python install", 1)
SETUP = Scoped("reader_setup_share", "at-most", 0.05, EVERY_FILE,
               "exercise functions that can end by naming something to install, over all exercise functions", "counter", "every exercise", 1)
UNASKED = Scoped("files_calling_out_unasked", "at-most", 0, EVERY_FILE,
                 "text scan: a key or endpoint read from the environment in a file with no --live flag", "guardrail", EVERY_LESSON, 2)
LEFT_BEHIND = Scoped("files_changed_by_a_run", "at-most", 0, "the same runs as the run-time metric",
                     "git status before and after the runs", "guardrail", "this repository's working tree", 2)
LINES = Scoped("practice_lines_total", "at-least", 10000, EVERY_FILE, "line count of every practice file", "activity", EVERY_LESSON, 0)

PASS = "pass: go on to the next batch of lessons"
FAIL = "fail: stop adding lessons and fix what broke"
AMBIGUOUS = "ambiguous: measure the doubtful metric directly on a plain install before deciding"
FLOOR = 0.9                       # below this share of exercises accounted for, it is a fail and not a doubt


def plan(*metrics: Scoped) -> MeasurementPlan:
    return MeasurementPlan(GOAL, list(QUESTIONS), list(metrics or (NAMED, SECONDS, SETUP, UNASKED, LEFT_BEHIND)))


def decide(measured: dict) -> str:
    """Pass, fail or ambiguous for a main.report(): the rule reads outcome, counter and guardrail metrics."""
    kinds = {metric["name"]: metric["kind"] for metric in measured["plan"]["metrics"]}
    results = {entry["name"]: entry for entry in measured["results"] if kinds[entry["name"]] in {"outcome", "counter", "guardrail"}}
    if any(entry.get("passed") is False for name, entry in results.items() if kinds[name] == "guardrail"):
        return FAIL
    if results.get(NAMED.name, {}).get("value", 1) < FLOOR:
        return FAIL
    return PASS if measured["status"] == "valid" and all(entry.get("passed") for entry in results.values()) else AMBIGUOUS


# ---------------------------------------------------------------------------
# Measurement. Nothing above this line reads the repository.
# ---------------------------------------------------------------------------

def practice_files() -> list[Path]:
    return sorted(REPO.glob("[0-9][0-9]-*/code/practice.py"))


def exercise_functions(file: Path) -> list[ast.FunctionDef]:
    return [node for node in ast.parse(file.read_text(encoding="utf-8")).body if isinstance(node, ast.FunctionDef) and re.match(r"ex\d+_", node.name)]


def exercises_asked(lesson: Path) -> int:
    section = (lesson / "docs" / "en.md").read_text(encoding="utf-8").split("## Exercises", 1)[1].split("\n## ", 1)[0]
    return len(re.findall(r"^\d+\. ", section, re.M))


def status() -> list[str]:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


@lru_cache(maxsize=None)
def measure() -> dict[str, float]:
    files = practice_files()
    functions = [(file, exercise_functions(file)) for file in files]
    sources = {file: file.read_text(encoding="utf-8") for file in files}
    before, seconds = status(), []
    for lesson in SAMPLE:
        started = time.perf_counter()
        run = subprocess.run([sys.executable, "-S", "-E", str(next(REPO.glob(f"{lesson}-*/code/practice.py")))], capture_output=True, timeout=120)
        seconds.append(time.perf_counter() - started if run.returncode == 0 else float("inf"))
    return {
        NAMED.name: sum(len(found) == exercises_asked(file.parents[1]) for file, found in functions) / len(files),
        SECONDS.name: round(median(seconds), 2),
        SETUP.name: round(sum(node.returns is not None and "str" in ast.unparse(node.returns) for _, found in functions for node in found)
                          / sum(len(found) for _, found in functions), 3),
        UNASKED.name: sum(bool(re.search(SWITCH, text)) and "--live" not in text for text in sources.values()),
        LEFT_BEHIND.name: len(set(before) ^ set(status())),
        LINES.name: sum(len(text.splitlines()) for text in sources.values()),
    }


# ---------------------------------------------------------------------------
# Exercise 1 - three questions derived from one outcome goal
#
# The goal has three clauses, and each becomes a question a number can
# answer: was everything accounted for, was it quick and free of setup, and
# was it harmless. Every metric in the plan answers exactly one of them, and
# no question is left without a metric.
# ---------------------------------------------------------------------------

def ex1_three_questions() -> None:
    drafted = plan()
    print(f"  goal: {GOAL}")
    for number, question in enumerate(QUESTIONS):
        answering = [metric.name for metric in drafted.metrics if metric.question == number]
        print(f"  Q{number + 1} {question}\n       answered by {', '.join(answering)}")
    assert len(QUESTIONS) == 3 and validate(drafted) == []
    assert all(any(metric.question == number for metric in drafted.metrics) for number in range(3))
    assert validate(MeasurementPlan(GOAL, [], [NAMED])) == ["questions are empty", "guardrail metric is missing"]


# ---------------------------------------------------------------------------
# Exercise 2 - a counter-metric that catches cost shifted to another role
#
# "Every exercise is accounted for" can be reached by leaning on packages the
# author has installed. The author's work is done and the reader inherits the
# setup. reader_setup_share counts the exercises that can end by naming
# something to install. Without it the same measurements read as a pass; with
# it they do not, and the number says how much was shifted.
# ---------------------------------------------------------------------------

def ex2_counter_metric() -> None:
    values = measure()
    without = decide(report(plan(NAMED, SECONDS, UNASKED, LEFT_BEHIND), values))
    with_it = decide(report(plan(), values))
    print(f"  exercises accounted for: {values[NAMED.name]:.0%}; exercises that can leave setup to the reader: {values[SETUP.name]:.1%} (limit {SETUP.threshold:.0%})")
    print(f"  without the counter-metric -> {without.split(':')[0]}; with it -> {with_it.split(':')[0]}")
    assert SETUP.kind == "counter" and values[NAMED.name] == 1.0 and values[SETUP.name] > SETUP.threshold
    assert without == PASS and with_it == AMBIGUOUS


# ---------------------------------------------------------------------------
# Exercise 3 - source, population and window for every metric
#
# main.Metric has a source and a window. Population is added here, because
# "share of what" changes the number: the setup share is over exercises, the
# others over files or lessons. A metric with any of the three missing cannot
# be measured the same way twice, so the check refuses it.
# ---------------------------------------------------------------------------

def unscoped(drafted: MeasurementPlan) -> list[str]:
    return [metric.name for metric in drafted.metrics if not (metric.source.strip() and metric.window.strip() and getattr(metric, "population", "").strip())]


def ex3_source_population_window() -> None:
    drafted = plan()
    for metric in drafted.metrics:
        print(f"  {metric.name:<26} {metric.direction} {metric.threshold:g}  [{metric.kind}]")
        print(f"       source: {metric.source}\n       population: {metric.population}; window: {metric.window}")
    loose = plan(NAMED, replace(SECONDS, window=" "), replace(SETUP, population=""), UNASKED)
    print(f"  a draft with two fields blank: main.validate finds {validate(loose)}; the population check finds {unscoped(loose)}")
    assert unscoped(drafted) == [] and validate(drafted) == []
    assert validate(loose) == [f"{SECONDS.name} lacks source or window"] and unscoped(loose) == [SECONDS.name, SETUP.name]


# ---------------------------------------------------------------------------
# Exercise 4 - pass, fail and ambiguous, written before the values
#
# The rule is decide(), at the top of the file with the plan. Each path is
# first walked with made-up values to show it can be reached. Then the real
# values go in. The result is ambiguous: every outcome and guardrail holds and
# the counter-metric is over its limit, so the next step is the one written
# down in advance and not a new threshold.
# ---------------------------------------------------------------------------

def ex4_decisions_before_values() -> None:
    clean = {NAMED.name: 1.0, SECONDS.name: 5, SETUP.name: 0.0, UNASKED.name: 0, LEFT_BEHIND.name: 0}
    walked = {
        "all inside": decide(report(plan(), clean)),
        "one file calls out": decide(report(plan(), clean | {UNASKED.name: 1})),
        "four in five accounted for": decide(report(plan(), clean | {NAMED.name: 0.8})),
        "slow runs": decide(report(plan(), clean | {SECONDS.name: 90})),
        "a value missing": decide(report(plan(), {name: value for name, value in clean.items() if name != SECONDS.name})),
    }
    for case, outcome in walked.items():
        print(f"  made-up: {case:<27} -> {outcome}")
    measured = report(plan(), measure())
    for entry in measured["results"]:
        print(f"  measured: {entry['name']:<26} {entry['value']:<6g} {'inside' if entry['passed'] else 'OUTSIDE'}")
    print(f"  -> {decide(measured)}")
    assert list(walked.values()) == [PASS, FAIL, FAIL, AMBIGUOUS, AMBIGUOUS]
    assert measured["status"] == "valid" and decide(measured) == AMBIGUOUS
    assert [entry["name"] for entry in measured["results"] if not entry["passed"]] == [SETUP.name]


# ---------------------------------------------------------------------------
# Exercise 5 - a metric that is easy to collect and cannot change the decision
#
# practice_lines_total is one line of code to collect and grows every batch.
# Pushing it far below or above its threshold leaves the decision where it
# was, while doing the same to any other metric moves it. The repository shows
# why: the shortest and the longest file differ by more than two times in
# length and by nothing in exercises accounted for. It is removed.
# ---------------------------------------------------------------------------

def moves_the_decision(drafted: MeasurementPlan, values: dict[str, float], metric: Metric) -> bool:
    """Whether some value of this one metric changes the decision, the others held where they are."""
    extremes = (metric.threshold - 1e9, metric.threshold + 1e9)
    return len({decide(report(drafted, values | {metric.name: value})) for value in extremes}) > 1


def ex5_remove_a_metric_that_decides_nothing() -> None:
    drafted, values = plan(NAMED, SECONDS, SETUP, UNASKED, LEFT_BEHIND, LINES), measure()
    moving = {metric.name: moves_the_decision(drafted, values | {SETUP.name: 0.0}, metric) for metric in drafted.metrics}
    for name, moved in moving.items():
        print(f"  {name:<26} measured {values[name]:<8g} can change the decision: {moved}")
    lengths = sorted((len(file.read_text(encoding='utf-8').splitlines()), len(exercise_functions(file)) == exercises_asked(file.parents[1])) for file in practice_files())
    print(f"  shortest file {lengths[0][0]} lines, longest {lengths[-1][0]}; every exercise accounted for in both: {lengths[0][1] and lengths[-1][1]}")
    kept = plan()
    assert [name for name, moved in moving.items() if not moved] == [LINES.name]
    assert LINES not in kept.metrics and validate(kept) == [] and decide(report(kept, values)) == decide(report(drafted, values))
    assert lengths[-1][0] > 2 * lengths[0][0] and all(full for _, full in lengths)


if __name__ == "__main__":
    print("Phase 14 - Lesson 52: Design Success Metrics Before the Result Exists - exercises")
    for exercise in (ex1_three_questions, ex2_counter_metric, ex3_source_population_window, ex4_decisions_before_values,
                     ex5_remove_a_metric_that_decides_nothing):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
