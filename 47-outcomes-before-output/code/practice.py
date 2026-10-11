"""Phase 14 - Lesson 47: Define the Outcome Before You Choose the Output - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The feature request is the one these practice files came from: "add a practice.py to every lesson
folder". Every fact used to frame it is read from this repository (needs git on the PATH).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from main import OutcomeFrame, decision, validate

REPO = Path(__file__).resolve().parents[2]
COURSE, FIRST, LATEST = "6a735e9", "90bedf6", "a08499a"   # the course as published; the first commit of practice files; the last one read here


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


def lessons() -> list[Path]:
    return sorted(path for path in REPO.glob("[0-9][0-9]-*") if path.is_dir())


def exercises_asked(lesson: Path) -> int:
    """How many numbered exercises end that lesson's notes."""
    section = (lesson / "docs" / "en.md").read_text(encoding="utf-8").split("## Exercises", 1)[1].split("\n## ", 1)[0]
    return len(re.findall(r"^\d+\. ", section, re.M))


# ---------------------------------------------------------------------------
# Exercise 1 - a feature request rewritten as an outcome frame
#
# The request names a thing to make. Copied into the frame it fails three ways:
# the outcome is the output, and nothing says what must stay true or what is
# left out. The rewrite asks who is helped and what they can do afterwards.
# main.validate only catches the output's exact name, so "a script with the
# solutions" gets through; leaks() looks for any word that names a product form.
# ---------------------------------------------------------------------------

REQUEST = "Add a practice.py to every lesson folder that solves the lesson's exercises"
FORMS = ("file", "script", "notebook", "dashboard", "page", "badge", "report", "suite", "tool")   # words that name a product form


def as_requested() -> OutcomeFrame:
    """The request copied into the frame as it was given."""
    return OutcomeFrame(
        user="the owner of this repository", situation="working through the course",
        current_behavior="a lesson folder holds the course's main.py and its notes",
        desired_outcome="every lesson folder has a practice.py that solves the lesson's exercises",
        constraints=[], non_goals=[], proposed_output="practice.py",
    )


def framed() -> OutcomeFrame:
    return OutcomeFrame(
        user="someone checking whether this course was worked through",
        situation="they open one lesson folder with a few minutes and a Python interpreter",
        current_behavior="the folder holds the course's own main.py and its notes; nothing in it shows whether the exercises that end the notes were attempted",
        desired_outcome="within a minute, and without taking the owner's word for it, they can tell which exercises of that lesson were solved and that the solutions work",
        constraints=[UNCHANGED],
        non_goals=["no rewrite of the lesson: the notes stay the explanation"],
        proposed_output="practice.py",
    )


def leaks(frame: OutcomeFrame) -> list[str]:
    """Product forms named in the desired outcome."""
    # ponytail: a word list, not a parser; it flags the sentence and a person decides
    return [form for form in FORMS if re.search(rf"\b{form}", frame.desired_outcome.lower())]


def ex1_outcome_frame() -> None:
    published = git("ls-tree", "-r", "--name-only", COURSE)
    asked = sum(exercises_asked(lesson) for lesson in lessons())
    print(f"  the course as published: {sum(name.endswith('code/main.py') for name in published)} lessons, {asked} exercises asked, "
          f"{sum('practice' in name for name in published)} files that answer one")
    print(f"  request: {REQUEST}")
    raw = decision(as_requested())
    print(f"  copied into the frame -> {raw['status']}: {'; '.join(raw['issues'])}")
    reworded = replace(framed(), desired_outcome="every lesson has a script that holds the solutions")
    print(f"  reworded -> main.validate finds {validate(reworded)}, leaks() finds {leaks(reworded)}")
    frame = framed()
    for part in ("user", "situation", "current_behavior", "desired_outcome"):
        print(f"  {part:<16} {getattr(frame, part)}")
    print(f"  -> {decision(frame)['status']}")
    assert raw["status"] == "needs-framing" and len(raw["issues"]) == 3 and "desired outcome names the proposed output" in raw["issues"]
    assert validate(reworded) == [] and leaks(reworded) == ["script"]
    assert decision(frame)["status"] == "ready-to-discover" and leaks(frame) == []
    assert asked == 272 and not any("practice" in name for name in published)


# ---------------------------------------------------------------------------
# Exercise 2 - one constraint that changes which solutions remain
#
# "main.py stays as published" removes one candidate. Adding "works offline on
# a plain Python install" removes three more: anything hosted and anything that
# needs a package. The repository is then checked against the new constraint:
# no practice file imports a package at module level, but one course main.py
# does, so in that lesson the two constraints cannot both hold. The older one
# wins: main.py is left alone and that lesson's file names the package it needs.
# ---------------------------------------------------------------------------

UNCHANGED = "the course's main.py files stay exactly as published"
OFFLINE = "it works offline on a plain Python install: nothing to download, no account, no key"


@dataclass(frozen=True)
class Output:
    name: str
    runs_code: bool = True         # the reader watches the solutions execute, instead of reading that they did
    edits_main: bool = False
    needs_network: bool = False
    needs_install: bool = False


CANDIDATES = [
    Output("a practice.py beside each main.py"),
    Output("a test module per lesson under code/tests, run with the unittest command the notes already use"),
    Output("a SOLUTIONS.md per lesson whose examples run under python -m doctest"),
    Output("answers written in prose under each exercise", runs_code=False),
    Output("the solutions added to main.py behind a flag", edits_main=True),
    Output("a pytest suite per lesson", needs_install=True),
    Output("a hosted notebook per lesson", needs_network=True),
    Output("a pass badge from a hosted build", runs_code=False, needs_network=True),
]
ALLOWS = {UNCHANGED: lambda output: not output.edits_main,
          OFFLINE: lambda output: not (output.needs_network or output.needs_install)}


def remaining(frame: OutcomeFrame) -> list[Output]:
    """The candidates every constraint of the frame allows."""
    return [output for output in CANDIDATES if all(ALLOWS[constraint](output) for constraint in frame.constraints if constraint in ALLOWS)]


def with_constraint() -> OutcomeFrame:
    return replace(framed(), constraints=[UNCHANGED, OFFLINE])


def module_level_packages(file: Path) -> list[str]:
    """Third-party packages the file imports unconditionally."""
    found = set()
    for node in ast.parse(file.read_text(encoding="utf-8")).body:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
        found |= {name.split(".")[0] for name in names}
    return sorted(found - set(sys.stdlib_module_names) - {path.stem for path in file.parent.glob("*.py")} - {""})


def plain_python(file: Path) -> subprocess.CompletedProcess:
    """Run a file with no installed packages visible (-S) and no PYTHON* variables (-E)."""
    return subprocess.run([sys.executable, "-S", "-E", str(file)], capture_output=True, text=True, timeout=120)


def ex2_constraint_that_narrows() -> None:
    before, after = remaining(framed()), remaining(with_constraint())
    print(f"  candidates: {len(CANDIDATES)}; allowed with one constraint: {len(before)}; with the offline constraint: {len(after)}")
    for output in before:
        print(f"    {'kept   ' if output in after else 'removed'} {output.name}")
    practice = [lesson / "code" / "practice.py" for lesson in lessons() if (lesson / "code" / "practice.py").is_file()]
    needy = {lesson.name: packages for lesson in lessons() if (packages := module_level_packages(lesson / "code" / "main.py"))}
    stock = plain_python(REPO / "15-crewai-role-based-crews" / "code" / "practice.py")
    print(f"  practice files that import a package at module level: {sum(bool(module_level_packages(file)) for file in practice)} of {len(practice)}")
    print(f"  course main.py files that do: {needy}")
    print(f"  lesson 15 on a plain install: exit {stock.returncode}, '{stock.stderr.strip().splitlines()[-1]}'")
    assert len(CANDIDATES) - len(before) == 1 and len(before) - len(after) == 3
    assert not any(module_level_packages(file) for file in practice)
    assert needy == {"15-crewai-role-based-crews": ["numpy"]} and stock.returncode == 1 and "needs numpy" in stock.stderr


# ---------------------------------------------------------------------------
# Exercise 3 - two non-goals that keep the first slice small
#
# A non-goal earns its place when it can turn work away, so each one is a test
# on the path a proposal would add. Both are then run over every file the
# practice commits really added: none is turned away, and the first commit
# covered ten lessons, not all of them.
# ---------------------------------------------------------------------------

STANDS_ALONE = "no shared package across lessons: each lesson's file stands on its own"
NO_QUIZ = "no answers to the quiz.json files"
TURNS_AWAY = {STANDS_ALONE: lambda path: path.endswith(".py") and not re.match(r"\d\d-", path),
              NO_QUIZ: lambda path: "quiz" in path.rsplit("/", 1)[-1]}
PROPOSALS = {   # tempting adjacent work -> a file it would add
    "pull the repeated helpers into one package": "common/helpers.py",
    "one runner at the top that executes every lesson": "run_all.py",
    "answer each lesson's quiz as well": "16-openai-agents-sdk/quiz-answers.json",
    "solve the exercises of lesson 16": "16-openai-agents-sdk/code/practice.py",
}


def final_frame() -> OutcomeFrame:
    frame = with_constraint()
    return replace(frame, non_goals=[*frame.non_goals, STANDS_ALONE, NO_QUIZ])


def turned_away_by(path: str, frame: OutcomeFrame) -> list[str]:
    return [goal for goal in frame.non_goals if goal in TURNS_AWAY and TURNS_AWAY[goal](path)]


def ex3_non_goals() -> None:
    frame = final_frame()
    verdicts = {proposal: turned_away_by(path, frame) for proposal, path in PROPOSALS.items()}
    for proposal, goals in verdicts.items():
        print(f"  {proposal:<50} -> {'turned away: ' + goals[0].split(':')[0] if goals else 'in the slice'}")
    added = [name for name in git("log", "--diff-filter=A", "--name-only", "--format=", f"{COURSE}..{LATEST}") if name]
    first = [name for name in git("show", "--diff-filter=A", "--name-only", "--format=", FIRST) if name.endswith("practice.py")]
    outside = [name for name in git("ls-files") if name.endswith(".py") and not re.match(r"\d\d-", name)]
    print(f"  files added from {FIRST} to {LATEST}: {len(added)}; turned away by a non-goal: {sum(bool(turned_away_by(name, frame)) for name in added)}")
    print(f"  the first commit covered {len(first)} of {len(lessons())} lessons; Python files outside a lesson folder today: {len(outside)}")
    assert [bool(goals) for goals in verdicts.values()] == [True, True, True, False]
    assert added and not any(turned_away_by(name, frame) for name in added) and len(first) == 10 and not outside
    assert decision(frame)["status"] == "ready-to-discover" and len(frame.non_goals) == 3


# ---------------------------------------------------------------------------
# Exercise 4 - the earliest observation that would disprove the outcome
#
# Each observation is something that would be seen if the outcome were false.
# The earliest is the one that needs the fewest lessons written and the least
# effort to look: run the first file on an interpreter with nothing installed.
# It is run here. It could have failed three ways (a crash, a slow run, an
# exercise with no named answer) and none happened. The second-earliest needs
# a reader who did not write the file; a script cannot make that observation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Observation:
    seen_if_false: str
    lessons_needed: int       # how many lessons must be written before it can be looked for
    minutes: float            # effort to look
    needs: str


def observations() -> list[Observation]:
    return [
        Observation("after every lesson is written, nobody but the owner has opened one", len(lessons()), 0, "time"),
        Observation("a reader who did not write the file cannot say which exercises it solves", 1, 5, "one reader"),
        Observation("the first file crashes, runs past a minute or leaves an exercise unnamed on a plain install", 1, 0.1, "one command"),
    ]


def ex4_earliest_disproof() -> None:
    ordered = sorted(observations(), key=lambda item: (item.lessons_needed, item.minutes))
    for item in ordered:
        print(f"  after {item.lessons_needed:>2} lesson(s), {item.minutes:>3} min, needs {item.needs:<11} {item.seen_if_false}")
    first_lesson = lessons()[0]
    started = time.perf_counter()
    run = plain_python(first_lesson / "code" / "practice.py")
    seconds = time.perf_counter() - started
    named = re.findall(r"^ex\d+_\w+", run.stdout, re.M)
    print(f"  made now on {first_lesson.name}: exit {run.returncode}, {seconds:.1f} s, {len(named)} of {exercises_asked(first_lesson)} exercises named in the output")
    assert ordered[0].needs == "one command" and ordered[0].lessons_needed == 1 and ordered[-1].lessons_needed == len(lessons())
    assert run.returncode == 0 and seconds < 60 and len(named) == exercises_asked(first_lesson) == 5


# ---------------------------------------------------------------------------
# Exercise 5 - three different outputs for the same outcome
#
# Three of the candidates survive both constraints and let the reader watch
# the code run. Each goes into the same frame as the proposed output and the
# frame does not change: same user, same outcome, same limits. The request as
# given could only ever have produced the first.
# ---------------------------------------------------------------------------

def ex5_three_outputs() -> None:
    frame = final_frame()
    three = [output for output in remaining(frame) if output.runs_code]
    frames = [replace(frame, proposed_output=output.name) for output in three]
    for candidate in frames:
        print(f"  {decision(candidate)['status']:<18} {candidate.proposed_output}")
    course_tests = len(list(REPO.glob("[0-9][0-9]-*/code/tests")))
    print(f"  lessons where the course already ships a code/tests folder for the second: {course_tests}")
    print(f"  outcome, unchanged across all three: {frame.desired_outcome}")
    assert len(three) == 3 and all(decision(candidate)["status"] == "ready-to-discover" and not leaks(candidate) for candidate in frames)
    assert len({tuple(sorted((key, str(value)) for key, value in asdict(candidate).items() if key != "proposed_output")) for candidate in frames}) == 1
    assert "practice.py" in REQUEST and sum("practice.py" in output.name for output in three) == 1 and course_tests == 12


if __name__ == "__main__":
    print("Phase 14 - Lesson 47: Define the Outcome Before You Choose the Output - exercises")
    for exercise in (ex1_outcome_frame, ex2_constraint_that_narrows, ex3_non_goals, ex4_earliest_disproof, ex5_three_outputs):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
