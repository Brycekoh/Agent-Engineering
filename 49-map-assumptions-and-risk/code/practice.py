"""Phase 14 - Lesson 49: Map Assumptions and Resolve the Riskiest One First - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The feature is the one lesson 47 framed: practice files that let a reader confirm the course was
worked through. Every assumption a script can test is tested here against this repository
(needs git on the PATH).
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from main import Assumption, next_experiment, prioritize, risk_score

REPO = Path(__file__).resolve().parents[2]
COURSE, LATEST = "6a735e9", "a08499a"     # the course as published; the last commit before this lesson's findings


def lessons() -> list[Path]:
    return sorted(path for path in REPO.glob("[0-9][0-9]-*") if path.is_dir())


def practice_files() -> list[Path]:
    return [lesson / "code" / "practice.py" for lesson in lessons() if (lesson / "code" / "practice.py").is_file()]


def exercises_asked(lesson: Path) -> int:
    section = (lesson / "docs" / "en.md").read_text(encoding="utf-8").split("## Exercises", 1)[1].split("\n## ", 1)[0]
    return len(re.findall(r"^\d+\. ", section, re.M))


@lru_cache(maxsize=None)
def exercise_functions(file: Path) -> tuple[ast.FunctionDef, ...]:
    return tuple(node for node in ast.parse(file.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.FunctionDef) and re.match(r"ex\d+_", node.name))


def module_level_packages(file: Path) -> list[str]:
    """Third-party packages the file imports unconditionally."""
    found = set()
    for node in ast.parse(file.read_text(encoding="utf-8")).body:
        names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
        found |= {name.split(".")[0] for name in names}
    return sorted(found - set(sys.stdlib_module_names) - {path.stem for path in file.parent.glob("*.py")} - {""})


def cannot_start_plain() -> list[str]:
    """Lessons where main.py or the practice file imports a package at module level."""
    return [lesson.name[:2] for lesson in lessons()
            if any(module_level_packages(file) for file in (lesson / "code" / "main.py", lesson / "code" / "practice.py") if file.is_file())]


@dataclass(frozen=True)
class Bet(Assumption):
    kind: str = ""            # value, usability, feasibility, viability or safety


# ---------------------------------------------------------------------------
# Exercise 1 - five assumptions for the feature
#
# Each is a sentence that could turn out false, with the test that would show
# it. Three can be tested by reading the repository, so they are, and the
# evidence line is what the test found (one of them is false for one lesson,
# one is false for three exercises). Two stay open, and the riskiest of all is
# one of those: nothing here shows that anyone but the owner has run a file.
# ---------------------------------------------------------------------------

READER = "Someone other than the owner opens a lesson and runs its file"
NAMED = "One run tells a reader which of the lesson's exercises were solved"
CHECKS = "Every exercise, even one that asks for an essay or an interview, can be answered by code that checks its own answer"
PLAIN = "Every file starts on a plain Python install"
LASTS = "The files keep passing when the course updates a main.py"


def named_in_full() -> tuple[int, int]:
    """(files with one exercise function per exercise the notes ask, files)."""
    files = practice_files()
    return sum(len(exercise_functions(file)) == exercises_asked(file.parents[1]) for file in files), len(files)


def self_checking() -> tuple[int, list[str]]:
    """(exercise functions in all, the ones that contain no assert)."""
    functions = [(file.parts[-3][:2], node) for file in practice_files() for node in exercise_functions(file)]
    return len(functions), [f"{lesson} {node.name}" for lesson, node in functions if not any(isinstance(inner, ast.Assert) for inner in ast.walk(node))]


def assumptions() -> list[Bet]:
    full, files = named_in_full()
    total, bare = self_checking()
    blocked = cannot_start_plain()
    return [
        Bet(READER, 5, 5, 2, "Hand one lesson folder to three people with no explanation and watch for five minutes", kind="value"),
        Bet(NAMED, 4, 2, 1, "Count the exercise functions in each file against the exercises its notes ask",
            f"{full} of {files} files have one named function per exercise asked", kind="usability"),
        Bet(CHECKS, 4, 3, 1, "Look for an assert inside every exercise function",
            f"{total - len(bare)} of {total} do; {len(bare)} ports to a framework only build an object and print it", kind="feasibility"),
        Bet(PLAIN, 3, 3, 1, "Start every file with no installed packages visible",
            f"false for {len(blocked)} of {len(lessons())} lessons: {', '.join(blocked)} (its main.py imports a package)", kind="feasibility"),
        Bet(LASTS, 3, 4, 2, "Replay every file against the course's next change to a main.py", kind="viability"),
    ]


def show(items: list[Assumption]) -> list[dict]:
    ranked = prioritize(items)
    for entry in ranked:
        print(f"  {entry['risk_score']:>2}  {entry['kind']:<11} {entry['status']:<6} {entry['statement']}")
        if entry["evidence"]:
            print(f"      {'':<18} {entry['evidence']}")
    return ranked


def ex1_five_assumptions() -> None:
    items = assumptions()
    ranked = show(items)
    riskiest = next_experiment(items)
    print(f"  next experiment: {riskiest.test}")
    total, bare = self_checking()
    assert [entry["statement"] for entry in ranked] == [READER, LASTS, CHECKS, PLAIN, NAMED] and {item.kind for item in items} == {"value", "usability", "feasibility", "viability"}
    assert riskiest.statement == READER and [entry["status"] for entry in ranked].count("open") == 2
    assert named_in_full()[0] == named_in_full()[1] and cannot_start_plain() == ["15"]
    assert bare == ["15 ex6_real_crewai", "16 ex5_sdk_port", "17 ex4_sdk_port"] and total == sum(exercises_asked(file.parents[1]) for file in practice_files())


# ---------------------------------------------------------------------------
# Exercise 2 - the safety assumption the list left out
#
# All five are about whether the files are worth having. None asks whether
# running one can do harm. The missing bet: a plain run never spends money or
# sends data anywhere. It ranks second, above everything that had been tested,
# because a charge or a leak cannot be taken back. The test is a scan for a
# key or an endpoint read from the environment in a file that takes no flag.
# Read against the files as committed it is false: four files called a paid or
# outside service as soon as a key was present. They now need --live as well.
# ---------------------------------------------------------------------------

SILENT = "A plain run never spends money or sends data anywhere; that takes a flag on the command line"
SWITCH = r'os\.environ\.get\("[A-Z_]+(_KEY|_ENDPOINT)"'       # a key or an endpoint read from the environment
FLAG = "--live"


def matching(pattern: str, at: str = "") -> set[str]:
    """Lesson numbers whose practice file matches: at that commit, or on disk when no commit is given."""
    if not at:
        return {file.parts[-3][:2] for file in practice_files() if re.search(pattern, file.read_text(encoding="utf-8"))}
    found = subprocess.run(["git", "grep", "-l", "-E", "-e", pattern, at, "--", "*/code/practice.py"], cwd=REPO, capture_output=True, text=True)
    return {line.split(":", 1)[1][:2] for line in found.stdout.splitlines()}     # git grep exits 1 when nothing matches


def calls_out_unasked(at: str = "") -> list[str]:
    """Lessons whose file reads a key or endpoint from the environment and takes no flag."""
    # ponytail: a text scan; it cannot see a call hidden behind a helper, and any mention of --live counts as the flag
    return sorted(matching(SWITCH, at) - matching(FLAG, at))


def with_safety() -> list[Bet]:
    then, now = calls_out_unasked(LATEST), calls_out_unasked()
    return [*assumptions(), Bet(SILENT, 5, 3, 5, "Scan every file for a key or endpoint read from the environment with no flag beside it",
                                f"false at {LATEST} for {len(then)} files ({', '.join(then)}); {len(now)} files today", kind="safety")]


def ex2_safety_assumption() -> None:
    items = with_safety()
    ranked = show(items)
    then, now = calls_out_unasked(LATEST), calls_out_unasked()
    print(f"  the safety bet enters at rank {[entry['statement'] for entry in ranked].index(SILENT) + 1} of {len(ranked)}; still open: {next_experiment(items).statement}")
    assert [entry["statement"] for entry in ranked][:2] == [READER, SILENT] and risk_score(items[-1]) == 20
    assert then == ["01", "09", "23", "24"] and now == [] and matching(FLAG) >= set(then)
    assert next_experiment(items).statement == READER


# ---------------------------------------------------------------------------
# Exercise 3 - a threshold that stops the build
#
# The rules were written before any value was read. Each names what is
# counted, where the build stops, and a band where the answer is to look again
# before deciding. Read against the files as committed, the first rule says
# stop, and that is what happened: the four files were gated before any more
# were written. The reader rule has no value yet. Its three outcomes are laid
# out so that the number, when it comes, cannot move the line.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StopRule:
    counted: str
    stop_above: float
    unclear_above: float          # between the two: look again before deciding


RULES = {
    SILENT: StopRule("files that call out without being asked", 0, 0),
    PLAIN: StopRule("share of lessons that cannot start on a plain install", 0.10, 0.02),
    READER: StopRule("of three people handed a lesson, those who do not run it", 2, 1),
}


def verdict(rule: StopRule, value: float) -> str:
    return "stop" if value > rule.stop_above else "unclear" if value > rule.unclear_above else "continue"


def ex3_stop_threshold() -> None:
    blocked_share = len(cannot_start_plain()) / len(lessons())
    read = [(SILENT, f"at {LATEST}", len(calls_out_unasked(LATEST))), (SILENT, "today", len(calls_out_unasked())), (PLAIN, "today", blocked_share)]
    for statement, when, value in read:
        print(f"  {RULES[statement].counted:<56} {when:<11} {value:<6.3g} -> {verdict(RULES[statement], value)}")
    table = {count: verdict(RULES[READER], count) for count in range(4)}
    print(f"  {RULES[READER].counted:<56} {'no value':<11} {'':<6} -> {table}")
    assert [verdict(RULES[statement], value) for statement, _, value in read] == ["stop", "continue", "continue"]
    assert list(table.values()) == ["continue", "continue", "unclear", "stop"]
    assert all(rule.unclear_above <= rule.stop_above for rule in RULES.values())


# ---------------------------------------------------------------------------
# Exercise 4 - a large experiment replaced by a cheaper decisive test
#
# The large experiment for "every file starts on a plain install" is to start
# every file that way. The cheaper test reads the imports: a file fails to
# start only when it, or the main.py it builds on, imports a package at module
# level. It is decisive for this one question and says nothing about whether
# the exercises then pass. Both are run on a lesson where the answer is yes
# and one where it is no, and they agree.
# ---------------------------------------------------------------------------

def starts_plain(lesson: str) -> bool:
    """The large experiment, for one lesson: start its file with no installed packages visible."""
    file = next(REPO.glob(f"{lesson}-*/code/practice.py"))
    run = subprocess.run([sys.executable, "-S", "-E", str(file)], capture_output=True, text=True, timeout=120)
    return "ModuleNotFoundError" not in run.stderr and "pip install" not in run.stderr


def ex4_cheaper_decisive_test() -> None:
    started = time.perf_counter()
    blocked = cannot_start_plain()
    scan_seconds = time.perf_counter() - started
    sample = ["01", "15"]
    started = time.perf_counter()
    observed = {lesson: starts_plain(lesson) for lesson in sample}
    per_file = (time.perf_counter() - started) / len(sample)
    predicted = {lesson: lesson not in blocked for lesson in sample}
    print(f"  import scan of all {len(lessons())} lessons: {scan_seconds:.2f} s -> cannot start: {blocked}")
    print(f"  starting the files: {per_file:.2f} s each on this sample, at least {per_file * len(lessons()):.0f} s for all")
    print(f"  on the sample: scan predicts {predicted}, starting them shows {observed}")
    assert predicted == observed == {"01": True, "15": False}
    assert scan_seconds < per_file * len(lessons())


# ---------------------------------------------------------------------------
# Exercise 5 - risk ranking against roadmap priority
#
# The roadmap is what the commits say they delivered, in order. It put
# evidence under three assumptions, and they are exactly the three with the
# lowest risk. The work went to what a script could confirm the same day,
# where uncertainty was already small. The two riskiest bets need what a
# builder's own script is worst at: a second person, and a look for harm.
# ---------------------------------------------------------------------------

DELIVERED = {   # assumption -> the words a commit message uses when that commit put evidence under it
    NAMED: "solving the five exercises",
    CHECKS: "asserts its own results",
    PLAIN: "needs numpy",
}


def first_evidence(statement: str) -> str:
    """The first commit whose message says it delivered evidence for the assumption, '' when none does."""
    if statement not in DELIVERED:
        return ""
    log = subprocess.run(["git", "log", "--reverse", "--format=%h %B%x00", f"{COURSE}..{LATEST}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    return next((" ".join(entry.split())[:7] for entry in log.split("\x00") if DELIVERED[statement] in " ".join(entry.split())), "")


def ex5_risk_against_roadmap() -> None:
    ranked = [entry["statement"] for entry in prioritize(with_safety())]
    served = {statement: first_evidence(statement) for statement in ranked}
    for rank, statement in enumerate(ranked, 1):
        print(f"  risk rank {rank}  first evidence on the roadmap: {served[statement] or 'none':<8} {statement[:80]}")
    on_roadmap = [statement for statement in ranked if served[statement]]
    print(f"  the roadmap served {len(on_roadmap)} assumptions: risk ranks {[ranked.index(statement) + 1 for statement in on_roadmap]}")
    assert on_roadmap == ranked[3:] and not any(served[statement] for statement in ranked[:3])
    assert served[NAMED] == served[CHECKS] == "90bedf6" and served[PLAIN] == "563dbed"


if __name__ == "__main__":
    print("Phase 14 - Lesson 49: Map Assumptions and Resolve the Riskiest One First - exercises")
    for exercise in (ex1_five_assumptions, ex2_safety_assumption, ex3_stop_threshold, ex4_cheaper_decisive_test, ex5_risk_against_roadmap):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
