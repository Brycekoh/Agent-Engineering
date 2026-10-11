"""Phase 14 - Lesson 48: Discover the Workflow People Actually Perform - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The workflow is the one that put the practice files of lessons 01-40 on main. Its log is this
repository's git history (needs git on the PATH); the range read is fixed, so the numbers hold
however the history grows.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from statistics import mean

from main import Evidence, WorkflowStep, audit

REPO = Path(__file__).resolve().parents[2]
COURSE, LATEST = "6a735e9", "a08499a"     # the course as published; the last commit read here
SUITE = "30-eval-driven-agent-development/code/practice.py"
PARTIAL_FOOTER = "for the rest" + '")'    # how a practice file ends its last line when a package was missing


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


@dataclass(frozen=True)
class Commit:
    sha: str
    subject: str
    body: str
    author: str
    added: tuple[str, ...]        # lesson numbers whose practice.py first appears here
    changed: tuple[str, ...]      # lesson numbers whose practice.py is changed here


@lru_cache(maxsize=None)
def commits() -> tuple[Commit, ...]:
    """Every commit after the course was published, oldest first."""
    found = []
    for sha in git("rev-list", "--reverse", f"{COURSE}..{LATEST}"):
        author, subject, *body = git("show", "-s", "--format=%an%n%s%n%b", sha)
        files = [line.split("\t") for line in git("show", "--name-status", "--format=", sha) if line.endswith("practice.py")]
        found.append(Commit(sha[:7], subject, "\n".join(body), author,
                            tuple(path[:2] for status, path in files if status == "A"),
                            tuple(path[:2] for status, path in files if status == "M")))
    return tuple(found)


def changed_later() -> Counter:
    """Lesson number -> how many commits changed its practice file after the one that added it."""
    return Counter(lesson for commit in commits() for lesson in commit.changed)


@lru_cache(maxsize=None)
def plain_python(path: str) -> subprocess.CompletedProcess:
    """Run a file with no installed packages visible (-S) and no PYTHON* variables (-E)."""
    return subprocess.run([sys.executable, "-S", "-E", str(REPO / path)], capture_output=True, text=True, timeout=120)


@dataclass(frozen=True)
class Step(WorkflowStep):
    authority: str = ""           # who alone may take this step, where that matters


def only_said(steps: list[WorkflowStep]) -> list[WorkflowStep]:
    """Steps that rest on what somebody said: no direct evidence under them."""
    return [step for step in steps if not any(item.direct for item in step.evidence)]


# ---------------------------------------------------------------------------
# Exercise 1 - a workflow reconstructed from a log, nobody interviewed
#
# Six commits give four steps. Three stand on what git recorded: which files
# appeared, who is named on the commit, which files were changed again. One
# stands on what a commit message says was done, so it is marked as reported.
# The friction is in the log too: lessons changed again after they were on main.
# The log is silent on who allowed each commit and on what anyone looked at.
# ---------------------------------------------------------------------------

def from_the_log() -> list[WorkflowStep]:
    log = commits()
    adds = [commit for commit in log if commit.added]
    reworks = [commit for commit in log if commit.changed]
    again = changed_later()
    return [
        WorkflowStep(1, "coding agent", "writes practice files for a run of consecutive lessons", (
            Evidence(f"git {', '.join(commit.sha for commit in adds)}", f"{[len(commit.added) for commit in adds]} new practice.py files", True, 1.0),
            Evidence("commit trailers", f"{sum('Co-Authored-By: Claude' in commit.body for commit in log)} of {len(log)} commits name a model as co-author", True, 1.0))),
        WorkflowStep(2, "coding agent", "runs each file before it is committed", (
            Evidence(f"message of {adds[0].sha}", "says each file asserts its own results", False, 0.7),)),
        WorkflowStep(3, "owner", "commits the run under one message that names the lesson range", (
            Evidence("git log", "; ".join(commit.subject.split("for ")[-1] for commit in adds), True, 1.0),)),
        WorkflowStep(4, "coding agent", "goes back over lessons that are already on main", (
            Evidence(f"git {', '.join(commit.sha for commit in reworks)}", f"{sum(len(commit.changed) for commit in reworks)} changes to files committed earlier", True, 1.0),),
            f"rework: {len(again)} of {sum(len(commit.added) for commit in adds)} lessons changed after their first commit, {sum(times > 1 for times in again.values())} of them twice"),
    ]


def ex1_workflow_from_a_log() -> None:
    steps = from_the_log()
    result = audit(steps)
    for step in steps:
        print(f"  {step.order}. {step.actor:<12} {step.action}")
        for item in step.evidence:
            print(f"       {'direct  ' if item.direct else 'reported'} {item.source}: {item.observation}")
    print(f"  {result['status']}, direct evidence ratio {result['direct_evidence_ratio']}; friction: {result['friction_points']}")
    print("  the log is silent on: who allowed each commit, what the owner looked at, whether anyone else has opened a file")
    assert len(commits()) == 6 and [len(commit.added) for commit in commits() if commit.added] == [10, 20, 10]
    assert len(changed_later()) == 21 and sorted(lesson for lesson, times in changed_later().items() if times > 1) == ["06", "09", "10"]
    assert result["status"] == "grounded" and result["direct_evidence_ratio"] == 0.8 and len(result["friction_points"]) == 1
    assert [step.order for step in only_said(steps)] == [2]


# ---------------------------------------------------------------------------
# Exercise 2 - an interview, with every claim that still lacks direct evidence marked
#
# RECORDED: the owner's instructions in the sessions that produced these files,
# paraphrased. Each claim is looked up in the repository. Two can be observed:
# one holds, one holds for ten lessons and fails for thirty. Two cannot be
# observed at all, because a yes lives in a chat and a reader leaves no trace
# here. main.audit still calls the whole thing grounded: one direct item
# anywhere is enough for it. only_said() is the stricter list.
# ---------------------------------------------------------------------------

def refined_before_commit() -> Evidence:
    again = changed_later()
    early, late = [n for n in again if n <= "30"], [n for n in again if n > "30"]
    return Evidence(f"git {COURSE}..{LATEST}", f"holds for lessons 31-40 ({len(late)} changed later), fails for 01-30 ({len(early)} changed later)", True, 1.0)


def note_names_the_need() -> Evidence:
    last = plain_python("13-langgraph-stateful-graphs/code/practice.py").stdout.strip().splitlines()[-1]
    return Evidence("lesson 13 run on a plain install", f"last line: '{last}'", True, 1.0)


INTERVIEW = [   # (what the owner said, the step it describes, how the repository can show it or None)
    ("The files are there to show that each chapter was completed", "a reader opens a lesson and checks it", None),
    ("Nothing is committed or pushed until I say yes, and a yes covers one batch", "the owner allows the commit", None),
    ("A batch is refined before it is committed", "the batch is refined before the commit", refined_before_commit),
    ("A note says what a path needs", "a file that skips an exercise says what to install", note_names_the_need),
]


def from_the_interview() -> list[WorkflowStep]:
    return [WorkflowStep(order, "owner", step, (Evidence("owner, in session", said, False, 0.8), *((check(),) if check else ())))
            for order, (said, step, check) in enumerate(INTERVIEW, 1)]


def ex2_interview_claims() -> None:
    steps = from_the_interview()
    lacking = only_said(steps)
    for step in steps:
        direct = [item for item in step.evidence if item.direct]
        print(f"  \"{step.evidence[0].observation}\"")
        print(f"       {'direct: ' + direct[0].observation if direct else 'STILL ONLY SAID: nothing in the repository records it'}")
    result = audit(steps)
    print(f"  main.audit: {result['status']}, ratio {result['direct_evidence_ratio']}; claims with no direct evidence: {len(lacking)} of {len(steps)}")
    assert [step.order for step in lacking] == [1, 2] and result["status"] == "grounded" and result["direct_evidence_ratio"] == 0.33
    assert "(0 changed later)" in steps[2].evidence[1].observation and "(21 changed later)" in steps[2].evidence[1].observation
    assert "install" in steps[3].evidence[1].observation


# ---------------------------------------------------------------------------
# Exercise 3 - one authority boundary and one failure-recovery step
#
# The boundary: only the owner lets a batch onto main, and the yes is for that
# batch. It becomes a step of its own, placed before the commit. What git can
# show is consistent with it (one author on every commit) but does not prove a
# yes was given. The recovery path is a branch, so the main order stays
# contiguous: it leaves at "runs each file" when a check gives different
# answers on different runs, and comes back to the same step. The refine
# messages record four such repairs, all made after the files were on main.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Branch:
    begins_at: int                # the order of the main step where it leaves
    when: str
    steps: tuple[WorkflowStep, ...]
    rejoins_at: int


def with_boundary() -> list[WorkflowStep]:
    write, run, commit, rework = from_the_log()
    authors = {item.author for item in commits()}
    yes = Step(3, "owner", "says yes to this batch: commit and push to main", (
        Evidence("owner, in session", INTERVIEW[1][0], False, 0.8),
        Evidence("git log", f"every commit has the same author ({len(authors)} name)", True, 0.6)), authority="owner only; one yes covers one batch")
    return [write, run, yes, replace(commit, order=4), replace(rework, order=5)]


def recovery() -> Branch:
    repairs = [line for commit in commits() for line in commit.body.splitlines() if re.search(r"clock|margin", line)]
    return Branch(2, "a check passes on one run and fails on another", (
        WorkflowStep(1, "coding agent", "finds what the check depends on besides the code", (
            Evidence("refine commit messages", f"{len(repairs)} repairs name a clock or a timing margin", True, 0.9),)),
        WorkflowStep(2, "coding agent", "removes the dependence and runs the file again", (
            Evidence("refine commit messages", "; ".join(line.lstrip("- ") for line in repairs[:2]), True, 0.9),),
            "the repair came after the commit: the file was on main first"),
    ), rejoins_at=2)


def ex3_authority_and_recovery() -> None:
    steps, branch = with_boundary(), recovery()
    for step in steps:
        print(f"  {step.order}. {step.actor:<12} {step.action}{'   [' + step.authority + ']' if getattr(step, 'authority', '') else ''}")
        if step.order == branch.begins_at:
            print(f"       branch when {branch.when}:")
            for item in branch.steps:
                print(f"         {branch.begins_at}.{item.order} {item.action}  ({item.evidence[0].observation})")
            print(f"         back to step {branch.rejoins_at}")
    bounded = [step for step in steps if getattr(step, "authority", "")]
    main_result, branch_result = audit(steps), audit(list(branch.steps))
    print(f"  main order {[step.order for step in steps]}: {main_result['status']}; branch: {branch_result['status']}; friction: {branch_result['friction_points']}")
    assert main_result["status"] == branch_result["status"] == "grounded" and not main_result["issues"]
    assert len(bounded) == 1 and bounded[0].order < next(step.order for step in steps if step.action.startswith("commits"))
    assert branch.begins_at == branch.rejoins_at == 2 and "4 repairs" in branch.steps[0].evidence[0].observation
    assert main_result["steps"][2]["authority"].startswith("owner only")


# ---------------------------------------------------------------------------
# Exercise 4 - two workflow variants, kept apart
#
# Lessons 01-30 were committed and then refined in later commits. Lessons
# 31-40 were refined first and committed once. Same people, same steps, a
# different order: the older and the current process. Averaged, a lesson's file
# reaches main 1.6 times, which is true of no lesson and hides that every
# repeat belongs to the first variant.
# ---------------------------------------------------------------------------

def variants() -> dict[str, list[WorkflowStep]]:
    write, run, yes, commit, rework = with_boundary()
    again = changed_later()
    early = sum(lesson <= "30" for lesson in again)
    refine_first = WorkflowStep(3, "coding agent", "refines the batch before anyone is asked", (
        Evidence("owner, in session", INTERVIEW[2][0], False, 0.8),
        Evidence(f"git {COURSE}..{LATEST}", f"{sum(lesson > '30' for lesson in again)} of 10 lessons changed after their commit", True, 1.0)))
    return {
        "commit, then refine (lessons 01-30)": [write, run, yes, commit, replace(rework, friction=f"{early} of 30 lessons changed while on main")],
        "refine, then commit (lessons 31-40)": [write, run, refine_first, replace(yes, order=4), replace(commit, order=5)],
    }


def times_committed() -> dict[str, int]:
    """Lesson number -> how many commits carried its practice file."""
    return dict(Counter(lesson for commit in commits() for lesson in commit.added + commit.changed))


def ex4_two_variants() -> None:
    both = variants()
    for name, steps in both.items():
        print(f"  {name}: {audit(steps)['status']}")
        print(f"       {' -> '.join(step.action.split()[0] for step in steps)}   friction: {audit(steps)['friction_points'] or 'none recorded'}")
    counts = times_committed()
    first, second = [counts[n] for n in counts if n <= "30"], [counts[n] for n in counts if n > "30"]
    merged = round(mean(counts.values()), 1)
    print(f"  commits per lesson: {mean(first):.1f} in the first variant, {mean(second):.1f} in the second; averaged {merged}, the count of {sum(value == merged for value in counts.values())} lessons")
    order = {name: [step.action.split()[0] for step in steps] for name, steps in both.items()}
    older, current = order.values()
    assert all(audit(steps)["status"] == "grounded" for steps in both.values()) and older != current
    assert older.index("commits") < older.index("goes") and current.index("refines") < current.index("commits")
    assert (len(first), len(second), round(mean(first), 1), mean(second), merged) == (30, 10, 1.8, 1, 1.6) and merged not in counts.values()


# ---------------------------------------------------------------------------
# Exercise 5 - a feature that removes a visible step and leaves hidden work
#
# The feature exists: lesson 30's --run-suite runs every lesson's file and
# prints one total. It removes the visible step, running the files one by one.
# As committed it judges a run by its exit code alone. The hidden work was
# never the running: it is reading the last line of each run, because a file
# whose optional package is missing skips those exercises, says so, and still
# exits 0. One total hid that line for every such file. The suite now reads
# the last line too and marks such a run PART.
# ---------------------------------------------------------------------------

def ex5_feature_that_leaves_hidden_work() -> None:
    committed = "\n".join(git("show", f"{LATEST}:{SUITE}"))
    partial = plain_python("13-langgraph-stateful-graphs/code/practice.py")
    last = partial.stdout.strip().splitlines()[-1]
    can_skip = [path.parts[-3][:2] for path in sorted(REPO.glob("[0-9][0-9]-*/code/practice.py")) if PARTIAL_FOOTER in path.read_text(encoding="utf-8")]
    print("  feature: one command that runs every lesson and prints a total (lesson 30, --run-suite)")
    print("  step it removes: running each file by hand")
    print(f"  how it judged a run at {LATEST}: {'exit code only' if 'returncode == 0' in committed and 'for the rest' not in committed else 'more than the exit code'}")
    print(f"  lesson 13 on a plain install: exit {partial.returncode}, last line '{last}'")
    print(f"  how it judges a run now: {'exit code and last line (PART)' if 'PART' in (REPO / SUITE).read_text(encoding='utf-8') else 'exit code only'}")
    print(f"  hidden work left: reading that last line; {len(can_skip)} files can exit 0 with exercises left over ({', '.join(can_skip)})")
    assert "returncode == 0" in committed and "for the rest" not in committed
    assert partial.returncode == 0 and "for the rest" in last and "13" in can_skip and "48" not in can_skip and len(can_skip) >= 13


if __name__ == "__main__":
    print("Phase 14 - Lesson 48: Discover the Workflow People Actually Perform - exercises")
    for exercise in (ex1_workflow_from_a_log, ex2_interview_claims, ex3_authority_and_recovery, ex4_two_variants,
                     ex5_feature_that_leaves_hidden_work):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
