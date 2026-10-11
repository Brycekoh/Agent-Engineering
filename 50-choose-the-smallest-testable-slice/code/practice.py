"""Phase 14 - Lesson 50: Choose the Smallest Slice That Can Change the Decision - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The outcome is lesson 47's: a reader can confirm the course was worked through. The proof a first
slice owes comes from lesson 49's two riskiest assumptions. Where a slice is one this repository
really took, its facts are read from git (needs git on the PATH).
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from main import Slice, choose, decision, score

REPO = Path(__file__).resolve().parents[2]
FIRST, LATEST = "90bedf6", "a08499a"      # the first commit of practice files; the last one before lesson 49's findings

READER_RUNS, RUN_IS_SILENT, ALL_ANSWERED = "reader-runs-a-file", "plain-run-is-silent", "every-exercise-answered"

# ---------------------------------------------------------------------------
# Exercise 2 comes first in the file because it comes first in time: the proof
# a slice owes is fixed here, before any slice exists to be scored.
#
# Lesson 49 ranked two assumptions above all the tested ones: someone other
# than the owner runs a file, and a plain run never spends or sends anything.
# A slice that leaves either unproven cannot change the decision to go on.
# ---------------------------------------------------------------------------

REQUIRED = {READER_RUNS, RUN_IS_SILENT}


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


@dataclass(frozen=True)
class Capability:
    name: str
    effort: int
    proves: tuple[str, ...] = ()


ONE_FILE = Capability("one lesson's exercises answered in a file that takes no key from the environment", 1, (RUN_IS_SILENT,))
TEN_FILES = Capability("ten lessons' files", 4, (RUN_IS_SILENT,))
ALL_FILES = Capability("every lesson's file", 10, (RUN_IS_SILENT, ALL_ANSWERED))
LIVE_PATH = Capability("a real model call for the exercise that asks for one", 1)
THREE_READERS = Capability("three readers handed the folder and watched for five minutes", 1, (READER_RUNS,))


def build(name: str, parts: list[Capability], outcome_value: int, uncertainty_reduced: int, consequence: int, reversible: bool) -> Slice:
    """A slice is the capabilities in it: effort adds up, and it proves what its parts prove."""
    return Slice(name, outcome_value, uncertainty_reduced, sum(part.effort for part in parts), consequence, reversible,
                 tuple(sorted({proof for part in parts for proof in part.proves})))


# ---------------------------------------------------------------------------
# Exercise 1 - three slices for the same outcome at different consequence levels
#
# A folder passed to three people touches nothing public. A branch on the
# public repository can be deleted. Every lesson on the public main cannot be
# called back from whoever cloned it. All three carry the readers, so all
# three can prove what is owed; they differ in what is spent to find out.
# ---------------------------------------------------------------------------

FOLDER_PARTS = [ONE_FILE, LIVE_PATH, THREE_READERS]


def three_slices() -> list[Slice]:
    return [
        build("one lesson's folder handed to three readers", FOLDER_PARTS, 2, 5, 1, True),
        build("ten lessons on a branch, three readers pointed at it", [TEN_FILES, LIVE_PATH, THREE_READERS], 3, 5, 2, True),
        build("every lesson on the public main, three readers asked afterwards", [ALL_FILES, LIVE_PATH, THREE_READERS], 5, 5, 4, False),
    ]


def ex1_three_slices() -> None:
    slices = three_slices()
    for item in slices:
        print(f"  consequence {item.consequence}, {'reversible' if item.reversible else 'one-way   '}, effort {item.effort:>2}, score {score(item):<5} {item.name}")
    chosen = choose(slices, REQUIRED)
    print(f"  chosen: {chosen.name}")
    assert [item.consequence for item in slices] == [1, 2, 4] and all(REQUIRED <= set(item.proves) for item in slices)
    assert [score(item) for item in slices] == sorted((score(item) for item in slices), reverse=True) and chosen == slices[0]


# ---------------------------------------------------------------------------
# Exercise 2 - the required proof set, stated before scoring
#
# With REQUIRED fixed above, two more candidates are put through the gate. One
# is cheaper than anything in exercise 1 and has the best score of all, and it
# stays out because it proves one of the two. The other is the slice this
# repository really took first: ten lessons straight to main, nobody asked.
# It proved that the files could be written, which was never in doubt.
# ---------------------------------------------------------------------------

def more_candidates() -> list[Slice]:
    pushed = sum(name.endswith("practice.py") for name in git("show", "--diff-filter=A", "--name-only", "--format=", FIRST))
    return [
        build("check one file for calls it makes unasked, show it to nobody", [ONE_FILE], 2, 3, 1, True),
        Slice(f"{pushed} lessons pushed to the public main, nobody asked (commit {FIRST})", 3, 2, TEN_FILES.effort + LIVE_PATH.effort, 3, False, ()),
    ]


def ex2_required_proof_first() -> None:
    candidates = three_slices() + more_candidates()
    print(f"  required proof: {sorted(REQUIRED)}")
    for item in sorted(candidates, key=score, reverse=True):
        missing = sorted(REQUIRED - set(item.proves))
        print(f"  score {score(item):<5} {'eligible' if not missing else 'out, lacks ' + ', '.join(missing):<52} {item.name}")
    record = decision(candidates, REQUIRED)
    cheap, taken = more_candidates()
    print(f"  selected: {record['selected']['name']} ({len(record['alternatives'])} alternatives on record)")
    assert score(cheap) == max(score(item) for item in candidates) and record["selected"]["name"] == three_slices()[0].name
    assert not REQUIRED <= set(cheap.proves) and taken.proves == () and taken.name.startswith("10 lessons")
    assert record["required_proof"] == sorted(REQUIRED) and len(record["alternatives"]) == 4


# ---------------------------------------------------------------------------
# Exercise 3 - one capability removed, the decisive evidence kept
#
# The chosen slice has three parts. Taking out the real model call costs no
# proof and makes the slice cheaper, so it scores higher. Taking out the
# readers saves the same effort and is the easier cut to make, because it is
# the only part that needs other people; it removes the proof the slice
# exists for.
# ---------------------------------------------------------------------------

def without(part: Capability) -> Slice:
    first = three_slices()[0]
    return build(f"{first.name}, without: {part.name}", [item for item in FOLDER_PARTS if item != part],
                 first.outcome_value, first.uncertainty_reduced, first.consequence, first.reversible)


def ex3_remove_a_capability() -> None:
    first = three_slices()[0]
    print(f"  as designed: effort {first.effort}, score {score(first)}, proves {list(first.proves)}")
    trimmed = {part.name: without(part) for part in FOLDER_PARTS}
    for name, item in trimmed.items():
        print(f"  without '{name[:44]}': effort {item.effort}, score {score(item)}, {'still eligible' if REQUIRED <= set(item.proves) else 'no longer eligible'}")
    lighter = trimmed[LIVE_PATH.name]
    assert set(lighter.proves) == set(first.proves) and lighter.effort < first.effort and score(lighter) > score(first)
    assert choose([first, lighter], REQUIRED) == lighter
    assert not REQUIRED <= set(trimmed[THREE_READERS.name].proves) and not REQUIRED <= set(trimmed[ONE_FILE.name].proves)


# ---------------------------------------------------------------------------
# Exercise 4 - a stop rule for a failed pilot
#
# Written for the two things the pilot can show: whether the file stayed
# silent, and how many of the three readers ran it. Two of the four answers
# are not "keep building". The rule is then read against the one result that
# exists: lesson 01's file as committed read a key from the environment and
# took no flag, so the rule said stop until that was closed. It is closed now.
# ---------------------------------------------------------------------------

def after_pilot(readers_who_ran: int, silent: bool) -> str:
    if not silent:
        return "stop: narrow authority first, every outbound call behind a flag"
    if readers_who_ran == 0:
        return "stop: no more files; test a different mechanism for the same outcome"
    if readers_who_ran == 1:
        return "hold: collect better evidence, three more readers, same folder"
    return "continue: the next ten lessons"


def silent(source: str) -> bool:
    """A file is silent when it reads no key from the environment, or takes --live before it uses one."""
    return "OPENAI_API_KEY" not in source or "--live" in source


def ex4_stop_rule() -> None:
    results = {(ran, quiet): after_pilot(ran, quiet) for quiet in (True, False) for ran in range(4)}
    for (ran, quiet), action in results.items():
        if quiet or ran == 0:
            print(f"  {'silent' if quiet else 'calls out unasked'}, {ran if quiet else 'any number'} of 3 readers ran it -> {action}")
    path = "01-the-agent-loop/code/practice.py"
    then, now = "\n".join(git("show", f"{LATEST}:{path}")), (REPO / path).read_text(encoding="utf-8")
    print(f"  lesson 01 as committed at {LATEST}: {after_pilot(3, silent(then)).split(':')[0]}; as it is now: silence no longer stops it")
    kinds = {action.split(":")[0] for action in results.values()}
    assert kinds == {"stop", "hold", "continue"} and sum(action.startswith("continue") for action in results.values()) == 2
    assert not silent(then) and silent(now) and after_pilot(3, silent(then)).startswith("stop")


# ---------------------------------------------------------------------------
# Exercise 5 - a reusable platform component that should wait
#
# The component is a shared helper module. The case for it is real: one
# helper is written out word for word in eleven files and another in three.
# But neither existed until lesson 31, so built before the first slice it
# would have been a guess at what thirty lessons then never used. As a slice
# it proves nothing that is owed, and no score gets it past the gate.
# ---------------------------------------------------------------------------

def copies(function: str) -> dict[str, list[str]]:
    """Source of a top-level function -> the lessons whose practice file contains exactly that source."""
    found: dict[str, list[str]] = {}
    for file in sorted(REPO.glob("[0-9][0-9]-*/code/practice.py")):
        for node in ast.parse(file.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.FunctionDef) and node.name == function:
                found.setdefault(ast.unparse(node), []).append(file.parts[-3][:2])
    return found


def ex5_platform_component_waits() -> None:
    repeated = {name: copies(name) for name in ("run_main", "receipt")}
    for name, variants in repeated.items():
        lessons = [lesson for group in variants.values() for lesson in group]
        print(f"  {name}(): {len(lessons)} copies in lessons {lessons[0]}-{lessons[-1]}, {len(variants)} distinct text")
    platform = Slice("a shared helper module for every lesson, built first", 5, 5, 1, 1, True, (ALL_ANSWERED,))
    candidates = three_slices() + [platform]
    print(f"  as a slice: score {score(platform)} (best of {len(candidates)}), proves {list(platform.proves)}; chosen instead: {choose(candidates, REQUIRED).name}")
    run_main = [lesson for group in repeated["run_main"].values() for lesson in group]
    assert len(run_main) == 11 and min(run_main) == "31" and len(repeated["run_main"]) == 1 and len(repeated["receipt"]) == 1
    assert score(platform) == max(score(item) for item in candidates) and choose(candidates, REQUIRED) == three_slices()[0]
    try:
        choose([platform], REQUIRED)
        raise AssertionError("a slice that proves nothing owed was chosen")
    except ValueError as refusal:
        print(f"  offered alone: {refusal}")


if __name__ == "__main__":
    print("Phase 14 - Lesson 50: Choose the Smallest Slice That Can Change the Decision - exercises")
    for exercise in (ex1_three_slices, ex2_required_proof_first, ex3_remove_a_capability, ex4_stop_rule, ex5_platform_component_waits):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
