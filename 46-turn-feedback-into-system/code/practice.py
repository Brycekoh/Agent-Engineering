"""Phase 14 - Lesson 46: Turn Every Agent Correction into a System Improvement - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The corrections are real ones from the sessions that wrote this repository's practice files. How
often each happened is read from the git history where the history shows it (needs git on the PATH).
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from main import Control, Correction, choose_target, promote, ratchet

REPO = Path(__file__).resolve().parents[2]
COURSE, FIRST, SECOND, BATCH, LATEST = "6a735e9", "90bedf6", "563dbed", "f7befa5", "a08499a"    # the course as published, then practice commits in order
DISCLAIMER = re.compile(r"(?i:\b(has|have|was|were|is|are) not (been )?(run|tested|verified|executed)\b)"
                        r"|\bNOT (RUN|DOWNLOADED|TESTED)\b|(?i:\bun(?:tested)\b)")


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


def disclaimers_at(commit: str) -> list[str]:
    """'lesson:line' for every line of a practice file, as it was at that commit, that the pattern matches."""
    found = []
    for entry in git("grep", "-n", "-e", "", commit, "--", "*/code/practice.py"):       # every line, as commit:path:line:text
        _, name, number, text = entry.split(":", 3)
        if DISCLAIMER.search(text):
            found.append(f"{name[:2]}:{number}")
    return found


def timing_repairs() -> list[str]:
    """The lines of the refine commit messages that record a check moved off the wall clock or given a wider margin."""
    return [line.lstrip("- ") for line in git("log", "--format=%B", f"{FIRST}..{LATEST}") if re.search(r"clock|margin", line)]


def out_of_range_edits() -> list[str]:
    """Lessons outside 11-30 whose file the commit for lessons 11-30 changed."""
    return [name[:2] for status, name in (line.split("\t") for line in git("show", "--name-status", "--format=", SECOND))
            if status == "M" and not "11" <= name[:2] <= "30"]


def rework_commits() -> list[str]:
    """Commits that changed a practice file that was already on main."""
    return [sha[:7] for sha in git("rev-list", f"{COURSE}..{LATEST}")
            if any(line.startswith("M") and line.endswith("practice.py") for line in git("show", "--name-status", "--format=", sha))]


@dataclass(frozen=True)
class Weighed(Correction):
    severity: int = 1           # 1 minor, 2 major, 3 severe: how bad one occurrence is


# ---------------------------------------------------------------------------
# Exercise 1 - five corrections from recent sessions, and their real owners
#
# All five happened while this repository's practice files were being
# written. The recurrence of three of them is counted from the git history:
# files in the first commit whose notes exercise 2's scan catches, repairs the
# refine messages record, and files the commit for lessons 11-30 changed
# outside that range. The other two happened once.
#
# The real owner is the earliest layer that would have stopped it, from the
# lesson's table. main.py's choose_target guesses the owner from keywords in
# the symptom and the cause. It agrees on two of the five. The other three
# all fall through to "instruction", its default, which is the weakest layer
# on the list: a paragraph asking to be remembered. Each of those three has
# an executable owner.
# ---------------------------------------------------------------------------

CORRECTIONS = [     # (correction, real owner, why that layer, how the control is verified)
    (Weighed("A note in a practice file disclaimed part of the work", "wording rule for notes was implicit",
             len({hit[:2] for hit in disclaimers_at(FIRST)}), "every such note had to be rewritten", severity=2),
     "test", "a scan of the files finds every case; nobody has to remember the rule (exercise 2)",
     "python 46-turn-feedback-into-system/code/practice.py"),
    (Weighed("A timing assertion could fail on a busy machine", "result depended on wall-clock sleeps", len(timing_repairs()),
             "a passing file fails at random", severity=2),
     "test", "the test is the thing that was wrong: it now runs on a virtual clock (lessons 16 and 22)",
     "no sleep call is left in the practice files of lessons 16 and 22"),
    (Weighed("A command resolved to the wrong program", "environment assumptions were implicit", 1,
             "one comparison could not run", severity=1),
     "automation", "a lookup that picks the right program (find_bash in lesson 42)",
     "python 42-agent-workbench-capstone/code/practice.py"),
    (Weighed("A comment said what an installer does, and it does not", "claim was described but not checked", 1,
             "a wrong statement was pushed", severity=3),
     "test", "the claim is now an assertion on the installer's output (lesson 42's mission check)",
     "python 42-agent-workbench-capstone/code/practice.py"),
    (Weighed("Files outside the task were edited", "scope was described but not checked", len(out_of_range_edits()),
             "found by the gate only after the fact", severity=2),
     "scope", "a contract checked against the diff (lessons 36 and 43)",
     "git diff --name-only main...HEAD | python 43-frame-the-task-before-code/code/practice.py --scope-receipt"),
]


def ex1_real_owners() -> None:
    guessed = [choose_target(entry[0]) for entry in CORRECTIONS]
    for (correction, owner, _, _), guess in zip(CORRECTIONS, guessed):
        print(f"  {correction.symptom[:52]:<52} real owner: {owner:<10} main.py: {guess:<11} {'' if guess == owner else '<- differs'}")
    agree = sum(guess == entry[1] for entry, guess in zip(CORRECTIONS, guessed))
    misfiled = {guess for entry, guess in zip(CORRECTIONS, guessed) if guess != entry[1]}
    print(f"  main.py agrees on {agree} of {len(CORRECTIONS)}; everything it misfiles lands in {sorted(misfiled)}")
    print(f"  counted from git: notes in lessons {sorted({hit[:2] for hit in disclaimers_at(FIRST)})} at {FIRST}; "
          f"{len(timing_repairs())} timing repairs; lessons {out_of_range_edits()} changed by {SECOND}")
    assert agree == 2 and misfiled == {"instruction"}
    assert [entry[1] for entry in CORRECTIONS].count("test") == 3
    assert [entry[0].recurrence for entry in CORRECTIONS] == [4, 4, 1, 1, 5]


# ---------------------------------------------------------------------------
# Exercise 2 - a prose rule replaced by an executable test
#
# The prose rule: a practice file says what a part needs ("needs pyyaml"); it
# does not disclaim the work. As prose it was broken in four files of the
# first commit before anyone noticed. As a test it is a scan over every
# practice file in the repository, and it runs here three ways: on the files
# as they are (clean), on the first commit (it finds the real cases), and on
# a planted file.
#
# The pattern (DISCLAIMER, at the top) is written so that it does not match
# its own source, and it leaves ordinary sentences alone ("a declined step
# does not run").
# ---------------------------------------------------------------------------

def disclaimers(root: Path) -> list[str]:
    return [f"{path.relative_to(root).as_posix()}:{number}"
            for path in sorted(root.glob("[0-9][0-9]-*/code/practice.py"))
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if DISCLAIMER.search(line)]


def ex2_rule_as_a_test() -> None:
    files = len(list(REPO.glob("[0-9][0-9]-*/code/practice.py")))
    found = disclaimers(REPO)
    with tempfile.TemporaryDirectory() as tmp:
        planted = Path(tmp) / "99-example" / "code" / "practice.py"
        planted.parent.mkdir(parents=True)
        planted.write_text("# The live call " + "was not " + "run here.\nprint('a declined step does not run')\n")
        caught = disclaimers(Path(tmp))
    then, after = disclaimers_at(FIRST), disclaimers_at(SECOND)
    print(f"  {files} practice files scanned, {len(found)} disclaimers")
    print(f"  the same scan on commit {FIRST}: {len(then)} lines in {len({hit[:2] for hit in then})} files ({', '.join(then)}); on {SECOND}: {len(after)}")
    print(f"  a planted file: caught at {caught}; its second line, an ordinary sentence, is left alone")
    assert files >= 40 and found == [] and caught == ["99-example/code/practice.py:1"]
    assert len(then) == 5 and {hit[:2] for hit in then} == {"01", "06", "08", "09"} and after == []


# ---------------------------------------------------------------------------
# Exercise 3 - consequence weighting
#
# main.py's ratchet promotes anything that has happened once, which is the
# one-off preference the lesson says not to promote. Here a correction is
# promoted when recurrence times severity reaches 3. A minor one has to come
# back three times. A severe one is promoted the first time, which is what
# the wrong statement that reached a pushed commit needed: waiting for it to
# happen again is the wrong experiment.
# ---------------------------------------------------------------------------

PROMOTE_AT = 3


def weighted_ratchet(corrections: list[Weighed]) -> list[Control]:
    return ratchet([correction for correction in corrections if correction.recurrence * correction.severity >= PROMOTE_AT])


def ex3_consequence_weighting() -> None:
    one_off = Weighed("A summary used 'firstly' where 'first' was wanted", "style preference was implicit", 1, "one word changed")
    corrections = [entry[0] for entry in CORRECTIONS] + [one_off]
    plain, weighted = ratchet(corrections), weighted_ratchet(corrections)
    promoted = {control.symptom for control in weighted}
    for correction in corrections:
        score = correction.recurrence * correction.severity
        print(f"  {correction.symptom[:52]:<52} {correction.recurrence:>2} x severity {correction.severity} = {score:>2}  "
              f"{'promoted' if correction.symptom in promoted else 'recorded, not promoted'}")
    print(f"  main.py promotes {len(plain)} of {len(corrections)}; weighted, {len(weighted)}")
    assert len(plain) == 6 and len(weighted) == 4 and one_off.symptom not in promoted
    assert CORRECTIONS[3][0].recurrence == 1 and CORRECTIONS[3][0].symptom in promoted       # severe, first occurrence
    assert CORRECTIONS[2][0].symptom not in promoted                                         # minor, first occurrence


# ---------------------------------------------------------------------------
# Exercise 4 - an owner and a retirement date on every control
#
# Each promoted correction is recorded under its real owner from exercise 1,
# not under main.py's guess, with what verifies it. A sixth correction from
# the same sessions is the one whose owner really is an instruction: the
# owner asked for a batch to be gone over again before it is committed, after
# commits kept changing files that were already on main. Two of the
# verifications are cheap enough to run here, and they are.
# The owner is a role that can change the control, by the layer it lives in.
# The review date depends on the layer too. An instruction costs context on
# every run and goes stale fastest, so it comes up for review after 90 days.
# A test costs nothing to keep, so a year. On the date the question is the
# lesson's: has the architecture changed, has a stronger control replaced it,
# has the failure stopped recurring, or does it cost more than it saves.
# ---------------------------------------------------------------------------

OWNER = {"test": "whoever maintains the suite", "scope": "whoever writes the task contracts",
         "automation": "whoever maintains the environment", "example": "whoever owns the canonical examples",
         "instruction": "whoever owns AGENTS.md"}
REVIEW_AFTER_DAYS = {"instruction": 90, "example": 180, "automation": 180, "scope": 365, "test": 365}


def record(control: Control, promoted_on: date) -> dict[str, Any]:
    return {**asdict(control), "owner": OWNER[control.target], "promoted_on": promoted_on.isoformat(),
            "review_on": (promoted_on + timedelta(days=REVIEW_AFTER_DAYS[control.target])).isoformat()}


ORDER_OF_WORK = Weighed("A batch was committed before it had been gone over again", "order of work was implicit", len(rework_commits()),
                        "files already on main kept changing")
REFINED_FIRST = ["git", "diff", "--quiet", BATCH, LATEST, "--", "3[1-9]-*", "40-*"]      # exits 0: the batch refined first was not changed later


def due(records: list[dict[str, Any]], today: date) -> list[str]:
    return [entry["rule"] for entry in records if date.fromisoformat(entry["review_on"]) <= today]


def ex4_owner_and_retirement() -> None:
    promoted_on = date(2026, 10, 9)
    promoted = {control.symptom for control in weighted_ratchet([entry[0] for entry in CORRECTIONS])}
    controls = [replace(promote(correction), target=owner, verification=check)
                for correction, owner, _, check in CORRECTIONS if correction.symptom in promoted]
    controls.append(replace(promote(ORDER_OF_WORK), verification=shlex.join(REFINED_FIRST)))
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "feedback-ratchet.json"
        output.write_text(json.dumps([record(control, promoted_on) for control in controls], indent=2) + "\n")
        records = json.loads(output.read_text())
    for entry in records:
        print(f"  {entry['target']:<11} {entry['rule'][:44]:<44} owner: {entry['owner']:<34} review on {entry['review_on']}")
        print(f"  {'':<11} verified by: {entry['verification']}")
    in_four_months, in_a_year = due(records, promoted_on + timedelta(days=120)), due(records, promoted_on + timedelta(days=365))
    print(f"  due for review after 120 days: {len(in_four_months)}; after a year: {len(in_a_year)} of {len(records)}")
    sleeping = [lesson for lesson in ("16", "22") if "sleep(" in next(REPO.glob(f"{lesson}-*/code/practice.py")).read_text(encoding="utf-8")]
    refined_first = subprocess.run(REFINED_FIRST, cwd=REPO, capture_output=True).returncode
    print(f"  run now: lessons 16 and 22 with a sleep call: {sleeping or 'none'}; the batch refined before its commit: exit {refined_first}")
    assert not sleeping and refined_first == 0 and choose_target(ORDER_OF_WORK) == "instruction" and len(rework_commits()) == 5
    assert all(entry["owner"] and entry["review_on"] > entry["promoted_on"] for entry in records)
    assert [entry["target"] for entry in records] == ["test", "test", "test", "scope", "instruction"]
    assert len(in_four_months) == sum(entry["target"] == "instruction" for entry in records) == 1
    assert len(in_a_year) == len(records)


# ---------------------------------------------------------------------------
# Exercise 5 - deleting an instruction, after proving a stronger control
#
# The instruction reviewed is a real one: the capstone pack's rule "Never
# edit a file outside the active task's scope contract". The stronger control
# is lesson 36's scope checker. The proof is not that the checker exists. It
# is that on every case the instruction is there to prevent, the checker
# refuses, and on a clean diff it does not, with nobody having read the
# rule. Only then is the rule taken out, and only from a copy.
#
# A second rule from the same file is reviewed and kept: "When confidence is
# below threshold, open a question note instead of guessing". Nothing
# executable in this repository can see a guess that was not reported, so
# the paragraph is still the only control there is.
# ---------------------------------------------------------------------------

RULES_FILE = REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack" / "docs" / "agent-rules.md"


def rule_blocks(text: str) -> dict[str, str]:
    return {block.split("\n", 1)[0].strip(): "## " + block for block in text.split("\n## ")[1:]}


def stronger_control_proven(control: Any, violations: list[Any], clean: Any) -> bool:
    return all(not control(case) for case in violations) and control(clean)


def ex5_delete_an_instruction() -> None:
    spec = importlib.util.spec_from_file_location("lesson36", REPO / "36-scope-contracts" / "code" / "main.py")
    scope_lesson = importlib.util.module_from_spec(spec)
    sys.modules["lesson36"] = scope_lesson
    spec.loader.exec_module(scope_lesson)
    contract = scope_lesson.ScopeContract(task_id="T-001", goal="add input validation to /signup", allowed_files=["app.py", "test_app.py"],
                                          forbidden_files=["scripts/**"], acceptance_criteria=[], rollback_plan="revert")

    def scope_checker(touched: list[str]) -> bool:
        return scope_lesson.scope_check(contract, scope_lesson.RunSummary(touched_files=touched, commands_run=[])).passed()

    rules = rule_blocks(RULES_FILE.read_text(encoding="utf-8"))
    out_of_scope, unsure = "forbidden/no-out-of-scope-writes", "uncertainty/open-question-note"
    proven = stronger_control_proven(scope_checker, violations=[["app.py", "config/settings.py"], ["scripts/release.sh"],
                                                                ["app.py", "billing/invoice.py", "billing/tax.py"]],
                                     clean=["app.py", "test_app.py"])
    decisions = {out_of_scope: proven, unsure: False}       # no executable control sees an unreported guess
    remaining = [slug for slug in rules if not decisions.get(slug)]
    print(f"  {out_of_scope}: {rules[out_of_scope].splitlines()[-1]}")
    print(f"    the scope checker refuses all three violating diffs and passes the clean one: {proven} -> delete")
    print(f"  {unsure}: no stronger control -> keep")
    print(f"  rules in the copy: {len(rules)} before, {len(remaining)} after; the pack's own file is not touched")
    assert len(rules) == 5 and proven and remaining == [slug for slug in rules if slug != out_of_scope]
    assert not stronger_control_proven(lambda touched: True, violations=[["scripts/release.sh"]], clean=["app.py"])     # a control that never refuses proves nothing


if __name__ == "__main__":
    print("Phase 14 - Lesson 46: Turn Every Agent Correction into a System Improvement - exercises")
    for exercise in (ex1_real_owners, ex2_rule_as_a_test, ex3_consequence_weighting, ex4_owner_and_retirement,
                     ex5_delete_an_instruction):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
