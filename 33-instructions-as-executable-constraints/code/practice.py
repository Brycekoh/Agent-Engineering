"""Phase 14 - Lesson 33: Instructions as Executable Constraints - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --ci <trace.json>        (exercise 3 as a command)
           python practice.py --classify <AGENTS.md>   (exercise 5 on any file)
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import main as lesson
from main import Rule, RuleChecker, TurnTrace

REPO = Path(__file__).resolve().parents[2]
SEVERITIES = ("block", "warn", "info")

# The five seed rules from main.py with a severity and an added date each,
# plus the two budget rules of exercise 1.
RULES = """\
# Agent Rules

## startup/state-file-fresh
- category: startup
- check: state_file_fresh
- severity: block
- added: 2026-06-01
Agent must read agent_state.json before any tool call.

## forbidden/no-release-script-edits
- category: forbidden
- check: no_release_script_edits
- severity: block
- added: 2026-06-01
- review_after_days: 365
Never edit scripts/release.sh outside an approved release task.

## done/tests-pass
- category: definition_of_done
- check: tests_pass
- severity: block
- added: 2026-06-01
A task is done only when its acceptance command exits zero.

## uncertainty/open-question-note
- category: uncertainty
- check: opened_question_when_unsure
- severity: warn
- added: 2026-06-01
When confidence is below threshold, write a question note instead of guessing.

## approval/new-dependency
- category: approval
- check: new_dependency_approved
- severity: block
- added: 2026-06-01
Adding a runtime dependency requires explicit human approval.

## budget/tool-call-limit
- category: budget
- check: within_tool_call_limit
- severity: block
- added: 2026-06-01
A turn stops at 200 tool calls and writes a blocker instead of continuing.

## budget/typical-tool-calls
- category: budget
- check: within_typical_tool_calls
- severity: info
- added: 2026-06-01
A turn that needs more than 50 tool calls is recorded so the trend is visible.
"""


@dataclass
class Trace(TurnTrace):
    tool_calls: int = 0
    read_board: bool = True
    active_task_status: str = "done"
    scope: list[str] | None = None          # the files the active task may edit; None when the turn had no contract


@dataclass
class FullRule(Rule):
    severity: str = "block"
    added: str = ""
    review_after_days: int = 90


class Checker(RuleChecker):
    def within_tool_call_limit(self, trace: Trace) -> bool:
        return trace.tool_calls <= 200

    def within_typical_tool_calls(self, trace: Trace) -> bool:
        return trace.tool_calls <= 50

    def board_read(self, trace: Trace) -> bool:
        return trace.read_board

    def active_task_marked_done(self, trace: Trace) -> bool:
        return trace.active_task_status == "done"

    def no_out_of_scope_writes(self, trace: Trace) -> bool:
        return trace.scope is not None and all(path in trace.scope for path in trace.edited_files)


def turn(**changes: object) -> Trace:
    """A clean turn, with the given fields changed."""
    clean = dict(read_state_file=True, edited_files=["app.py", "test_app.py"], confidence=0.9, asked_for_help=False,
                 tests_exit_code=0, added_dependencies=[], tool_calls=12, scope=["app.py", "test_app.py"])
    return Trace(**{**clean, **changes})        # type: ignore[arg-type]


def parse_base(text: str) -> list[Rule]:
    """main.py's parser, pointed at a text instead of the file next to it.

    main.py drops a rule that has no category or check line and says nothing.
    A dropped rule is a rule nobody enforces, so here it is an error.
    """
    original = lesson.RULES_PATH
    with tempfile.TemporaryDirectory() as tmp:
        lesson.RULES_PATH = Path(tmp) / "agent-rules.md"
        lesson.RULES_PATH.write_text(text)
        try:
            rules = lesson.parse_rules()
        finally:
            lesson.RULES_PATH = original
    headings = len(re.findall(r"^## ", text, re.M))
    if len(rules) != headings:
        raise ValueError(f"{headings} rule headings but {len(rules)} rules parsed: a rule is missing its category or check")
    return rules


def parse(text: str) -> list[FullRule]:
    rules = []
    for rule, block in zip(parse_base(text), re.split(r"^## ", text, flags=re.M)[1:]):
        extra = dict(re.findall(r"^- (severity|added|review_after_days): (\S+)", block, re.M))
        if extra.get("severity", "block") not in SEVERITIES or "added" not in extra:
            raise ValueError(f"{rule.slug}: needs an added date and a severity from {SEVERITIES}")
        rules.append(FullRule(**asdict(rule), severity=extra.get("severity", "block"), added=extra["added"],
                              review_after_days=int(extra.get("review_after_days", 90))))
    return rules


# ---------------------------------------------------------------------------
# Exercise 1 - a sixth category: budget
#
# ASSUMPTION about the product: the agent runs unattended, so nobody is
# watching when it gets stuck in a loop.
#
# Why budget does not collapse into one of the five:
#   forbidden           is a property of one action. No single call out of 400
#                       is forbidden; the sum is the problem.
#   approval            unlocks an action a human has looked at. There is no
#                       action here to look at.
#   definition_of_done  describes the end state. A looping agent can still
#                       arrive at a passing end state.
#   uncertainty         depends on the agent noticing it is unsure. A looping
#                       agent is usually confident.
#   startup             is checked once, before the first call.
# A budget rule reads a running counter, and breaking it stops the loop.
# ---------------------------------------------------------------------------

def ex1_sixth_category() -> None:
    rules = parse(RULES)
    stuck = turn(tool_calls=400)
    results = lesson.score(rules, Checker(), stuck)
    by_category: dict[str, list[bool]] = {}
    for result in results:
        by_category.setdefault(str(result["category"]), []).append(bool(result["passed"]))
    for category, passed in by_category.items():
        print(f"  {category:<19} {'pass' if all(passed) else 'FAIL'}")
    print("  a turn of 400 tool calls passes every rule in the five categories; only budget notices")
    assert [category for category, passed in by_category.items() if not all(passed)] == ["budget"]
    assert len(by_category) == 6


# ---------------------------------------------------------------------------
# Exercise 2 - a severity on each rule, and a report that aggregates by it
#
# block  the turn does not count as done
# warn   the turn stands, and a human is shown the failure
# info   recorded for the trend, nobody is interrupted
# The verdict is the worst severity that failed. A severity that is not one
# of the three is rejected when the file is parsed: a misspelt "block" must
# not turn into a rule that no longer blocks.
# ---------------------------------------------------------------------------

def report(rules: list[FullRule], trace: Trace) -> dict[str, object]:
    results = lesson.score(rules, Checker(), trace)
    failed = {severity: [rule.slug for rule, result in zip(rules, results) if rule.severity == severity and not result["passed"]]
              for severity in SEVERITIES}
    verdict = "fail" if failed["block"] else "warn" if failed["warn"] else "pass"
    return {"verdict": verdict, "failed": failed, "passed": sum(bool(result["passed"]) for result in results), "total": len(results)}


def ex2_severity() -> None:
    rules = parse(RULES)
    turns = {
        "clean turn": turn(),
        "long turn (60 calls)": turn(tool_calls=60),
        "guessed while unsure": turn(confidence=0.4),
        "everything wrong": turn(read_state_file=False, edited_files=["scripts/release.sh"], confidence=0.4, tests_exit_code=1,
                                 added_dependencies=["fastapi"], tool_calls=300),
    }
    verdicts = {}
    for name, trace in turns.items():
        summary = report(rules, trace)
        failed = summary["failed"]
        counts = ", ".join(f"{len(failed[severity])} {severity}" for severity in SEVERITIES)       # type: ignore[index]
        print(f"  {name:<21} {summary['verdict']:<4} ({summary['passed']}/{summary['total']} pass; failed: {counts})")
        verdicts[name] = summary["verdict"]
    assert list(verdicts.values()) == ["pass", "pass", "warn", "fail"]
    assert report(rules, turns["long turn (60 calls)"])["failed"]["info"] == ["budget/typical-tool-calls"]      # type: ignore[index]

    rejected = []
    for broken in (RULES.replace("severity: warn", "severity: wran"), RULES.replace("- check: tests_pass\n", "")):
        try:
            parse(broken)
        except ValueError as error:
            rejected.append(str(error))
    print(f"  misspelt severity    rejected: {rejected[0]}")
    print(f"  rule with no check   rejected: {rejected[1]}")
    assert len(rejected) == 2


# ---------------------------------------------------------------------------
# Exercise 3 - the checker in CI
#
# The build fails when a block rule fails on the latest agent run. It also
# fails when there is no trace to check: a gate that passes on missing
# evidence is not a gate. Warn and info failures are printed and the build
# continues.
#
# ASSUMPTION: the agent run writes its trace to outputs/latest_trace.json.
# The workflow below is kept as text; copy it to .github/workflows/ to use it.
# ---------------------------------------------------------------------------

WORKFLOW = """\
name: agent-rules
on: [pull_request]
jobs:
  rules:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: python 33-instructions-as-executable-constraints/code/practice.py --ci outputs/latest_trace.json
"""


def ci(trace_path: Path) -> tuple[int, str]:
    if not trace_path.exists():
        return 1, f"FAIL: no trace at {trace_path.name}"
    summary = report(parse(RULES), Trace(**json.loads(trace_path.read_text())))
    failed = summary["failed"]
    lines = [f"{severity}: {slug}" for severity in SEVERITIES for slug in failed[severity]]      # type: ignore[index]
    verdict = "FAIL" if summary["verdict"] == "fail" else "ok"
    return int(verdict == "FAIL"), "\n".join([f"{verdict}: {summary['passed']}/{summary['total']} rules pass", *lines])


def ex3_ci_gate() -> None:
    cases = {"clean": turn(), "warn_only": turn(confidence=0.4, tool_calls=60), "blocked": turn(tests_exit_code=1)}
    codes = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, trace in cases.items():
            (Path(tmp) / f"{name}.json").write_text(json.dumps(asdict(trace)))
        for name in (*cases, "missing"):
            done = subprocess.run([sys.executable, __file__, "--ci", str(Path(tmp) / f"{name}.json")], capture_output=True, text=True)
            codes[name] = done.returncode
            print(f"  {name:<10} exit {done.returncode}  {' | '.join(done.stdout.split(chr(10))).strip(' |')}")
    assert codes == {"clean": 0, "warn_only": 0, "blocked": 1, "missing": 1}
    assert "practice.py --ci" in WORKFLOW


# ---------------------------------------------------------------------------
# Exercise 4 - an expiry on each rule
#
# Each rule carries review_after_days (90 unless it says otherwise) and the
# date it was added. The last failure of each rule is kept beside the rules,
# not in them, so the rules file only changes when a person edits it.
#
# A rule that has not failed for that long is put up for review, not removed.
# A person decides which of three things is true: the check can no longer
# fail, the situation it guarded against is gone, or the rule is doing its
# job quietly. The release-script rule sets 365 days for that last reason: it
# guards a rare event, and a quiet year is the rule working.
# ---------------------------------------------------------------------------

def record_failures(history: dict[str, str], rules: list[FullRule], trace: Trace, today: date) -> None:
    for result in lesson.score(rules, Checker(), trace):
        if not result["passed"]:
            history[str(result["slug"])] = today.isoformat()


def up_for_review(rules: list[FullRule], history: dict[str, str], today: date) -> list[str]:
    return [rule.slug for rule in rules
            if (today - date.fromisoformat(history.get(rule.slug, rule.added))).days >= rule.review_after_days]


def ex4_rule_expiry() -> None:
    rules = parse(RULES)
    start, history = date(2026, 6, 1), {}
    for day in range(130):          # SIMULATED: one agent run a day
        changes: dict[str, object] = {"tool_calls": 60 if day >= 80 else 12}
        if day == 10:
            changes["confidence"] = 0.4
        if day == 120:
            changes["tests_exit_code"] = 1
        record_failures(history, rules, turn(**changes), start + timedelta(days=day))
    today = start + timedelta(days=129)
    review = up_for_review(rules, history, today)
    for rule in rules:
        quiet = (today - date.fromisoformat(history.get(rule.slug, rule.added))).days
        last = f"last failed {quiet}d ago" if rule.slug in history else f"never failed in {quiet}d"
        print(f"  {rule.slug:<34} {last:<22} limit {rule.review_after_days:>3}d  {'REVIEW' if rule.slug in review else ''}")
    assert review == ["startup/state-file-fresh", "uncertainty/open-question-note", "approval/new-dependency", "budget/tool-call-limit"]


# ---------------------------------------------------------------------------
# Exercise 5 - a real AGENTS.md rewritten as five-category rules
#
# Two real files from this repository: the AGENTS.md that lesson 32 lays
# down, and the one the course ships in its capstone pack (lesson 42). A line
# counts as operational when it names something a check can look at: a file,
# a command, a field or a number. It is aspirational when it names nothing a
# check could look at. Headings and lead-in lines are structure.
#
# Measured.
#   lesson 32    9 lines of text: 7 operational, 0 aspirational, 2 structure
#   the pack    11 lines of text: 8 operational, 1 aspirational, 2 structure
# The pack's one line that no check can look at is its opening sentence,
# which sets the scene and asks for nothing.
#
# Lesson 32's 7 operational lines hold 4 rules, in 2 of the 5 categories.
# Nothing in it says what is forbidden, what to do when unsure, or what needs
# approval; it points to docs/agent-rules.md for that, a file lesson 32 never
# creates. The pack ships that file, and main.py's parser reads it as 5 rules
# in all 5 categories, so in the pack the pointer leads somewhere.
#
# One of the pack's five names a check, no_out_of_scope_writes, that main.py's
# RuleChecker does not have. main.py scores a rule with no check as failed,
# so that rule fails on every turn, a clean one included, until the check is
# written. The Checker in this file has it.
# ---------------------------------------------------------------------------

FIVE_CATEGORIES = {"startup", "forbidden", "definition_of_done", "uncertainty", "approval"}
PACK = REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack"

# ponytail: "names something checkable" is a backtick or a number; label by hand when a file needs finer judgement
CHECKABLE = re.compile(r"`[^`]+`|\b\d+\b")

REWRITTEN = """\
# Agent Rules

## startup/read-state
- category: startup
- check: state_file_fresh
Read agent_state.json before acting.

## startup/read-board
- category: startup
- check: board_read
Read task_board.json before acting.

## done/board-status-done
- category: definition_of_done
- check: active_task_marked_done
The active task has status done on task_board.json.

## done/verification-exits-zero
- category: definition_of_done
- check: tests_pass
python3 -m pytest -x, the command in the task's acceptance, has exited 0.
"""


def classify(line: str) -> str:
    text = re.sub(r"^(\d+\.|[-*])\s+", "", line.strip())
    if not text or text.startswith("#") or text.endswith(":"):
        return "structure"
    return "operational" if CHECKABLE.search(text) else "aspirational"


def count_lines(text: str) -> dict[str, int]:
    labels = [classify(line) for line in text.splitlines() if line.strip()]
    return {label: labels.count(label) for label in ("operational", "aspirational", "structure")}


def ex5_rewrite_agents_md() -> None:
    made_up = {"Write clean, maintainable code.": "aspirational", "- Follow best practices.": "aspirational",
               "- Run `npm test` before every commit.": "operational", "Keep functions under 40 lines.": "operational"}
    assert {line: classify(line) for line in made_up} == made_up        # the classifier, on lines written for this test

    spec = importlib.util.spec_from_file_location("lesson32", REPO / "32-minimal-agent-workbench" / "code" / "main.py")
    lesson32 = importlib.util.module_from_spec(spec)
    sys.modules["lesson32"] = lesson32
    spec.loader.exec_module(lesson32)
    counts = count_lines(lesson32.AGENTS_MD)
    print(f"  lesson 32's AGENTS.md: {counts}")
    assert counts == {"operational": 7, "aspirational": 0, "structure": 2}

    rules = parse_base(REWRITTEN)
    covered = {rule.category for rule in rules}
    gaps = sorted(FIVE_CATEGORIES - covered)
    print(f"  rewritten as {len(rules)} rules in {sorted(covered)}")
    print(f"  categories with no rule: {gaps}")
    clean = lesson.score(rules, Checker(), turn())
    sloppy = lesson.score(rules, Checker(), turn(read_board=False, active_task_status="in_progress"))
    print(f"  every rewritten rule has a check: clean turn {sum(bool(r['passed']) for r in clean)}/4, "
          f"sloppy turn {sum(bool(r['passed']) for r in sloppy)}/4")
    assert len(rules) == 4 and gaps == ["approval", "forbidden", "uncertainty"]
    assert all(result["passed"] for result in clean) and sum(bool(result["passed"]) for result in sloppy) == 2

    pack_counts = count_lines((PACK / "AGENTS.md").read_text(encoding="utf-8"))
    pack_rules = parse_base((PACK / "docs" / "agent-rules.md").read_text(encoding="utf-8"))
    print(f"  the capstone pack's AGENTS.md: {pack_counts}")
    print(f"  its docs/agent-rules.md: {len(pack_rules)} rules in {len({rule.category for rule in pack_rules})} categories")
    assert pack_counts == {"operational": 8, "aspirational": 1, "structure": 2}
    assert len(pack_rules) == 5 and {rule.category for rule in pack_rules} == FIVE_CATEGORIES

    def failed(checker: RuleChecker, trace: Trace) -> list[str]:
        return [str(result["slug"]) for result in lesson.score(pack_rules, checker, trace) if not result["passed"]]

    print(f"  the pack's rules on a clean turn, main.py's checker: fails {failed(RuleChecker(), turn())}")
    print(f"  with the missing check written: clean turn fails {failed(Checker(), turn())}, "
          f"a turn that also edits README.md fails {failed(Checker(), turn(edited_files=['app.py', 'README.md']))}")
    assert failed(RuleChecker(), turn()) == ["forbidden/no-out-of-scope-writes"] and failed(Checker(), turn()) == []
    assert failed(Checker(), turn(edited_files=["app.py", "README.md"])) == ["forbidden/no-out-of-scope-writes"]
    assert failed(Checker(), turn(scope=None)) == ["forbidden/no-out-of-scope-writes"]         # no contract is not a pass


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder.
#   exits zero                               yes
#   prints the rule set, the run trace and   the rule set and pass or fail per
#   pass or fail per rule                    rule; the trace is written to
#                                            rule_report.json, not printed
#   rule_report.json catches the two         the demo run breaks all five
#   intentional violations                   rules, and all five are caught
# The mission also lists an aggregate severity in the report. main.py's report
# has none; exercise 2 adds it.
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
    (exit_code, stdout, files), = run_main()
    written = json.loads(files["code/rule_report.json"])
    caught = [row["slug"] for row in written["bad"] if not row["passed"]]
    print(f"  main.py exits {exit_code}; rule_report.json catches {len(caught)} of the {len(written['bad'])} rules the bad trace breaks")
    assert exit_code == 0 and "rules parsed:" in stdout and stdout.count("FAIL") == 5 and stdout.count("PASS") == 5
    assert len(caught) == 5 and all(row["passed"] for row in written["good"]) and written["trace_bad"]["tests_exit_code"] == 1
    assert "severity" not in json.dumps(written) and report(parse(RULES), turn())["verdict"] == "pass"


if __name__ == "__main__":
    if sys.argv[1:2] == ["--ci"]:
        code, message = ci(Path(sys.argv[2]))
        print(message)
        sys.exit(code)
    if sys.argv[1:2] == ["--classify"]:
        print(count_lines(Path(sys.argv[2]).read_text(encoding="utf-8")))
        sys.exit(0)
    print("Phase 14 - Lesson 33: Instructions as Executable Constraints - exercises")
    for exercise in (ex1_sixth_category, ex2_severity, ex3_ci_gate, ex4_rule_expiry, ex5_rewrite_agents_md, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
