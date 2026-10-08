"""Phase 14 - Lesson 38: Verification Gates - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
Exercise 5 needs git and this repository's history.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any
from unittest import mock

import main as lesson
from main import Artifacts, Finding, VerdictReport, verify

REPO = Path(__file__).resolve().parents[2]
ACCEPTANCE = "pytest -x test_app.py::test_signup_rejects_short_password"


def artifacts(**changes: Any) -> Artifacts:
    """A task that did everything right, with the given fields changed."""
    clean = dict(task_id="T-001", acceptance_commands=[ACCEPTANCE], feedback=[{"command": ACCEPTANCE, "exit_code": 0}],
                 scope_report={"forbidden_writes": [], "off_scope_writes": []},
                 rule_report=[{"slug": "done/tests-pass", "passed": True}],
                 coverage_report={"current": 0.84, "previous": 0.84}, head_commit="a1b2c3d")
    return Artifacts(**{**clean, **changes})


def codes(report: VerdictReport) -> list[str]:
    return [f"{finding.severity}:{finding.code}" for finding in report.findings]


# ---------------------------------------------------------------------------
# Exercise 1 - a coverage floor of 80%
#
# main.py has the floor. Three reports get past it, all measured below:
#   - no report at all is a warning, so the cheapest way to meet the floor is
#     not to produce the report;
#   - a coverage of NaN is not below anything, so it is not below the floor;
#   - a report in percent (62 for 62%) clears a floor written as 0.80.
# gate() makes a missing report a block and rejects any value that is not a
# fraction between 0 and 1.
#
# Which artifact carries the floor. Not the coverage report: the test command
# writes it, and it should hold measurements only. Not the gate's command
# line: whoever runs the gate can pass --floor 0. The floor is a project rule,
# so it sits with the project's rules, written by a person before the task. A
# task's scope contract may raise it for that task and can never lower it.
# ---------------------------------------------------------------------------

PROJECT_FLOOR = 0.80


def effective_floor(task_floor: float | None = None) -> float:
    return max(PROJECT_FLOOR, task_floor or 0.0)


def gate(art: Artifacts, task_floor: float | None = None, strict: bool = False) -> VerdictReport:
    report = verify(art, strict=strict, coverage_floor=effective_floor(task_floor))
    findings = [replace(finding, severity="block") if finding.code == "coverage.missing" else finding for finding in report.findings]
    values = list((art.coverage_report or {}).values())
    if not all(isinstance(value, (int, float)) and 0.0 <= value <= 1.0 for value in values):
        findings.append(Finding("coverage.invalid", "block", f"coverage must be a fraction between 0 and 1, got {values}"))
    return replace(report, findings=findings, passed=not any(finding.severity == "block" for finding in findings))


def ex1_coverage_floor() -> None:
    reports = {"84%, was 85%": {"current": 0.84, "previous": 0.85}, "62%, was 80%": {"current": 0.62, "previous": 0.80},
               "no coverage report": None, "coverage is NaN": {"current": float("nan")},
               "62% written as 62": {"current": 62.0, "previous": 62.0}}
    verdicts = {}
    for label, coverage in reports.items():
        shipped, guarded = verify(artifacts(coverage_report=coverage)), gate(artifacts(coverage_report=coverage))
        verdicts[label] = (shipped.passed, guarded.passed)
        print(f"  {label:<20} main.py: {'pass' if shipped.passed else 'FAIL':<5} gate(): {'pass' if guarded.passed else 'FAIL':<5} "
              f"{codes(guarded)}")
    assert list(verdicts.values()) == [(True, True), (False, False), (True, False), (True, False), (True, False)]

    at_84 = artifacts(coverage_report={"current": 0.84, "previous": 0.84})
    floors = {"floor passed on the command line as 0": verify(at_84, coverage_floor=0.0).passed,
              "project floor 80%": gate(at_84).passed,
              "task contract raises it to 90%": gate(at_84, task_floor=0.90).passed,
              "task contract asks for 50%": effective_floor(0.50) == PROJECT_FLOOR and gate(at_84, task_floor=0.50).passed}
    for label, passed in floors.items():
        print(f"  84% coverage, {label:<38} {'pass' if passed else 'FAIL'}")
    assert list(floors.values()) == [True, True, False, True]


# ---------------------------------------------------------------------------
# Exercise 2 - --strict: every warning blocks
#
# main.py has the flag. When strict is the right default:
#   - release branches and main: a warning there ships;
#   - unattended runs: a warning nobody reads is a pass;
#   - a diff that touches auth, payments, migrations, scripts or CI config:
#     the cost of being wrong is not the cost of a typo;
#   - a new model or agent version, until its warnings have been looked at.
# When it is the wrong default: day-to-day feature branches with a person in
# the loop. If most runs fail on something the person then waves through, the
# override becomes a reflex and the block findings lose their weight too.
# strict_by_default writes the first three cases down as a check.
# ---------------------------------------------------------------------------

SENSITIVE_PATHS = ["auth/**", "payments/**", "migrations/**", "scripts/**", ".github/**"]


def strict_by_default(branch: str, unattended: bool, touched: list[str]) -> str | None:
    """The reason strict mode applies, or None when the lenient default is fine."""
    if branch in ("main", "master") or branch.startswith("release/"):
        return "release branch"
    if unattended:
        return "nobody is watching this run"
    sensitive = [path for path in touched if any(fnmatch.fnmatch(path, pattern) for pattern in SENSITIVE_PATHS)]
    return f"touches {sensitive[0]}" if sensitive else None


def ex2_strict_mode() -> None:
    warned = artifacts(scope_report={"forbidden_writes": [], "off_scope_writes": ["CHANGELOG.md"]},
                       coverage_report={"current": 0.845, "previous": 0.85})
    lenient, strict = verify(warned), verify(warned, strict=True)
    print(f"  two warnings, default: {'pass' if lenient.passed else 'FAIL'} {codes(lenient)}")
    print(f"  two warnings, strict : {'pass' if strict.passed else 'FAIL'} {codes(strict)}")
    assert lenient.passed and not strict.passed and len(strict.findings) == 2
    assert verify(artifacts(), strict=True).passed                      # strict does not fail a clean task

    cases = [("feature/signup", False, ["app.py"]), ("release/2026.10", False, ["app.py"]),
             ("feature/signup", True, ["app.py"]), ("feature/signup", False, ["app.py", "migrations/0007_add_index.sql"])]
    reasons = [strict_by_default(*case) for case in cases]
    for (branch, unattended, touched), reason in zip(cases, reasons):
        print(f"  {branch:<16} unattended={unattended!s:<5} {len(touched)} files -> {'strict: ' + reason if reason else 'lenient'}")
    assert reasons == [None, "release branch", "nobody is watching this run", "touches migrations/0007_add_index.sql"]


# ---------------------------------------------------------------------------
# Exercise 3 - a Markdown summary beside the JSON
#
# Both files are written from the same VerdictReport, so they cannot disagree.
#
# What belongs in the summary is what changes the reader's decision:
#   the verdict            first, in the heading
#   the task and commit    a verdict with no commit can be pasted under any diff
#   blocking findings      what has to be fixed, with the detail
#   warnings               what the reader accepts by letting it through
#   coverage and the floor one line
#   strict or not          the same findings mean different things
# What stays in the JSON is what proves it: checks that passed, info findings,
# the feedback log, timestamps.
#
# Finding details carry file names and commands the agent chose, so each one
# is flattened to a single line before it is written. A file name with a line
# break in it cannot add a second heading that says PASS.
# ---------------------------------------------------------------------------

def one_line(text: object) -> str:
    return " ".join(str(text).split()).replace("`", "'")


def to_markdown(report: VerdictReport, floor: float = PROJECT_FLOOR) -> str:
    blocking = [finding for finding in report.findings if finding.severity == "block"]
    warnings = [finding for finding in report.findings if finding.severity == "warn"]
    lines = [f"## Verification: {'PASS' if report.passed else 'FAIL'} - {one_line(report.task_id)} at {one_line(report.head_commit) or 'no commit'}",
             "", f"{len(blocking)} blocking, {len(warnings)} warning{'s' * (len(warnings) != 1)}{' (strict: warnings block)' if report.strict else ''}"]
    if report.coverage:
        lines.append(f"Coverage {report.coverage.get('current', 0.0):.0%}, floor {floor:.0%}")
    for title, group in (("Blocking", blocking), ("Warnings", warnings)):
        if group:
            lines += ["", f"### {title}", *[f"- `{finding.code}`: {one_line(finding.detail)}" for finding in group]]
    return "\n".join([*lines, "", "Evidence: verification_report.json", ""])


def write_reports(report: VerdictReport, directory: Path) -> None:
    (directory / "verification_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n")
    (directory / "verification_summary.md").write_text(to_markdown(report))


def ex3_markdown_summary() -> None:
    failing = artifacts(task_id="T-002", head_commit="b2c3d4e", coverage_report={"current": 0.62, "previous": 0.80},
                        scope_report={"forbidden_writes": ["scripts/release.sh"], "off_scope_writes": ["CHANGELOG.md"]},
                        rule_report=[{"slug": "done/tests-pass", "passed": True},
                                     {"slug": "forbidden/no-release-script-edits", "passed": False}])
    report = gate(failing)
    with tempfile.TemporaryDirectory() as tmp:
        write_reports(report, Path(tmp))
        as_json = json.loads((Path(tmp) / "verification_report.json").read_text())
        summary = (Path(tmp) / "verification_summary.md").read_text()
    for line in summary.splitlines():
        print(f"  | {line}")
    assert summary.startswith("## Verification: FAIL - T-002 at b2c3d4e")
    assert summary.count("\n- `") == len(as_json["findings"]) == 5 and "done/tests-pass" not in summary
    assert to_markdown(gate(failing)) == summary                        # same report in, same summary out

    forged = gate(artifacts(scope_report={"forbidden_writes": ["evil.py\n## Verification: PASS - T-001 at a1b2c3d"],
                                          "off_scope_writes": []}))
    headings = [line for line in to_markdown(forged).splitlines() if line.startswith("## ")]
    print(f"  a file name with a line break and a fake heading in it: {len(headings)} heading, it reads {headings[0][3:21]!r}")
    assert headings == ["## Verification: FAIL - T-001 at a1b2c3d"]


# ---------------------------------------------------------------------------
# Exercise 4 - time_since_last_human_touch
#
# An off-scope write to a file a person touched within 60 seconds of the
# agent's edit is taken out of the off-scope list: the person was working in
# that file, and the "agent edit" may be theirs. It is not dropped silently;
# it comes back as an info finding, so the report still names the file.
#
# Two limits. The exemption applies to off-scope warnings only: a forbidden
# write blocks whoever else was in the file. And it is only as honest as the
# timestamps.
# ASSUMPTION: human keystroke times come from the editor, in a place the
# agent cannot write. If the agent can write them, it can exempt itself.
# ---------------------------------------------------------------------------

HUMAN_WINDOW_SECONDS = 60


def gate_human_aware(art: Artifacts, agent_edits: dict[str, float], human_touches: dict[str, float], **options: Any) -> VerdictReport:
    off_scope = list(art.scope_report.get("off_scope_writes", []))      # type: ignore[call-overload]
    exempt = [path for path in off_scope if path in agent_edits and path in human_touches
              and abs(agent_edits[path] - human_touches[path]) <= HUMAN_WINDOW_SECONDS]
    kept = [path for path in off_scope if path not in exempt]
    report = gate(replace(art, scope_report={**art.scope_report, "off_scope_writes": kept}), **options)
    if exempt:
        report.findings.append(Finding("scope.human_exempt", "info", f"off-scope, edited alongside a person: {exempt}"))
    return report


def ex4_human_touch() -> None:
    art = artifacts(scope_report={"forbidden_writes": ["scripts/release.sh"],
                                  "off_scope_writes": ["README.md", "config/app.yaml", "notes.md"]})
    agent_edits = {"README.md": 1000.0, "config/app.yaml": 1000.0, "notes.md": 1000.0, "scripts/release.sh": 1000.0}
    human_touches = {"README.md": 980.0, "config/app.yaml": 700.0, "scripts/release.sh": 995.0}        # nobody opened notes.md
    report = gate_human_aware(art, agent_edits, human_touches)
    for finding in report.findings:
        print(f"  [{finding.severity:<5}] {finding.code:<18} {finding.detail}")
    details = {finding.code: finding.detail for finding in report.findings}
    assert "README.md" in details["scope.human_exempt"] and "README.md" not in details["scope.off_scope"]
    assert "config/app.yaml" in details["scope.off_scope"] and "notes.md" in details["scope.off_scope"]
    assert "scripts/release.sh" in details["scope.forbidden"] and not report.passed
    only_readme = replace(art, scope_report={"forbidden_writes": [], "off_scope_writes": ["README.md"]})
    assert codes(gate_human_aware(only_readme, agent_edits, human_touches, strict=True)) == ["info:scope.human_exempt"]


# ---------------------------------------------------------------------------
# Exercise 5 - the gate on a real agent diff
#
# The diff is commit 563dbed of this repository, "Add exercise solutions for
# lessons 11-30", written by an agent. Everything fed to the gate is read
# from git: the touched files, a scope report from lesson 36's checker
# against the contract as the task was given, two project rules checked on
# the committed bytes, and an acceptance check that compiles every committed
# Python file.
#
# Measured: the gate passes, with 2 warnings covering 7 items.
#   Real, 5: practice files for lessons 01, 06, 08, 09 and 10. The task was
#     lessons 11 to 30. The owner asked mid-task for wording changes in the
#     earlier files, so the edits were wanted, but nothing written down says
#     so. The gate is right to ask.
#   Noise, 2: practice.ts in lesson 18, which the lesson's own exercise asks
#     for and the contract's "*.py" pattern did not foresee; and the missing
#     coverage report, in a repository that has no coverage tool and never
#     will, which makes it a warning on every run.
#
# Where the gate needs to grow, each one measured below:
#   - A scope change agreed in conversation has nowhere to go. An override
#     can be recorded and signed, and verify() never reads the override log,
#     so the strict verdict stays FAIL.
#   - It reads every feedback record ever written. An acceptance command that
#     failed once and then passed still blocks, which is every real agent
#     loop. It needs the latest run of each command.
#   - It never ties a green run to the commit. A pass recorded before the
#     last edit counts the same as one recorded after.
#   - A check that cannot apply to a repository needs to be switched off for
#     it, not left as a permanent warning.
# ---------------------------------------------------------------------------

REAL_COMMIT = "563dbed"
SYNTAX_CHECK = "compile every committed Python file"


def latest_per_command(feedback: list[dict[str, object]]) -> list[dict[str, object]]:
    return list({str(record.get("command")): record for record in feedback}.values())


def ex5_real_agent_diff() -> str | None:
    def git(*args: str) -> bytes:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True, timeout=10).stdout

    try:
        touched = git("show", "--name-only", "--format=", REAL_COMMIT).decode().split()
        blobs = {path: git("show", f"{REAL_COMMIT}:{path}") for path in touched}
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print("  needs git and this repository's history")
        return "git"

    def compiles(path: str) -> bool:
        try:
            compile(blobs[path].decode("utf-8"), path, "exec")
            return True
        except (SyntaxError, UnicodeDecodeError):
            return False

    spec = importlib.util.spec_from_file_location("lesson36", REPO / "36-scope-contracts" / "code" / "main.py")
    lesson36 = importlib.util.module_from_spec(spec)
    sys.modules["lesson36"] = lesson36
    spec.loader.exec_module(lesson36)
    contract = lesson36.ScopeContract(
        task_id="T-11-30", goal="exercise solutions for lessons 11 to 30", rollback_plan=f"git revert {REAL_COMMIT}",
        allowed_files=["1[1-9]-*/code/practice.py", "2[0-9]-*/code/practice.py", "30-*/code/practice.py"],
        forbidden_files=["*/code/main.py", "*/docs/*", "*/quiz.json"], acceptance_criteria=[SYNTAX_CHECK])
    scope = lesson36.scope_check(contract, lesson36.RunSummary(touched_files=touched, commands_run=[SYNTAX_CHECK]))
    python_files = [path for path in touched if path.endswith(".py")]
    art = Artifacts(
        task_id=contract.task_id, acceptance_commands=[SYNTAX_CHECK], head_commit=REAL_COMMIT,
        feedback=[{"command": SYNTAX_CHECK, "exit_code": 0 if all(compiles(path) for path in python_files) else 1}],
        scope_report={"forbidden_writes": scope.forbidden_writes, "off_scope_writes": scope.off_scope_writes},
        rule_report=[{"slug": "done/ascii-only", "passed": all(max(blob, default=0) < 128 for blob in blobs.values())},
                     {"slug": "done/lf-line-endings", "passed": not any(bytes([13, 10]) in blob for blob in blobs.values())}])
    report, strict = verify(art), verify(art, strict=True)
    print(f"  commit {REAL_COMMIT}: {len(touched)} files touched, all {len(python_files)} Python files compile, 2 rules checked")
    print(f"  gate: {'pass' if report.passed else 'FAIL'}; strict: {'pass' if strict.passed else 'FAIL'}")
    for finding in report.findings:
        print(f"    [{finding.severity}] {finding.code}: {finding.detail[:96]}")
    earlier = [path for path in scope.off_scope_writes if path[:2] <= "10"]
    print(f"  off scope: {len(earlier)} practice files from lessons 01-10, plus {sorted(set(scope.off_scope_writes) - set(earlier))}")
    assert report.passed and not strict.passed and codes(report) == ["warn:scope.off_scope", "warn:coverage.missing"]
    assert len(earlier) == 5 and set(scope.off_scope_writes) - set(earlier) == {"18-agno-and-mastra-runtimes/code/practice.ts"}
    assert scope.forbidden_writes == [] and len(touched) == 26

    with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VERIFY_OVERRIDE_SECRET": "practice-file-only"}):
        saved, lesson.OVERRIDES_PATH = lesson.OVERRIDES_PATH, Path(tmp) / "overrides.jsonl"
        try:
            entry = lesson.record_override(contract.task_id, "scope.off_scope", "owner asked mid-task for wording changes in lessons 1-10",
                                           "Brycekoh", REAL_COMMIT)
            signed, tampered = lesson.verify_signature(entry), lesson.verify_signature({**entry, "reason": "because"})
        finally:
            lesson.OVERRIDES_PATH = saved
    print(f"  override recorded: signature checks {signed}, edited entry checks {tampered}; strict gate afterwards: "
          f"{'pass' if verify(art, strict=True).passed else 'FAIL'}")
    assert signed and not tampered and not verify(art, strict=True).passed

    retried = replace(art, feedback=[{"command": SYNTAX_CHECK, "exit_code": 1}, {"command": SYNTAX_CHECK, "exit_code": 0}])
    all_runs, last_runs = verify(retried), verify(replace(retried, feedback=latest_per_command(retried.feedback)))
    print(f"  failed once, then passed: gate {'pass' if all_runs.passed else 'FAIL'}; on the latest run of each command: "
          f"{'pass' if last_runs.passed else 'FAIL'}")
    assert not all_runs.passed and "block:acceptance.failed" in codes(all_runs) and last_runs.passed

    stale = replace(art, feedback=[{"command": SYNTAX_CHECK, "exit_code": 0, "head_commit": "90bedf6"}])
    print(f"  a green run recorded on the commit before: gate {'pass' if verify(stale).passed else 'FAIL'}")
    assert codes(verify(stale)) == codes(report)
    return None


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder.
#   exits zero                               yes
#   the clean pass reports passed: true,     yes
#   the other two passed: false
#   each scenario writes its own report      three separate reports, written
#   under outputs/verification/              next to the script instead
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
    reports = {name: json.loads(text) for name, text in files.items() if "verification_report" in name}
    passed = {report["task_id"]: report["passed"] for report in reports.values()}
    print(f"  main.py exits {exit_code}; verdicts {passed}")
    print(f"  reports: {list(reports)}")
    assert exit_code == 0 and passed == {"T-001": True, "T-002": False, "T-003": False}
    assert list(reports) == [f"code/verification_report_T-00{number}.json" for number in (1, 2, 3)]


if __name__ == "__main__":
    print("Phase 14 - Lesson 38: Verification Gates - exercises")
    missing = []
    for exercise in (ex1_coverage_floor, ex2_strict_mode, ex3_markdown_summary, ex4_human_touch, ex5_real_agent_diff, mission_acceptance):
        print(f"\n{exercise.__name__}")
        missing.append(exercise())
    missing = [name for name in missing if name]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
