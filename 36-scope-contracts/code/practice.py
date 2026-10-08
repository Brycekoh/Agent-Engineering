"""Phase 14 - Lesson 36: Scope Contracts and Task Boundaries - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import glob
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from main import RunSummary, ScopeContract, ScopeReport, matches_any, merge_contracts, scope_check

ACCEPTANCE = "pytest -x test_app.py::test_signup_rejects_short_password"


def contract(**changes: Any) -> ScopeContract:
    base = dict(task_id="T-001", goal="add input validation to /signup", allowed_files=["app.py", "test_app.py"],
                forbidden_files=["migrations/**"], acceptance_criteria=[ACCEPTANCE], rollback_plan="revert the commit")
    return ScopeContract(**{**base, **changes})


def run(**changes: Any) -> RunSummary:
    base = dict(touched_files=["app.py", "test_app.py"], commands_run=[ACCEPTANCE], elapsed_minutes=12.0)
    return RunSummary(**{**base, **changes})


def verdict(report: ScopeReport) -> str:
    codes = [f"{finding.severity}:{finding.code}" for finding in report.findings]
    return ("pass" if report.passed() else "FAIL") + (f"  {codes}" if codes else "")


# ---------------------------------------------------------------------------
# Exercise 1 - network_egress: an allowlist of hosts
#
# main.py has the field and checks it at close-out. Two things come out of
# running it:
#   - The default is None, which means no enforcement. A contract that leaves
#     the field out allows every host. Deny-all has to be written as [].
#   - The check reads a summary of a run that is over. By the time it reports
#     a host, the request has been sent. Refusing means asking before the
#     connection, which is what may_connect is for.
# The close-out check compares strings exactly, so an allowed host written
# with a port or in capitals is reported. That errs on the safe side.
# ---------------------------------------------------------------------------

def host_name(host: str) -> str:
    # ponytail: host names and IPv4 only; an IPv6 literal needs urllib.parse
    return host.strip().lower().split(":")[0].rstrip(".")


def may_connect(scope: ScopeContract, host: str) -> bool:
    if scope.network_egress is None:
        return True
    return host_name(host) in {host_name(allowed) for allowed in scope.network_egress}


def ex1_network_egress() -> None:
    allowlist = ["api.anthropic.com"]
    cases = [("allowed host", allowlist, ["api.anthropic.com"]),
             ("one host off the list", allowlist, ["api.anthropic.com", "evil.example"]),
             ("lookalike host", allowlist, ["api.anthropic.com.evil.example"]),
             ("allowed host, with port", allowlist, ["API.Anthropic.com:443"]),
             ("deny-all, any host", [], ["api.anthropic.com"]),
             ("field left out", None, ["evil.example"])]
    passed = {}
    for label, egress, hosts in cases:
        report = scope_check(contract(network_egress=egress), run(network_hosts=hosts))
        passed[label] = report.passed()
        print(f"  {label:<24} egress={str(egress):<24} {verdict(report)}")
    assert list(passed.values()) == [True, False, False, False, False, True]

    scope = contract(network_egress=allowlist)
    wanted = ["api.anthropic.com", "API.Anthropic.com:443", "evil.example", "api.anthropic.com.evil.example"]
    reached = [host for host in wanted if may_connect(scope, host)]       # SCRIPTED: the hosts an agent tries, in order
    print(f"  asked before connecting: {len(reached)} of {len(wanted)} connections made, evil.example never contacted")
    assert reached == wanted[:2] and not may_connect(contract(network_egress=[]), "api.anthropic.com")


# ---------------------------------------------------------------------------
# Exercise 2 - soft on docs/**, hard on scripts/**
#
# main.py is already soft on docs: an off-scope write under docs_paths_soft
# is an info finding. Off-scope writes anywhere else are warnings, and a
# contract with a violation budget of 1 lets one of them through, including
# an edit to a deploy script. scope_check_hard turns an off-scope write under
# scripts/** into a blocking finding, unless the contract names that file.
#
# Why the asymmetry. A wrong sentence in the docs is read by a person, shows
# in the diff, and is undone by a revert. A script is executed, often by CI
# and often with credentials, by people who never saw the task; no test in
# the task covers it, and a release that ran cannot be un-run. Blocking docs
# edits would also make the gate noisy, because an agent that changes a flag
# and updates the README has done the right thing.
#
# Measured on the way: "**/*.md" does not cover a markdown file at the repo
# root, because fnmatch needs the slash. README.md is soft only because
# main.py lists it by name; CHANGELOG.md is an ordinary off-scope warning.
# ---------------------------------------------------------------------------

HARD_PATHS = ["scripts/**"]


def scope_check_hard(scope: ScopeContract, summary: RunSummary) -> ScopeReport:
    hard = [glob.escape(path) for path in summary.touched_files
            if matches_any(path, HARD_PATHS) and not matches_any(path, scope.allowed_files)]
    return scope_check(replace(scope, forbidden_files=scope.forbidden_files + hard), summary)


def ex2_soft_docs_hard_scripts() -> None:
    lenient = contract(violation_budget=1)
    script_task = contract(goal="fix the build script", allowed_files=["scripts/build.sh"], violation_budget=1)
    cases = [("docs/api.md", lenient, ["app.py", "docs/api.md"]),
             ("CHANGELOG.md", lenient, ["app.py", "CHANGELOG.md"]),
             ("scripts/deploy.sh", lenient, ["app.py", "scripts/deploy.sh"]),
             ("scripts/build.sh, allowed", script_task, ["scripts/build.sh"])]
    results = {}
    for label, scope, touched in cases:
        before, after = scope_check(scope, run(touched_files=touched)), scope_check_hard(scope, run(touched_files=touched))
        results[label] = (before.passed(), after.passed())
        print(f"  {label:<26} main.py: {verdict(before):<34} hard on scripts: {verdict(after)}")
    assert results == {"docs/api.md": (True, True), "CHANGELOG.md": (True, True),
                       "scripts/deploy.sh": (True, False), "scripts/build.sh, allowed": (True, True)}
    assert scope_check(lenient, run(touched_files=["CHANGELOG.md"])).off_scope_writes == ["CHANGELOG.md"]


# ---------------------------------------------------------------------------
# Exercise 3 - allowed_files derived from the goal by static rules
#
# The rules map a word in the goal to the files that word usually means.
# The first goal works. The first edge case is the second task on lesson 32's
# own board, "document the new /signup contract": the word signup is there as
# the subject of the documentation, the rule reads it as the target of the
# work, and a docs task comes out allowed to edit app.py. A run that rewrites
# app.py under that task passes the scope check.
#
# Word rules cannot tell what the task acts on from what it mentions, so they
# widen scope without saying so. A negation fails the same way, and a goal
# with no known word gets an empty list and blocks every write. The derived
# list is safe as a proposal a person confirms, not as the contract.
# ---------------------------------------------------------------------------

GOAL_RULES = [("signup", ["app.py", "test_app.py"]), ("login", ["auth.py", "test_auth.py"]),
              ("document", ["docs/**"]), ("migration", ["migrations/**"])]


def derive_allowed(goal: str) -> list[str]:
    return sorted({pattern for word, patterns in GOAL_RULES if word in goal.lower() for pattern in patterns})


def ex3_allowed_from_goal() -> None:
    goals = ["add input validation to /signup", "document the new /signup contract",
             "fix the login typo, do not touch signup", "speed up the nightly report"]
    derived = {goal: derive_allowed(goal) for goal in goals}
    for goal, allowed in derived.items():
        print(f"  {goal!r:<44} -> {allowed}")
    assert derived[goals[0]] == contract().allowed_files                # matches the hand-written contract
    docs_task = contract(goal=goals[1], allowed_files=derived[goals[1]], acceptance_criteria=[])
    rewrite = scope_check(docs_task, run(touched_files=["app.py"], commands_run=[]))
    print(f"  the docs task rewrites app.py: {verdict(rewrite)}, in scope {rewrite.in_scope_writes}")
    assert rewrite.passed() and rewrite.in_scope_writes == ["app.py"]
    assert "app.py" in derived[goals[2]] and derived[goals[3]] == []
    blocked = scope_check(contract(goal=goals[3], allowed_files=[]), run(touched_files=["report.py"]))
    print(f"  the goal with no known word, one write: {verdict(blocked)}")
    assert not blocked.passed()


# ---------------------------------------------------------------------------
# Exercise 4 - time_budget_minutes, and refusing to continue past it
#
# main.py has the field and reports an overrun at close-out, after the time
# is spent. Refusing to continue means looking at the clock before each step.
# The overrun is then at most one step long. The close-out check still fails
# the task; the difference is how much was burnt before anyone stopped it.
# A step that never returns is out of reach of both and needs its own timeout.
# ---------------------------------------------------------------------------

def run_steps(scope: ScopeContract, steps: list[Callable[[], None]], clock: Callable[[], float],
              enforce: bool = True) -> tuple[int, float, str]:
    """Run steps until done or out of time. Returns (steps completed, minutes elapsed, why it stopped)."""
    started = clock()
    for done, step in enumerate(steps):
        elapsed = (clock() - started) / 60
        if enforce and scope.time_budget_minutes is not None and elapsed >= scope.time_budget_minutes:
            return done, elapsed, f"refused step {done + 1}: {elapsed:.0f} min used of {scope.time_budget_minutes}"
        step()
    return len(steps), (clock() - started) / 60, "all steps ran"


def ex4_time_budget() -> None:
    def scenario(scope: ScopeContract, enforce: bool) -> tuple[int, float, str, bool]:
        now = [0.0]

        def step() -> None:
            now[0] += 7 * 60            # SIMULATED: every step takes seven minutes

        done, minutes, reason = run_steps(scope, [step] * 10, lambda: now[0], enforce)
        return done, minutes, reason, scope_check(scope, run(elapsed_minutes=minutes)).passed()

    results = {"close-out check only": scenario(contract(time_budget_minutes=30), enforce=False),
               "clock checked each step": scenario(contract(time_budget_minutes=30), enforce=True),
               "no budget on the contract": scenario(contract(), enforce=True)}
    for label, (done, minutes, reason, passed) in results.items():
        print(f"  {label:<26} {done:>2} steps, {minutes:>2.0f} min, {reason}; close-out {'pass' if passed else 'FAIL'}")
    assert results["close-out check only"][:2] == (10, 70.0) and not results["close-out check only"][3]
    assert results["clock checked each step"][:2] == (5, 35.0) and not results["clock checked each step"][3]
    assert results["no budget on the contract"][:2] == (10, 70.0) and results["no budget on the contract"][3]


# ---------------------------------------------------------------------------
# Exercise 5 - two contracts, one diff
#
# The right semantics when both apply is least privilege: a write is in scope
# only if both contracts allow it, forbidden if either forbids it, and the
# tighter budget wins. The simplest way to get exactly that is not to merge
# the contracts at all: check the diff against each one and pass only if
# both pass.
#
# main.py's merge_contracts gets two cases wrong, both measured below.
#   - It intersects the allowed lists as strings. The project contract allows
#     lib/**/*.py and the task contract allows lib/auth/session.py; no string
#     is in both lists, the merged list is empty, and a file both contracts
#     allow is reported off scope.
#   - It unions docs_paths_soft. A task contract that set the list to empty,
#     to be strict about docs, gets the project's soft paths back, and the
#     docs write it would have failed goes through as an info finding.
# ---------------------------------------------------------------------------

def check_all(contracts: list[ScopeContract], summary: RunSummary) -> tuple[bool, list[ScopeReport]]:
    reports = [scope_check(scope, summary) for scope in contracts]
    return all(report.passed() for report in reports), reports


def ex5_two_contracts() -> None:
    project = contract(task_id="P-PROJECT", goal="project-wide defaults", allowed_files=["lib/**/*.py", "tests/**/*.py"],
                       forbidden_files=["scripts/release.sh"], acceptance_criteria=[], time_budget_minutes=60)
    task = contract(task_id="T-007", goal="rotate the session key", allowed_files=["lib/auth/session.py", "tests/auth/test_session.py"],
                    forbidden_files=["migrations/**"], time_budget_minutes=30, docs_paths_soft=[])
    merged = merge_contracts(project, task)
    diffs = {"the two files the task names": ["lib/auth/session.py", "tests/auth/test_session.py"],
             "plus a file only the project allows": ["lib/auth/session.py", "lib/billing/invoice.py"],
             "plus the release script": ["lib/auth/session.py", "scripts/release.sh"],
             "plus the README": ["lib/auth/session.py", "README.md"]}
    results = {}
    for label, touched in diffs.items():
        summary = run(touched_files=touched)
        both, _ = check_all([project, task], summary)
        results[label] = (both, scope_check(merged, summary).passed())
        print(f"  {label:<36} each contract checked: {'pass' if both else 'FAIL'}   merge_contracts: "
              f"{'pass' if results[label][1] else 'FAIL'}")
    print(f"  merged allowed_files: {merged.allowed_files}, merged docs_paths_soft: {merged.docs_paths_soft}")
    assert merged.allowed_files == [] and merged.time_budget_minutes == 30
    assert results == {"the two files the task names": (True, False), "plus a file only the project allows": (False, False),
                       "plus the release script": (False, False), "plus the README": (False, False)}

    strict_docs = replace(task, allowed_files=["lib/**/*.py"])           # same strings as the project, so the merge keeps them
    loose_merge = merge_contracts(replace(project, allowed_files=["lib/**/*.py"]), strict_docs)
    readme = run(touched_files=["lib/auth/session.py", "README.md"])
    print(f"  README edit, task contract strict about docs: task alone {verdict(scope_check(strict_docs, readme))}")
    print(f"  {'':<45} merged     {verdict(scope_check(loose_merge, readme))}")
    assert not scope_check(strict_docs, readme).passed() and scope_check(loose_merge, readme).passed()


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run in a temp copy of this folder.
#   exits zero                               yes
#   the in-scope run reports zero            yes
#   violations
#   the creeping run reports the exact       yes: the forbidden writes, the
#   files and the reason for each            missing acceptance run, the time
#                                            overrun and the unlisted host
# The mission asks for a scope_report.json next to the script. main.py archives
# to closed/<task>.json instead, and both demo runs carry the task id T-001, so
# the creeping run's report overwrites the clean one: one file is left, and it
# says failed.
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
    clean, creep = stdout.split("clean run findings:")[1].split("creep run findings:")
    archived = json.loads(files["code/closed/T-001.json"])
    print(f"  main.py exits {exit_code}; clean run: {clean.strip()}; creeping run: {creep.count('[block]')} blocking findings")
    print(f"  reports left on disk: {list(files)}, passed={archived['passed']}")
    assert exit_code == 0 and clean.strip() == "passed=True over_budget=False"
    assert "passed=False" in creep and all(name in creep for name in ("scripts/release.sh", "migrations/001_init.sql", "evil.example"))
    assert list(files) == ["code/closed/T-001.json"] and archived["passed"] is False


if __name__ == "__main__":
    print("Phase 14 - Lesson 36: Scope Contracts and Task Boundaries - exercises")
    for exercise in (ex1_network_egress, ex2_soft_docs_hard_scripts, ex3_allowed_from_goal, ex4_time_budget, ex5_two_contracts, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
