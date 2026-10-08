"""Phase 14 - Lesson 35: Initialization Scripts for Agents - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --fix [--dry-run] [--approve NAME]   (exercise 3 as a command)
Exercise 1 needs git on PATH. Part of exercise 4 needs pyyaml (pip install pyyaml).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest import mock

import main as lesson
from main import Probe

PATCHED = ("HERE", "WORK", "STATE_PATH", "REPORT_PATH", "LOCK_PATH", "LKG_PATH", "REQUIRED_TEST_COMMAND", "REQUIRED_DEPS")


@contextlib.contextmanager
def temp_workbench() -> Iterator[Path]:
    """Point every path main.py writes to at a temp directory, and put them back afterwards."""
    saved = {name: getattr(lesson, name) for name in PATCHED}
    with tempfile.TemporaryDirectory() as tmp:
        lesson.HERE, lesson.WORK = Path(tmp) / "repo", Path(tmp) / "workdir"
        lesson.HERE.mkdir()
        lesson.WORK.mkdir()
        for name in PATCHED[2:6]:
            setattr(lesson, name, lesson.WORK / saved[name].name)
        lesson.REQUIRED_TEST_COMMAND = sys.executable       # main.py asks for "python3", which not every machine has
        try:
            yield lesson.HERE
        finally:
            for name, value in saved.items():
                setattr(lesson, name, value)


def run_init(*argv: str) -> int:
    """main.py's command line, with what it prints kept off the screen."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return lesson.main(list(argv))


# ---------------------------------------------------------------------------
# Exercise 1 - a probe that diffs HEAD against the last-known-good commit
#
# main.py has the probe (probe_lkg_diff). Here it runs against a real git
# repository built in a temp directory, on both sides of the 50-file limit.
#
# One gap shows up: the probe compares two commits, so files a session
# changed and never committed are invisible to it. probe_uncommitted counts
# those with the same budget.
# ---------------------------------------------------------------------------

@lesson._timed
def probe_uncommitted() -> Probe:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True,
                             timeout=2.0, cwd=lesson.HERE)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return Probe("uncommitted", "warn", "git unavailable or slow; skipped")
    changed = len(out.stdout.splitlines())
    status = "fail" if changed > lesson.LKG_FILE_DIFF_BUDGET else "pass"
    return Probe("uncommitted", status, f"{changed} uncommitted files (budget {lesson.LKG_FILE_DIFF_BUDGET})")


def ex1_lkg_diff_probe() -> str | None:
    if not shutil.which("git"):
        print("  needs git on PATH")
        return "git"

    def git(*args: str) -> None:
        identity = ["-c", "user.name=practice", "-c", "user.email=practice@example.invalid", "-c", "commit.gpgsign=false"]
        subprocess.run(["git", *identity, *args], cwd=repo, capture_output=True, text=True, check=True)

    def commit(count: int, prefix: str) -> None:
        for number in range(count):
            (repo / f"{prefix}_{number}.py").write_text(f"VALUE = {number}\n")
        git("add", f"{prefix}_*.py")
        git("commit", "-q", "-m", f"add {count} {prefix} files")

    with temp_workbench() as repo:
        git("init", "-q")
        commit(1, "baseline")
        unpinned = lesson.probe_lkg_diff()
        assert run_init("--write-lkg") == 0                     # main.py pins HEAD as last-known-good
        commit(50, "feature")
        at_limit = lesson.probe_lkg_diff()
        for number in range(60):
            (repo / f"scratch_{number}.py").write_text("")
        dirty, uncommitted = lesson.probe_lkg_diff(), probe_uncommitted()
        git("add", "scratch_0.py")
        git("commit", "-q", "-m", "one more")
        over, exit_code = lesson.probe_lkg_diff(), run_init("--no-cache")
        report = json.loads(lesson.REPORT_PATH.read_text())

    for label, probe in (("nothing pinned yet", unpinned), ("50 files committed", at_limit),
                         ("+60 never committed", dirty), ("", uncommitted), ("51 files committed", over)):
        print(f"  {label:<20} {probe.name:<12} {probe.status:<4}  {probe.detail}")
    print(f"  init script exit code with 51 files changed: {exit_code}")
    assert [probe.status for probe in (unpinned, at_limit, dirty, uncommitted, over)] == ["warn", "pass", "pass", "fail", "fail"]
    assert exit_code == 1 and not report["ok"] and "(budget 50)" in over.detail
    return None


# ---------------------------------------------------------------------------
# Exercise 2 - prereqs.lock, and no launch on a lock older than seven days
#
# main.py writes the lock after a clean probe pass and uses it for 24 hours
# as a reason to skip the probes. The seven-day rule is the other side of the
# same file: the agent launcher reads it and refuses to start unless an init
# pass is on record, for the current prerequisites, within seven days.
#
#   lock younger than 24h     init skips its probes, launch allowed
#   24h to 7 days             init runs the probes again, launch allowed
#   older than 7 days, absent launch refused until init has passed again
#   prerequisites changed     launch refused at any age
# ---------------------------------------------------------------------------

LOCK_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
DAY = 24 * 60 * 60


def launch_gate(now: float | None = None) -> tuple[bool, str]:
    try:
        lock = json.loads(lesson.LOCK_PATH.read_text())
        age = (now or time.time()) - float(lock["written_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return False, "no readable prereqs.lock; run the init script"
    if lock.get("fingerprint") != lesson._deps_fingerprint():
        return False, "prerequisites changed since the lock was written; run the init script"
    if age > LOCK_MAX_AGE_SECONDS:
        return False, f"prereqs.lock is {age / DAY:.0f} days old (limit 7); run the init script"
    return True, f"init passed {age / DAY:.0f} days ago"


def ex2_prereqs_lock() -> None:
    with temp_workbench():
        gates = {"before any init run": launch_gate()}
        assert run_init("--no-cache") == 0 and lesson.LOCK_PATH.exists()        # a clean pass writes the lock
        written = json.loads(lesson.LOCK_PATH.read_text())["written_at"]
        lesson.REPORT_PATH.unlink()
        assert run_init() == 0 and not lesson.REPORT_PATH.exists()              # within 24h: probes skipped, no new report
        gates["right after init"] = launch_gate()
        gates["6 days later"] = launch_gate(now=written + 6 * DAY)
        gates["8 days later"] = launch_gate(now=written + 8 * DAY)
        lesson.REQUIRED_DEPS = [*lesson.REQUIRED_DEPS, "sqlite3"]
        gates["a prerequisite was added"] = launch_gate()
    for label, (allowed, reason) in gates.items():
        print(f"  {label:<25} {'launch' if allowed else 'REFUSE':<6}  {reason}")
    assert [allowed for allowed, _ in gates.values()] == [False, True, True, False, False]


# ---------------------------------------------------------------------------
# Exercise 3 - --fix: install missing dev dependencies, never runtime ones
#
# A dev dependency changes the workbench. A runtime dependency changes what
# ships, so it waits for a person, the same line lesson 33 draws with its
# approval rule. A name that is on neither list is never installed: the
# script installs what the repo declared, not what an error message asked for.
#
# The exercise below hands fix() a recorder in place of pip, so running this
# file installs nothing.
# ---------------------------------------------------------------------------

DEV_DEPS = {"pytest": "pytest", "yaml": "pyyaml"}           # import name -> the name pip installs
RUNTIME_DEPS = {"numpy": "numpy", "pydantic": "pydantic"}


def plan_fixes(missing: list[str], approved: set[str]) -> tuple[list[list[str]], list[str]]:
    commands, held = [], []
    for name in missing:
        if name in DEV_DEPS or (name in RUNTIME_DEPS and name in approved):
            commands.append([sys.executable, "-m", "pip", "install", {**RUNTIME_DEPS, **DEV_DEPS}[name]])
        elif name in RUNTIME_DEPS:
            held.append(f"{name}: runtime dependency, needs --approve {name}")
        else:
            held.append(f"{name}: not a declared dependency, never installed")
    return commands, held


def fix(missing: list[str], approved: set[str], run: Callable[..., Any] = subprocess.run) -> list[str]:
    commands, held = plan_fixes(missing, approved)
    for command in commands:
        run(command, check=True)
    return held


def ex3_fix_flag() -> None:
    # SCRIPTED: what a dependency probe might report. "reqeusts" is misspelt on purpose: the kind of name an
    # error message or a model suggests, and somebody else may have registered.
    missing = ["yaml", "pydantic", "reqeusts"]
    installed: list[str] = []

    def recorder(command: list[str], check: bool) -> None:
        installed.append(command[-1])

    held = fix(missing, approved=set(), run=recorder)
    print(f"  --fix                     installs {installed}")
    for line in held:
        print(f"  {'':<25} holds {line}")
    assert installed == ["pyyaml"] and len(held) == 2

    installed.clear()
    held = fix(missing, approved={"pydantic", "reqeusts"}, run=recorder)
    print(f"  --fix --approve pydantic  installs {installed}; still holds {held}")
    assert installed == ["pyyaml", "pydantic"] and held == ["reqeusts: not a declared dependency, never installed"]

    dry = subprocess.run([sys.executable, __file__, "--fix", "--dry-run"], capture_output=True, text=True)
    print(f"  the command line, --fix --dry-run: exit {dry.returncode}, {len(dry.stdout.splitlines())} lines, nothing installed")
    would_install = [line.split()[1] for line in dry.stdout.splitlines() if line.startswith("install ")]
    assert dry.returncode == 0 and set(would_install) <= set(DEV_DEPS.values())


# ---------------------------------------------------------------------------
# Exercise 4 - probes in a YAML registry
#
# The registry holds which probes run and what each one needs. The behaviour
# stays in main.py's probe functions; the runner fills in their settings from
# the file. Kinds are a closed set, and the whole file is checked before any
# probe runs, so a misspelt kind stops the script instead of skipping a check.
#
# The trade-off.
#   For:     adding or tightening a check is a one-line change that anyone
#            can review, and one runner can serve several repos.
#   Against: YAML decides the types. Measured with pyyaml: needs: 3.10 loads
#            as the number 3.1, and the runtime check quietly drops to Python
#            3.1 unless the value is quoted.
#            Mistakes surface when the script runs, not when it is imported.
#            The first probe that fits no kind invites a "command:" kind, and
#            then a config file is running shell without the review code gets.
#   Verdict: worth it once several repos share the probes. For one repo and
#            six probes, the functions in main.py are less machinery.
# ---------------------------------------------------------------------------

REGISTRY = """\
# probe: a kind the runner knows   needs: what it checks   budget: seconds
- probe: runtime
  needs: "3.10"
  budget: 0.5
- probe: dependencies
  needs: json, dataclasses
- probe: env
  needs: WORKBENCH_TOKEN
  budget: 0.5
"""


def comma_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


KINDS: dict[str, tuple[str, Callable[[str], Any], Callable[[], Probe]]] = {
    # kind -> (the main.py setting it fills, how to read the value, the main.py probe)
    "runtime": ("REQUIRED_PYTHON", lambda text: tuple(int(part) for part in text.split(".")), lesson.probe_runtime),
    "dependencies": ("REQUIRED_DEPS", comma_list, lesson.probe_dependencies),
    "test_command": ("REQUIRED_TEST_COMMAND", str, lesson.probe_test_command),
    "env": ("REQUIRED_ENV_VARS", comma_list, lesson.probe_env),
}


def parse_registry(text: str) -> list[dict[str, str]]:
    # ponytail: reads only the flat subset the registry uses (a list of mappings, scalar values, all kept
    # as text); use yaml.safe_load when the file needs anything more
    entries: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.startswith("- "):
            entries.append({})
        key, _, value = line[2:].partition(": ")
        entries[-1][key.strip()] = value.strip().strip('"')
    return entries


def run_with_budget(probe: Callable[[], Probe], budget_s: float) -> Probe:
    """Run a probe wrapped by main.py's _timed against its own budget instead of the global one."""
    saved, lesson.PROBE_BUDGET_SECONDS = lesson.PROBE_BUDGET_SECONDS, budget_s
    try:
        return probe()
    finally:
        lesson.PROBE_BUDGET_SECONDS = saved


def run_registry(entries: list[dict[str, str]]) -> list[tuple[Probe, float]]:
    unknown = [entry.get("probe") for entry in entries if entry.get("probe") not in KINDS]
    if unknown:
        raise ValueError(f"unknown probe kinds {unknown}; known kinds are {sorted(KINDS)}")
    results = []
    for entry in entries:
        setting, read, probe = KINDS[entry["probe"]]
        budget_s = float(entry.get("budget", lesson.PROBE_BUDGET_SECONDS))
        saved = getattr(lesson, setting)
        setattr(lesson, setting, read(entry["needs"]))
        try:
            results.append((run_with_budget(probe, budget_s), budget_s))
        finally:
            setattr(lesson, setting, saved)
    return results


def ex4_yaml_registry() -> str | None:
    entries = parse_registry(REGISTRY)
    with mock.patch.dict(os.environ, {"WORKBENCH_TOKEN": ""}):
        results = run_registry(entries)
    for probe, budget_s in results:
        print(f"  {probe.name:<13} {probe.status:<4}  budget {budget_s}s  {probe.detail}")
    assert [probe.status for probe, _ in results] == ["pass", "pass", "fail"]
    assert lesson.REQUIRED_ENV_VARS == [] and lesson.REQUIRED_PYTHON == (3, 10)        # main.py's settings are put back

    try:
        run_registry(parse_registry(REGISTRY.replace("probe: runtime", "probe: runtme")))
        typo = ""
    except ValueError as error:
        typo = str(error)
    print(f"  misspelt kind: {typo}")
    assert typo

    if importlib.util.find_spec("yaml") is None:
        print("  needs pyyaml: pip install pyyaml")
        return "pyyaml"
    import yaml
    loaded = [{key: str(value) for key, value in entry.items()} for entry in yaml.safe_load(REGISTRY)]
    unquoted = yaml.safe_load("needs: 3.10")["needs"]
    print(f"  pyyaml reads the registry the same way: {loaded == entries}")
    print(f"  pyyaml reads an unquoted 3.10 as {unquoted!r}, so the check would ask for Python >= {KINDS['runtime'][1](str(unquoted))}")
    assert loaded == entries and unquoted == 3.1
    return None


# ---------------------------------------------------------------------------
# Exercise 5 - a timing budget per probe
#
# main.py has one budget for every probe, 3 seconds, and marks a slow pass as
# a warning. Here each probe carries its own budget (the registry's budget
# field, or 3 seconds), and the list of smells is built from the measured
# durations, because main.py's wrapper only annotates a probe that passed: a
# probe that fails slowly keeps its plain "fail" and says nothing about time.
#
# A budget is measured after the probe returns. It does not stop a probe that
# never returns; a probe that can hang needs its own timeout, as
# probe_lkg_diff has on its git call.
# ---------------------------------------------------------------------------

def smells(timed: list[tuple[Probe, float]]) -> list[str]:
    return [f"{probe.name} took {probe.duration_ms} ms against a budget of {budget_s * 1000:.0f} ms"
            for probe, budget_s in timed if probe.duration_ms > budget_s * 1000]


@lesson._timed
def slow_pass() -> Probe:
    time.sleep(0.12)                # STAND-IN for a probe that calls a slow service
    return Probe("slow_pass", "pass", "reachable")


@lesson._timed
def slow_fail() -> Probe:
    time.sleep(0.12)
    return Probe("slow_fail", "fail", "unreachable")


def ex5_timing_budget() -> None:
    timed = [(run_with_budget(slow_pass, 3.0), 3.0), (run_with_budget(slow_pass, 0.05), 0.05), (run_with_budget(slow_fail, 0.05), 0.05)]
    for probe, budget_s in timed:
        print(f"  {probe.name:<10} budget {budget_s:>4}s  {probe.status:<4}  {probe.detail}")
    found = smells(timed)
    print(f"  smells: {len(found)} of {len(timed)}, including the slow failure main.py leaves unmarked")
    assert [probe.status for probe, _ in timed] == ["pass", "warn", "fail"]
    assert "slow" in timed[1][0].detail and "slow" not in timed[2][0].detail and len(found) == 2

    with temp_workbench():
        real = [(probe, lesson.PROBE_BUDGET_SECONDS) for probe in lesson.run_probes()]
    slowest = max(probe.duration_ms for probe, _ in real)
    print(f"  main.py's {len(real)} probes on this machine: slowest {slowest} ms, smells {smells(real)}")
    assert smells(real) == []


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py's command line is run in a temp workbench, as in the exercises.
#   exits zero on the happy path             yes
#   twice in a row is a no-op except for     yes: the same probes, statuses
#   the timestamp                            and details; the measured
#                                            durations can differ
#   a missing env var shows in the report    yes, once one is required.
#   and flips the exit code                  main.py requires none, so the
#                                            check requires one and unsets it
# ---------------------------------------------------------------------------

def mission_acceptance() -> None:
    def probes(report: dict[str, Any]) -> list[tuple[str, str, str]]:
        return [(probe["name"], probe["status"], probe["detail"]) for probe in report["probes"]]

    with temp_workbench():
        happy = run_init("--no-cache")
        first = json.loads(lesson.REPORT_PATH.read_text())
        again = run_init("--no-cache")
        second = json.loads(lesson.REPORT_PATH.read_text())
        saved, lesson.REQUIRED_ENV_VARS = lesson.REQUIRED_ENV_VARS, ["WORKBENCH_TOKEN"]
        try:
            with mock.patch.dict(os.environ, {"WORKBENCH_TOKEN": ""}):
                blocked = run_init("--no-cache")
            report = json.loads(lesson.REPORT_PATH.read_text())
        finally:
            lesson.REQUIRED_ENV_VARS = saved
    failing = [name for name, status, _ in probes(report) if status == "fail"]
    print(f"  happy path exits {happy}, and {again} again with the same {len(first['probes'])} probe results")
    print(f"  with a required env var unset: exit {blocked}, report ok={report['ok']}, failing probes {failing}")
    assert happy == again == 0 and first["ok"] and probes(first) == probes(second)
    assert blocked == 1 and not report["ok"] and failing == ["env"]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fix"]:
        absent = [name for name in {**DEV_DEPS, **RUNTIME_DEPS} if importlib.util.find_spec(name) is None]
        approvals = {sys.argv[index + 1] for index, word in enumerate(sys.argv[:-1]) if word == "--approve"}
        planned, on_hold = plan_fixes(absent, approvals)
        for line in [f"install {command[-1]}" for command in planned] + [f"hold {line}" for line in on_hold]:
            print(line)
        if "--dry-run" not in sys.argv:
            fix(absent, approvals)
        sys.exit(0)
    print("Phase 14 - Lesson 35: Initialization Scripts for Agents - exercises")
    missing = []
    for exercise in (ex1_lkg_diff_probe, ex2_prereqs_lock, ex3_fix_flag, ex4_yaml_registry, ex5_timing_budget, mission_acceptance):
        print(f"\n{exercise.__name__}")
        missing.append(exercise())
    missing = [name for name in missing if name]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
