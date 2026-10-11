"""Phase 14 - Lesson 42: Capstone - Ship a Reusable Agent Workbench Pack - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --install <target> [--force] [--dry-run]    (exercise 2 as a command)
           python practice.py --uninstall <target> [--force]              (exercise 3 as a command)
           python practice.py --lint-pack <pack> [target]                 (exercise 4 as a command)
The pack used throughout is the one the course ships in ../outputs; it is only read. Everything is
installed into temp directories. Parts of exercises 2 and 3 need bash, and part of exercise 3 needs git.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
SHIPPED_IN = Path(__file__).resolve().parents[1]        # the pack's root, when this file is shipped inside a pack as bin/uninstall.py
PACK = SHIPPED_IN if (SHIPPED_IN / "VERSION").is_file() else REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack"
SURFACE = ("AGENTS.md", "docs", "schemas", "scripts")
LOCK = ".workbench-version"
STATE_FILES = ("agent_state.json", "task_board.json", "feedback_record.jsonl")


def load_lesson(folder: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / folder / "code" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tree(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*"), key=str)
            if path.is_file() and ".git" not in path.parts}


def run_script(target: Path, script: str, *args: str) -> tuple[int, str]:
    """Run one of the installed pack's own scripts in the target repo."""
    done = subprocess.run([sys.executable, f"scripts/{script}", *args], cwd=target, capture_output=True, text=True,
                          timeout=60, env={**os.environ, "WORKBENCH_TEST_COMMAND": sys.executable})
    return done.returncode, done.stdout


# ---------------------------------------------------------------------------
# Exercise 2 (first, because the others install with it) - the installer in
# Python, with --dry-run
#
# The plan is built as data before anything is written, so --dry-run is the
# same code with the last step left out, and a refusal happens before the
# first copy.
#
# Against bin/install.sh, measured below on the same pack:
#   - both leave the same files, byte for byte, and both refuse to overwrite
#     an existing AGENTS.md;
#   - install.sh takes its one flag by position. "--dry-run" is not a flag it
#     knows, so it ignores it and installs for real; and "--force" given
#     second is not seen, so the script refuses. The Python version rejects
#     what it does not know;
#   - install.sh needs bash, which on Windows means Git Bash or WSL. The
#     pack's four scripts already need Python, so a Python installer adds no
#     requirement;
#   - install.sh is one page of shell with nothing to import, and anyone can
#     read it in a minute. That is worth something, and it is the only thing
#     lost.
# ---------------------------------------------------------------------------

class Refused(RuntimeError):
    pass


def plan_install(pack: Path, target: Path, force: bool = False, skip: tuple[str, ...] = ()) -> list[tuple[Path, Path]]:
    """Every copy an install would make, as (source, destination). Raises before anything is written."""
    absent = [name for name in (*SURFACE, "VERSION") if not (pack / name).exists()]
    if absent:
        raise Refused(f"missing pack source: {absent}")
    if (target / "AGENTS.md").exists() and not force and "AGENTS.md" not in skip:
        raise Refused("AGENTS.md already exists; pass --force to overwrite")
    sources = [pack / "AGENTS.md", *sorted((path for folder in SURFACE[1:] for path in (pack / folder).rglob("*") if path.is_file()), key=str)]
    copies = [(source, target / source.relative_to(pack)) for source in sources if source.relative_to(pack).parts[0] not in skip]
    return [*copies, (pack / "VERSION", target / LOCK)]


def install(pack: Path, target: Path, force: bool = False, dry_run: bool = False, skip: tuple[str, ...] = ()) -> list[str]:
    copies = plan_install(pack, target, force, skip)
    if not dry_run:
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return [destination.relative_to(target).as_posix() for _, destination in copies]


def find_bash() -> str | None:
    """A bash that can run a script by its path. On Windows that is Git's, not the WSL launcher in System32."""
    candidates = [shutil.which("bash")]
    if git := shutil.which("git"):
        candidates += [str(Path(git).resolve().parents[up] / "bin" / "bash.exe") for up in (1, 2)]
    return next((found for found in candidates if found and Path(found).exists() and "system32" not in found.lower()), None)


def bash_runs(script: Path, runs: list[tuple[Path, str]]) -> list[int] | None:
    """Run a shell script once per (directory to run it in, flags), in one shell. Exit codes, or None without bash."""
    bash, python = find_bash(), Path(sys.executable).as_posix()
    if bash is None:
        return None
    lines = [f'cd "{target.as_posix()}" && PYTHON="{python}" bash "{script.as_posix()}" {flags}; echo "exit:$?"' for target, flags in runs]
    done = subprocess.run([bash, "-c", "\n".join(lines)], capture_output=True, text=True, timeout=120)
    codes = [int(code) for code in re.findall(r"^exit:(\d+)$", done.stdout, re.M)]
    return codes if len(codes) == len(runs) else None


def ex2_python_installer() -> str | None:
    with tempfile.TemporaryDirectory() as tmp:
        targets = {name: Path(tmp) / name for name in ("python", "bash", "unknown-flag")}
        for target in targets.values():
            target.mkdir()

        planned = install(PACK, targets["python"], dry_run=True)
        assert tree(targets["python"]) == {}                            # the dry run wrote nothing
        written = install(PACK, targets["python"])
        try:
            install(PACK, targets["python"])
            second = "installed again"
        except Refused as refusal:
            second = f"refused: {refusal}"
        install(PACK, targets["python"], force=True)
        print(f"  --dry-run lists {len(planned)} files and writes 0; the install writes the same {len(written)}")
        print(f"  second install without --force: {second}")
        assert planned == written and set(written) == set(tree(targets["python"])) and second.startswith("refused")

        codes = bash_runs(PACK / "bin" / "install.sh", [(targets["bash"], ""), (targets["bash"], ""), (targets["unknown-flag"], "--dry-run"),
                                                        (targets["bash"], "--verbose --force")])
        if codes is None or codes[0] != 0:
            print("  needs bash for the comparison with bin/install.sh")
            return "bash"
        first, again, dry, late_force = codes
        same = tree(targets["bash"]) == tree(targets["python"])
        shell_lines = len((PACK / "bin" / "install.sh").read_text(encoding="utf-8").splitlines())
        print(f"  install.sh ({shell_lines} lines) leaves the same {len(tree(targets['bash']))} files, byte for byte: {same}; a second run exits {again}")
        print(f"  install.sh --dry-run: exit {dry}, files written {len(tree(targets['unknown-flag']))}")
        print(f"  install.sh --verbose --force on an installed repo: exit {late_force}")
        assert same and again == 1
        assert dry == 0 and len(tree(targets["unknown-flag"])) == len(written)      # it installed for real
        assert late_force == 1                                                      # --force was not seen
    return None


# ---------------------------------------------------------------------------
# Exercise 1 - the fifth doc: docs/scope-contract.md
#
# The rule used to choose: a doc is promoted when a script in the pack
# depends on something that only that doc would explain. Measured on the
# pack: its scripts read and write six paths under outputs/, and none of the
# four docs, AGENTS.md or the README names a single one of them.
#
# One of those gaps is worse than the rest. The gate reads the task's
# acceptance criteria from outputs/scope/closed/<task>.json. With no contract
# there, it has nothing to require: on a freshly installed pack
# `verify_agent.py T-001` reports passed for a task on which nothing has run.
# Written the way the doc below says, the same gate fails until the
# acceptance command has run and passes after.
#
# The cut. An artifact map of all six paths: reference material, and each
# script's --help can carry it. An override policy: one paragraph of the
# reliability policy already, and the pack ships nothing that records an
# override. A migration guide: needed once per major version, so it belongs
# in release notes (and exercise 5). Onboarding: the lesson rules it out.
# ---------------------------------------------------------------------------

SCOPE_CONTRACT_MD = """\
# Scope Contract

One per task, written before the first edit and kept when the task closes.

Where it lives: `outputs/scope/closed/<task_id>.json`, in the shape of
`schemas/scope_contract.schema.json`. The verification gate reads
`acceptance_criteria` from that file. With no contract on disk the gate has
nothing to require, and it passes.

Required: task_id, goal, allowed_files, forbidden_files, acceptance_criteria,
rollback_plan.

- allowed_files and forbidden_files are globs. A write outside allowed_files
  is off scope. A write that matches forbidden_files blocks the task.
- Each acceptance criterion is a command, written exactly as it will be run
  through `scripts/run_with_feedback.py`. The gate matches the text.
- The scope report goes beside the contract as `<task_id>.report.json`, with
  `forbidden_writes` and `off_scope_writes`.
"""


def ex1_fifth_doc() -> None:
    prose = [path.read_text(encoding="utf-8") for path in (*sorted((PACK / "docs").glob("*.md")), PACK / "AGENTS.md", PACK / "README.md")]
    used = sorted({found for script in (PACK / "scripts").glob("*.py")
                   for found in re.findall(r"outputs/[a-z_]+", script.read_text(encoding="utf-8").replace('" / "', "/"))})
    unexplained = [path for path in used if not any(path in text for text in prose)]
    print(f"  paths the pack's scripts use under outputs/: {used}")
    print(f"  named in a doc, AGENTS.md or the README: {len(used) - len(unexplained)} of {len(used)}")
    assert len(used) == 6 and unexplained == used

    acceptance = [sys.executable, "-c", "pass"]
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        install(PACK, target)
        (target / "docs" / "scope-contract.md").write_text(SCOPE_CONTRACT_MD)
        vacuous = run_script(target, "verify_agent.py", "T-001")
        contract = {"task_id": "T-001", "goal": "add input validation to /signup", "allowed_files": ["app.py", "test_app.py"],
                    "forbidden_files": ["scripts/release.sh"], "acceptance_criteria": [" ".join(acceptance)],
                    "rollback_plan": "revert the commit"}
        (target / "outputs" / "scope" / "closed").mkdir(parents=True)
        (target / "outputs" / "scope" / "closed" / "T-001.json").write_text(json.dumps(contract))
        required = run_script(target, "verify_agent.py", "T-001")
        run_script(target, "run_with_feedback.py", "--", *acceptance)
        met = run_script(target, "verify_agent.py", "T-001")
    for label, (exit_code, output) in (("fresh install, no contract", vacuous), ("contract written", required),
                                       ("acceptance command run", met)):
        report = json.loads(output)
        print(f"  gate, {label:<27} exit {exit_code}, passed={report['passed']}, findings {[f['code'] for f in report['findings']]}")
    assert vacuous[0] == 0 and json.loads(vacuous[1]) == {"task_id": "T-001", "passed": True, "findings": []}
    assert required[0] == 1 and [f["code"] for f in json.loads(required[1])["findings"]] == ["acceptance.missing"]
    assert met[0] == 0 and json.loads(met[1])["passed"]

    schema = json.loads((PACK / "schemas" / "scope_contract.schema.json").read_text())
    assert all(field in SCOPE_CONTRACT_MD for field in schema["required"])          # the doc and the schema agree
    assert "outputs/scope/closed" in SCOPE_CONTRACT_MD and len(SCOPE_CONTRACT_MD.splitlines()) < 25


# ---------------------------------------------------------------------------
# Exercise 3 - uninstall, refusing on state with non-trivial history
#
# The uninstaller never deletes state: agent_state.json, task_board.json, the
# feedback log and outputs/ belong to the repository, not to the pack. It
# removes what an install laid down, and only where the file is still
# identical to the pack's copy. A rules file the team has edited is theirs
# now, so it is kept and reported.
#
# What counts as non-trivial is not "has ever been used"; finished history is
# left in place and is safe. It is state that removing the pack would strand:
#   - a task is active in agent_state.json;
#   - the board has a task in progress or blocked;
#   - a state file has changes that are not committed, so the repository has
#     no record of them yet.
# Any of these refuses the uninstall until the work is closed out, or until
# someone passes --force and owns the consequence.
#
# In the pack this ships as bin/uninstall.py (this file under that name: it
# finds the pack it sits in), with the three-line bin/uninstall.sh below in
# front of it, the way the pack's own scripts are Python. Where bash is
# available both are shipped into a copy of the pack and the shell script is
# run from there. `python practice.py --uninstall <target>` runs the same
# function.
# ---------------------------------------------------------------------------

UNINSTALL_SH = """\
#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" "$(cd "$(dirname "$0")" && pwd)/uninstall.py" --uninstall "$(pwd)" "$@"
"""


def read_json(path: Path, default: Any) -> Any:
    return json.loads(path.read_text()) if path.exists() else default


def stranded_by_removal(target: Path) -> list[str]:
    reasons = []
    active = read_json(target / "agent_state.json", {}).get("active_task_id")
    if active:
        reasons.append(f"task {active} is active")
    open_tasks = [task["id"] for task in read_json(target / "task_board.json", []) if task.get("status") in ("in_progress", "blocked")]
    if open_tasks:
        reasons.append(f"the board has work in flight: {open_tasks}")
    if (target / ".git").exists():
        dirty = subprocess.run(["git", "status", "--porcelain", "--", *STATE_FILES], cwd=target, capture_output=True,
                               text=True, timeout=10).stdout.split()[1::2]
        if dirty:
            reasons.append(f"uncommitted changes in {dirty}")
    return reasons


def uninstall(pack: Path, target: Path, force: bool = False) -> tuple[list[str], list[str]]:
    """Returns (files removed, files kept because they were edited since the install)."""
    reasons = stranded_by_removal(target)
    if reasons and not force:
        raise Refused("; ".join(reasons))
    removed, kept = [], []
    for name in install(pack, target, force=True, dry_run=True):        # exactly the files an install lays down
        installed, source = target / name, pack / ("VERSION" if name == LOCK else name)
        if installed.exists():
            if installed.read_bytes() == source.read_bytes():
                installed.unlink()
                removed.append(name)
            else:
                kept.append(name)
    for folder in SURFACE[1:]:
        if (target / folder).is_dir() and not any((target / folder).iterdir()):
            (target / folder).rmdir()
    return removed, kept


def ex3_uninstall() -> str | None:
    idle = {"schema_version": 1, "active_task_id": None, "touched_files": [], "next_action": "pick next task"}
    board = [{"id": "T-001", "goal": "validate /signup", "owner": "builder", "acceptance": ["pytest -x"], "status": "done"}]

    def installed_repo(root: Path, state: dict[str, Any]) -> Path:
        root.mkdir()
        install(PACK, root)
        (root / "agent_state.json").write_text(json.dumps(state))
        (root / "task_board.json").write_text(json.dumps(board))
        (root / "docs" / "team-notes.md").write_text("our own notes\n")
        (root / "docs" / "agent-rules.md").write_text((root / "docs" / "agent-rules.md").read_text() + "\n## forbidden/no-prod-config\n")
        return root

    with tempfile.TemporaryDirectory() as tmp:
        quiet = installed_repo(Path(tmp) / "quiet", idle)
        removed, kept = uninstall(PACK, quiet)
        left = sorted(tree(quiet))
        print(f"  idle workbench: removed {len(removed)} pack files, kept {kept} (edited since install)")
        print(f"  left in the repo: {left}")
        assert kept == ["docs/agent-rules.md"] and len(removed) == len(install(PACK, quiet, dry_run=True, force=True)) - 1
        assert left == ["agent_state.json", "docs/agent-rules.md", "docs/team-notes.md", "task_board.json"]

        busy = installed_repo(Path(tmp) / "busy", {**idle, "active_task_id": "T-002"})
        try:
            uninstall(PACK, busy)
            outcome = "removed"
        except Refused as refusal:
            outcome = f"refused: {refusal}"
        print(f"  a task is active: {outcome}")
        assert outcome == "refused: task T-002 is active" and (busy / "AGENTS.md").exists()
        assert uninstall(PACK, busy, force=True)[0] and not (busy / "AGENTS.md").exists()

        missing = []
        shipped = shutil.copytree(PACK, Path(tmp) / "pack")                         # a pack that ships the uninstaller
        (shipped / "bin" / "uninstall.sh").write_text(UNINSTALL_SH, newline="\n")
        shutil.copy(__file__, shipped / "bin" / "uninstall.py")
        at_rest, at_work = installed_repo(Path(tmp) / "at-rest", idle), installed_repo(Path(tmp) / "at-work", {**idle, "active_task_id": "T-002"})
        codes = bash_runs(shipped / "bin" / "uninstall.sh", [(at_rest, ""), (at_work, ""), (at_work, "--force")])
        if codes is None:
            print("  needs bash to run bin/uninstall.sh")
            missing.append("bash")
        else:
            print(f"  bin/uninstall.sh run with bash from a pack that ships it: idle exit {codes[0]}, task active exit {codes[1]}, --force exit {codes[2]}")
            assert codes == [0, 1, 0] and not (at_rest / "AGENTS.md").exists() and not (at_work / "AGENTS.md").exists()
            assert (at_rest / "docs" / "agent-rules.md").exists() and (at_rest / "agent_state.json").exists()      # edited rules and state are kept

        if not shutil.which("git"):
            print("  needs git for the uncommitted-changes check")
            missing.append("git")
        else:
            tracked = installed_repo(Path(tmp) / "tracked", idle)
            identity = ["-c", "user.name=practice", "-c", "user.email=practice@example.invalid", "-c", "commit.gpgsign=false"]
            for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "workbench installed"]):
                subprocess.run(["git", *identity, *args], cwd=tracked, capture_output=True, check=True)
            assert stranded_by_removal(tracked) == []
            (tracked / "agent_state.json").write_text(json.dumps({**idle, "next_action": "write the release notes"}))
            reasons = stranded_by_removal(tracked)
            print(f"  state edited and not committed: refused: {reasons[0]}")
            assert reasons == ["uncommitted changes in ['agent_state.json']"]
    return ", ".join(missing) or None


# ---------------------------------------------------------------------------
# Exercise 4 - lint_pack: fail when the pack drifts from VERSION
#
# VERSION is the contract, so everything that repeats it is checked against
# it: the "Pack version" line in AGENTS.md, the schema_version the state
# schema accepts (it has to be VERSION's major, because a major bump is what
# a state migration means), and, in an installed repo, the lock file the
# installer wrote. It also checks that every doc AGENTS.md routes to is in
# the pack and that every script compiles.
#
# The pack as shipped is clean. The workflow below is for the pack's own
# repository and is kept as text; copy it to .github/workflows/ to use it.
# lint_pack.py is this file under that name: its --lint-pack command needs
# nothing else from the lesson. The workflow's run step is executed below in
# a copy of the pack, the way a checkout would run it.
# ---------------------------------------------------------------------------

WORKFLOW = """\
name: lint-pack
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: python lint_pack.py --lint-pack .
"""


def lint_pack(pack: Path, target: Path | None = None) -> list[str]:
    version = (pack / "VERSION").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return [f"VERSION is {version!r}, not MAJOR.MINOR.PATCH"]
    problems = []
    agents = (pack / "AGENTS.md").read_text(encoding="utf-8")
    stated = re.search(r"^Pack version: (\S+)$", agents, re.M)
    if not stated or stated[1] != version:
        problems.append(f"AGENTS.md says pack version {stated[1] if stated else 'nothing'}, VERSION says {version}")
    accepted = json.loads((pack / "schemas" / "agent_state.schema.json").read_text())["properties"]["schema_version"]["enum"]
    if accepted != [int(version.split(".")[0])]:
        problems.append(f"the state schema accepts schema_version {accepted}, VERSION's major is {version.split('.')[0]}")
    problems += [f"AGENTS.md routes to {doc}, which is not in the pack"
                 for doc in re.findall(r"`(docs/[\w.-]+)`", agents) if not (pack / doc).exists()]
    for script in sorted((pack / "scripts").glob("*.py")):
        try:
            compile(script.read_text(encoding="utf-8"), script.name, "exec")
        except SyntaxError as error:
            problems.append(f"scripts/{script.name} does not compile: {error.msg}")
    if target is not None:
        lock = (target / LOCK).read_text().strip() if (target / LOCK).exists() else "missing"
        if lock != version:
            problems.append(f"{target.name}/{LOCK} is {lock}, the pack is {version}")
    return problems


def ex4_lint_pack() -> None:
    def drifted(root: Path, name: str, change: Any) -> Path:
        copy = shutil.copytree(PACK, root / name)
        change(copy)
        return copy

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = {
            "the pack as shipped": PACK,
            "VERSION bumped to 1.1.0 only": drifted(root, "minor", lambda pack: (pack / "VERSION").write_text("1.1.0\n")),
            "VERSION bumped to 2.0.0 only": drifted(root, "major", lambda pack: (pack / "VERSION").write_text("2.0.0\n")),
            "a doc deleted": drifted(root, "doc", lambda pack: (pack / "docs" / "handoff-protocol.md").unlink()),
        }
        found = {label: lint_pack(pack) for label, pack in cases.items()}
        for label, problems in found.items():
            print(f"  {label:<29} {len(problems)} problems{': ' + '; '.join(problems) if problems else ''}"[:170])
        assert [len(problems) for problems in found.values()] == [0, 1, 2, 1]

        repo = root / "installed"
        repo.mkdir()
        install(PACK, repo)
        assert lint_pack(PACK, target=repo) == []
        stale = lint_pack(cases["VERSION bumped to 1.1.0 only"], target=repo)
        print(f"  an installed repo against a newer pack: {stale[-1]}")
        assert stale[-1] == f"installed/{LOCK} is 1.0.0, the pack is 1.1.0"

        step = re.search(r"- run: python (.+)", WORKFLOW)[1].split()         # the workflow's own step
        as_ci = []
        for checkout in (shutil.copytree(PACK, root / "checkout"), cases["VERSION bumped to 2.0.0 only"]):
            shutil.copy(__file__, checkout / "lint_pack.py")
            as_ci.append(subprocess.run([sys.executable, *step], cwd=checkout, capture_output=True, text=True).returncode)
    print(f"  the workflow's step (python {' '.join(step)}): exit {as_ci[0]} in a checkout of the shipped pack, exit {as_ci[1]} in the drifted one")
    assert as_ci == [0, 1] and step == ["lint_pack.py", "--lint-pack", "."]


# ---------------------------------------------------------------------------
# Exercise 5 - migrating a hand-rolled workbench to the pack
#
# The order that keeps downtime smallest: everything additive first,
# everything that only reads next, and the router last. Docs, schemas and
# scripts can be laid beside the old workbench without anything reading them.
# Validation and the state migration happen on copies. AGENTS.md is the file
# every session reads first, so swapping it is the cutover, and it is the
# only step during which agents have to stop: one commit, made between two
# tasks.
#
# The runbook is run below on lesson 32's three-file workbench. What it
# found on the way: that workbench's state has no schema_version, so it fails
# the pack's schema until the migration adds one; and lesson 32's AGENTS.md
# points at docs/agent-rules.md, which does not exist until step 2 puts it
# there.
# ---------------------------------------------------------------------------

RUNBOOK = [     # (step, agents keep working during it, how to undo it)
    ("tag the current commit", True, "nothing to undo"),
    ("install docs, schemas and scripts beside the old workbench; leave its AGENTS.md", True, "delete the three folders"),
    ("check the board and the state against the pack's schemas", True, "nothing was changed"),
    ("write the migrated state to a copy and validate the copy", True, "delete the copy"),
    ("run scripts/init_agent.py", True, "delete init_report.json"),
    ("between two tasks: swap in the migrated state and the pack's AGENTS.md, as one commit", False, "revert the commit"),
    ("run the next task through the new gate; drop the old scripts after a few clean tasks", True, "go back to the tag"),
]


def ex5_migration_runbook() -> None:
    lesson_32, lesson_34 = load_lesson("32-minimal-agent-workbench", "lesson32"), load_lesson("34-repo-memory-and-state", "lesson34")
    schemas = {name: json.loads((PACK / "schemas" / f"{name}.schema.json").read_text()) for name in ("agent_state", "task_board")}

    def valid(value: Any, schema: str) -> str:
        try:
            lesson_34.validate(value, schemas[schema])
            return "valid"
        except lesson_34.SchemaError as error:
            return str(error)

    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        lesson_32.write_initial(repo / "agent_state.json", repo / "task_board.json", repo / "AGENTS.md")     # the hand-rolled workbench
        old_router = (repo / "AGENTS.md").read_bytes()
        dangling_before = not (repo / "docs" / "agent-rules.md").exists()

        install(PACK, repo, skip=("AGENTS.md",))                                                # step 2
        router_untouched = (repo / "AGENTS.md").read_bytes() == old_router
        old_state = json.loads((repo / "agent_state.json").read_text())
        checks = (valid(json.loads((repo / "task_board.json").read_text()), "task_board"), valid(old_state, "agent_state"))     # step 3
        migrated = {"schema_version": 1, **old_state}                                           # step 4
        (repo / "agent_state.migrated.json").write_text(json.dumps(migrated, indent=2) + "\n")
        copy_check = valid(migrated, "agent_state")
        init_exit, _ = run_script(repo, "init_agent.py")                                        # step 5
        os.replace(repo / "agent_state.migrated.json", repo / "agent_state.json")               # step 6, the cutover
        install(PACK, repo, force=True)
        routed = re.findall(r"`([\w./-]+\.(?:md|json))`", (repo / "AGENTS.md").read_text(encoding="utf-8"))
        after = (lint_pack(PACK, target=repo), [name for name in routed if not (repo / name).exists()],
                 valid(json.loads((repo / "agent_state.json").read_text()), "agent_state"))

    for number, (step, working, undo) in enumerate(RUNBOOK, 1):
        print(f"  {number}. {step:<84} {'agents working' if working else 'AGENTS PAUSED'}   undo: {undo}")
    print(f"  step 2: old AGENTS.md untouched: {router_untouched}; its pointer to docs/agent-rules.md was dangling: {dangling_before}")
    print(f"  step 3: board {checks[0]}; state: {checks[1]}")
    print(f"  step 4: migrated copy {copy_check};  step 5: init exits {init_exit}")
    print(f"  after the cutover: lint problems {after[0]}, files AGENTS.md points at that are missing {after[1]}, state {after[2]}")
    assert [working for _, working, _ in RUNBOOK].count(False) == 1 and not RUNBOOK[5][1]
    assert router_untouched and dangling_before and checks == ("valid", "$: missing required field 'schema_version'")
    assert copy_check == "valid" and init_exit == 0 and after == ([], [], "valid")


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run twice in a temp copy of this folder, so the pack in
# ../outputs is not rewritten.
#   exits zero and prints the pack tree      yes
#   re-running the assembler is idempotent   yes, and what it builds is the
#                                            pack committed in ../outputs
#   install into a fresh target leaves       seven of the nine. The rules,
#   state, board, rules, scope, init,        the scope schema, init, the
#   runner, gate, reviewer and handoff in    runner, the gate, the reviewer's
#   place                                    rubric and the handoff generator
#                                            are there. The state file and the
#                                            board are not: the installer
#                                            copies surface files only and
#                                            leaves state to the repository.
# Until someone writes those two files, the first two things AGENTS.md tells
# an agent to read do not exist.
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


IN_PLACE = {"state": "agent_state.json", "board": "task_board.json", "rules": "docs/agent-rules.md",
            "scope": "schemas/scope_contract.schema.json", "init": "scripts/init_agent.py",
            "runner": "scripts/run_with_feedback.py", "gate": "scripts/verify_agent.py",
            "reviewer": "docs/reviewer-rubric.md", "handoff": "scripts/generate_handoff.py"}


def mission_acceptance() -> None:
    first, second = run_main(times=2)
    built = {name.removeprefix("outputs/agent-workbench-pack/"): text for name, text in second[2].items()}
    # compared as text: main.py writes with the platform's default encoding, the committed pack is UTF-8
    committed = {name: path.read_text(encoding="utf-8") for name in built if (path := PACK / name).exists()}
    with tempfile.TemporaryDirectory() as tmp:
        install(PACK, Path(tmp))
        there = {surface: (Path(tmp) / name).exists() for surface, name in IN_PLACE.items()}
    print(f"  exit codes {first[0]} and {second[0]}; the tree lists {len(first[1].splitlines())} files; second run identical: {first[2] == second[2]}")
    print(f"  what it builds matches the committed pack: {built == committed}")
    print(f"  after an install: {sum(there.values())} of {len(there)} in place; not there: {[s for s, ok in there.items() if not ok]}")
    assert first[0] == second[0] == 0 and first[2] == second[2] and len(first[1].splitlines()) == len(built) == 15
    assert built == committed and [surface for surface, ok in there.items() if not ok] == ["state", "board"]


if __name__ == "__main__":
    flags = {"--force", "--dry-run"}
    if sys.argv[1:2] in (["--install"], ["--uninstall"], ["--lint-pack"]):
        unknown = [arg for arg in sys.argv[3:] if arg.startswith("--") and arg not in flags]
        if len(sys.argv) < 3 or unknown:
            sys.exit(f"usage: see the top of this file; unknown flags: {unknown}")
        try:
            if sys.argv[1] == "--install":
                print("\n".join(install(PACK, Path(sys.argv[2]), force="--force" in sys.argv, dry_run="--dry-run" in sys.argv)))
            elif sys.argv[1] == "--uninstall":
                gone, edited = uninstall(PACK, Path(sys.argv[2]), force="--force" in sys.argv)
                print(f"removed {len(gone)} files; kept because edited: {edited}")
            else:
                found = lint_pack(Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
                print("\n".join(found) or "pack ok")
                sys.exit(1 if found else 0)
        except Refused as refusal:
            sys.exit(f"refused: {refusal}")
        sys.exit(0)
    print("Phase 14 - Lesson 42: Ship a Reusable Agent Workbench Pack - exercises")
    outcomes = []
    for exercise in (ex1_fifth_doc, ex2_python_installer, ex3_uninstall, ex4_lint_pack, ex5_migration_runbook,
                     mission_acceptance):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
