"""Phase 14 - Lesson 53: Choose Prototype, Pilot, or Production Deliberately - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The three projects are things this repository holds, and the pilot is the practice files of
lessons 01-40 on main. Facts about them are read from the repository; the rollback is rehearsed in
a throwaway clone (needs git on the PATH).
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path

from main import BuildDecision, choose_stage, plan, required_controls

REPO = Path(__file__).resolve().parents[2]
COURSE, LATEST = "6a735e9", "a08499a"     # the course as published; the last commit of the pilot
PACK = REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack"


def git(*args: str, cwd: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


# ---------------------------------------------------------------------------
# Exercise 1 - three current projects classified by learning stage
#
# Deployment status is what the project looks like from outside: a version
# number, a place on main, a merged flag. The stage comes from what is still
# unknown and what finding out would touch. Two of the three look released and
# are pilots; nothing here is production, because nobody has taken on keeping
# any of it working. The pack cannot become production by being declared
# ready: it ships no way to take it out again.
# ---------------------------------------------------------------------------

def projects() -> dict[str, tuple[str, BuildDecision]]:
    """Name -> (deployment status as read from the repository, the decision that sets the stage)."""
    removable = any("uninstall" in path.name for path in PACK.rglob("*"))
    on_main = sum(name.endswith("code/practice.py") for name in git("ls-tree", "-r", "--name-only", LATEST).splitlines())
    flagged = "--live" in next(REPO.glob("41-*/code/practice.py")).read_text(encoding="utf-8")
    return {
        "the agent-workbench pack": (
            f"VERSION {(PACK / 'VERSION').read_text(encoding='utf-8').strip()}, ships an installer",
            BuildDecision("Does the pack help an agent in a repository it was not written for?", True, True, 3, removable, False)),
        "the practice files": (
            f"{on_main} files on the main branch",
            BuildDecision("Does a reader other than the owner use a file to check a lesson?", True, False, 2, True, False)),
        "lesson 41's model-written patch": (
            f"in the file behind --live: {flagged}",
            BuildDecision("Can a model write the patch that the scripted policy stands in for?", False, False, 1, True, False)),
    }


def ex1_three_projects_by_stage() -> None:
    stages = {}
    for name, (status, decision) in projects().items():
        stages[name] = choose_stage(decision)
        print(f"  {name:<32} status: {status:<36} stage: {stages[name]}")
        print(f"  {'':<32} still unknown: {decision.unknown}")
    pack, files = (decision for _, decision in list(projects().values())[:2])
    print(f"  if someone took on running them: practice files -> {choose_stage(replace(files, operational_readiness=True))}, "
          f"pack -> {choose_stage(replace(pack, operational_readiness=True))} (it ships no uninstaller)")
    assert list(stages.values()) == ["pilot", "pilot", "prototype"] and "production" not in stages.values()
    assert projects()["the agent-workbench pack"][0].startswith("VERSION 1.0.0") and not pack.reversible
    assert choose_stage(replace(files, operational_readiness=True)) == "production" and choose_stage(replace(pack, operational_readiness=True)) == "pilot"
    assert choose_stage(replace(pack, operational_readiness=True, reversible=True)) == "production"


# ---------------------------------------------------------------------------
# Exercise 2 - pilot exit criteria that include a stop decision
#
# The pilot is the practice files in front of readers. It ends one of three
# ways and each is written down with its number: expand when two of three
# readers run a file, revise when fewer do, stop when none does, when a
# guardrail breaks, or when thirty days pass with nobody asked. The last
# keeps a pilot from turning into the product by never being looked at.
# Read as of the pilot's last commit, the answer is to ask the readers.
# ---------------------------------------------------------------------------

PILOT_DAYS = 30


def exit_decision(asked: int, ran: int, guardrail_broken: bool, days: int) -> str:
    if guardrail_broken:
        return "stop: a run spent or sent something unasked"
    if asked == 0:
        return f"stop: {PILOT_DAYS} days and nobody was asked" if days > PILOT_DAYS else "revise: no evidence yet, ask three readers"
    if ran == 0:
        return "stop: nobody who was asked ran a file"
    return "expand: the next ten lessons, same three readers" if ran / asked >= 2 / 3 else "revise: find out why the others did not run it"


def ex2_pilot_exit_criteria() -> None:
    cases = {"three asked, three ran": (3, 3, False, 10), "three asked, one ran": (3, 1, False, 10), "three asked, none ran": (3, 0, False, 10),
             "a guardrail broke": (3, 3, True, 10), "nobody asked, day 45": (0, 0, False, 45)}
    for case, numbers in cases.items():
        print(f"  {case:<24} -> {exit_decision(*numbers)}")
    started, ended = (date.fromisoformat(git("log", "-1", "--format=%cs", commit)) for commit in (git("rev-list", f"{COURSE}..{LATEST}").splitlines()[-1], LATEST))
    now = exit_decision(0, 0, False, (ended - started).days)
    print(f"  the pilot as of {LATEST} (day {(ended - started).days}, nobody asked on record) -> {now}")
    kinds = [exit_decision(*numbers).split(":")[0] for numbers in cases.values()]
    assert kinds == ["expand", "revise", "stop", "stop", "stop"] and "exit criteria" in required_controls("pilot")
    assert now.startswith("revise") and (ended - started).days == 2


# ---------------------------------------------------------------------------
# Exercise 3 - a technical control that keeps a prototype off production data
#
# Production data here is what the repository has committed. This lesson's
# main.py writes into ../outputs when it runs, which is exactly that. The
# control is an audit hook: a build marked prototype may open files for
# writing inside its own folder and nowhere else, and a build with no stage
# marker does not start. The stage is read from the environment, so the
# system itself says what it is. main.py is run three ways under it.
# ---------------------------------------------------------------------------

# ponytail: the hook sees one process and Python-level opens; a child process or a C extension would need an OS sandbox
STAGE_GUARD = r'''
import os, runpy, sys
stage = os.environ.get("BUILD_STAGE")
if stage not in ("prototype", "pilot", "production"):
    sys.exit("refused: BUILD_STAGE is not set, and a build of unknown stage does not start")
own = os.path.normcase(os.path.realpath(os.environ["BUILD_OWN_DIR"])) + os.sep
def hook(event, args):
    if stage == "prototype" and event == "open" and isinstance(args[0], str) and isinstance(args[1], str) and set(args[1]) & set("wax+"):
        if not os.path.normcase(os.path.realpath(args[0])).startswith(own):
            raise PermissionError("refused: a prototype may not write to " + args[0])
sys.dont_write_bytecode = True
sys.addaudithook(hook)
runpy.run_path(sys.argv[1], run_name="__main__")
'''


def staged(script: Path, own: Path, stage: str | None) -> subprocess.CompletedProcess:
    env = {name: value for name, value in os.environ.items() if name != "BUILD_STAGE"} | {"BUILD_OWN_DIR": str(own)} | ({"BUILD_STAGE": stage} if stage else {})
    return subprocess.run([sys.executable, "-c", STAGE_GUARD, str(script)], capture_output=True, text=True, timeout=60, env=env)


def ex3_prototype_cannot_reach_production_data() -> None:
    here = Path(__file__).resolve().parent
    committed = here.parent / "outputs" / "stage-decisions.json"
    before = committed.read_bytes() if committed.exists() else None
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "code" / "main.py"
        copy.parent.mkdir()
        (Path(tmp) / "outputs").mkdir()
        copy.write_bytes((here / "main.py").read_bytes())
        in_place = staged(here / "main.py", Path(tmp), "prototype")
        in_own_folder = staged(copy, Path(tmp), "prototype")
        unmarked = staged(copy, Path(tmp), None)
        written = (Path(tmp) / "outputs" / "stage-decisions.json").is_file()
    after = committed.read_bytes() if committed.exists() else None
    print(f"  prototype, main.py where it lives : exit {in_place.returncode}, {in_place.stderr.strip().splitlines()[-1][:76]}...")
    print(f"  prototype, main.py in its own copy: exit {in_own_folder.returncode}, wrote its output there: {written}")
    print(f"  no stage marker                   : exit {unmarked.returncode}, {unmarked.stderr.strip()}")
    print(f"  the committed outputs file is unchanged: {before == after}")
    assert in_place.returncode != 0 and "a prototype may not write to" in in_place.stderr and before == after
    assert in_own_folder.returncode == 0 and written and unmarked.returncode != 0 and "BUILD_STAGE is not set" in unmarked.stderr
    assert "synthetic or recorded inputs" in required_controls("prototype")


# ---------------------------------------------------------------------------
# Exercise 4 - the first operational responsibility that makes it production
#
# Seven controls separate production from a pilot. Each is listed with the
# one it cannot exist without. Six of them need somebody who is answerable: a
# service level is a promise by someone, monitoring pages someone, recovery is
# done by someone. Only "on-call owner" needs none of the others, so it comes
# first, and taking it on is the moment the build stops being a pilot.
# ---------------------------------------------------------------------------

RESTS_ON = {   # production-only control -> the control it presupposes
    "service level objective": "on-call owner",
    "continuous monitoring": "on-call owner",
    "recovery": "on-call owner",
    "security review": "on-call owner",
    "cost and capacity controls": "service level objective",
    "retirement path": "on-call owner",
    "on-call owner": "",
}


def rests_on_in_the_end(control: str) -> str:
    """Follow RESTS_ON down to the control that rests on nothing."""
    return control if not RESTS_ON[control] else rests_on_in_the_end(RESTS_ON[control])


def ex4_first_operational_responsibility() -> None:
    added = [control for control in required_controls("production") if control not in required_controls("pilot")]
    first = [control for control in added if not RESTS_ON[control]]
    for control in added:
        print(f"  {control:<27} rests on: {RESTS_ON[control] or 'nothing else on the list'}")
    files = projects()["the practice files"][1]
    stages = [choose_stage(replace(files, operational_readiness=ready)) for ready in (False, True)]
    answerable = [name for name in ("CODEOWNERS", ".github/CODEOWNERS", "SECURITY.md") if (REPO / name).exists()]
    print(f"  first: {first[0]}; the practice files without it: {stages[0]}, with it: {stages[1]}")
    print(f"  files in this repository that name someone answerable: {answerable or 'none'}")
    assert set(added) == set(RESTS_ON) and len(added) == 7 and first == ["on-call owner"]
    assert all(RESTS_ON[control] in added for control in added if RESTS_ON[control]) and stages == ["pilot", "production"]
    assert {rests_on_in_the_end(control) for control in added} == {"on-call owner"}      # every one of the seven comes down to it


# ---------------------------------------------------------------------------
# Exercise 5 - a rollback receipt for the bounded pilot
#
# The pilot is bounded by two commits, so rolling it back is one command over
# that range. A rollback that has never been run is a hope, so it is run in a
# throwaway clone. The receipt records what was undone, the command, the tree
# it was meant to restore and the tree it produced. It also records that the
# pilot's commits are still in the history (a rollback is not an erasure) and
# that the real repository did not move.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RollbackReceipt:
    pilot: str
    commits_undone: int
    command: str
    tree_to_restore: str
    tree_after_rollback: str
    pilot_still_in_history: bool
    real_repository_moved: bool

    @property
    def restored(self) -> bool:
        return self.tree_to_restore == self.tree_after_rollback


def rehearse_rollback() -> RollbackReceipt:
    head, state = git("rev-parse", "HEAD"), git("status", "--porcelain")
    command = ["revert", "--no-edit", f"{COURSE}..{LATEST}"]
    with tempfile.TemporaryDirectory() as tmp:
        git("clone", "--quiet", "--shared", "--no-checkout", str(REPO), tmp)       # shares objects: nothing is copied or changed
        git("checkout", "--quiet", "--detach", LATEST, cwd=Path(tmp))
        git("-c", "user.name=rollback rehearsal", "-c", "user.email=rehearsal@example.invalid", *command, cwd=Path(tmp))
        after = git("rev-parse", "HEAD^{tree}", cwd=Path(tmp))
        undone = int(git("rev-list", "--count", f"{LATEST}..HEAD", cwd=Path(tmp)))
        kept = subprocess.run(["git", "merge-base", "--is-ancestor", LATEST, "HEAD"], cwd=tmp).returncode == 0
    return RollbackReceipt(f"practice files of lessons 01-40 ({COURSE}..{LATEST})", undone, "git " + " ".join(command),
                           git("rev-parse", f"{COURSE}^{{tree}}"), after, kept,
                           (git("rev-parse", "HEAD"), git("status", "--porcelain")) != (head, state))


def ex5_rollback_receipt() -> None:
    receipt = rehearse_rollback()
    for field, value in asdict(receipt).items():
        print(f"  {field:<22} {value}")
    print(f"  restored: {receipt.restored}")
    assert receipt.restored and receipt.commits_undone == 6 and receipt.pilot_still_in_history and not receipt.real_repository_moved
    assert {"rollback", "audit trail"} <= set(plan(projects()["the practice files"][1])["required_controls"])


if __name__ == "__main__":
    print("Phase 14 - Lesson 53: Choose Prototype, Pilot, or Production Deliberately - exercises")
    for exercise in (ex1_three_projects_by_stage, ex2_pilot_exit_criteria, ex3_prototype_cannot_reach_production_data,
                     ex4_first_operational_responsibility, ex5_rollback_receipt):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
