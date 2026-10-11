"""Phase 14 - Lesson 54: Build a Feedback Ratchet with Ownership and Retirement - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

Both signals are real: an incident that lesson 49's safety scan found in this repository's practice
files, and a complaint the owner made while they were being written. Counts are read from git
(needs git on the PATH), and every verification command is run.
"""

from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from main import RatchetAction, Signal, backlog, destination, promote

REPO = Path(__file__).resolve().parents[2]
COURSE, BATCH, LATEST = "6a735e9", "f7befa5", "a08499a"   # the course as published; the commit of lessons 31-40; the last commit read here
SWITCH = r'os\.environ\.get\("[A-Z_]+(_KEY|_ENDPOINT)"'
SCAN = "49-map-assumptions-and-risk/code/practice.py"


def git(*args: str) -> list[str]:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.splitlines()


# ---------------------------------------------------------------------------
# Exercise 1 - one incident and one user complaint turned into ratchet actions
#
# The incident: practice files that called a paid or outside service as soon
# as a key was present. The complaint (RECORDED, the owner in session,
# paraphrased): lessons were being committed first and refined afterwards.
# Frequency is counted, not guessed: the files the scan finds at the last
# commit, and the commits that changed a file already on main. main.promote
# gives each an owner, a priority, an artifact and a retirement line. It files
# both under "backlog", because neither sentence contains one of its keywords.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def signals() -> tuple[Signal, Signal]:
    called_out = git("grep", "-l", "-E", "-e", SWITCH, LATEST, "--", "*/code/practice.py")
    reworking = [sha for sha in git("rev-list", f"{COURSE}..{LATEST}")
                 if any(line.startswith("M") and line.endswith("practice.py") for line in git("show", "--name-status", "--format=", sha))]
    return (
        Signal("incident: lesson 49's safety scan", "practice files called an outside service as soon as a key was in the environment",
               4, len(called_out), "repository owner", 180),
        Signal("complaint: the owner, in session", "lessons were committed first and refined afterwards, so files on main kept changing",
               2, len(reworking), "repository owner", 90),
    )


def ex1_incident_and_complaint() -> None:
    actions = backlog(list(signals()))
    for signal, action in zip(sorted(signals(), key=lambda item: -item.severity * item.frequency), actions):
        print(f"  {signal.source}")
        print(f"    severity {signal.severity} x seen {signal.frequency} times = priority {action.priority}; main.destination: {action.destination}; owner: {action.owner}")
        print(f"    {action.change}\n    artifact: {action.durable_artifact}; retire: {action.retirement_check}")
    assert [signal.frequency for signal in signals()] == [4, 5] and [action.priority for action in actions] == [16, 10]
    assert {destination(signal) for signal in signals()} == {"backlog"} and all(action.owner and action.retirement_check for action in actions)


# ---------------------------------------------------------------------------
# Exercise 2 - the earliest layer that can prevent each recurrence
#
# The layers run from the most mechanical to the least. For each signal the
# first one that would stop the next occurrence is the answer. The incident
# recurs when a new file reads a key and forgets the flag, and a scan in the
# suite catches that before a commit; a flag check inside each file is the
# very thing a new file forgets. The complaint is about order, and no check
# knows whether a batch has been looked at twice: that is a condition on the
# owner's yes, so the earliest layer is policy. Neither is "backlog".
# ---------------------------------------------------------------------------

LAYERS = ["runtime", "evaluation", "policy", "context", "backlog"]
PREVENTS = {   # signal number -> layer -> why it would, or would not, stop the next occurrence
    0: {"runtime": "", "evaluation": "a scan fails the run when a file reads a key or endpoint and takes no flag"},
    1: {"runtime": "", "evaluation": "", "policy": "the owner's yes is asked for only after the batch has been refined"},
}
ARTIFACTS = {"evaluation": SCAN, "policy": "the owner's standing instructions to the agent (kept outside this repository)"}


def earliest_layer(number: int) -> str:
    return next(layer for layer in LAYERS if PREVENTS[number].get(layer))


def routed() -> list[RatchetAction]:
    """main's actions, moved to the layer that can prevent the recurrence."""
    return [replace(promote(signal), destination=earliest_layer(number), durable_artifact=ARTIFACTS[earliest_layer(number)])
            for number, signal in enumerate(signals())]


def ex2_earliest_layer() -> None:
    for number, (signal, action) in enumerate(zip(signals(), routed())):
        print(f"  {signal.source}: main says {destination(signal)}, earliest that prevents it: {action.destination}")
        print(f"    because {PREVENTS[number][action.destination]}\n    artifact: {action.durable_artifact}")
    scan = (REPO / SCAN).read_text(encoding="utf-8")
    print(f"  the scan exists and its result is asserted: {'now == []' in scan}")
    assert [action.destination for action in routed()] == ["evaluation", "policy"] and (REPO / SCAN).is_file()
    assert "def calls_out_unasked" in scan and "now == []" in scan
    assert all(LAYERS.index(action.destination) < LAYERS.index(destination(signal)) for signal, action in zip(signals(), routed()))


# ---------------------------------------------------------------------------
# Exercise 3 - verification commands and observations in the lab output
#
# main writes "Record review against the outcome frame" for both actions,
# which nobody can run. Each action gets a command that exits 0 while its
# control holds, plus what to look for. The backlog is written the way main
# writes it (to a temp folder, not into the repository), read back, and every
# command in it is run.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verified(RatchetAction):
    verification_command: str = ""        # run from the repository root; exit 0 means the control holds
    verification_observation: str = ""


def verified() -> list[Verified]:
    incident, complaint = routed()
    return [
        Verified(**asdict(incident), verification_command=f"python {SCAN}",
                 verification_observation="its second exercise reports 0 files that call out without --live"),
        Verified(**asdict(complaint), verification_command=f'git diff --quiet {BATCH} {LATEST} -- "3[1-9]-*" "40-*"',
                 verification_observation="no lesson of the batch that was refined first is changed by a later commit"),
    ]


@lru_cache(maxsize=None)
def run(command: str) -> int:
    words = shlex.split(command)
    return subprocess.run([sys.executable if words[0] == "python" else words[0], *words[1:]], cwd=REPO, capture_output=True, timeout=300).returncode


def ex3_verification_in_the_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "feedback-backlog.json"
        output.write_text(json.dumps([asdict(action) for action in verified()], indent=2) + "\n", encoding="utf-8")
        written = json.loads(output.read_text(encoding="utf-8"))
    codes = [run(entry["verification_command"]) for entry in written]
    for entry, code in zip(written, codes):
        print(f"  [{entry['destination']}] main's wording: {entry['verification_evidence']}")
        print(f"    command: {entry['verification_command']} -> exit {code}\n    look for: {entry['verification_observation']}")
    earlier = run(f'git diff --quiet {COURSE} {LATEST} -- "0[1-9]-*"')
    print(f"  the second command on lessons 01-09, which were committed and then refined: exit {earlier}")
    assert codes == [0, 0] and all(entry["verification_command"] and entry["verification_observation"] for entry in written)
    assert len({entry["verification_evidence"] for entry in written}) == 1 and earlier == 1


# ---------------------------------------------------------------------------
# Exercise 4 - a retirement condition for a policy rule
#
# The rule: a batch is committed and pushed only on the owner's yes for that
# batch. main would retire it after ninety quiet days. For a permission rule,
# quiet is the rule working, so time alone never retires it here. It goes
# when a lower layer enforces the same boundary (the host refuses a push to
# main that the owner has not approved), or when it has blocked more
# legitimate work than it has prevented harm. Today neither is true.
# ---------------------------------------------------------------------------

RULE = "A batch is committed and pushed only on the owner's yes for that batch"


def retire(quiet_days: int, enforced_lower_down: bool, blocked_legitimate: int = 0, prevented: int = 0) -> tuple[bool, str]:
    if enforced_lower_down:
        return True, "a lower layer now enforces the boundary; keep one test that it still does"
    if blocked_legitimate > prevented:
        return True, "it has blocked more legitimate work than it has prevented harm"
    return False, f"{quiet_days} quiet days are the rule working, and nothing below it holds the line"


def ex4_retirement_condition() -> None:
    by_main = promote(Signal("standing rule", RULE, 3, 1, "repository owner", 90)).retirement_check
    hook = REPO / git("rev-parse", "--git-path", "hooks/pre-push")[0]
    authors = set(git("log", "--format=%an", f"{COURSE}..{LATEST}"))
    print(f"  rule: {RULE}\n  main's retirement line: {by_main}")
    cases = {"day 400, nothing below it": retire(400, False), "the host refuses unapproved pushes": retire(10, True),
             "blocked 6 good batches, stopped 1 bad one": retire(10, False, 6, 1)}
    for case, (gone, why) in cases.items():
        print(f"  {case:<42} -> {'retire' if gone else 'keep'}: {why}")
    today = retire(2, hook.is_file())
    print(f"  today: a pre-push hook in this clone: {hook.is_file()}; authors on the {len(git('rev-list', f'{COURSE}..{LATEST}'))} commits: {len(authors)} -> {'retire' if today[0] else 'keep'}")
    assert "90 days without recurrence" in by_main and [gone for gone, _ in cases.values()] == [False, True, True]
    assert not today[0] and len(authors) == 1


# ---------------------------------------------------------------------------
# Exercise 5 - one accepted correction traced into the next task frame
#
# The next task is lesson 49's riskiest open bet: put a lesson folder in front
# of three readers. Its frame is lesson 47's OutcomeFrame. The incident's
# control enters as a constraint that carries its origin and its check, so
# the chain can be walked both ways: signal -> action -> constraint, and from
# the constraint back to the command that proves it. A constraint that no
# action stands behind is reported as untraceable.
# ---------------------------------------------------------------------------

def lesson_47():
    spec = importlib.util.spec_from_file_location("lesson47_main", REPO / "47-outcomes-before-output" / "code" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def as_constraint(action: Verified, source: str) -> str:
    return f"a plain run of anything the readers are handed makes no outside call [from {source}; check: {action.verification_command}]"


def origin(constraint: str, actions: list[Verified]) -> Verified | None:
    """The action a constraint came from: the one whose verification command it carries."""
    return next((action for action in actions if f"check: {action.verification_command}]" in constraint), None)


def ex5_correction_into_the_next_frame() -> None:
    frames = lesson_47()
    incident = verified()[0]
    frame = frames.OutcomeFrame(
        user="owner deciding whether to go on adding lessons",
        situation="every tested assumption is one a script could check; the riskiest one is about other people",
        current_behavior="nothing on record shows that anyone but the owner has run a practice file",
        desired_outcome="knows from what three readers did with one lesson folder whether it lets them check the lesson",
        constraints=[as_constraint(incident, signals()[0].source), "the readers get no explanation beyond the folder"],
        non_goals=["more lessons before the readers have been asked", "a survey in place of watching"],
        proposed_output="reader trial",
    )
    verdict = frames.decision(frame)
    traced = {constraint: origin(constraint, verified()) for constraint in frame.constraints}
    for constraint, action in traced.items():
        print(f"  constraint: {constraint[:96]}")
        print(f"    {'from: ' + action.change[:80] + '; verified: exit ' + str(run(action.verification_command)) if action else 'untraceable: no action stands behind it'}")
    print(f"  the frame: {verdict['status']}; {verdict['next_question']}")
    assert verdict["status"] == "ready-to-discover" and [action is not None for action in traced.values()] == [True, False]
    assert traced[frame.constraints[0]].destination == "evaluation" and signals()[0].source in frame.constraints[0]
    assert origin(frame.constraints[0], verified()[1:]) is None      # take the action out of the backlog and the constraint loses its origin


if __name__ == "__main__":
    print("Phase 14 - Lesson 54: Build a Feedback Ratchet with Ownership and Retirement - exercises")
    for exercise in (ex1_incident_and_complaint, ex2_earliest_layer, ex3_verification_in_the_output, ex4_retirement_condition,
                     ex5_correction_into_the_next_frame):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
