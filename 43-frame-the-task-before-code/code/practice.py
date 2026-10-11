"""Phase 14 - Lesson 43: Frame the Task Before the Agent Writes Code - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --acceptance      (the frame's proof: exits 1 while the bug is there)
           python practice.py --scope-receipt   (changed file names on stdin; exits 1 if any is out of scope)

The bug framed here is a real one in this repository: lesson 16's Runner.run returns an empty
string when it runs out of hops. Framing it changes no file.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from main import RepositoryFact, TaskFrame, render, validate

REPO = Path(__file__).resolve().parents[2]
RUNNER = "16-openai-agents-sdk/code/main.py"


def receipt(path: str, text: str) -> str:
    """'path:line' for the line of that file that contains the text, or '' when no line does."""
    file = REPO / path
    if file.is_file():
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if text in line:
                return f"{path}:{number}"
    return ""


def hop_limit_outcome() -> str:
    """What lesson 16's Runner gives back when two agents hand a request to each other until the limit."""
    spec = importlib.util.spec_from_file_location("lesson16", REPO / RUNNER)
    sdk: Any = importlib.util.module_from_spec(spec)
    sys.modules["lesson16"] = sdk
    spec.loader.exec_module(sdk)
    billing = sdk.Agent("billing", "refunds", policy=lambda text: {"kind": "handoff", "to": "triage", "input": text})
    triage = sdk.Agent("triage", "routing", policy=lambda text: {"kind": "handoff", "to": "billing", "input": text},
                       handoffs=[sdk.Handoff(target=billing)])
    billing.handoffs.append(sdk.Handoff(target=triage))
    try:
        return sdk.Runner(max_hops=4).run(triage, "refund please")
    except Exception as error:      # raising is one way to make the run tell-apart-able; the frame leaves that open
        return f"raised {type(error).__name__}"


ACCEPTANCE = "python 43-frame-the-task-before-code/code/practice.py --acceptance"
SCOPE_RECEIPT = "git diff --name-only main...HEAD | python 43-frame-the-task-before-code/code/practice.py --scope-receipt"


# ---------------------------------------------------------------------------
# Exercise 1 - a real bug, framed without proposing a solution
#
# The goal names the behaviour that has to change and stops there. Whether
# the run should raise, return a message or return a typed result is not in
# it; that choice is exercise 3's.
#
# Every repository fact carries a receipt, and the receipts are looked up, not
# typed in: receipt() finds the line that holds the quoted text, so a fact
# whose text is no longer in the file comes back without one and main.py's
# validator blocks the frame. The behaviour itself is a command result, which
# the lesson ranks above a file and line.
#
# The proof is written before any patch: `practice.py --acceptance` runs the
# scenario against the repository's own lesson 16 and fails today.
# ---------------------------------------------------------------------------

def first_draft() -> TaskFrame:
    return TaskFrame(
        goal="A run that uses up max_hops can be told apart from an agent that answered with nothing",
        allowed_paths=["16-openai-agents-sdk/**"],
        forbidden_paths=["16-openai-agents-sdk/docs/**", "16-openai-agents-sdk/quiz.json"],
        acceptance=[ACCEPTANCE],
        facts=[
            RepositoryFact("Runner.run starts from an empty final_output", receipt(RUNNER, 'final_output = ""')),
            RepositoryFact("The loop is bounded by max_hops and nothing runs when it is used up",
                           receipt(RUNNER, "for hop in range(self.max_hops):")),
            RepositoryFact("Two agents that hand off to each other end the run with an empty string",
                           f"{ACCEPTANCE} -> exit 1"),
            RepositoryFact("No caller depends on the empty string", ""),             # exercise 2
        ],
        unknowns=["discoverable: which checks in the repository assert the current behaviour"],
    )


def ex1_frame_a_real_bug() -> None:
    frame = first_draft()
    supported = [fact for fact in frame.facts if fact.evidence]
    proof = subprocess.run([sys.executable, __file__, "--acceptance"], capture_output=True, text=True)
    print(f"  goal: {frame.goal}")
    for fact in supported:
        print(f"  fact: {fact.claim}  [{fact.evidence}]")
    print(f"  the proof today: exit {proof.returncode} ({proof.stdout.strip()})")
    assert [fact.evidence for fact in supported[:2]] == [f"{RUNNER}:83", f"{RUNNER}:84"]
    assert proof.returncode == 1 and hop_limit_outcome() == ""                  # the command result is real
    assert not any(word in frame.goal.lower() for word in ("raise", "return", "exception", "add ", "fix"))
    assert receipt(RUNNER, "this text is not in the file") == ""


# ---------------------------------------------------------------------------
# Exercise 2 - a claim that was an assumption, replaced with evidence
#
# The first draft says "No caller depends on the empty string". It has no
# receipt, so main.py's validator already refuses the frame. Looking for the
# evidence turns the claim over: three checks in this repository pin the empty
# string. The practice files of lessons 16, 19 and 30 each assert the current
# behaviour, so a patch to lesson 16 alone turns three passing files red.
# (This file is a fourth, on purpose: its receipts and its proof describe the
# bug as it stands, and they stop holding the day it is fixed.)
#
# That is the difference the lesson draws: an unknown would have been "which
# checks assert this?"; the assumption answered it without looking.
# ---------------------------------------------------------------------------

PINS = [    # (what pins the behaviour, file, the line that does it)
    ("Lesson 16's practice file asserts the run returns ''", "16-openai-agents-sdk/code/practice.py", 'assert silent == ""'),
    ("Lesson 19's task t3 expects the hop-limit test to fail before its patch", "19-benchmarks-swebench-gaia/code/practice.py",
     "def t3_hop_limit_reported"),
    ("Lesson 30's eval case asserts the agent does not pass it yet", "30-eval-driven-agent-development/code/practice.py",
     "assert not now.passed and after_fix.passed"),
]


def with_evidence(frame: TaskFrame) -> TaskFrame:
    facts = [fact for fact in frame.facts if fact.evidence]
    facts += [RepositoryFact(claim, receipt(path, text)) for claim, path, text in PINS]
    return replace(frame, facts=facts, unknowns=[])         # the discoverable unknown is now answered


def ex2_assumption_to_evidence() -> None:
    draft, revised = first_draft(), with_evidence(first_draft())
    print(f"  first draft : {validate(draft)}")
    for fact in revised.facts[-3:]:
        print(f"  found       : {fact.claim}  [{fact.evidence}]")
    print(f"  revised     : {validate(revised) or 'no issues'}")
    assert validate(draft) == ["unsupported fact: No caller depends on the empty string"]
    assert validate(revised) == [] and all(fact.evidence for fact in revised.facts) and len(revised.facts) == 6
    assert "No caller depends" not in render(revised)


# ---------------------------------------------------------------------------
# Exercise 3 - a human unknown that would change the public contract
#
# "When the limit is hit, does run() raise, or return a message?" Callers of
# Runner.run handle one or the other, so the answer changes its contract, and
# the lesson's rule is that the agent pauses at a human unknown before the
# choice is buried in code.
#
# main.py's validator does not read unknowns: a frame with this question open
# still renders as READY. may_start() adds the pause, and lets the other three
# kinds through.
# ---------------------------------------------------------------------------

CONTRACT_QUESTION = ("human: when max_hops is used up, should run() raise or return a message? "
                     "Callers handle one or the other, so this sets the public contract of Runner.run")


def may_start(frame: TaskFrame) -> tuple[bool, list[str]]:
    waiting = validate(frame) + [unknown for unknown in frame.unknowns if unknown.startswith("human:")]
    return not waiting, waiting


def ex3_human_unknown() -> None:
    frame = with_evidence(first_draft())
    frame.unknowns = [CONTRACT_QUESTION, "decidable: the wording of the message, if a message is chosen",
                      "deferred: whether the default of 3 hops is the right default"]
    started, waiting = may_start(frame)
    print(f"  main.py renders the frame as: {render(frame).splitlines()[2]}")
    print(f"  may_start: {started}; waiting on: {waiting[0][:72]}...")
    answered = replace(frame, unknowns=[unknown for unknown in frame.unknowns if not unknown.startswith("human:")])
    print(f"  once a person has answered: may_start {may_start(answered)[0]}, with {len(answered.unknowns)} unknowns still listed")
    assert "Status: READY" in render(frame) and not started and waiting == [CONTRACT_QUESTION]
    assert may_start(answered) == (True, [])


# ---------------------------------------------------------------------------
# Exercise 4 - one broad allowed path, split into the smallest safe set
#
# The first draft allows 16-openai-agents-sdk/**. Two things are wrong with
# it. It contains the docs folder the frame forbids, and main.py's validator
# does not notice, because it compares the two lists as strings. And it is
# both too wide and too narrow: it lets the change into every file of lesson
# 16, and leaves out the two files in other lessons that exercise 2 showed
# have to change.
#
# The smallest safe set is the set of files a fact points at: the one that
# holds the bug and the three that pin it.
# ---------------------------------------------------------------------------

def repository_files() -> list[str]:
    return sorted(path.relative_to(REPO).as_posix() for path in REPO.glob("[0-9][0-9]-*/**/*")
                  if path.is_file() and "__pycache__" not in path.parts)


def covered(patterns: list[str], files: list[str]) -> list[str]:
    return [name for name in files if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]


def smallest_safe_set(frame: TaskFrame) -> list[str]:
    return sorted({fact.evidence.rsplit(":", 1)[0] for fact in frame.facts if fact.evidence.rsplit(":", 1)[-1].isdigit()})


def ex4_split_the_allowed_path() -> None:
    files = repository_files()
    broad = with_evidence(first_draft())
    narrow = replace(broad, allowed_paths=smallest_safe_set(broad))
    contradictions = sorted(set(covered(broad.allowed_paths, files)) & set(covered(broad.forbidden_paths, files)))
    print(f"  broad  {broad.allowed_paths}: {len(covered(broad.allowed_paths, files))} files, "
          f"{len(contradictions)} of them also forbidden; main.py's validator says {validate(broad) or 'no issues'}")
    print(f"  narrow: {narrow.allowed_paths}")
    needed = {path for _, path, _ in PINS} | {RUNNER}
    print(f"  files the change has to touch that the broad path leaves out: {sorted(needed - set(covered(broad.allowed_paths, files)))}")
    assert validate(broad) == [] and contradictions                              # the overlap main.py does not see
    assert set(narrow.allowed_paths) == needed and len(narrow.allowed_paths) == 4
    assert not set(covered(narrow.allowed_paths, files)) & set(covered(narrow.forbidden_paths, files))
    assert len(needed - set(covered(broad.allowed_paths, files))) == 2


# ---------------------------------------------------------------------------
# Exercise 5 - a scope receipt in the acceptance evidence
#
# "Tests pass" says nothing about where the change landed. The receipt is a
# second acceptance command: the changed file names go in, and it exits 1 if
# any of them is outside the allowed set or inside a forbidden one. It is
# evidence about the diff, produced by a command, like every other fact in
# the frame.
# ---------------------------------------------------------------------------

def final_frame() -> TaskFrame:
    frame = with_evidence(first_draft())
    return replace(frame, allowed_paths=smallest_safe_set(frame), acceptance=[ACCEPTANCE, SCOPE_RECEIPT],
                   unknowns=[CONTRACT_QUESTION])


def scope_receipt(frame: TaskFrame, touched: list[str]) -> tuple[bool, str]:
    forbidden = covered(frame.forbidden_paths, touched)
    outside = [name for name in touched if name not in covered(frame.allowed_paths, touched) and name not in forbidden]
    summary = f"{len(touched)} files changed, {len(outside)} outside the allowed paths {outside}, {len(forbidden)} forbidden {forbidden}"
    return not outside and not forbidden, summary


def ex5_scope_receipt() -> None:
    frame = final_frame()
    diffs = {"the four files the frame allows": frame.allowed_paths,
             "the fix plus a tidy-up of the docs": [RUNNER, "16-openai-agents-sdk/docs/en.md"],
             "the fix plus an unrelated lesson": [RUNNER, "17-claude-agent-sdk/code/main.py"]}
    codes = {}
    for label, touched in diffs.items():
        done = subprocess.run([sys.executable, __file__, "--scope-receipt"], input="\n".join(touched), capture_output=True, text=True)
        codes[label] = done.returncode
        print(f"  {label:<35} exit {done.returncode}: {done.stdout.strip()}")
    print(f"  acceptance evidence: {len(frame.acceptance)} commands; frame status {render(frame).splitlines()[2]}")
    assert list(codes.values()) == [0, 1, 1] and validate(frame) == []
    assert frame.acceptance == [ACCEPTANCE, SCOPE_RECEIPT] and "--scope-receipt" in render(frame)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--acceptance"]:
        outcome = hop_limit_outcome()
        print(f"run ended with {outcome!r}")
        sys.exit(0 if outcome.strip() else 1)
    if sys.argv[1:2] == ["--scope-receipt"]:
        ok, message = scope_receipt(final_frame(), [line.strip() for line in sys.stdin if line.strip()])
        print(message)
        sys.exit(0 if ok else 1)
    print("Phase 14 - Lesson 43: Frame the Task Before the Agent Writes Code - exercises")
    for exercise in (ex1_frame_a_real_bug, ex2_assumption_to_evidence, ex3_human_unknown, ex4_split_the_allowed_path,
                     ex5_scope_receipt):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
