"""Phase 14 - Lesson 51: Write Specifications That Preserve Judgment - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The ticket is the one lesson 49's safety check produced: four practice files called an outside
service as soon as a key was in the environment. This is the specification the fix was made
against, and its proof receipts are run on the fixed files (needs git on the PATH).
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from main import Decision, Specification, compile_contract, validate

REPO = Path(__file__).resolve().parents[2]
LATEST = "a08499a"                                        # the last commit before the fix
SWITCH = r'os\.environ\.get\("[A-Z_]+(_KEY|_ENDPOINT)"'   # a key or an endpoint read from the environment
SWITCHES = {"OPENAI_API_KEY": "practice-file-only", "LANGFUSE_PUBLIC_KEY": "pk-practice", "LANGFUSE_SECRET_KEY": "sk-practice",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://192.0.2.1:4318"}     # 192.0.2.1 is a documentation address; the guard refuses it anyway

# Runs a script with an audit hook that refuses, before any packet leaves, every lookup of or
# connection to a machine other than this one, and prints the host it refused.
# ponytail: the hook sees one process; a call made by a child process would go unseen
GUARD = r'''
import os, runpy, sys
LOCAL = {"127.0.0.1", "::1", "localhost"}
def hook(event, args):
    host = args[1][0] if event == "socket.connect" and isinstance(args[1], tuple) else args[0] if event == "socket.getaddrinfo" else None
    host = host.decode() if isinstance(host, bytes) else host
    if host is not None and host not in LOCAL:
        print(f"OUTSIDE {host}", flush=True)
        raise PermissionError(f"connection to {host} refused: this run has no --live")
sys.addaudithook(hook)
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
runpy.run_path(sys.argv[0], run_name="__main__")
'''


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


@lru_cache(maxsize=None)
def guarded(file: str, *args: str, keys: bool = True) -> tuple[int, tuple[str, ...], str]:
    """Run a script under the guard. (exit code, hosts it tried to reach, everything it printed)."""
    env = {name: value for name, value in os.environ.items() if name not in SWITCHES} | (SWITCHES if keys else {})
    done = subprocess.run([sys.executable, "-c", GUARD, file, *args], capture_output=True, text=True, timeout=120, env=env)
    return done.returncode, tuple(re.findall(r"^OUTSIDE (\S+)", done.stdout, re.M)), done.stdout


def reads_a_switch() -> list[Path]:
    """The practice files that read a key or an endpoint from the environment."""
    return [file for file in sorted(REPO.glob("[0-9][0-9]-*/code/practice.py")) if re.search(SWITCH, file.read_text(encoding="utf-8"))]


# ---------------------------------------------------------------------------
# Exercise 1 - a backlog ticket turned into the six specification surfaces
#
# The ticket says what is wrong and "make them stop". As a specification that
# is an outcome and nothing else: five surfaces are empty. The full version
# says what must always hold, shows cases, fences off the neighbouring work,
# sorts the decisions, and names the proof.
# ---------------------------------------------------------------------------

TICKET = "Lessons 01, 09, 23 and 24 call an outside service when a key is in the environment. Make them stop."
INVARIANTS = ["a run without --live connects to no machine but this one",
              "the gate removes no check: every assert in the file is still there"]
EXAMPLES = ["python practice.py with OPENAI_API_KEY set: prints how to turn the call on, connects nowhere, exits 0",
            "python practice.py --live with no key: prints the same line, connects nowhere, exits 0"]


def decisions() -> list[Decision]:
    return [
        Decision("What is the flag called?", "locked", "--live: lesson 41 already takes it, and two names for one thing is one too many"),
        Decision("May the change be committed and pushed?", "locked", "Only on the owner's yes for this batch"),
        Decision("Which files may be edited?", "bounded", "Only practice files that read a key or an endpoint from the environment; never a main.py"),
        Decision("How does a file say that the call was skipped?", "delegated", ""),
        Decision("Where in the file does the check go?", "delegated", ""),
    ]


def specification() -> Specification:
    return Specification(
        outcome="Running a practice file never spends money or sends data unless the person running it asked for that on the command line",
        invariants=list(INVARIANTS),
        examples=list(EXAMPLES),
        non_goals=["changing what the live call does once it is asked for", "replacing the outside services with fakes",
                   "a flag for files that make no outside call"],
        decisions=decisions(),
        proof=["a guarded run of every file that reads a key, with every key set and no flag", "a count of asserts before and after"],
    )


def ex1_ticket_to_six_surfaces() -> None:
    bare = compile_contract(Specification(TICKET, [], [], [], [], []))
    print(f"  the ticket alone: {bare['status']}, {len(bare['issues'])} surfaces empty")
    contract = compile_contract(specification())
    for surface in ("outcome", "invariants", "examples", "non_goals", "proof"):
        value = contract["contract"][surface]
        print(f"  {surface:<10} {value if isinstance(value, str) else '; '.join(value)}"[:170])
    print(f"  decisions  {len(contract['human_checkpoint'])} locked, {len(contract['bounded_decisions'])} bounded, {len(contract['agent_may_decide'])} delegated -> {contract['status']}")
    assert bare["status"] == "incomplete" and len(bare["issues"]) == 5
    assert contract["status"] == "executable" and not contract["issues"] and all(contract["contract"].values())


# ---------------------------------------------------------------------------
# Exercise 2 - three implementation instructions replaced by one invariant and two examples
#
# The instructions say where to put a check. A reviewer ticks them off by
# finding the check in the text. SYNTHETIC: the script below follows the
# instruction to the letter (the check is there) and still reaches out, because
# its condition is an "or". The invariant is about what a run does, so it is
# checked by running: the guard catches the script, and both examples hold on
# the real lesson 01.
# ---------------------------------------------------------------------------

INSTRUCTIONS = ["In lesson 01, put a check for --live in front of the Responses API call",
                "In lesson 09, put a check for --live in front of the mem0 client",
                "In lessons 23 and 24, put a check for --live in front of the two uploads"]
FOLLOWS_THE_INSTRUCTION = '''import os, sys, urllib.request
KEY = "OPENAI_API_KEY"
if "--live" in sys.argv or os.environ.get(KEY):          # the check is in front of the call
    urllib.request.urlopen("http://192.0.2.1/", timeout=1)
print("all exercises passed")
'''


def follows(source: str) -> bool:
    """How an instruction gets ticked off: the check is in the text."""
    return '"--live" in sys.argv' in source or '"--live" not in sys.argv' in source


def ex2_invariant_and_examples() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "practice.py"
        script.write_text(FOLLOWS_THE_INSTRUCTION, encoding="utf-8")
        code, outside, _ = guarded(str(script))
    print(f"  {len(INSTRUCTIONS)} instructions -> 1 invariant: {INVARIANTS[0]}")
    print(f"  synthetic script: instruction followed {follows(FOLLOWS_THE_INSTRUCTION)}; guarded run exit {code}, tried to reach {list(outside)}")
    first = str(next(REPO.glob("01-*/code/practice.py")))
    with_key, with_flag = guarded(first), guarded(first, "--live", keys=False)
    for example, (exit_code, reached, printed) in zip(EXAMPLES, (with_key, with_flag)):
        line = next(line.strip() for line in printed.splitlines() if "--live" in line)
        print(f"  example: {example[:62]}... -> exit {exit_code}, reached {list(reached)}, '{line}'")
    assert follows(FOLLOWS_THE_INSTRUCTION) and code != 0 and outside == ("192.0.2.1",)
    assert all(exit_code == 0 and not reached and "run with --live" in printed for exit_code, reached, printed in (with_key, with_flag))


# ---------------------------------------------------------------------------
# Exercise 3 - every decision marked, every locked or bounded one justified
#
# A first draft locks the flag's name without saying why, and the validator
# refuses it: a lock nobody can explain cannot be revised later either. The
# reason is then checked against the repository, because a rationale that
# cites a fact should survive a look at the fact.
# ---------------------------------------------------------------------------

def ex3_decisions_and_reasons() -> None:
    draft = replace(specification(), decisions=[Decision("What is the flag called?", "locked", ""), *decisions()[1:]])
    print(f"  draft: {validate(draft)}")
    contract = compile_contract(specification())
    for item in specification().decisions:
        print(f"  {item.mode:<9} {item.question:<46} {item.rationale or '(the agent decides and says why)'}")
    lesson_41 = next(REPO.glob("41-*/code/practice.py")).read_text(encoding="utf-8")
    edited = [name for name in git("diff", "--name-only", LATEST, "--", "01-*", "09-*", "23-*", "24-*").splitlines()]
    print(f"  the facts behind two reasons: lesson 41 takes --live: {'python practice.py --live' in lesson_41}; files edited for the fix: {[name[:2] for name in edited]}")
    assert validate(draft) == ["constrained decision lacks rationale: What is the flag called?"] and contract["status"] == "executable"
    assert (len(contract["human_checkpoint"]), len(contract["bounded_decisions"]), len(contract["agent_may_decide"])) == (2, 1, 2)
    assert "python practice.py --live" in lesson_41 and edited and all(name.endswith("code/practice.py") for name in edited)
    assert validate(replace(specification(), decisions=[Decision("What is the flag called?", "optional", "any name")])) == ["invalid decision mode: optional"]


# ---------------------------------------------------------------------------
# Exercise 4 - a proof receipt for every invariant
#
# A receipt is something that was run and what it showed. The first invariant
# is about connections, so its receipt is a guarded run of every file that
# reads a key, with every key set and no flag. The second is about the text,
# so counting asserts is the right layer for it. The first receipt is also run
# on lesson 24 as it was committed, where it fails: the receipt can tell the
# two versions apart.
# ---------------------------------------------------------------------------

def asserts(source: str) -> int:
    return sum(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(source)))


def receipts() -> dict[str, tuple[bool, str]]:
    files = reads_a_switch()
    runs = [guarded(str(file)) for file in files]
    counts = [(asserts(git("show", f"{LATEST}:{file.relative_to(REPO).as_posix()}")), asserts(file.read_text(encoding="utf-8"))) for file in files]
    return {
        INVARIANTS[0]: (all(code == 0 and not outside for code, outside, _ in runs),
                        f"guarded run of lessons {', '.join(file.parts[-3][:2] for file in files)} with {len(SWITCHES)} switches set: "
                        f"{sum(len(outside) for _, outside, _ in runs)} hosts reached, exit codes {[code for code, _, _ in runs]}"),
        INVARIANTS[1]: (all(now >= before for before, now in counts), f"asserts at {LATEST} and now: {counts}"),
    }


def ex4_proof_receipts() -> None:
    proven = receipts()
    for invariant, (held, receipt) in proven.items():
        print(f"  {'held  ' if held else 'BROKEN'} {invariant}\n         {receipt}")
    contract = compile_contract(replace(specification(), proof=[f"{invariant} <- {receipt}" for invariant, (_, receipt) in proven.items()]))
    with tempfile.TemporaryDirectory() as tmp:
        old = Path(tmp) / "code" / "practice.py"
        old.parent.mkdir()
        lesson = next(REPO.glob("24-*"))
        (old.parent / "main.py").write_bytes((lesson / "code" / "main.py").read_bytes())
        old.write_text(git("show", f"{LATEST}:{lesson.name}/code/practice.py"), encoding="utf-8")
        code, outside, _ = guarded(str(old))
    print(f"  the same receipt on lesson 24 as committed at {LATEST}: exit {code}, tried to reach {list(outside)}")
    assert set(proven) == set(specification().invariants) and all(held for held, _ in proven.values())
    assert [file.parts[-3][:2] for file in reads_a_switch()] == ["01", "09", "23", "24"] and contract["status"] == "executable"
    assert code != 0 and outside == ("cloud.langfuse.com",)


# ---------------------------------------------------------------------------
# Exercise 5 - a constraint with no evidence or risk behind it, removed
#
# "Parse the flag with argparse" was a lock with no reason given, and none
# can be found: no file that passes the receipts uses argparse. Nothing fails
# without it, so it protected nothing. It becomes the agent's choice. The
# lock on the flag's name stays, because that one cites a fact that holds.
# ---------------------------------------------------------------------------

def ex5_remove_an_unjustified_constraint() -> None:
    lock = Decision("How is the flag parsed?", "locked", "")
    drafted = replace(specification(), decisions=[*decisions(), lock])
    print(f"  with the lock: {compile_contract(drafted)['status']}: {validate(drafted)}")
    using = [file.parts[-3][:2] for file in reads_a_switch() if "argparse" in file.read_text(encoding="utf-8")]
    held = all(ok for ok, _ in receipts().values())
    print(f"  evidence for it: files that pass every receipt: {len(reads_a_switch())}; of those using argparse: {len(using)}")
    freed = replace(drafted, decisions=[*decisions(), replace(lock, mode="delegated")])
    contract = compile_contract(freed)
    print(f"  delegated instead: {contract['status']}; the agent may decide {len(contract['agent_may_decide'])} questions, {len(contract['human_checkpoint'])} stay locked")
    assert validate(drafted) == ["constrained decision lacks rationale: How is the flag parsed?"] and held and not using
    assert contract["status"] == "executable" and lock.question in contract["agent_may_decide"] and len(contract["human_checkpoint"]) == 2
    assert all(item.rationale for item in freed.decisions if item.mode != "delegated")


if __name__ == "__main__":
    print("Phase 14 - Lesson 51: Write Specifications That Preserve Judgment - exercises")
    for exercise in (ex1_ticket_to_six_surfaces, ex2_invariant_and_examples, ex3_decisions_and_reasons, ex4_proof_receipts,
                     ex5_remove_an_unjustified_constraint):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
