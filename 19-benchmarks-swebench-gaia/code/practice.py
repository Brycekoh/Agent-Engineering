"""Phase 14 - Lesson 19: Benchmarks - SWE-bench, GAIA, AgentBench - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import re
import sys
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from main import TaskResult, gaia_level

REPO = Path(__file__).resolve().parents[2]
Test = Callable[[Any], None]                    # gets the loaded module, raises if the behaviour is wrong
Step = tuple[str, tuple[str, ...]]              # ("read", (path,)), ("edit", (old, new)), ("run_tests", ())


# ---------------------------------------------------------------------------
# Exercise 1 - the harness on a real repo, with three FAIL_TO_PASS tests
#
# The real repo is this one. The three bugs are real: I ran into each of them
# in the lessons' own main.py files while doing the earlier exercises.
#   t1  lesson 06: validate() lets "nan" through a minimum/maximum check
#   t2  lesson 17: SessionStore.delete() orphans grandchild sessions
#   t3  lesson 16: Runner returns "" when it runs out of hops
# A task's patch is applied to a copy of the source held in memory. The files
# in the repo are never modified.
#
# ponytail: tests run in this process against the patched copy. SWE-bench
# gives every task its own container; do that once tasks can touch the disk.
# ---------------------------------------------------------------------------

@dataclass
class RepoTask:
    tid: str
    issue: str
    file: str
    trajectory: list[Step]                      # what the (scripted) agent did; its edits are the patch
    fail_to_pass: dict[str, Test]
    pass_to_pass: dict[str, Test]


def load_source(source: str, name: str) -> Any:
    module = types.ModuleType(name)
    sys.modules[name] = module                  # dataclasses look their module up by name
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def passing(module: Any, tests: dict[str, Test]) -> int:
    count = 0
    for test in tests.values():
        try:
            test(module)
            count += 1
        except Exception:
            pass
    return count


def run_repo_task(task: RepoTask) -> tuple[TaskResult, int]:
    """Returns the SWE-bench verdict and how many FAIL_TO_PASS tests passed before the patch."""
    before = (REPO / task.file).read_text(encoding="utf-8")
    after = before
    for kind, args in task.trajectory:
        if kind == "edit":
            old, new = args
            if old not in after:
                raise ValueError(f"{task.tid}: patch does not apply")
            after = after.replace(old, new, 1)
    pre, post = load_source(before, f"swe_{task.tid}_pre"), load_source(after, f"swe_{task.tid}_post")
    ftp_pre, ptp_pre = passing(pre, task.fail_to_pass), passing(pre, task.pass_to_pass)
    ftp_post, ptp_post = passing(post, task.fail_to_pass), passing(post, task.pass_to_pass)
    resolved = ftp_post == len(task.fail_to_pass) and ptp_post >= ptp_pre
    return TaskResult(task.tid, ftp_post, len(task.fail_to_pass), ptp_post, len(task.pass_to_pass), resolved), ftp_pre


RANGE = {"type": "object", "properties": {"p": {"type": "number", "minimum": 0, "maximum": 1}}, "required": ["p"]}


def t1_nan_is_rejected(m: Any) -> None:
    assert m.validate({"p": "nan"}, RANGE)[1], "nan passed the range check"


def t1_in_range_accepted(m: Any) -> None:
    assert m.validate({"p": 0.5}, RANGE) == ({"p": 0.5}, [])


def t1_out_of_range_rejected(m: Any) -> None:
    assert m.validate({"p": 2}, RANGE)[1]


def t1_numeric_string_coerced(m: Any) -> None:
    assert m.validate({"p": "0.5"}, RANGE) == ({"p": 0.5}, [])


def nested_store(m: Any) -> Any:
    store = m.SessionStore()
    for session in ("root", "root.a", "root.a.x", "other"):
        store.append(session, m.Turn("user", "hi"))
    store.link_sub("root", "root.a")
    store.link_sub("root.a", "root.a.x")
    return store


def t2_grandchildren_deleted(m: Any) -> None:
    store = nested_store(m)
    store.delete("root")
    assert store.list_sessions() == ["other"], store.list_sessions()


def t2_children_deleted(m: Any) -> None:
    store = nested_store(m)
    store.delete("root")
    assert "root" not in store.list_sessions() and "root.a" not in store.list_sessions()


def t2_unrelated_session_kept(m: Any) -> None:
    store = nested_store(m)
    store.delete("root")
    assert "other" in store.list_sessions()


def ping_pong(m: Any) -> Any:
    billing = m.Agent("billing", "x", policy=lambda text: {"kind": "handoff", "to": "triage", "input": text})
    triage = m.Agent("triage", "x", policy=lambda text: {"kind": "handoff", "to": "billing", "input": text},
                     handoffs=[m.Handoff(target=billing)])
    billing.handoffs.append(m.Handoff(target=triage))
    return triage


def t3_hop_limit_reported(m: Any) -> None:
    assert "max hops" in m.Runner(max_hops=4).run(ping_pong(m), "refund please")


def t3_normal_handoff_works(m: Any) -> None:
    billing = m.Agent("billing", "x", policy=m._billing_policy)
    triage = m.Agent("triage", "x", policy=m._triage_policy, handoffs=[m.Handoff(target=billing)])
    assert m.Runner().run(triage, "refund for invoice 4711").startswith("billing handled")


def t3_input_guardrail_trips(m: Any) -> None:
    runner = m.Runner(input_guardrails=[m.InputGuardrail("pii", m._pii_check)])
    try:
        runner.run(m.Agent("a", "x", policy=m._billing_policy), "share my ssn")
    except m.GuardrailTripped:
        return
    raise AssertionError("guardrail did not trip")


TASKS = [
    RepoTask(
        tid="t1",
        issue="validate() accepts the string 'nan' for a number field declared with minimum 0 and maximum 1.",
        file="06-tool-use-and-function-calling/code/main.py",
        trajectory=[
            ("read", ("06-tool-use-and-function-calling/code/main.py",)),
            ("search", ("minimum",)),
            ("edit", ('and coerced < prop["minimum"]:', 'and not coerced >= prop["minimum"]:')),
            ("edit", ('and coerced > prop["maximum"]:', 'and not coerced <= prop["maximum"]:')),
            ("run_tests", ()),
        ],
        fail_to_pass={"nan is rejected": t1_nan_is_rejected},
        pass_to_pass={"in range accepted": t1_in_range_accepted, "out of range rejected": t1_out_of_range_rejected,
                      "numeric string coerced": t1_numeric_string_coerced},
    ),
    RepoTask(
        tid="t2",
        issue=("SessionStore.delete('root') leaves grandchild sessions behind. delete() should recurse: "
               "`for sub in self._subkeys.pop(session_id, []):` then `self.delete(sub)`."),
        file="17-claude-agent-sdk/code/main.py",
        trajectory=[
            ("read", ("17-claude-agent-sdk/code/main.py",)),
            ("edit", ("        for sub in self._subkeys.get(session_id, []):\n"
                      "            self._sessions.pop(sub, None)\n"
                      "        self._subkeys.pop(session_id, None)\n",
                      "        for sub in self._subkeys.pop(session_id, []):\n"
                      "            self.delete(sub)\n")),
            ("run_tests", ()),
        ],
        fail_to_pass={"grandchildren deleted": t2_grandchildren_deleted},
        pass_to_pass={"children deleted": t2_children_deleted, "unrelated session kept": t2_unrelated_session_kept},
    ),
    RepoTask(
        tid="t3",
        issue="Runner.run returns an empty string when two agents keep handing off until max_hops is used up.",
        file="16-openai-agents-sdk/code/main.py",
        trajectory=[
            ("read", ("16-openai-agents-sdk/code/main.py",)),
            ("search", ("max_hops",)),
            ("edit", ('            final_output = f"error: unknown policy kind {kind}"\n            break\n',
                      '            final_output = f"error: unknown policy kind {kind}"\n            break\n'
                      '        else:\n            final_output = f"error: max hops ({self.max_hops}) exceeded"\n')),
            ("run_tests", ()),
        ],
        fail_to_pass={"hop limit reported": t3_hop_limit_reported},
        pass_to_pass={"normal handoff works": t3_normal_handoff_works, "input guardrail trips": t3_input_guardrail_trips},
    ),
]

# A second attempt at t1 that overreaches: it "fixes" nan by refusing every string.
OVERREACH = replace(TASKS[0], tid="t1-overreach", trajectory=[
    ("read", ("06-tool-use-and-function-calling/code/main.py",)),
    ("edit", ('                return float(value), None\n            except ValueError:\n'
              '                return value, f"cannot coerce string {value!r} to number"',
              '                raise ValueError\n            except ValueError:\n'
              '                return value, f"cannot coerce string {value!r} to number"')),
    ("run_tests", ()),
])


def ex1_real_repo_tasks() -> None:
    for task in TASKS + [OVERREACH]:
        result, failing_before = run_repo_task(task)
        print(f"  {result.tid:<12} FAIL_TO_PASS {result.ftp_passed}/{result.ftp_total} "
              f"(passing before the patch: {failing_before})   PASS_TO_PASS {result.ptp_passed}/{result.ptp_total}"
              f"   resolved: {result.resolved}")
        assert failing_before == 0                      # each FAIL_TO_PASS test does fail on the repo as it is
        assert result.resolved == (task is not OVERREACH)
    assert run_repo_task(OVERREACH)[0].ftp_passed == 1  # it does fix the bug; it breaks something else


# ---------------------------------------------------------------------------
# Exercise 2 - a step-count metric
#
# SCRIPTED TRAJECTORIES: the steps are the ones I wrote into each task, not a
# model's. The metric and the two ways of reporting it are the point: steps
# per resolved task flatters the agent, because it leaves out what was spent
# on attempts that did not resolve anything.
# ---------------------------------------------------------------------------

def ex2_step_count() -> None:
    attempts = [(task, run_repo_task(task)[0].resolved) for task in TASKS + [OVERREACH]]
    for task, resolved in attempts:
        kinds = [kind for kind, _ in task.trajectory]
        print(f"  {task.tid:<12} {len(kinds)} steps {kinds}  {'resolved' if resolved else 'not resolved'}")
    resolved_steps = [len(task.trajectory) for task, resolved in attempts if resolved]
    all_steps = sum(len(task.trajectory) for task, _ in attempts)
    print(f"  steps per resolved task        : {sum(resolved_steps) / len(resolved_steps):.1f}")
    print(f"  steps per resolution, all-in   : {all_steps / len(resolved_steps):.1f}  (failed attempt charged)")
    assert resolved_steps == [5, 3, 4] and all_steps == 15


# ---------------------------------------------------------------------------
# Exercise 3 - SWE-bench+: a solution-leakage check
#
# The paper looked at SWE-Agent + GPT-4's successful SWE-bench patches and
# found the solution was given in the issue or its comments for 32.67% of
# them, and another 31.08% passed only because the tests were weak. With both
# removed the resolution rate fell from 12.47% to 3.97%.
#
# The check below is the pattern match the exercise asks for: does a line the
# fix adds already appear in the issue text?
#
# ponytail: it catches verbatim leaks only. "delete should recurse into the
# children" gives the fix away just as well and matches nothing here.
#
# The second category is shown with the overreaching patch from exercise 1:
# take one PASS_TO_PASS test away and the same wrong patch counts as resolved.
# ---------------------------------------------------------------------------

def added_lines(trajectory: list[Step]) -> list[str]:
    lines = []
    for kind, args in trajectory:
        if kind == "edit":
            old, new = args
            lines += [line.strip() for line in new.splitlines() if line.strip() and line.strip() not in old]
    return lines


def leaked_lines(issue: str, trajectory: list[Step], min_length: int = 12) -> list[str]:
    squash = lambda text: re.sub(r"\s+", " ", text).lower()
    issue_text = squash(issue)
    return [line for line in added_lines(trajectory) if len(line) >= min_length and squash(line) in issue_text]


def ex3_solution_leakage() -> None:
    leaks = {task.tid: leaked_lines(task.issue, task.trajectory) for task in TASKS}
    for tid, lines in leaks.items():
        print(f"  {tid}: {'LEAKED ' + str(lines) if lines else 'clean'}")
    clean_rate = sum(not lines for lines in leaks.values())
    print(f"  resolved 3/3 as reported, {clean_rate}/3 once leaked tasks are set aside")
    assert leaks["t2"] == ["for sub in self._subkeys.pop(session_id, []):", "self.delete(sub)"]
    assert leaks["t1"] == [] and leaks["t3"] == []

    weak_suite = {name: test for name, test in OVERREACH.pass_to_pass.items() if name != "numeric string coerced"}
    weak = run_repo_task(replace(OVERREACH, pass_to_pass=weak_suite))[0]
    print(f"  overreaching patch against a weaker test suite: resolved={weak.resolved}")
    assert weak.resolved                                # wrong patch, passing verdict


# ---------------------------------------------------------------------------
# Exercise 4 - trace a GAIA question
#
# The GAIA dataset is gated, so this uses the Level 1 sample printed in the
# GAIA paper (its first figure), paraphrased. The trace is what a GPT-4-class
# agent has to do, written out step by step.
# ---------------------------------------------------------------------------

GAIA_QUESTION = ("Find the enrollment count of the clinical trial on H. pylori in acne vulgaris patients "
                 "that ran from January to May 2018, as listed on the NIH website.")
GAIA_ANSWER = "90"
GAIA_TRACE = [
    ("web_search", "H. pylori acne vulgaris clinical trial NIH", "a ClinicalTrials.gov record among the results"),
    ("open_page", "that trial's record", "the study page"),
    ("read_page", "study start and completion dates", "confirms January to May 2018, so it is the right trial"),
    ("read_page", "the Enrollment field", "90, marked as actual rather than estimated"),
    ("answer", GAIA_ANSWER, "scored by quasi exact match, so only the number"),
]


def ex4_gaia_trace() -> None:
    for i, (tool, target, outcome) in enumerate(GAIA_TRACE, 1):
        print(f"  {i}. {tool:<10} {target} -> {outcome}")
    tools = sorted({tool for tool, _, _ in GAIA_TRACE} - {"answer"})
    print(f"  tools needed: {tools} (one capability, web browsing; no files, code or images)")
    print("  where it goes wrong: the wrong trial, the estimated count, or an answer with words around the number")
    print(f"  main.gaia_level's keyword guess: level {gaia_level(GAIA_QUESTION)}; the paper labels it level 1")
    assert len(GAIA_TRACE) == 5 and GAIA_TRACE[-1][1] == "90"
    assert gaia_level(GAIA_QUESTION) in (1, 2, 3)


# ---------------------------------------------------------------------------
# Exercise 5 - AgentBench's environments
#
# ASSUMPTION: "your product surface" is taken to be DevOps tooling, an agent
# that works through a shell. Swap the key below if yours is different.
# Figures are from the AgentBench paper (2023), not from a current
# leaderboard, so "SOTA" here means at publication.
# ---------------------------------------------------------------------------

AGENTBENCH = {
    "OS": "bash tasks in a Docker Ubuntu container",
    "DB": "SQL generation and database manipulation",
    "KG": "long multi-step queries over a knowledge graph",
    "DCG": "a turn-based digital card game",
    "LTP": "lateral thinking puzzles solved by questioning",
    "HH": "household tasks (ALFWorld)",
    "WS": "web shopping (WebShop)",
    "WB": "web browsing (Mind2Web)",
}
MY_SURFACE = "OS"


def ex5_agentbench_environment() -> None:
    for code, what in AGENTBENCH.items():
        print(f"  {code:<3} {what}{'   <- closest to DevOps tooling' if code == MY_SURFACE else ''}")
    print("  OS at publication: 144 test tasks, scored by success rate; the best model, GPT-4, solved 42.4%.")
    print("  Overall: GPT-4 scored 4.01; open-source models averaged 0.51 against 2.15 for API models.")
    print("  How runs ended matters more than the average: where 'task limit exceeded' dominates, the agent")
    print("  was looping, not failing fast. For a shell agent, track that rate next to the success rate.")
    assert len(AGENTBENCH) == 8 and MY_SURFACE in AGENTBENCH


if __name__ == "__main__":
    print("Phase 14 - Lesson 19: SWE-bench, GAIA, AgentBench - exercises")
    for exercise in (ex1_real_repo_tasks, ex2_step_count, ex3_solution_leakage, ex4_gaia_trace,
                     ex5_agentbench_environment):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
