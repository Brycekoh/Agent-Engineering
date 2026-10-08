"""Phase 14 - Lesson 26: Failure Modes - Why Agents Break - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py            (every exercise asserts its own result)
           python practice.py --ci MODE  (exit 1 if MODE tags 5% or more of the traces)
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from collections import Counter
from typing import Any, Callable

from main import Trace, TraceStep, detect_cascading_errors, detect_success_hallucination, tag


def call(name: str, status: str = "ok", result: str = "", **args: Any) -> TraceStep:
    return TraceStep("tool_call", name, args, status, result)


# ---------------------------------------------------------------------------
# Exercise 1 - a success-hallucination detector that probes the state
#
# main.detect_success_hallucination trusts two things it is handed: a
# target_state_changed flag, and a list of write-like words in the request.
# So "deploy the service" or "send the invoice" slip past it. The detector
# here takes the state after the run and a check for what the request should
# have changed, and it also treats "success" claimed straight after a failed
# tool call as a hallucination whatever the state looks like.
# ---------------------------------------------------------------------------

def probe_success_hallucination(trace: Trace, world_after: dict[str, Any],
                                achieved: Callable[[dict[str, Any]], bool]) -> str | None:
    if not trace.final_success_claim:
        return None
    tool_calls = [step for step in trace.steps if step.kind == "tool_call"]
    if tool_calls and tool_calls[-1].status == "error":
        return "success_hallucination"
    return None if achieved(world_after) else "success_hallucination"


def ex1_success_hallucination() -> None:
    cases = [   # (trace, state after the run, what success would look like)
        (Trace("deploy", "deploy the service to staging", [], [call("search", query="deploy steps")], True, False),
         {"staging_version": "v1"}, lambda world: world["staging_version"] == "v2"),
        (Trace("invoice", "send the invoice to finance", [], [call("search", status="error", query="finance address")], True, False),
         {"sent": []}, lambda world: bool(world["sent"])),
        (Trace("pr", "create a PR", [], [call("search", query="PR template")], True, False),
         {"open_prs": 0}, lambda world: world["open_prs"] == 1),
        (Trace("readme", "update the readme", [], [call("write_file", path="README.md", content="notes")], True, True),
         {"README.md": "notes"}, lambda world: world["README.md"] == "notes"),
        (Trace("find", "find the config file", [], [call("search", query="config")], True, False),
         {}, lambda world: True),
    ]
    flagged = {"main": [], "probe": []}
    for trace, world, achieved in cases:
        by_main = detect_success_hallucination(trace) is not None
        by_probe = probe_success_hallucination(trace, world, achieved) is not None
        flagged["main"] += [trace.tid] * by_main
        flagged["probe"] += [trace.tid] * by_probe
        print(f"  {trace.user_request:<30} main: {'flag' if by_main else '-':<5} probe: {'flag' if by_probe else '-'}")
    assert flagged["main"] == ["pr"]
    assert flagged["probe"] == ["deploy", "invoice", "pr"]


# ---------------------------------------------------------------------------
# Exercise 2 - tag 100 traces: which mode dominates, and what fixing it costs
#
# SYNTHETIC TRACES: there are no product traces here, so 100 are generated
# with a known mix and tagged with main.tag. That checks the tagger recovers
# the mix; running tag() over a product's real traces is the same two lines.
# The cost column is a judgement about where each fix lives.
# ---------------------------------------------------------------------------

MAKERS: dict[str, Callable[[str], Trace]] = {
    "clean": lambda tid: Trace(tid, "find the config file", [], [
        call("search", query="config"), call("read_file", path="config.yml")], True, False),
    "tool_misuse": lambda tid: Trace(tid, "read the log", [], [call("read_file", file="app.log")], False, False),
    "scope_creep": lambda tid: Trace(tid, "find the config file", [], [
        call("search", query="config"), call("write_file", path="config.yml", content="tidied")], True, True),
    "cascading_errors": lambda tid: Trace(tid, "look up invoice 4711", [], [
        call("search", status="error", query="invoice 4711"), call("read_file", path="/tmp/guess"),
        call("list_dir", path="/tmp")], False, False),
    "hallucinated_action": lambda tid: Trace(tid, "list project files", [], [call("magic_scanner", path="/")], False, False),
    "context_loss": lambda tid: Trace(tid, "update the release notes", ["do not deploy on fridays"], [
        call("write_file", path="notes.md", content="deploy friday build")], True, True),
    "success_hallucination": lambda tid: Trace(tid, "create a PR", [], [call("search", query="PR template")], True, False),
}
MIX = {"clean": 55, "tool_misuse": 15, "scope_creep": 10, "cascading_errors": 8, "hallucinated_action": 6,
       "context_loss": 3, "success_hallucination": 3}
FIX_COST = {
    "tool_misuse": ("validate arguments against the tool schema and return the error to the model", "small"),
    "scope_creep": ("allowlist of tools per request type, confirmation before writes", "medium"),
    "cascading_errors": ("stop or replan on the first tool error instead of continuing", "medium"),
    "hallucinated_action": ("reject unknown tool names at dispatch", "small"),
    "context_loss": ("re-inject constraints every turn and check them before each write", "medium"),
    "success_hallucination": ("probe the target state before reporting success", "medium"),
}


def hundred_traces(seed: int = 0) -> list[Trace]:
    kinds = [kind for kind, count in MIX.items() for _ in range(count)]
    random.Random(seed).shuffle(kinds)
    return [MAKERS[kind](f"t{i:03d}") for i, kind in enumerate(kinds)]


def ex2_tag_hundred_traces() -> None:
    distribution: Counter[str] = Counter()
    for trace in hundred_traces():
        distribution.update(tag(trace))
    for mode, count in distribution.most_common():
        fix, effort = FIX_COST[mode]
        print(f"  {mode:<22} {count:>3}%   fix: {fix} ({effort})")
    dominant = distribution.most_common(1)[0][0]
    print(f"  dominant mode: {dominant}; clean traces: {100 - sum(distribution.values())}")
    assert dict(distribution) == {mode: count for mode, count in MIX.items() if mode != "clean"}
    assert dominant == "tool_misuse"


# ---------------------------------------------------------------------------
# Exercise 3 - cascade radius
#
# Given a failure at step N, how many later steps did it affect? Counting
# every later step overstates it. The radius here follows the data: a step is
# affected if its arguments use something the failed step returned, or
# something an affected step returned. Unrelated steps after the failure do
# not count.
# ---------------------------------------------------------------------------

def identifiers(text: str) -> set[str]:
    return {token for token in re.findall(r"[\w./-]+", text) if len(token) >= 5 and re.search(r"[\d/_-]", token)}


def cascade_radius(trace: Trace, failed_index: int) -> int:
    tainted = identifiers(trace.steps[failed_index].result)
    affected = 0
    for step in trace.steps[failed_index + 1:]:
        if any(token in json.dumps(step.args) for token in tainted):
            affected += 1
            tainted |= identifiers(step.result)
    return affected


def ex3_cascade_radius() -> None:
    trace = Trace("phantom", "order the blue widget", [], [
        call("search", status="error", result="no match; closest sku-PHANTOM-9", query="blue widget"),
        call("read_file", result="price-0.00-PHANTOM", path="catalog/sku-PHANTOM-9.json"),
        call("write_file", result="wrote orders/draft-17.json", path="orders/draft-17.json",
             content="sku-PHANTOM-9 at price-0.00-PHANTOM"),
        call("list_dir", result="3 files", path="/tmp"),
        call("read_file", result="draft order", path="orders/draft-17.json"),
    ], True, True)
    radius = cascade_radius(trace, failed_index=0)
    every_later_step = len(trace.steps) - 1
    print(f"  failure at step 0; steps after it: {every_later_step}; steps it actually reached: {radius}")
    print(f"  main tags the trace as: {detect_cascading_errors(trace)}")
    assert radius == 3 and every_later_step == 4               # list_dir was never touched by the bad sku
    assert cascade_radius(trace, failed_index=3) == 0           # a failure there would have reached nothing


# ---------------------------------------------------------------------------
# Exercise 4 - three of MASFT's fourteen failure modes, as detectors
#
# The taxonomy has three categories: specification issues (FM-1.x),
# inter-agent misalignment (FM-2.x) and task verification (FM-3.x). The three
# picked here are the ones a single tool-using agent in DevOps work runs
# into, which is an assumption about the product:
#   FM-1.1  fail to follow task specification
#   FM-1.3  step repetition
#   FM-3.2  no or incomplete verification
# The first also covers a gap in main: detect_context_loss looks for the word
# after "do not" inside the arguments, so "do not modify src/" followed by a
# write to src/ goes untagged.
# ---------------------------------------------------------------------------

def detect_task_spec_violation(trace: Trace) -> str | None:
    for constraint in trace.constraints:
        found = re.search(r"do not (?:modify|touch|edit|write to) (\S+)", constraint.lower())
        if not found:
            continue
        target = found[1].rstrip("/.")
        for step in trace.steps:
            path = str(step.args.get("path", "")).lower()
            if step.name == "write_file" and (target in ("any", "anything") or path.startswith(target)):
                return "FM-1.1 fail to follow task specification"
    return None


def detect_step_repetition(trace: Trace, limit: int = 3) -> str | None:
    counts = Counter((step.name, json.dumps(step.args, sort_keys=True)) for step in trace.steps)
    return "FM-1.3 step repetition" if counts and max(counts.values()) >= limit else None


def detect_no_verification(trace: Trace) -> str | None:
    writes = [i for i, step in enumerate(trace.steps) if step.name == "write_file" and step.status == "ok"]
    if not trace.final_success_claim or not writes:
        return None
    checked = any(step.name in ("read_file", "list_dir") for step in trace.steps[writes[-1] + 1:])
    return None if checked else "FM-3.2 no or incomplete verification"


MASFT_DETECTORS = (detect_task_spec_violation, detect_step_repetition, detect_no_verification)


def ex4_masft_detectors() -> None:
    traces = [
        Trace("touches-src", "update readme with release notes", ["do not modify src/"], [
            call("write_file", path="README.md", content="notes"), call("read_file", path="README.md"),
            call("write_file", path="src/foo.py", content="also notes"), call("read_file", path="src/foo.py")], True, True),
        Trace("retry-loop", "find the failing test", [], [call("search", status="error", query="FAILED")] * 4, False, False),
        Trace("unverified", "save the report", [], [call("write_file", path="report.md", content="done")], True, True),
        Trace("careful", "save the report", [], [
            call("write_file", path="report.md", content="done"), call("read_file", path="report.md")], True, True),
    ]
    found = {}
    for trace in traces:
        found[trace.tid] = [label for label in (detect(trace) for detect in MASFT_DETECTORS) if label]
        print(f"  {trace.tid:<12} MASFT: {found[trace.tid] or '[clean]'}   main: {tag(trace) or '[clean]'}")
    assert found["touches-src"] == ["FM-1.1 fail to follow task specification"]
    assert "context_loss" not in tag(traces[0])                 # the gap in main's detector
    assert found["retry-loop"] == ["FM-1.3 step repetition"]
    assert found["unverified"] == ["FM-3.2 no or incomplete verification"] and found["careful"] == []


# ---------------------------------------------------------------------------
# Exercise 5 - one detector wired into CI
#
# ci_gate returns the exit code. `python practice.py --ci MODE` uses it, so
# the job is one line. The workflow below is the wiring, kept here as text:
# it is not installed in .github/workflows.
# ---------------------------------------------------------------------------

CI_WORKFLOW = """\
name: agent-failure-gate
on: [pull_request]
jobs:
  failure-modes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: python 26-failure-modes-agentic/code/practice.py --ci tool_misuse
"""


def ci_gate(traces: list[Trace], mode: str, threshold: float = 0.05) -> tuple[int, str]:
    tagged = sum(mode in tag(trace) for trace in traces)
    rate = tagged / len(traces)
    verdict = "FAIL" if rate >= threshold else "ok"
    return int(rate >= threshold), f"{verdict}: {mode} tags {tagged}/{len(traces)} traces ({rate:.0%}, limit {threshold:.0%})"


def ex5_ci_gate() -> None:
    traces = hundred_traces()
    for mode in ("tool_misuse", "success_hallucination"):
        print(f"  {ci_gate(traces, mode)[1]}")
    exactly_five = [MAKERS["hallucinated_action"](f"h{i}") for i in range(5)] + [MAKERS["clean"](f"c{i}") for i in range(95)]
    assert ci_gate(traces, "tool_misuse")[0] == 1 and ci_gate(traces, "success_hallucination")[0] == 0
    assert ci_gate(exactly_five, "hallucinated_action")[0] == 1         # 5% is already a failure

    as_ci_runs_it = subprocess.run([sys.executable, __file__, "--ci", "tool_misuse"], capture_output=True, text=True)
    print(f"  as a CI step: exit code {as_ci_runs_it.returncode}, output {as_ci_runs_it.stdout.strip()!r}")
    assert as_ci_runs_it.returncode == 1
    assert "practice.py --ci tool_misuse" in CI_WORKFLOW


if __name__ == "__main__":
    if sys.argv[1:2] == ["--ci"]:
        code, message = ci_gate(hundred_traces(), sys.argv[2])
        print(message)
        sys.exit(code)
    print("Phase 14 - Lesson 26: Failure Modes - exercises")
    for exercise in (ex1_success_hallucination, ex2_tag_hundred_traces, ex3_cascade_radius, ex4_masft_detectors,
                     ex5_ci_gate):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
