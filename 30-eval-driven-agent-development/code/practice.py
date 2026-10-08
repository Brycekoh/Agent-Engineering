"""Phase 14 - Lesson 30: Eval-Driven Agent Development - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py              (every exercise asserts its own result)
           python practice.py --ci         (run the suite, exit 1 on a regression of 5% or more)
           python practice.py --run-suite  (execute every lesson's practice file as an eval case)
"""

from __future__ import annotations

import ast
import importlib.util
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from main import (CaseResult, EvalCase, _benchmark_case, _custom_llm_judge_case, _flaky_benchmark_case,
                  _online_guardrail_case, ci_gate, evaluator_optimizer)

REPO = Path(__file__).resolve().parents[2]


def load_lesson(folder: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / folder / "code" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Exercise 1 - a failure turned into an eval case
#
# The failure is a real one from this repo: lesson 16's Runner returns an
# empty string when two agents hand a request back and forth until the hop
# limit. The case reproduces it and judges the answer.
#
# Does the agent pass it now? No. main.py in lesson 16 still has the bug, so
# the case fails, which is what a regression case should do until the fix
# lands. With a fix in place (a shim here, not a change to that file) the
# same case passes, and from then on it guards the behaviour.
# ---------------------------------------------------------------------------

def handoff_loop_answer(fixed: bool) -> str:
    sdk = load_lesson("16-openai-agents-sdk", "lesson16_main")
    billing = sdk.Agent("billing", "refunds", policy=lambda text: {"kind": "handoff", "to": "triage", "input": text})
    triage = sdk.Agent("triage", "routing", policy=lambda text: {"kind": "handoff", "to": "billing", "input": text},
                       handoffs=[sdk.Handoff(target=billing)])
    billing.handoffs.append(sdk.Handoff(target=triage))
    answer = sdk.Runner(max_hops=4).run(triage, "refund please")
    if fixed and not answer:
        answer = "handoff limit reached after 4 hops; escalating to a human"
    return answer


def handoff_loop_case(fixed: bool) -> EvalCase:
    def judge(candidate: str) -> tuple[bool, str]:
        if "limit" in candidate and candidate.strip():
            return True, "ends with an explicit refusal"
        return False, f"agent answered {candidate!r} instead of refusing"

    return EvalCase(cid="regress_handoff_loop", category="custom",
                    description="agents that keep handing off must end with an explicit refusal",
                    proposer=lambda feedback: handoff_loop_answer(fixed), judge=judge, max_rounds=1)


def ex1_failure_as_eval_case() -> None:
    now, after_fix = evaluator_optimizer(handoff_loop_case(fixed=False)), evaluator_optimizer(handoff_loop_case(fixed=True))
    print(f"  agent as it is : {'PASS' if now.passed else 'FAIL'} ({now.reason})")
    print(f"  with the fix   : {'PASS' if after_fix.passed else 'FAIL'} ({after_fix.reason})")
    assert not now.passed and after_fix.passed


# ---------------------------------------------------------------------------
# Exercise 2 - a three-dimension judge rubric, scored on 50 sessions
#
# SCRIPTED JUDGE over SYNTHETIC SESSIONS: each line of the rubric is a rule,
# standing in for the prompt an LLM judge would get, and the sessions are
# generated with known defects. Each dimension scores 0 to 2. A session
# passes with at least 5 of 6 and no zero. The factual line is grounded on
# the session's own tool results, which is the part a judge must not do from
# memory.
# ---------------------------------------------------------------------------

RUBRIC = {
    "factual": {2: "every figure appears in a tool result", 1: "no figures to check", 0: "a figure with no source"},
    "tone": {2: "acknowledges the problem, no blame", 1: "neutral", 0: "blames or dismisses the customer"},
    "scope": {2: "stays within support work", 1: "raises an out-of-scope topic and declines it", 0: "acts out of scope"},
}


def judge_session(session: dict[str, Any]) -> dict[str, int]:
    answer = session["answer"].lower()
    grounded = set(re.findall(r"\d+(?:\.\d+)?", " ".join(session["tool_results"])))
    figures = re.findall(r"\d+(?:\.\d+)?", answer)
    factual = 1 if not figures else 2 if all(f in grounded for f in figures) else 0
    tone = 0 if any(p in answer for p in ("you should have", "as i already said")) else 2 if "sorry" in answer else 1
    scope = 0 if "delete_account" in session["tools"] else 1 if "legal advice" in answer else 2
    return {"factual": factual, "tone": tone, "scope": scope}


def fifty_sessions(seed: int = 3, broken_share: float = 0.0) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sessions = []
    for i in range(50):
        amount = f"{rng.randint(10, 300)}.00"
        session = {"id": f"s{i:02d}", "tools": ["lookup_invoice", "issue_refund"],
                   "tool_results": [f"invoice {5000 + i} charged {amount} twice"],
                   "answer": f"Sorry about that. I refunded {amount} on invoice {5000 + i}.",
                   "steps": rng.choice((3, 3, 3, 4, 4, 5, 7)), "gold_steps": 3}
        defect = rng.choice(["none"] * 16 + ["invented figure", "blame", "curt", "declines legal", "wrong tool"])
        if rng.random() < broken_share:
            defect = "invented figure"
        if defect == "invented figure":
            session["answer"] = session["answer"].replace(amount, "999.00")
        elif defect == "blame":
            session["answer"] = "You should have checked the invoice. " + session["answer"]
        elif defect == "curt":
            session["answer"] = session["answer"].replace("Sorry about that. ", "")
        elif defect == "declines legal":
            session["answer"] += " I cannot give legal advice on chargebacks."
        elif defect == "wrong tool":
            session["tools"] = session["tools"] + ["delete_account"]
        sessions.append(session)
    return sessions


def session_case(session: dict[str, Any]) -> EvalCase:
    def judge(candidate: str) -> tuple[bool, str]:
        scores = judge_session(session)
        return sum(scores.values()) >= 5 and min(scores.values()) > 0, str(scores)

    return EvalCase(cid=session["id"], category="custom", description="support answer meets the rubric",
                    proposer=lambda feedback: session["answer"], judge=judge, max_rounds=1)


def ex2_rubric_on_fifty_sessions() -> None:
    sessions = fifty_sessions()
    scores = [judge_session(session) for session in sessions]
    results = [evaluator_optimizer(session_case(session)) for session in sessions]
    for dimension, levels in RUBRIC.items():
        mean = sum(score[dimension] for score in scores) / len(scores)
        zeros = sum(score[dimension] == 0 for score in scores)
        print(f"  {dimension:<8} mean {mean:.2f}/2, scored zero in {zeros} sessions   (2 = {levels[2]})")
    passed = sum(result.passed for result in results)
    print(f"  {passed}/50 sessions pass")
    assert len(results) == 50 and 35 <= passed < 50
    assert all(set(score) == set(RUBRIC) and all(v in (0, 1, 2) for v in score.values()) for score in scores)


# ---------------------------------------------------------------------------
# Exercise 3 - the suite in CI, failing the build on a 5% regression
#
# main.ci_gate blocks when the regression is greater than the threshold. The
# exercise wants 5% itself to fail, and that needs care with floats: 0.95
# minus 0.90 is 0.04999999999999993, so a plain >= would let exactly 5%
# through. The gate below compares with a tolerance.
#
# `python practice.py --ci` runs the suite and exits with the gate's code.
# The workflow is the wiring, kept here as text; it is not installed in
# .github/workflows.
# ---------------------------------------------------------------------------

CI_WORKFLOW = """\
name: agent-evals
on: [pull_request]
jobs:
  evals:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: python 30-eval-driven-agent-development/code/practice.py --ci
"""


def gate(results: list[CaseResult], baseline: float, threshold: float = 0.05) -> tuple[int, str]:
    pass_rate = sum(result.passed for result in results) / len(results)
    regression = baseline - pass_rate
    failed = regression >= threshold - 1e-9
    return int(failed), f"{'BLOCK' if failed else 'ALLOW'}: pass rate {pass_rate:.1%}, baseline {baseline:.1%}, regression {regression:+.1%}"


def run_suite(broken_share: float = 0.0) -> list[CaseResult]:
    cases = [_benchmark_case(), _flaky_benchmark_case(), _custom_llm_judge_case(), _online_guardrail_case()]
    cases += [session_case(session) for session in fifty_sessions(broken_share=broken_share)]
    return [evaluator_optimizer(case) for case in cases]


BASELINE = sum(result.passed for result in run_suite()) / len(run_suite())


def ex3_ci_gate() -> None:
    twenty = [CaseResult(f"c{i}", "custom", i < 18, 1, "", "") for i in range(20)]         # 90% against a 95% baseline
    print(f"  main.ci_gate at exactly 5%  : {'ALLOW' if ci_gate(twenty, 0.95)[0] else 'BLOCK'}")
    print(f"  gate() at exactly 5%        : {gate(twenty, 0.95)[1]}")
    assert ci_gate(twenty, 0.95)[0] is True and gate(twenty, 0.95)[0] == 1

    healthy = subprocess.run([sys.executable, __file__, "--ci"], capture_output=True, text=True)
    regressed = subprocess.run([sys.executable, __file__, "--ci", "--simulate-regression"], capture_output=True, text=True)
    print(f"  --ci                        : exit {healthy.returncode}  {healthy.stdout.strip()}")
    print(f"  --ci --simulate-regression  : exit {regressed.returncode}  {regressed.stdout.strip()}")
    assert healthy.returncode == 0 and regressed.returncode == 1
    assert "practice.py --ci" in CI_WORKFLOW


# ---------------------------------------------------------------------------
# Exercise 4 - a trajectory-efficiency metric
#
# Steps taken divided by the gold trajectory's steps, per session. A pass
# rate cannot see an agent that reaches the right answer in twice the steps;
# this can, so it is reported next to the pass rate and given its own limit.
# For the evaluator-optimizer cases in main the same ratio is rounds used
# over one round.
# ---------------------------------------------------------------------------

def efficiency(steps: int, gold_steps: int) -> float:
    return steps / gold_steps


def ex4_trajectory_efficiency() -> None:
    sessions = fifty_sessions()
    ratios = [efficiency(session["steps"], session["gold_steps"]) for session in sessions]
    mean = sum(ratios) / len(ratios)
    over = sum(ratio > 2.0 for ratio in ratios)
    worst = max(sessions, key=lambda session: session["steps"])
    print(f"  50 sessions: mean {mean:.2f}x gold, {sum(r == 1.0 for r in ratios)} at 1.0x, {over} over 2.0x "
          f"(worst {worst['id']}: {worst['steps']} steps for a {worst['gold_steps']}-step task)")
    rounds = {case.cid: evaluator_optimizer(case).rounds for case in
              (_benchmark_case(), _flaky_benchmark_case(), _custom_llm_judge_case(), _online_guardrail_case())}
    print(f"  main's cases, rounds over a one-round gold: {rounds}")
    assert 1.0 < mean < 2.0 and over > 0
    assert all(value == 2 for value in rounds.values())             # every case needed one retry: 2.0x


# ---------------------------------------------------------------------------
# Exercise 5 - map every lesson to an eval case
#
# The suite for this curriculum is the practice files: each one asserts its
# lesson's behaviour and exits non-zero when an assertion fails, which is all
# an eval case is. The map is built by reading the repo, so it cannot drift:
# a lesson counts as covered when it has a practice.py with at least five
# exercise functions. Lessons without one are the gap.
# `python practice.py --run-suite` executes every covered lesson as a case.
# ---------------------------------------------------------------------------

def lesson_coverage() -> dict[str, int]:
    """lesson folder -> number of exercise functions in its practice.py (0 if there is none)."""
    coverage = {}
    for folder in sorted(p for p in REPO.iterdir() if p.is_dir() and re.match(r"\d\d-", p.name)):
        practice = folder / "code" / "practice.py"
        if not practice.exists():
            coverage[folder.name] = 0
            continue
        tree = ast.parse(practice.read_text(encoding="utf-8"))
        coverage[folder.name] = sum(isinstance(node, ast.FunctionDef) and re.match(r"ex\d+_", node.name) is not None
                                    for node in tree.body)
    return coverage


def run_lesson_cases() -> dict[str, bool]:
    here = Path(__file__).resolve()
    outcomes = {}
    for folder, exercises in lesson_coverage().items():
        practice = REPO / folder / "code" / "practice.py"
        if exercises and practice.resolve() != here:
            outcomes[folder] = subprocess.run([sys.executable, str(practice)], capture_output=True).returncode == 0
    return outcomes


def ex5_lesson_coverage() -> None:
    coverage = lesson_coverage()
    covered = [name for name, exercises in coverage.items() if exercises >= 5]
    gaps = [name for name, exercises in coverage.items() if exercises < 5]
    print(f"  {len(coverage)} lessons in the repo, {len(covered)} with an eval case, "
          f"{sum(coverage.values())} exercise checks in total")
    print(f"  covered: {covered[0]} ... {covered[-1]}")
    print(f"  gaps to close ({len(gaps)}): {', '.join(name[:2] for name in gaps) or 'none'}")
    assert all(coverage[name] >= 5 for name in coverage if int(name[:2]) <= 40)
    assert "30-eval-driven-agent-development" in covered


if __name__ == "__main__":
    if "--ci" in sys.argv:
        code, message = gate(run_suite(broken_share=0.2 if "--simulate-regression" in sys.argv else 0.0), BASELINE)
        print(message)
        sys.exit(code)
    if "--run-suite" in sys.argv:
        outcomes = run_lesson_cases()
        for folder, passed in outcomes.items():
            print(f"  {'PASS' if passed else 'FAIL'}  {folder}")
        print(f"{sum(outcomes.values())} of {len(outcomes)} lesson cases pass")
        sys.exit(0 if all(outcomes.values()) else 1)
    print("Phase 14 - Lesson 30: Eval-Driven Agent Development - exercises")
    for exercise in (ex1_failure_as_eval_case, ex2_rubric_on_fifty_sessions, ex3_ci_gate, ex4_trajectory_efficiency,
                     ex5_lesson_coverage):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
