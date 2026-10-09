"""Phase 14 - Lesson 02: ReWOO and Plan-and-Execute - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from main import (REFERENCE_RE, Plan, PlanStep, ScriptedPlanner, ScriptedSolver, ToolRegistry,
                  fake_search, resolve_references, rounded_million, run_workers, topological)

QUESTION = "What is the population of the capital of France, rounded?"
SOLVER = ScriptedSolver("The capital of France is {E1}; rounded population is {E3}.")
GOOD_PLAN = Plan(steps=[
    PlanStep("E1", "search", {"query": "capital of France"}),
    PlanStep("E2", "search", {"query": "population of #E1"}),
    PlanStep("E3", "round_million", {"text": "#E2"}),
])


def demo_tools(delay: float = 0.0) -> ToolRegistry:
    """The lesson's tools, each with `delay` seconds of simulated latency."""
    def slow(fn):
        def wrapped(**kwargs: Any) -> str:
            time.sleep(delay)
            return fn(**kwargs)
        return wrapped

    tools = ToolRegistry()
    tools.register("search", slow(fake_search))
    tools.register("round_million", slow(rounded_million))
    tools.register("upper", slow(lambda text: text.upper()))
    return tools


def failed(evidence: dict[str, str]) -> list[str]:
    return [k for k, v in evidence.items() if v.startswith(("error", "no result", "unknown"))]


# ---------------------------------------------------------------------------
# Exercise 1 - run independent plan nodes in parallel
#
# On a 6-node DAG with 2 parallel groups the wall clock drops from 6 tool
# latencies to 2. Tokens do not change: the same calls are made. Parallelism
# buys latency, not cost.
# ---------------------------------------------------------------------------

def levels(plan: Plan) -> list[list[PlanStep]]:
    """Group steps so each level depends only on the levels before it."""
    done: set[str] = set()
    pending = list(plan.steps)
    out: list[list[PlanStep]] = []
    while pending:
        ready = [s for s in pending
                 if all(f"E{r}" in done for r in REFERENCE_RE.findall(str(s.args)))]
        if not ready:
            raise RuntimeError("cyclic plan or unresolved reference")
        out.append(ready)
        done |= {s.id for s in ready}
        pending = [s for s in pending if s.id not in done]
    return out


def run_workers_parallel(plan: Plan, tools: ToolRegistry) -> dict[str, str]:
    evidence: dict[str, str] = {}
    with ThreadPoolExecutor() as pool:
        for level in levels(plan):
            bound = [{k: resolve_references(v, evidence) for k, v in s.args.items()} for s in level]
            results = list(pool.map(tools.dispatch, [s.tool for s in level], bound))
            evidence.update(zip([s.id for s in level], results))
    return evidence


SIX_NODE_PLAN = Plan(steps=[
    PlanStep("E1", "search", {"query": "capital of France"}),
    PlanStep("E2", "search", {"query": "capital of Germany"}),
    PlanStep("E3", "search", {"query": "population of Paris"}),
    PlanStep("E4", "search", {"query": "population of #E1"}),
    PlanStep("E5", "upper", {"text": "#E2"}),
    PlanStep("E6", "round_million", {"text": "#E3"}),
])


def ex1_parallel_workers() -> None:
    tools = demo_tools(delay=0.1)
    start = time.perf_counter()
    sequential = run_workers(SIX_NODE_PLAN, tools)
    t_seq = time.perf_counter() - start
    start = time.perf_counter()
    parallel = run_workers_parallel(SIX_NODE_PLAN, tools)
    t_par = time.perf_counter() - start

    print(f"  levels     : {[[s.id for s in level] for level in levels(SIX_NODE_PLAN)]}")
    print(f"  sequential : {t_seq:.2f}s   parallel: {t_par:.2f}s   speedup: {t_seq / t_par:.1f}x")
    assert parallel == sequential
    assert len(levels(SIX_NODE_PLAN)) == 2
    assert t_par < 0.75 * t_seq                 # 2 levels against 6 calls: a third, with room for a busy machine


# ---------------------------------------------------------------------------
# Exercise 2 - a replanner that fires when a worker fails
#
# The smallest change that turns ReWOO into Plan-and-Execute is one edge:
# evidence flows back to the planner. ReWOO's planner never sees observations;
# here it sees them, but only when something failed.
# ---------------------------------------------------------------------------

class ReplanningPlanner(ScriptedPlanner):
    """Scripted stand-in for an LLM replanner: given the failed evidence, return a repaired plan."""

    def __init__(self, plan: Plan, repaired: Plan) -> None:
        super().__init__(plan)
        self.repaired = repaired

    def replan(self, question: str, plan: Plan, evidence: dict[str, str]) -> Plan:
        return self.repaired


def run_plan_and_execute(question: str, planner: ReplanningPlanner, tools: ToolRegistry,
                         solver: ScriptedSolver, max_replans: int = 2) -> tuple[str, int]:
    plan = planner.plan_for(question)
    replans = 0
    while True:
        # ponytail: a replan re-runs steps that already succeeded; cache evidence by
        # (tool, args) once tool calls cost real money.
        evidence = run_workers(plan, tools)
        if not failed(evidence) or replans == max_replans:
            return solver.solve(question, plan, evidence), replans
        plan = planner.replan(question, plan, evidence)     # the one new edge
        replans += 1


def ex2_replanner() -> None:
    typo_plan = Plan(steps=[PlanStep("E1", "search", {"query": "capitol of France"}),
                            *GOOD_PLAN.steps[1:]])
    tools = demo_tools()
    rewoo_answer = SOLVER.solve(QUESTION, typo_plan, run_workers(typo_plan, tools))
    answer, replans = run_plan_and_execute(QUESTION, ReplanningPlanner(typo_plan, GOOD_PLAN), tools, SOLVER)

    print(f"  ReWOO, no replanner : {rewoo_answer}")
    print(f"  with replanner      : {answer}  (replans: {replans})")
    assert "no result" in rewoo_answer
    assert answer == "The capital of France is Paris; rounded population is 11 million."
    assert replans == 1


# ---------------------------------------------------------------------------
# Exercise 3 - small planner, frontier solver
#
# SIMULATED: no models on this machine. The "7B" plan below makes the mistake
# small planners typically make, losing the #E1 variable binding. A real
# comparison needs two real models and a question set.
#
# Where the split fails: at the plan. The solver has no tools, so the best a
# frontier solver can do with broken evidence is refuse; it cannot repair it.
# The opposite split (frontier planner, small solver) is safe here because the
# solver only has to compose evidence that is already correct.
# ---------------------------------------------------------------------------

class CarefulSolver(ScriptedSolver):
    """A stronger solver: notices failed evidence and refuses instead of composing garbage."""

    def solve(self, question: str, plan: Plan, evidence: dict[str, str]) -> str:
        bad = failed(evidence)
        if bad:
            return f"cannot answer: evidence {', '.join(bad)} failed"
        return super().solve(question, plan, evidence)


def ex3_small_planner() -> None:
    small_plan = Plan(steps=[
        PlanStep("E1", "search", {"query": "capital of France"}),
        PlanStep("E2", "search", {"query": "population of the capital"}),   # dropped the #E1 reference
        PlanStep("E3", "round_million", {"text": "#E2"}),
    ])
    tools = demo_tools()
    careful = CarefulSolver(SOLVER.template)
    results = {
        "frontier planner + frontier solver": careful.solve(QUESTION, GOOD_PLAN, run_workers(GOOD_PLAN, tools)),
        "small planner    + basic solver   ": SOLVER.solve(QUESTION, small_plan, run_workers(small_plan, tools)),
        "small planner    + frontier solver": careful.solve(QUESTION, small_plan, run_workers(small_plan, tools)),
    }
    for label, answer in results.items():
        print(f"  {label}: {answer}")
    answers = list(results.values())
    assert "11 million" in answers[0]
    assert "unknown" in answers[1]                      # confidently wrong
    assert answers[2].startswith("cannot answer")       # detected, still not fixed


# ---------------------------------------------------------------------------
# Exercise 4 - planner distillation (ReWOO paper, Sections 2.3 and 3.3 on arXiv)
#
# Training data: (question -> plan) pairs written by the 175B teacher planner
# (GPT-3.5) on HotpotQA and TriviaQA. No observations are needed, because the
# planner never sees any - that is what makes it distillable. The paper
# generates about 4,000 plans and keeps the roughly 2,000 that led to a
# correct final answer, then LoRA-fine-tunes Alpaca-7B on them.
#
# Scoring plan quality: there is no gold plan to compare against, so score a
# plan by what it does. It must parse into an acyclic DAG, every step must
# execute, and the solved answer must match the gold answer. The last check is
# the only one that measures quality; the first two are cheap filters.
# ---------------------------------------------------------------------------

def score_plan(plan: Plan, tools: ToolRegistry, gold: str) -> dict[str, bool]:
    try:
        topological(plan)
    except RuntimeError:
        return {"valid": False, "executes": False, "correct": False}
    evidence = run_workers(plan, tools)
    executes = not failed(evidence)
    return {"valid": True, "executes": executes,
            "correct": executes and gold in SOLVER.solve(QUESTION, plan, evidence)}


def ex4_distillation() -> None:
    teacher_plans = {
        "good": GOOD_PLAN,
        "typo": Plan(steps=[PlanStep("E1", "search", {"query": "capitol of France"}), *GOOD_PLAN.steps[1:]]),
        "cyclic": Plan(steps=[PlanStep("E1", "search", {"query": "#E2"}),
                              PlanStep("E2", "search", {"query": "#E1"})]),
    }
    tools = demo_tools()
    scores = {name: score_plan(plan, tools, gold="11 million") for name, plan in teacher_plans.items()}
    dataset = [(QUESTION, teacher_plans[name]) for name, score in scores.items() if all(score.values())]

    for name, score in scores.items():
        print(f"  teacher plan {name:<6}: {score}")
    print(f"  kept for fine-tuning: {len(dataset)} of {len(teacher_plans)}")
    assert [name for name, s in scores.items() if all(s.values())] == ["good"]


# ---------------------------------------------------------------------------
# Exercise 5 - Plan-and-Act's shape: the plan is a sequence, not a DAG
#
# Tradeoffs: a sequence is easier to generate and to fine-tune on, and it has
# an obvious replan point after every step, which is what long web tasks need.
# It loses fan-in and parallelism: a step sees the previous observation only,
# so the 6-node DAG above costs 6 serial steps instead of 2 levels.
# ---------------------------------------------------------------------------

def run_sequence(steps: list[tuple[str, dict[str, str]]], tools: ToolRegistry) -> list[str]:
    trace: list[str] = []
    for tool, args in steps:
        prev = trace[-1] if trace else ""
        trace.append(tools.dispatch(tool, {k: v.replace("{prev}", prev) for k, v in args.items()}))
    return trace


def ex5_sequence_plan() -> None:
    tools = demo_tools()
    trace = run_sequence([
        ("search", {"query": "capital of France"}),
        ("search", {"query": "population of {prev}"}),
        ("round_million", {"text": "{prev}"}),
    ], tools)
    dag_evidence = run_workers(GOOD_PLAN, tools)

    print(f"  sequence trace : {trace}")
    print(f"  DAG evidence   : {dag_evidence}")
    print(f"  6-node plan    : {len(levels(SIX_NODE_PLAN))} levels as a DAG, "
          f"{len(SIX_NODE_PLAN.steps)} steps as a sequence")
    assert trace == list(dag_evidence.values())


if __name__ == "__main__":
    print("Phase 14 - Lesson 02: ReWOO and Plan-and-Execute - exercises")
    for exercise in (ex1_parallel_workers, ex2_replanner, ex3_small_planner,
                     ex4_distillation, ex5_sequence_plan):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
