"""Phase 14 - Lesson 11: Planning with HTN and Evolutionary Search - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import itertools
import random
import sqlite3
from typing import Callable

from main import HTNPlanner, Method, Operator, ScriptedLLM

# ---------------------------------------------------------------------------
# Exercise 1 - backtracking when an operator's postcondition fails at runtime
#
# main.plan picks the first applicable method and plans on paper: it assumes
# every operator delivers its effects. execute() runs the operators for real,
# checks each postcondition, and when one fails it throws away the state that
# method built and tries the next method from the rollback point.
#
# ponytail: rollback restores the planner's symbolic state only. An operator
# that already changed the outside world needs a compensating operator.
# ---------------------------------------------------------------------------

Runtime = dict[str, Callable[[set[str]], set[str]]]     # what an operator really does when executed


class BacktrackingPlanner(HTNPlanner):
    def execute(self, task: str, state: set[str], runtime: Runtime, log: list[str]) -> set[str] | None:
        if task in self.operators:
            op = self.operators[task]
            if not op.applicable(state):
                return None
            after = runtime.get(task, op.apply)(set(state))
            if not all(fact in after for fact in op.effects_add):
                log.append(f"{task}: postcondition failed")
                return None
            log.append(f"{task}: ok")
            return after
        for method in self.methods.get(task, []):
            if not method.applicable(state):
                continue
            current: set[str] | None = set(state)       # a copy; `state` stays the rollback point
            for subtask in method.subtasks:
                current = self.execute(subtask, current, runtime, log)
                if current is None:
                    break
            if current is not None:
                return current
            log.append(f"rollback {method.name}")
        return None


def ex1_backtracking() -> None:
    operators = {
        "build_image": Operator("build_image", ("code_merged",), ("image_built",)),
        "push_registry": Operator("push_registry", ("image_built",), ("image_pushed",)),
        "rollout_k8s": Operator("rollout_k8s", ("image_pushed",), ("deployed",)),
        "upload_artifact": Operator("upload_artifact", ("image_built",), ("artifact_uploaded",)),
        "rollout_vm": Operator("rollout_vm", ("artifact_uploaded",), ("deployed",)),
    }
    methods = {"deploy": [
        Method("deploy_via_k8s", "deploy", ("code_merged",), ("build_image", "push_registry", "rollout_k8s")),
        Method("deploy_via_vm", "deploy", ("code_merged",), ("build_image", "upload_artifact", "rollout_vm")),
    ]}
    planner = BacktrackingPlanner(operators=operators, methods=methods, llm=ScriptedLLM({}))
    registry_is_down: Runtime = {"push_registry": lambda state: state}      # runs, changes nothing

    log: list[str] = []
    final = planner.execute("deploy", {"code_merged"}, registry_is_down, log)
    print(f"  plan on paper : {planner.plan('deploy', {'code_merged'})}")
    for line in log:
        print(f"  {line}")
    assert final is not None and "deployed" in final and "image_pushed" not in final
    assert log == ["build_image: ok", "push_registry: postcondition failed", "rollback deploy_via_k8s",
                   "build_image: ok", "upload_artifact: ok", "rollout_vm: ok"]


# ---------------------------------------------------------------------------
# Exercise 2 - an LLM-method cache keyed on the state pattern
#
# main caches an LLM decomposition under the task name alone, so it replays it
# in states where it cannot work and never asks again. Here the decomposition
# is stored as a real Method in the library, and its preconditions are the
# state pattern: the facts the resulting plan needed from the starting state.
# "Check the library first" then needs no extra code - a learned method is
# just a method.
# ---------------------------------------------------------------------------

class LearningPlanner(HTNPlanner):
    def plan(self, task: str, state: set[str], depth: int = 0, max_depth: int = 12) -> list[str] | None:
        in_library = task in self.operators or any(m.applicable(state) for m in self.methods.get(task, []))
        if in_library or depth > max_depth:
            return super().plan(task, state, depth, max_depth)
        suggested = self.llm.decompose(task, state)
        if suggested is None or not all(s in self.operators or s in self.methods for s in suggested):
            return None
        plan = self._expand(list(suggested), state, depth)
        if plan is not None:
            name = f"learned_{task}_{len(self.methods.get(task, [])) + 1}"
            self.methods.setdefault(task, []).append(
                Method(name, task, self._required_facts(plan), tuple(suggested)))
        return plan

    def _required_facts(self, plan: list[str]) -> tuple[str, ...]:
        needed: set[str] = set()
        produced: set[str] = set()
        for step in plan:
            op = self.operators[step]
            needed |= set(op.preconditions) - produced
            produced |= set(op.effects_add)
        return tuple(sorted(needed))


class StateAwareLLM(ScriptedLLM):
    """Scripted stand-in whose answer depends on the state it is shown."""

    def decompose(self, task: str, state: set[str]) -> tuple[str, ...] | None:
        steps = super().decompose(task, state)
        if steps is None or "logged_in" in state:
            return steps
        return ("login", *steps)


def shipping_domain() -> tuple[dict[str, Operator], dict[str, tuple[str, ...]]]:
    operators = {
        "login": Operator("login", (), ("logged_in",)),
        "open_editor": Operator("open_editor", ("logged_in",), ("editor_open",)),
        "write_tests": Operator("write_tests", ("editor_open",), ("tests_written",)),
        "run_tests": Operator("run_tests", ("tests_written",), ("tests_passing",)),
        "open_pr": Operator("open_pr", ("tests_passing",), ("pr_open",)),
    }
    scripts = {"ship_feature": ("open_editor", "write_tests", "run_tests", "open_pr")}
    return operators, scripts


def ex2_method_cache() -> None:
    operators, scripts = shipping_domain()
    planner = LearningPlanner(operators=operators, methods={}, llm=StateAwareLLM(scripts))
    calls = []
    for state in ({"logged_in"}, {"logged_in"}, set(), set()):
        plan = planner.plan("ship_feature", state)
        calls.append(len(planner.llm.calls))
        print(f"  state {sorted(state) or '{}'}: {len(plan)} steps, LLM calls so far {calls[-1]}")
    for method in planner.methods["ship_feature"]:
        print(f"  library: {method.name} when {method.preconditions or '()'} -> {method.subtasks[0]}, ...")

    baseline = HTNPlanner(operators=operators, methods={}, llm=ScriptedLLM(scripts))
    baseline.plan("ship_feature", {"logged_in"})
    print(f"  main, logged out after caching: {baseline.plan('ship_feature', set())}")
    assert calls == [1, 1, 2, 2]                                # one LLM call per new state pattern
    assert planner.plan("ship_feature", set())[0] == "login"
    assert baseline.plan("ship_feature", set()) is None         # task-only cache replays the wrong plan


# ---------------------------------------------------------------------------
# Exercise 3 - a real test suite as the evaluator: evolve a sort function
#
# The program being evolved is a sorting network for 5 values: a list of
# compare-and-swap pairs. Mutations are random (AlphaEvolve uses an LLM
# ensemble there); the evaluator is the test suite, pass or fail per case.
#
# The second result matters more than the generation count. Twenty random
# cases are a weak evaluator: a network can pass all of them and still fail
# other inputs. Evolution finds whatever the evaluator rewards, nothing more.
# The 32 zero/one inputs are a complete evaluator for a 5-wire network (the
# 0-1 principle), and what evolves against them is correct on everything.
# ---------------------------------------------------------------------------

Network = list[tuple[int, int]]
WIDTH = 5


def run_network(network: Network, values: list[int]) -> list[int]:
    out = list(values)
    for i, j in network:
        if out[i] > out[j]:
            out[i], out[j] = out[j], out[i]
    return out


def passes(network: Network, tests: list[list[int]]) -> int:
    return sum(run_network(network, case) == sorted(case) for case in tests)


def mutate(network: Network, rng: random.Random) -> Network:
    child = list(network)
    i, j = sorted(rng.sample(range(WIDTH), 2))
    kind = rng.choice(("add", "add", "replace", "drop"))
    if kind == "add" or not child:
        child.insert(rng.randint(0, len(child)), (i, j))
    elif kind == "replace":
        child[rng.randrange(len(child))] = (i, j)
    else:
        child.pop(rng.randrange(len(child)))
    return child


def evolve_sort(tests: list[list[int]], seed: int, max_generations: int = 3000) -> tuple[Network, int]:
    rng = random.Random(seed)
    population: list[Network] = [[] for _ in range(8)]       # seed program: does nothing
    for generation in range(1, max_generations + 1):
        children = [mutate(parent, rng) for parent in population for _ in range(3)]
        # children first: on equal fitness the newer program survives, which lets the search drift
        population = sorted(children + population, key=lambda n: -passes(n, tests))[:8]
        if passes(population[0], tests) == len(tests):
            return population[0], generation
    return population[0], max_generations


RANDOM_TESTS = [random.Random(case).sample(range(100), WIDTH) for case in range(20)]
ZERO_ONE_TESTS = [list(bits) for bits in itertools.product((0, 1), repeat=WIDTH)]
ALL_PERMUTATIONS = [list(p) for p in itertools.permutations(range(WIDTH))]


def ex3_evolve_sort() -> None:
    results = {}
    for label, tests in (("20 random test cases", RANDOM_TESTS), ("32 zero/one cases   ", ZERO_ONE_TESTS)):
        runs = [evolve_sort(tests, seed) for seed in range(5)]
        generations = [g for _, g in runs]
        holdout = [passes(net, ALL_PERMUTATIONS) for net, _ in runs]
        results[label] = (generations, holdout)
        print(f"  {label}: converged in {generations} generations (mean {sum(generations) / 5:.0f}); "
              f"sorts {holdout} of 120 permutations")
    assert all(g < 3000 for generations, _ in results.values() for g in generations)
    assert all(h == 120 for h in results["32 zero/one cases   "][1])
    assert min(results["20 random test cases"][1]) < 120        # passed its tests, still wrong


# ---------------------------------------------------------------------------
# Exercise 4 - an evaluator for SQL query optimisation
#
# AlphaEvolve's evaluator is a function from a candidate program to scalar
# metrics, and the search optimises whatever it returns. Designing one for
# "make this query faster" means three properties:
#   - machine-checkable and deterministic: cost is SQLite VM steps, counted by
#     a progress handler, not wall-clock time;
#   - staged, cheap checks first: (1) it parses and finishes inside a step
#     budget, (2) it returns the reference rows on several seeded databases,
#     (3) only then is cost recorded;
#   - correctness gates the metric: a candidate that fails stage 2 gets no
#     cost at all, or the search would learn to be fast by being wrong.
# ---------------------------------------------------------------------------

REFERENCE_SQL = ("SELECT c.id, (SELECT SUM(o.amount) FROM orders o WHERE o.customer_id = c.id "
                 "AND o.status = 'paid') AS total FROM customers c WHERE c.region = 'EU' ORDER BY c.id")
CANDIDATES = {
    "reference (subquery)": REFERENCE_SQL,
    "left join + group by": ("SELECT c.id, SUM(o.amount) AS total FROM customers c LEFT JOIN orders o "
                             "ON o.customer_id = c.id AND o.status = 'paid' WHERE c.region = 'EU' "
                             "GROUP BY c.id ORDER BY c.id"),
    "inner join": ("SELECT c.id, SUM(o.amount) AS total FROM customers c JOIN orders o "
                   "ON o.customer_id = c.id WHERE c.region = 'EU' AND o.status = 'paid' "
                   "GROUP BY c.id ORDER BY c.id"),
    "runaway cross join": ("SELECT c.id, COUNT(*) AS total FROM customers c, orders a, orders b, orders d "
                           "WHERE c.region = 'EU' GROUP BY c.id ORDER BY c.id"),
    "syntax error": "SELEC c.id FROM customers c",
}


def build_db(seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript("CREATE TABLE customers(id INTEGER PRIMARY KEY, region TEXT);"
                     "CREATE TABLE orders(id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER, status TEXT);")
    for customer in range(1, 40):
        db.execute("INSERT INTO customers VALUES (?, ?)", (customer, rng.choice(("EU", "US", "APAC"))))
    db.execute("INSERT INTO customers VALUES (40, 'EU')")           # edge case: a customer with no orders
    for order in range(1, 401):
        db.execute("INSERT INTO orders VALUES (?, ?, ?, ?)",
                   (order, rng.randint(1, 39), rng.randint(5, 500), rng.choice(("paid", "paid", "refunded", "pending"))))
    return db


def run_metered(db: sqlite3.Connection, sql: str, budget: int = 2_000_000) -> tuple[list[tuple], int]:
    steps = 0

    def tick() -> int:
        nonlocal steps
        steps += 100
        return steps > budget          # non-zero makes SQLite abort the query

    db.set_progress_handler(tick, 100)
    try:
        return db.execute(sql).fetchall(), steps
    finally:
        db.set_progress_handler(None, 0)


def evaluate_query(sql: str, databases: list[sqlite3.Connection]) -> dict[str, object]:
    cost = 0
    for db in databases:
        try:
            rows, steps = run_metered(db, sql)
        except sqlite3.OperationalError as error:
            return {"stage": 1, "valid": False, "correct": False, "cost": None, "why": str(error)}
        if rows != run_metered(db, REFERENCE_SQL)[0]:
            return {"stage": 2, "valid": True, "correct": False, "cost": None, "why": "rows differ from reference"}
        cost += steps
    return {"stage": 3, "valid": True, "correct": True, "cost": cost, "why": ""}


def ex4_sql_evaluator() -> None:
    databases = [build_db(seed) for seed in range(3)]
    scores = {name: evaluate_query(sql, databases) for name, sql in CANDIDATES.items()}
    for name, score in scores.items():
        verdict = f"cost {score['cost']} VM steps" if score["correct"] else f"rejected at stage {score['stage']}: {score['why']}"
        print(f"  {name:<21} {verdict}")
    survivors = {name: s["cost"] for name, s in scores.items() if s["correct"]}
    print(f"  selected: {min(survivors, key=survivors.get)}")
    assert set(survivors) == {"reference (subquery)", "left join + group by"}
    assert scores["inner join"]["stage"] == 2           # runs fine, wrong rows
    assert scores["runaway cross join"]["stage"] == 1 and scores["syntax error"]["stage"] == 1


# ---------------------------------------------------------------------------
# Exercise 5 - HTN decomposes, evolutionary search fills in a primitive
#
# Where it shines: a primitive with a complete, machine-checkable spec and a
# large search space (sort_scores, evolved against the 32 zero/one cases).
# Where it over-engineers: a primitive that is one obvious line (take_top3).
# Evolving that would spend evaluator calls to rediscover a slice, and for a
# primitive with no checkable spec there is nothing to evolve against at all.
# ---------------------------------------------------------------------------

def ex5_htn_plus_evolution() -> None:
    operators = {
        "sort_scores": Operator("sort_scores", ("scores_loaded",), ("scores_sorted",)),
        "take_top3": Operator("take_top3", ("scores_sorted",), ("leaderboard_ready",)),
    }
    methods = {"make_leaderboard": [Method("leaderboard_m1", "make_leaderboard", ("scores_loaded",),
                                           ("sort_scores", "take_top3"))]}
    plan = HTNPlanner(operators=operators, methods=methods, llm=ScriptedLLM({})).plan(
        "make_leaderboard", {"scores_loaded"})

    network, generations = evolve_sort(ZERO_ONE_TESTS, seed=0)
    implementations: dict[str, Callable[[list[int]], list[int]]] = {
        "sort_scores": lambda scores: run_network(network, scores),         # evolved
        "take_top3": lambda scores: scores[-3:][::-1],                      # written by hand
    }
    data = [41, 7, 99, 23, 58]
    for step in plan:
        data = implementations[step](data)
    print(f"  HTN plan      : {plan}")
    print(f"  sort_scores   : evolved in {generations} generations, {len(network)} comparators")
    print(f"  take_top3     : one line, no search")
    print(f"  leaderboard   : {data}")
    assert plan == ["sort_scores", "take_top3"] and data == [99, 58, 41]


if __name__ == "__main__":
    print("Phase 14 - Lesson 11: HTN and Evolutionary Search - exercises")
    for exercise in (ex1_backtracking, ex2_method_cache, ex3_evolve_sort, ex4_sql_evaluator,
                     ex5_htn_plus_evolution):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
