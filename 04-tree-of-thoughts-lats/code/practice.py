"""Phase 14 - Lesson 04: Tree of Thoughts and LATS - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import math
import random
from typing import Callable

from main import NUMBERS, Node, _all_leaves, backprop, expand, tot_bfs, uct, value


def fresh_root(numbers: list[int] = NUMBERS) -> Node:
    return Node(state=tuple(sorted(numbers, reverse=True)), trace=[])


def mcts(root: Node, iterations: int, rng: random.Random, c: float = 1.4,
         score: Callable[[Node], float] = value) -> Node:
    """main.mcts with the exploration constant and the scorer as parameters."""
    for _ in range(iterations):
        path, cur = [root], root
        while cur.children:
            cur = max(cur.children, key=lambda ch: uct(cur, ch, c))
            path.append(cur)
        if cur.visits > 0 and len(cur.state) > 1:
            cur.children = expand(cur)
            if cur.children:
                cur = cur.children[0]
                path.append(cur)
        leaf = cur
        for _ in range(max(0, 3 - len(cur.trace))):         # random rollout to a terminal state
            options = expand(leaf)
            if not options:
                break
            leaf = rng.choice(options)
        backprop(path, score(leaf))
    return root


def solved(root: Node) -> bool:
    """Does the tree contain a finished expression that equals 24?"""
    return any(value(leaf) == 1.0 for leaf in _all_leaves(root))


# ---------------------------------------------------------------------------
# Exercise 1 - UCT with c=0.1 versus c=2.0
#
# The exploration term is c * sqrt(ln N / n). With a small c the search trusts
# its early value estimates and visits pile onto a few root children. With a
# large c it keeps returning to neglected children, so visits spread almost
# evenly across all 24 of them.
# ---------------------------------------------------------------------------

def tree_shape(c: float, iterations: int = 300) -> dict[str, float]:
    root = mcts(fresh_root(), iterations, random.Random(7), c=c)
    visits = sorted((ch.visits for ch in root.children), reverse=True)
    return {"root children": len(visits),
            "top child share": round(visits[0] / sum(visits), 2),
            "children visited >5x": sum(v > 5 for v in visits)}


def ex1_exploration_constant() -> None:
    greedy, curious = tree_shape(0.1), tree_shape(2.0)
    print(f"  c=0.1 : {greedy}")
    print(f"  c=2.0 : {curious}")
    assert greedy["top child share"] > curious["top child share"]
    assert greedy["children visited >5x"] < curious["children visited >5x"]


# ---------------------------------------------------------------------------
# Exercise 2 - a noisy value function
#
# Gaussian jitter is added to every score the search sees. The agent then has
# to commit without an oracle: it answers with the leaf whose mean observed
# reward is highest.
#
# There are two signals, and they fail at different noise levels. Guidance
# between partial states is weak (values differ by about 0.05), so sigma=0.1
# already costs some runs: the search stops finding the solution. The gap
# between a solution (1.0) and the best wrong leaf (about 0) is large, and the
# search only collapses once sigma approaches it.
# ---------------------------------------------------------------------------

def commit_rate(sigma: float, seeds: int = 40, iterations: int = 400) -> float:
    hits = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        root = mcts(fresh_root(), iterations, rng, score=lambda n: value(n) + rng.gauss(0, sigma))
        visited = [leaf for leaf in _all_leaves(root) if leaf.visits]
        answer = max(visited, key=lambda leaf: leaf.q)
        hits += value(answer) == 1.0
    return hits / seeds


def ex2_noisy_value() -> None:
    rates = {sigma: commit_rate(sigma) for sigma in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)}
    for sigma, rate in rates.items():
        print(f"  sigma={sigma:<4}  commits to a real solution in {rate:.0%} of runs")
    broken = next(s for s, r in rates.items() if r < 0.5 * rates[0.0])
    print(f"  below half the noise-free rate at sigma={broken}: a signal-to-noise ratio of about {1 / broken:.0f}:1")
    assert rates[0.0] >= 0.9
    assert rates[2.0] < rates[0.0]


# ---------------------------------------------------------------------------
# Exercise 3 - beam-search ToT versus full BFS
#
# main.tot_bfs already keeps the top-k nodes per level, so it is the beam;
# an unlimited k makes it plain BFS. Every expansion is one value-prompt call,
# so expansions stand in for tokens. Measured over random solvable puzzles.
#
# On a tight budget the beam is better: BFS simply does not fit. But a beam is
# only as good as its value function, and "distance of the closest number to
# 24" is a weak one, so narrow beams prune most real solutions away.
# ---------------------------------------------------------------------------

def solvable_puzzles(count: int = 30, seed: int = 1) -> list[list[int]]:
    rng = random.Random(seed)
    puzzles: list[list[int]] = []
    while len(puzzles) < count:
        numbers = [rng.randint(1, 9) for _ in range(4)]
        best, _ = tot_bfs(fresh_root(numbers), max_expansions_per_level=10 ** 9)
        if best is not None and value(best) == 1.0:
            puzzles.append(numbers)
    return puzzles


def ex3_beam_vs_bfs() -> None:
    puzzles = solvable_puzzles()
    table: dict[int, tuple[float, float]] = {}
    for k in (1, 3, 8, 20, 10 ** 9):
        runs = [tot_bfs(fresh_root(p), max_expansions_per_level=k) for p in puzzles]
        solve_rate = sum(best is not None and value(best) == 1.0 for best, _ in runs) / len(runs)
        table[k] = (solve_rate, sum(n for _, n in runs) / len(runs))
    for k, (solve_rate, expansions) in table.items():
        label = "BFS (no pruning)" if k == 10 ** 9 else f"beam k={k}"
        print(f"  {label:<17} solved {solve_rate:>4.0%}   mean expansions {expansions:>4.0f}"
              f"   solves per 100 expansions {100 * solve_rate / expansions:.2f}")
    best_k = max(table, key=lambda k: table[k][0] / table[k][1])
    print(f"  most solves per token: beam k={best_k}")
    assert table[10 ** 9][0] == 1.0                     # BFS is complete on these puzzles
    assert table[1][1] < table[10 ** 9][1]              # the beam is far cheaper
    assert table[1][0] <= table[20][0]                  # and pays for it in solve rate
    assert best_k != 10 ** 9


# ---------------------------------------------------------------------------
# Exercise 4 - how many rollouts for the HumanEval result (LATS paper)
#
# The paper reports pass@1 of 92.7% with GPT-4 on HumanEval using at most k=8
# search iterations with n=5 sampled children per expansion. (HumanEval is in
# the programming section; Section 5.1 is HotPotQA, which uses k=50.) Code
# needs few rollouts because the value function is a test run, which is nearly
# noise-free.
#
# Reproduced in shape only: the toy needs hundreds of iterations because its
# rollouts are random rather than model-guided, and the tree has to be three
# levels deep before it can hold an answer. The last line asks how good one
# rollout must be for 8 of them to reach 92.7%.
# ---------------------------------------------------------------------------

def ex4_rollout_budget() -> None:
    seeds = 40
    rates = {}
    for iterations in (100, 200, 300, 400):
        rates[iterations] = sum(solved(mcts(fresh_root(), iterations, random.Random(s)))
                                for s in range(seeds)) / seeds
        print(f"  toy Game of 24, {iterations:>3} iterations: solved in {rates[iterations]:.0%} of runs")
    per_rollout = 1 - (1 - 0.927) ** (1 / 8)
    print(f"  8 independent rollouts reach 92.7% if each one passes {per_rollout:.0%} of the time")
    assert rates[400] > rates[100]
    assert math.isclose(1 - (1 - per_rollout) ** 8, 0.927)


# ---------------------------------------------------------------------------
# Exercise 5 - when LATS helps less: a decision rule
# ---------------------------------------------------------------------------

DECISION_RULE = """\
  Use plain ReAct when one trajectory usually works or the task is short; search
  only multiplies cost. Add a verifier (CRITIC) when mistakes can be checked but
  not undone. Use beam-search ToT when the problem is pure reasoning with a
  cheap, trustworthy scorer and a tight budget. Use LATS only when all three
  hold: the environment can be reverted to an earlier state, the value signal
  is reliable (tests, an exact target), and correctness matters more than
  tokens and wall-clock time. Where actions are irreversible, or where feedback
  is generic rather than specific (the paper's WebShop result), LATS buys
  little, and with a noisy scorer it finds a well-scoring wrong answer."""


def ex5_decision_rule() -> None:
    print(DECISION_RULE)


if __name__ == "__main__":
    print("Phase 14 - Lesson 04: Tree of Thoughts and LATS - exercises")
    for exercise in (ex1_exploration_constant, ex2_noisy_value, ex3_beam_vs_bfs,
                     ex4_rollout_budget, ex5_decision_rule):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
