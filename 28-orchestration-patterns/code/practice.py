"""Phase 14 - Lesson 28: Orchestration Patterns - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import random
from typing import Callable

import main as lesson
from main import SPECIALISTS, classify, debate, hierarchical, supervisor_worker, swarm

KEYWORDS = {"refund": ("refund", "money back", "charged"), "bug": ("crash", "error", "bug"),
            "sales": ("pricing", "quote", "seats")}


# ---------------------------------------------------------------------------
# Exercise 1 - a supervisor-worker turned into a swarm
#
# main.swarm still calls the global classifier inside every agent, which is a
# router by another name. Here the router is really gone: whoever holds the
# task checks it against its own keywords, and if it is not theirs passes it
# to the one peer they think of.
#
# What improves: a task that lands on the right agent costs one call instead
# of two, and there is no router to be a bottleneck or a single point of
# failure. What breaks: a task nobody recognises circulates until something
# stops it; an ambiguous task goes to whoever held it first, where the
# supervisor would have made one consistent choice; and no single place
# records why a task went where it did.
# ---------------------------------------------------------------------------

NEXT_PEER = {"refund": "bug", "bug": "sales", "sales": "refund"}


def local_swarm(task: str, entry: str = "refund", max_handoffs: int = 3) -> tuple[str | None, list[str], int]:
    current, trace, ops = entry, [], 0
    for _ in range(max_handoffs + 1):
        ops += 1
        if any(word in task.lower() for word in KEYWORDS[current]):
            trace.append(f"{current}: {SPECIALISTS[current](task)}")
            return current, trace, ops
        trace.append(f"{current} -> {NEXT_PEER[current]}")
        current = NEXT_PEER[current]
    return None, trace, ops


def ex1_remove_the_router() -> None:
    tasks = ["I need a refund for invoice 4711", "refund the duplicate charge please", "the CLI crashes on ctrl-c",
             "do you offer volume pricing?", "the app crashed after I was charged twice",
             "my invoice shows the wrong company name"]
    total = {"supervisor": 0, "swarm": 0}
    for task in tasks:
        _, supervisor_ops = supervisor_worker([task])
        handler, trace, swarm_ops = local_swarm(task)
        total["supervisor"] += supervisor_ops
        total["swarm"] += swarm_ops
        print(f"  {task[:42]:<42} supervisor -> {classify(task):<6} ({supervisor_ops} ops)   "
              f"swarm -> {handler or 'nobody':<6} ({swarm_ops} ops)")
    print(f"  total ops: supervisor {total['supervisor']}, swarm {total['swarm']}")
    assert local_swarm(tasks[0])[2] == 1                                    # the entry agent owns it: one call
    assert local_swarm(tasks[4])[0] == "refund" and classify(tasks[4]) == "bug"     # they disagree on the ambiguous one
    assert local_swarm(tasks[5])[0] is None and classify(tasks[5]) == "sales"       # dropped here, guessed there


# ---------------------------------------------------------------------------
# Exercise 2 - a hop counter on the swarm
#
# Does "refuse after 3 handoffs" catch A -> B -> A bouncing? It ends it, on
# the third handoff, without knowing it was a bounce, and it also refuses a
# legitimate chain that happens to be four agents long. Remembering who has
# already held the task catches the bounce one handoff earlier, names it, and
# lets the long chain through. Keep both: the visited set for loops, the
# counter as the hard ceiling.
# ---------------------------------------------------------------------------

def handoff_chain(start: str, next_agent: dict[str, str | None], max_handoffs: int | None,
                  detect_revisit: bool) -> tuple[str, list[str]]:
    path = [start]
    while next_agent[path[-1]] is not None:
        target = next_agent[path[-1]]
        if detect_revisit and target in path:
            return f"refused: {target} already held this task", path
        if max_handoffs is not None and len(path) - 1 >= max_handoffs:
            return f"refused: {max_handoffs} handoffs used", path
        path.append(target)
    return f"handled by {path[-1]}", path


def ex2_hop_counter() -> None:
    bounce = {"A": "B", "B": "A"}
    long_chain = {"A": "B", "B": "C", "C": "D", "D": "E", "E": None}
    results = {}
    for label, routes in (("A <-> B bounce", bounce), ("A to E, four handoffs", long_chain)):
        results[label] = (handoff_chain("A", routes, max_handoffs=3, detect_revisit=False),
                          handoff_chain("A", routes, max_handoffs=None, detect_revisit=True))
        for name, (outcome, path) in zip(("hop counter ", "visited set "), results[label]):
            print(f"  {label:<22} {name}: {' -> '.join(path)}  {outcome}")
    assert results["A <-> B bounce"][0] == ("refused: 3 handoffs used", ["A", "B", "A", "B"])
    assert results["A <-> B bounce"][1] == ("refused: A already held this task", ["A", "B"])
    assert results["A to E, four handoffs"][0][0].startswith("refused")         # a false alarm
    assert results["A to E, four handoffs"][1][0] == "handled by E"


# ---------------------------------------------------------------------------
# Exercise 3 - two levels for a 12-specialist domain
#
# Where the context budget fails without nesting: a flat supervisor has to
# hold every specialist's card (description plus tool schemas) in the prompt
# of every routing call. The card size and the budget below are assumptions;
# with them the flat router is over budget from the ninth specialist. Nested,
# the top level holds four short group cards and each group lead holds three
# specialist cards, at the price of one more call per task.
# ---------------------------------------------------------------------------

DOMAIN = {
    "billing": {"refund": ("refund", "money back"), "invoice": ("invoice", "receipt"), "tax": ("vat", "tax")},
    "technical": {"bug": ("crash", "bug"), "outage": ("down", "outage"), "api": ("api", "webhook")},
    "sales": {"pricing": ("pricing", "quote"), "upgrade": ("upgrade", "plan change"), "trial": ("trial", "demo")},
    "account": {"login": ("password", "login"), "privacy": ("gdpr", "delete my data"), "closure": ("close my account", "cancel")},
}
SPECIALIST_CARD_TOKENS, GROUP_CARD_TOKENS, ROUTING_BUDGET_TOKENS = 450, 120, 4000


def route_two_levels(task: str) -> tuple[str, str, int] | None:
    text, ops = task.lower(), 1
    for group, specialists in DOMAIN.items():                   # top level: which group
        if any(word in text for words in specialists.values() for word in words):
            ops += 1
            for specialist, words in specialists.items():       # group lead: which specialist
                if any(word in text for word in words):
                    return group, specialist, ops + 1           # + the specialist's own call
    return None


def ex3_two_level_hierarchy() -> None:
    tasks = {"refund": "I want my money back", "invoice": "resend the receipt", "tax": "add our vat number",
             "bug": "it will crash on save", "outage": "the dashboard is down", "api": "webhook retries fail",
             "pricing": "quote for 50 seats", "upgrade": "plan change to enterprise", "trial": "extend our trial",
             "login": "reset my password", "privacy": "please delete my data", "closure": "close my account"}
    routed = {expected: route_two_levels(task) for expected, task in tasks.items()}
    assert all(route is not None and route[1] == expected and route[2] == 3 for expected, route in routed.items())

    specialists = sum(len(group) for group in DOMAIN.values())
    flat = specialists * SPECIALIST_CARD_TOKENS
    top = len(DOMAIN) * GROUP_CARD_TOKENS
    lead = max(len(group) for group in DOMAIN.values()) * SPECIALIST_CARD_TOKENS
    first_over = next(n for n in range(1, 50) if n * SPECIALIST_CARD_TOKENS > ROUTING_BUDGET_TOKENS)
    print(f"  12/12 tasks routed correctly through two levels, 3 calls each")
    print(f"  flat supervisor prompt : {flat} tokens for {specialists} cards (budget {ROUTING_BUDGET_TOKENS}); "
          f"over budget from specialist {first_over}")
    print(f"  nested prompts         : top {top} tokens, group lead {lead} tokens")
    assert flat > ROUTING_BUDGET_TOKENS and first_over == 9
    assert max(top, lead) < ROUTING_BUDGET_TOKENS


# ---------------------------------------------------------------------------
# Exercise 4 - the four patterns on one workload
#
# SIMULATED WORKLOAD: 300 support tasks, and a classifier that is wrong 15% of
# the time. The noise is injected by swapping main.classify, so main's four
# pattern functions run unchanged. Cost is calls per task. Latency counts
# calls on the critical path, with debate's three proposers in parallel.
# Debuggability is a judgement, stated per pattern.
#
# No pattern wins everywhere, which is the lesson's point about choosing the
# problem first.
# ---------------------------------------------------------------------------

HANDLED_BY = {"refund handled": "refund", "bug logged": "bug", "quote sent": "sales"}
DEBUGGABILITY = {
    "supervisor-worker": "one routing decision, in one place",
    "swarm": "decisions spread across agents; follow the handoffs",
    "hierarchical": "two decisions, in two layers",
    "debate": "three proposals and a vote, all logged together",
}


def profile(pattern: Callable[[list[str]], tuple[list[str], int]], tasks: list[str], error_rate: float,
            seed: int) -> dict[str, float]:
    rng = random.Random(seed)

    def noisy(text: str) -> str:
        truth = classify(text)
        return truth if rng.random() >= error_rate else rng.choice([label for label in SPECIALISTS if label != truth])

    lesson.classify = noisy
    try:
        calls = correct = dropped = latency = 0
        for task in tasks:
            trace, ops = pattern([task])
            handler = next((label for line in trace for text, label in HANDLED_BY.items() if text in line), None)
            calls += ops
            latency += ops - 2 if pattern is debate else ops
            correct += handler == classify(task)
            dropped += handler is None
    finally:
        lesson.classify = classify
    n = len(tasks)
    return {"cost": calls / n, "latency": latency / n, "accuracy": correct / n, "dropped": dropped / n}


def ex4_profile_patterns() -> None:
    rng = random.Random(1)
    templates = ["I need a refund for invoice {n}", "the CLI crashes on build {n}", "pricing for {n} seats please"]
    tasks = [rng.choice(templates).format(n=rng.randint(10, 999)) for _ in range(300)]
    patterns = {"supervisor-worker": supervisor_worker, "swarm": swarm, "hierarchical": hierarchical, "debate": debate}
    results = {name: profile(fn, tasks, error_rate=0.15, seed=2) for name, fn in patterns.items()}
    for name, r in results.items():
        print(f"  {name:<18} cost {r['cost']:.2f}  latency {r['latency']:.2f}  accuracy {r['accuracy']:.1%}  "
              f"dropped {r['dropped']:.1%}   {DEBUGGABILITY[name]}")
    winners = {metric: (min if metric != "accuracy" else max)(results, key=lambda name: results[name][metric])
               for metric in ("cost", "latency", "accuracy")}
    print(f"  wins: {winners}")
    assert winners["accuracy"] == "debate" and results["debate"]["cost"] == max(r["cost"] for r in results.values())
    assert results["swarm"]["dropped"] > 0 and results["supervisor-worker"]["dropped"] == 0


# ---------------------------------------------------------------------------
# Exercise 5 - map production flows onto the four patterns
#
# STAND-IN FLOWS for a DevOps tooling product; replace them with real ones.
# "Building Effective Agents" argues for the simplest thing that works:
# single calls and fixed workflows before agents, agents before teams of
# agents. That shows up in the mapping. Two of the seven flows do not map onto
# any of the four multi-agent patterns, because they are one agent following
# a workflow, and forcing them into a topology would be the "topology-first"
# mistake the lesson warns about.
# ---------------------------------------------------------------------------

PATTERNS = {"supervisor-worker", "swarm", "hierarchical", "debate", "none: single-agent workflow"}
FLOWS = [
    ("CI failure triage", "supervisor-worker", "one router sends each failure to a flaky-test, regression or infra handler"),
    ("incident investigation", "supervisor-worker", "an orchestrator fans out to log, metric and deploy investigators, then synthesises"),
    ("review of a risky change", "debate", "independent reviewers must agree before auto-merge"),
    ("on-call handoff between service bots", "swarm", "the bot that owns the service takes over directly; hop limit needed"),
    ("support desk across 40 services", "hierarchical", "too many specialists for one router's prompt"),
    ("nightly dependency bump", "none: single-agent workflow", "fixed steps in a fixed order: a prompt chain"),
    ("release notes with a checker", "none: single-agent workflow", "draft, evaluate, revise: an evaluator-optimizer loop"),
]


def ex5_map_flows() -> None:
    for flow, pattern, why in FLOWS:
        print(f"  {flow:<37} {pattern:<28} {why}")
    unmapped = [flow for flow, pattern, _ in FLOWS if pattern.startswith("none")]
    print(f"  do not map onto a multi-agent pattern: {unmapped}")
    assert all(pattern in PATTERNS for _, pattern, _ in FLOWS) and len(unmapped) == 2


if __name__ == "__main__":
    print("Phase 14 - Lesson 28: Orchestration Patterns - exercises")
    for exercise in (ex1_remove_the_router, ex2_hop_counter, ex3_two_level_hierarchy, ex4_profile_patterns,
                     ex5_map_flows):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
