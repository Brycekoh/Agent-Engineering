"""Phase 14 - Lesson 12: Anthropic's Workflow Patterns - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import importlib.util
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from main import evaluator_optimizer, parallel_vote, prompt_chain, route

LABELS = ("refund", "bug", "sales")

# ---------------------------------------------------------------------------
# Exercise 1 - routing with a confidence threshold
#
# Where the threshold lands: where automating a ticket stops being cheaper
# than a human touching it. If the classifier is calibrated (a call made with
# confidence c is right c of the time), automating costs (1 - c) * misroute and
# escalating costs one human touch, so the break-even is
#     threshold = 1 - human_cost / misroute_cost.
# With a misroute costing five human touches, that is 0.8. The sweep below
# measures it on simulated tickets. A real classifier's confidence has to be
# calibrated on labelled tickets before this rule means anything.
# ---------------------------------------------------------------------------

Classifier = Callable[[str], tuple[str, float]]


def route_with_confidence(text: str, classifier: Classifier, handlers: dict[str, Callable[[str], str]],
                          threshold: float) -> tuple[str, str]:
    """main.route, sent to the default (human) handler when the classifier is unsure."""
    label, confidence = classifier(text)
    gated = label if confidence >= threshold else "default"
    return route(text, lambda _: gated, handlers)


def simulated_tickets(n: int = 20000, seed: int = 0) -> list[tuple[str, str, float]]:
    """(true label, predicted label, confidence) from a calibrated classifier."""
    rng = random.Random(seed)
    tickets = []
    for _ in range(n):
        truth = rng.choice(LABELS)
        confidence = rng.uniform(0.34, 1.0)
        wrong = rng.choice([label for label in LABELS if label != truth])
        tickets.append((truth, truth if rng.random() < confidence else wrong, confidence))
    return tickets


def ex1_confidence_threshold() -> None:
    handlers = {label: (lambda text, label=label: f"{label} flow") for label in LABELS}
    handlers["default"] = lambda text: "escalate to human"
    unsure: Classifier = lambda text: ("refund", 0.55)
    print(f"  0.55 confidence, threshold 0.5 -> {route_with_confidence('money back?', unsure, handlers, 0.5)}")
    print(f"  0.55 confidence, threshold 0.8 -> {route_with_confidence('money back?', unsure, handlers, 0.8)}")
    assert route_with_confidence("money back?", unsure, handlers, 0.5) == ("refund", "refund flow")
    assert route_with_confidence("money back?", unsure, handlers, 0.8) == ("default", "escalate to human")

    tickets = simulated_tickets()
    human_cost, misroute_cost = 1.0, 5.0
    costs = {}
    for threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        automated = [(truth, predicted) for truth, predicted, conf in tickets if conf >= threshold]
        misrouted = sum(truth != predicted for truth, predicted in automated)
        escalated = len(tickets) - len(automated)
        costs[threshold] = (escalated * human_cost + misrouted * misroute_cost) / len(tickets) * 100
        print(f"  threshold {threshold:<4}: automated {len(automated) / len(tickets):>4.0%}, "
              f"misrouted {misrouted / max(len(automated), 1):>4.0%} of those, cost per 100 tickets {costs[threshold]:.1f}")
    best = min(costs, key=costs.get)
    print(f"  cheapest threshold: {best}  (rule predicts {1 - human_cost / misroute_cost})")
    assert best == 0.8


# ---------------------------------------------------------------------------
# Exercise 2 - a timeout on parallel_vote
#
# What happens when one call hangs: main.parallel_vote runs the calls one
# after another, so the whole vote waits on the slowest. With a deadline the
# vote returns on time and the hung call simply never arrives.
#
# Aggregating with missing votes: count them as abstentions, and require a
# majority of the votes requested, not of the votes received. A missing vote
# can then block a decision but can never swing one, and "no decision" goes to
# a human instead of being reported as a win.
# ---------------------------------------------------------------------------

def parallel_vote_with_timeout(prompt: str, llm: Callable[[str], str], n: int = 5,
                               timeout_s: float = 0.2) -> tuple[str | None, Counter, int]:
    pool = ThreadPoolExecutor(max_workers=n)
    futures = [pool.submit(llm, prompt) for _ in range(n)]
    done, pending = wait(futures, timeout=timeout_s)
    pool.shutdown(wait=False, cancel_futures=True)          # do not wait for the hung call
    counts = Counter(f.result() for f in done if f.exception() is None)
    winner, votes = counts.most_common(1)[0] if counts else (None, 0)
    return (winner if votes > n / 2 else None), counts, len(pending)


HANG_S = 0.6


def scripted_voters(script: list[str], hang_s: float = HANG_S) -> Callable[[str], str]:
    answers, lock = iter(script), threading.Lock()

    def llm(prompt: str) -> str:
        with lock:
            answer = next(answers)
        if answer == "HANG":
            time.sleep(hang_s)
            return "yes"
        return answer

    return llm


def ex2_vote_timeout() -> None:
    question = "is this code safe to ship?"
    start = time.perf_counter()
    parallel_vote(question, scripted_voters(["yes", "yes", "no", "yes", "HANG"]), n=5)
    blocked = time.perf_counter() - start

    start = time.perf_counter()
    winner, counts, missing = parallel_vote_with_timeout(question, scripted_voters(["yes", "yes", "no", "yes", "HANG"]))
    bounded = time.perf_counter() - start
    split, split_counts, split_missing = parallel_vote_with_timeout(
        question, scripted_voters(["yes", "yes", "no", "HANG", "HANG"]))

    print(f"  main.parallel_vote, one hung call : returned after {blocked:.2f}s")
    print(f"  with a 0.2s deadline              : {winner!r} {dict(counts)}, {missing} missing, {bounded:.2f}s")
    print(f"  two hung calls                    : {split!r} {dict(split_counts)}, {split_missing} missing -> escalate")
    assert bounded < 0.9 * HANG_S < blocked             # back before the hung call would have finished
    assert winner == "yes" and missing == 1             # 3 of 5 requested is still a majority
    assert split is None and split_counts["yes"] == 2   # 2 of 5 is not, even though yes leads 2 to 1


# ---------------------------------------------------------------------------
# Exercise 3 - evaluator-optimizer as a bandit that keeps the top two
#
# main.evaluator_optimizer returns the last candidate when nothing passes, so
# a bad final round overwrites a good earlier one. Here the two best
# candidates are kept across rounds and the best is returned. They also act as
# the two arms: rounds alternate between refining the leader and the
# runner-up, so one misleading critique cannot trap the search.
# ---------------------------------------------------------------------------

Scorer = Callable[[str, str], tuple[float, str]]


def evaluator_optimizer_top2(task: str, proposer: Callable[[str, str | None], str], scorer: Scorer,
                             max_iter: int = 5, pass_score: float = 0.95) -> tuple[str, list[tuple[float, str, str]]]:
    board: list[tuple[float, str, str]] = []        # (score, candidate, critique), best first, at most two
    feedback: str | None = None
    for i in range(max_iter):
        candidate = proposer(task, feedback)
        score, critique = scorer(task, candidate)
        board = sorted(board + [(score, candidate, critique)], key=lambda row: -row[0])[:2]
        if score >= pass_score:
            break
        parent = board[i % len(board)]              # alternate between the two arms
        feedback = f"improve '{parent[1]}': {parent[2]}"
    return board[0][1], board


def ex3_top2_bandit() -> None:
    quality = {"draft A": 0.40, "draft B": 0.70, "draft C": 0.90, "draft D": 0.30}
    parents: list[str | None] = []

    def scripted_proposer() -> Callable[[str, str | None], str]:
        drafts = iter(quality)

        def proposer(task: str, feedback: str | None) -> str:
            parents.append(feedback)
            return next(drafts)

        return proposer

    last, _ = evaluator_optimizer("summarise ReAct", scripted_proposer(),
                                  lambda task, cand: (False, f"score {quality[cand]}"), max_iter=4)
    parents.clear()
    best, board = evaluator_optimizer_top2("summarise ReAct", scripted_proposer(),
                                           lambda task, cand: (quality[cand], "tighten it"), max_iter=4)
    print(f"  main returns the last draft : {last} (quality {quality[last]})")
    print(f"  top-2 returns the best      : {best} (quality {quality[best]}), runner-up {board[1][1]}")
    print(f"  parent refined each round   : {[p.split(chr(39))[1] if p else None for p in parents]}")
    assert last == "draft D" and best == "draft C"
    assert [row[1] for row in board] == ["draft C", "draft B"]


# ---------------------------------------------------------------------------
# Exercise 4 - a router that picks one of three chains, against one big prompt
#
# MEASURED ON MY OWN INSTRUCTION TEXT with a scripted model; words stand in
# for tokens. The shape is what carries over: the routed chain sends only the
# instructions for the route it took, so it reads fewer tokens, and pays for
# that with three round trips instead of one. Prompt caching narrows the gap,
# because the big prompt's instructions are a stable prefix.
# ---------------------------------------------------------------------------

INSTRUCTIONS = {
    "refund": ("Check the order date against the thirty day refund window, confirm the item is eligible, "
               "and state the refundable amount in the currency of the original payment. Order: {text}",
               "Write a short reply that confirms the refund amount, says when the money will arrive, "
               "and apologises once without offering anything beyond policy. Decision: {text}"),
    "bug": ("Extract the product area, the exact steps that trigger the failure, the expected behaviour "
            "and the observed behaviour, and rate severity from one to four. Report: {text}",
            "Write a short reply that thanks the reporter, restates the problem in one sentence, "
            "and gives the ticket severity and what happens next. Triage: {text}"),
    "sales": ("Identify the team size, the plan they are asking about and any volume or term discount "
              "they qualify for under the current price list. Enquiry: {text}",
              "Write a short reply that quotes the plan and price, mentions the discount if one applies, "
              "and offers a call with an account manager. Quote: {text}"),
}
CLASSIFY = "Classify this support message as refund, bug or sales. Answer with one word. Message: {text}"


class MeteredLLM:
    """Scripted model that counts calls and words in and out."""

    def __init__(self) -> None:
        self.calls = self.words_in = self.words_out = 0

    def __call__(self, prompt: str) -> str:
        if prompt.startswith("Classify"):
            reply = next(label for label in LABELS if label in prompt.lower().split("message:")[1])
        else:
            reply = "Thanks for getting in touch, here is the outcome and what happens next for you."
        self.calls += 1
        self.words_in += len(prompt.split())
        self.words_out += len(reply.split())
        return reply


def ex4_routed_chains() -> None:
    messages = ["I would like a refund for order 4711 placed last week",
                "there is a bug where the export button does nothing",
                "what does the sales team charge for fifty seats"]
    routed, single = MeteredLLM(), MeteredLLM()
    for message in messages:
        handlers = {label: (lambda text, label=label: prompt_chain(
            text, routed, [("decide", INSTRUCTIONS[label][0]), ("reply", INSTRUCTIONS[label][1])])[-1][1])
            for label in LABELS}
        label, _ = route(message, lambda text: routed(CLASSIFY.format(text=text)), handlers)
        assert label in message

        every_instruction = " ".join(step.format(text="(see message)") for steps in INSTRUCTIONS.values() for step in steps)
        single("Decide whether this is a refund, bug or sales message, then follow only the matching "
               f"instructions. {every_instruction} Message: {message}")
    for label, meter in (("router + chain  ", routed), ("one big prompt  ", single)):
        print(f"  {label}: {meter.calls / 3:.0f} calls per message, {meter.words_in / 3:.0f} words in, "
              f"{meter.words_out / 3:.0f} words out")
    print(f"  the routed chain reads {routed.words_in / single.words_in:.0%} of the big prompt's input")
    assert routed.calls == 9 and single.calls == 3
    assert routed.words_in < single.words_in


# ---------------------------------------------------------------------------
# Exercise 5 - draw a feature as a workflow graph and count steps
#
# The feature is one this repository has: the support flow that lesson 13
# builds. The graph is read from that lesson's code, not redrawn by hand, so
# it cannot drift from it.
#
#   classify -+-> refund -+
#             +-> bug ----+-> human_gate -> send
#             +-> sales --+
#
# Would an agent be better? No. The graph has no cycle, one branch point and
# four steps on every run, so each run is bounded and auditable: routing plus
# a short chain covers it, and the human gate is a pause, not a loop. An
# agent earns its cost where the number of steps is not known in advance. Add
# one edge that says "not resolved, classify again" and the longest run has
# no bound: that is where a turn budget, and an agent, start to make sense.
# ---------------------------------------------------------------------------

def support_graph() -> tuple[dict[str, list[str]], str]:
    """The graph lesson 13 builds, read from its code: node -> next nodes, and the entry node."""
    path = Path(__file__).resolve().parents[2] / "13-langgraph-stateful-graphs" / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("lesson13_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    graph = module.build_graph()
    return {node: [edge.dst for edge in graph.edges.get(node, []) if edge.dst != module.END]
            for node in graph.nodes}, graph.entry


def longest_path(graph: dict[str, list[str]], node: str, seen: tuple[str, ...] = ()) -> int | None:
    """Steps on the longest run from `node`, or None if a cycle makes it unbounded."""
    if node in seen:
        return None
    lengths = [longest_path(graph, nxt, seen + (node,)) for nxt in graph[node]]
    if None in lengths:
        return None
    return 1 + max(lengths, default=0)


def ex5_workflow_or_agent() -> None:
    graph, entry = support_graph()
    steps = longest_path(graph, entry)
    branches = [node for node, nexts in graph.items() if len(nexts) > 1]
    for node, nexts in graph.items():
        print(f"  {node:<10} -> {', '.join(nexts) or 'END'}")
    print(f"  nodes: {len(graph)}   branch points: {branches}   longest run: {steps} steps")
    print(f"  verdict: {'workflow' if steps is not None else 'agent'} (bounded, no cycle)")
    looping = {**graph, "send": ["classify"]}                # "not resolved, classify again"
    print(f"  with a send -> classify edge: longest run {longest_path(looping, entry)} -> agent territory")
    assert (entry, len(graph), steps, branches) == ("classify", 6, 4, ["classify"])
    assert longest_path(looping, entry) is None


if __name__ == "__main__":
    print("Phase 14 - Lesson 12: Anthropic's Workflow Patterns - exercises")
    for exercise in (ex1_confidence_threshold, ex2_vote_timeout, ex3_top2_bandit, ex4_routed_chains,
                     ex5_workflow_or_agent):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
