"""Phase 14 - Lesson 25: Multi-Agent Debate - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

Every debater in this file is scripted, as in main.py. The numbers describe
those scripts; what carries over to real models is the mechanism each one shows.
"""

from __future__ import annotations

import random
from collections import Counter

from main import Debater, _make_debater, full_mesh_round, run_debate

TRUTH = {"capital_of_portugal": "Lisbon", "is_2_plus_2_equal_4": "yes", "chess_legal_e4": "legal"}
ALTERNATIVES = {"capital_of_portugal": ["Lisbon", "Madrid", "Porto", "Seville", "Faro"],
                "is_2_plus_2_equal_4": ["yes", "no", "unsure"],
                "chess_legal_e4": ["legal", "illegal", "unsure"]}
KNOWS_ALL = dict(TRUTH)


def main_debaters() -> list[Debater]:
    """The three debaters main.py builds inside main()."""
    return [
        _make_debater("alpha", bias="Lisbon", corrections={k: v for k, v in TRUTH.items() if k != "capital_of_portugal"}),
        _make_debater("beta", bias="Madrid", corrections=KNOWS_ALL),
        _make_debater("gamma", bias="Porto", corrections=KNOWS_ALL),
    ]


def debate_from(debaters: list[Debater], question: str, openings: dict[str, str], rounds: int = 3) -> tuple[str, int]:
    """Full-mesh debate starting from given opening proposals. Returns (answer, round of consensus or -1)."""
    prior, converged = dict(openings), -1
    for r in range(rounds):
        prior, _ = full_mesh_round(debaters, question, prior)
        if converged == -1 and len(set(prior.values())) == 1:
            converged = r + 1
    return Counter(prior.values()).most_common(1)[0][0], converged


# ---------------------------------------------------------------------------
# Exercise 1 - forced disagreement in round 1
#
# Rule: opening proposals must all differ. A debater whose answer is already
# taken has to open with its next choice.
#
# Effect on convergence speed: it costs rounds, and with main's debaters it
# can cost consensus. Without the rule they agree in round 1 on all three
# questions. With it, one question takes three rounds and the other two are
# still swapping answers after three, though the majority answer stays right.
# (main's debaters keep no memory of their own last answer and follow the
# first-listed peer on a tie, which is what lets them oscillate.)
#
# What the rule buys shows in the second half: three debaters who share the
# same wrong first answer converge on it immediately, and only the forced
# alternatives put the right answer on the table for them to recognise.
# ---------------------------------------------------------------------------

def distinct_openings(preferences: dict[str, list[str]]) -> dict[str, str]:
    """preferences: debater -> answers in order of preference. First come, first served."""
    taken: set[str] = set()
    openings = {}
    for name, ranked in preferences.items():
        openings[name] = next(answer for answer in ranked if answer not in taken)
        taken.add(openings[name])
    return openings


class Opinionated:
    """A debater that follows its peers unless it sees an answer it recognises as right."""

    def __init__(self, name: str, ranked: list[str], recognises: set[str]) -> None:
        self.name, self.ranked, self.recognises = name, ranked, recognises
        self.current = ranked[0]

    def drift(self, question: str, peers: list[str]) -> str:
        recognised = [answer for answer in peers if answer in self.recognises]
        if recognised:
            self.current = recognised[0]
        elif self.current not in self.recognises and peers:
            self.current = Counter(peers + [self.current]).most_common(1)[0][0]
        return self.current


def ex1_forced_disagreement() -> None:
    for question, truth in TRUTH.items():
        debaters = main_debaters()
        free = {d.name: d.drift(question, []) for d in debaters}
        forced = distinct_openings({name: [answer] + [a for a in ALTERNATIVES[question] if a != answer]
                                    for name, answer in free.items()})
        plain, forced_result = debate_from(debaters, question, free), debate_from(debaters, question, forced)
        print(f"  {question:<20} free: {plain[0]} in round {plain[1]}   forced: {forced_result[0]} in round {forced_result[1]}")
        assert plain == (truth, 1) and forced_result[0] == truth and forced_result[1] != 1

    def team() -> list[Opinionated]:
        return [Opinionated("alpha", ["Madrid", "Porto"], {"Lisbon"}),
                Opinionated("beta", ["Madrid", "Lisbon"], {"Lisbon"}),
                Opinionated("gamma", ["Madrid", "Seville"], {"Lisbon"})]

    outcomes = {}
    for label, force in (("free", False), ("forced", True)):
        agents = team()
        openings = distinct_openings({a.name: a.ranked for a in agents}) if force else {a.name: a.ranked[0] for a in agents}
        for agent in agents:
            agent.current = openings[agent.name]
        outcomes[label] = debate_from([Debater(a.name, a.drift) for a in agents], "capital_of_portugal", openings)
        print(f"  shared wrong prior, {label:<6}: opens {sorted(openings.values())} -> {outcomes[label][0]}")
    assert outcomes["free"][0] == "Madrid" and outcomes["forced"][0] == "Lisbon"


# ---------------------------------------------------------------------------
# Exercise 2 - confidence-weighted aggregation
#
# Does it help? Only as far as the confidences mean something. When a
# debater's confidence tracks how often it is right, weighting by it beats a
# plain majority. When debaters are most confident exactly when they are
# wrong, weighting hands them the vote and does worse than counting heads.
# ---------------------------------------------------------------------------

def aggregate(votes: list[tuple[str, float]]) -> tuple[str, str]:
    majority = Counter(answer for answer, _ in votes).most_common(1)[0][0]
    weight: Counter[str] = Counter()
    for answer, confidence in votes:
        weight[answer] += confidence
    return majority, weight.most_common(1)[0][0]


def aggregation_accuracy(calibrated: bool, trials: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    majority_right = weighted_right = 0
    for _ in range(trials):
        votes = []
        for _ in range(3):
            skill = rng.uniform(0.4, 0.9)
            right = rng.random() < skill
            if calibrated:
                confidence = skill
            else:
                confidence = rng.uniform(0.4, 0.7) if right else rng.uniform(0.8, 1.0)
            votes.append(("right" if right else rng.choice(("wrong-a", "wrong-b")), confidence))
        majority, weighted = aggregate(votes)
        majority_right += majority == "right"
        weighted_right += weighted == "right"
    return majority_right / trials, weighted_right / trials


def ex2_confidence_weighting() -> None:
    example = [("Lisbon", 0.9), ("Madrid", 0.4), ("Madrid", 0.35)]
    print(f"  one confident expert against two unsure: majority {aggregate(example)[0]}, weighted {aggregate(example)[1]}")
    calibrated, overconfident = aggregation_accuracy(True), aggregation_accuracy(False)
    print(f"  calibrated confidence  : majority {calibrated[0]:.1%}   weighted {calibrated[1]:.1%}")
    print(f"  confident when wrong   : majority {overconfident[0]:.1%}   weighted {overconfident[1]:.1%}")
    assert aggregate(example) == ("Madrid", "Lisbon")
    assert calibrated[1] > calibrated[0] and overconfident[1] < overconfident[0]


# ---------------------------------------------------------------------------
# Exercise 3 - one debater swapped for a different model
#
# Three copies of one model make the same mistakes, so voting among them is
# worth nothing. Swap one for a model with independent mistakes and a
# majority vote is still worth nothing: when the two copies are wrong
# together they outvote it. The different model only pays off through the
# debate itself, when the others can recognise its better answer once they
# see it (assumed here to happen 80% of the time), or when all three differ.
# ---------------------------------------------------------------------------

def team_accuracy(team: str, trials: int = 20000, seed: int = 0, skill: float = 0.7, recognise: float = 0.8) -> float:
    rng = random.Random(seed)
    right = 0
    for _ in range(trials):
        shared, other, third = (rng.random() < skill for _ in range(3))
        if team == "three copies, vote":
            right += shared
        elif team == "two copies + one different, vote":
            right += shared                             # the pair always forms the majority
        elif team == "two copies + one different, debate":
            right += shared or (other and rng.random() < recognise)
        else:                                           # three different models, vote
            right += shared + other + third >= 2
    return right / trials


def ex3_heterogeneity() -> None:
    teams = ("three copies, vote", "two copies + one different, vote",
             "two copies + one different, debate", "three different models, vote")
    accuracy = {team: team_accuracy(team) for team in teams}
    for team, value in accuracy.items():
        print(f"  {team:<36} {value:.1%}")
    assert abs(accuracy[teams[0]] - accuracy[teams[1]]) < 0.01      # a lone dissenter changes nothing in a vote
    assert accuracy[teams[2]] > accuracy[teams[3]] > accuracy[teams[0]]


# ---------------------------------------------------------------------------
# Exercises 4 and 5 share a five-debater team.
# ---------------------------------------------------------------------------

def five_debaters(bad_hub: bool = False) -> list[Debater]:
    first = (_make_debater("zeta", bias="Madrid", corrections={k: v for k, v in TRUTH.items() if k != "capital_of_portugal"})
             if bad_hub else main_debaters()[0])
    return [first, *main_debaters()[1:],
            _make_debater("delta", bias="Seville", corrections=KNOWS_ALL),
            _make_debater("epsilon", bias="Faro", corrections=KNOWS_ALL)]


# ---------------------------------------------------------------------------
# Exercise 4 - cost of full mesh against sparse, on the three questions
#
# Cost is main's count of critique operations, one per peer answer read. The
# plot is text: one row per configuration, sorted by cost.
# ---------------------------------------------------------------------------

def ex4_cost_vs_accuracy() -> None:
    rows = []
    for size, debaters in ((3, main_debaters()), (5, five_debaters())):
        for topology in ("sparse_star", "full_mesh"):
            results = [run_debate(debaters, question, rounds=3, topology=topology) for question in TRUTH]
            cost = sum(ops for _, _, ops in results)
            correct = sum(answer == TRUTH[q] for (answer, _, _), q in zip(results, TRUTH))
            rows.append((cost, f"{topology} N={size}", correct))
    for cost, label, correct in sorted(rows):
        print(f"  {label:<18} {cost:>4} ops |{'#' * (cost // 6):<30}| accuracy {correct}/3")
    costs = {label: cost for cost, label, _ in rows}
    assert costs["full_mesh N=3"] == 54 and costs["sparse_star N=3"] == 36
    assert costs["full_mesh N=5"] == 180 and costs["sparse_star N=5"] == 72
    assert all(correct == 3 for _, _, correct in rows)          # same accuracy, 2.5x the cost at N=5


# ---------------------------------------------------------------------------
# Exercise 5 - the toy at N=5, R=3
#
# The Society of Minds experiments ran three agents for two rounds, for cost,
# and found that more agents and more rounds help on hard problems.
#
# What gets better at N=5: full mesh absorbs one wrong debater in a single
# round, because the other four outvote it in everyone's view.
#
# What breaks:
#   - cost: 60 critique operations per question against 18 at N=3;
#   - ties: a debater does not count its own answer, so with two of five
#     wrong the three who are right each see a 2-2 split among their peers.
#     The tie goes to the answer listed first, the wrong one, and they switch,
#     in the same round that the two who were wrong are won over. The camps
#     swap answers every round and there is no consensus after three;
#   - the star: put the one wrong debater at the hub and every spoke reads
#     only its answer. Spokes copy the hub while the hub copies the spokes,
#     and the group ends on the wrong answer.
# ---------------------------------------------------------------------------

def ex5_five_debaters_three_rounds() -> None:
    question = "capital_of_portugal"
    mesh = run_debate(five_debaters(bad_hub=True), question, rounds=3, topology="full_mesh")
    star = run_debate(five_debaters(bad_hub=True), question, rounds=3, topology="sparse_star")
    print(f"  one wrong debater, full mesh       : {mesh[0]}, consensus in round {mesh[1]}, {mesh[2]} ops")
    print(f"  the wrong debater as hub of a star : {star[0]}, consensus in round {star[1]}, {star[2]} ops")

    stubborn = [_make_debater("zeta", "Madrid", {}), _make_debater("eta", "Madrid", {}), *five_debaters()[2:]]
    two_wrong = run_debate(stubborn, question, rounds=3, topology="full_mesh")
    print(f"  two of five wrong, full mesh       : {two_wrong[0]}, consensus in round {two_wrong[1]}")
    assert mesh[0] == "Lisbon" and mesh[1] == 1 and mesh[2] == 60
    assert star[0] == "Madrid" and star[1] == -1 and star[2] == 24
    assert two_wrong[1] == -1                       # a 2-2 split among peers never settles


if __name__ == "__main__":
    print("Phase 14 - Lesson 25: Multi-Agent Debate - exercises")
    for exercise in (ex1_forced_disagreement, ex2_confidence_weighting, ex3_heterogeneity, ex4_cost_vs_accuracy,
                     ex5_five_debaters_three_rounds):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
