"""Phase 14 - Lesson 03: Reflexion and Verbal Reinforcement Learning - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from main import TARGET, Actor, EpisodicMemory, Reflection, SelfReflector, run_reflexion


def nudge(attempt: list[int], step: int) -> list[int]:
    """Move the sum of three picks by `step`, keeping every pick inside 1..9."""
    out = sorted(attempt)
    for _ in range(abs(step)):
        if step > 0:
            i = next((i for i in range(3) if out[i] < 9), None)
        else:
            i = next((i for i in (2, 1, 0) if out[i] > 1), None)
        if i is None:
            break
        out[i] += 1 if step > 0 else -1
    return out


# ---------------------------------------------------------------------------
# Exercise 1 - binary evaluator versus a scalar distance
#
# Yes, it converges faster. A binary signal only says "wrong", so the actor can
# do no better than sweep one step at a time. A distance says how far, so the
# reflection can carry the size of the fix and the next trial lands on target.
# ---------------------------------------------------------------------------

def trials_to_converge(feedback: str, max_trials: int = 40) -> int:
    attempt = [1, 2, 3]
    for trial in range(1, max_trials + 1):
        distance = abs(sum(attempt) - TARGET)
        if distance == 0:
            return trial
        step = distance if feedback == "scalar" else 1      # binary: "wrong, try a bit more"
        attempt = nudge(attempt, step)
    return max_trials


def ex1_scalar_evaluator() -> None:
    binary, scalar = trials_to_converge("binary"), trials_to_converge("scalar")
    print(f"  binary evaluator : {binary} trials")
    print(f"  scalar evaluator : {scalar} trials")
    assert scalar == 2 and binary == 15


# ---------------------------------------------------------------------------
# Exercise 2 - a TTL of 10 trials on reflections
#
# While the task holds still, old reflections are harmless but redundant: they
# all say the same thing and only cost prompt space. Once the task shifts they
# hurt, because stale advice outvotes fresh evidence. The actor below follows
# the majority of the reflections it can see; the target moves at trial 16.
# ---------------------------------------------------------------------------

@dataclass
class TTLMemory(EpisodicMemory):
    ttl: int = 10

    def expire(self, now: int) -> None:
        self.items = [r for r in self.items if now - r.trial < self.ttl]


def recovery_trial(memory: TTLMemory, trials: int = 60, shift_at: int = 16) -> int | None:
    """Trial at which the actor is back on target after the target moves from 20 to 12."""
    attempt, target, reflector = [1, 2, 3], 20, SelfReflector()
    for trial in range(1, trials + 1):
        if trial == shift_at:
            target = 12
        memory.expire(trial)
        delta = sum(attempt) - target
        if delta == 0:
            if trial >= shift_at:
                return trial
            continue
        memory.add(Reflection(trial, reflector.reflect(attempt, delta)))
        votes = sum(1 if "larger" in r.text else -1 for r in memory.items)
        direction = (votes > 0) - (votes < 0) or (1 if delta < 0 else -1)   # tie: newest wins
        attempt = nudge(attempt, direction)
    return None


def ex2_reflection_ttl() -> None:
    forever = 10 ** 9
    results = {
        "keep everything      ": recovery_trial(TTLMemory(max_len=forever, ttl=forever)),
        "TTL of 10 trials     ": recovery_trial(TTLMemory(max_len=forever, ttl=10)),
        "last 6 (main default)": recovery_trial(TTLMemory(max_len=6, ttl=forever)),
    }
    for label, trial in results.items():
        print(f"  {label}: back on target at trial {trial}")
    keep_all, ttl10, last6 = results.values()
    assert last6 < ttl10 < keep_all


# ---------------------------------------------------------------------------
# Exercise 3 - heuristic evaluator: the same action twice means "stuck"
#
# How it interacts with the Self-Reflector: it needs no ground truth, so it
# fires even when the scalar evaluator has nothing new to say, and it should
# produce a different reflection. Writing "6 short, pick larger" a second time
# adds nothing; "you repeated yourself, the last reflection had no effect" does.
# It also ends hopeless runs early instead of spending the whole trial budget.
# ---------------------------------------------------------------------------

def run_with_stuck_check(use_memory: bool, max_trials: int = 4) -> tuple[int, str]:
    actor, reflector, memory = Actor(), SelfReflector(), EpisodicMemory()
    attempts: list[list[int]] = []
    for trial in range(1, max_trials + 1):
        attempt = actor.act(memory if use_memory else EpisodicMemory())
        attempts.append(attempt)
        delta = sum(attempt) - TARGET
        if delta == 0:
            return trial, "success"
        if len(attempts) >= 2 and attempts[-1] == attempts[-2]:
            return trial, f"stuck: repeated {attempt}; the last reflection changed nothing, switch strategy"
        memory.add(Reflection(trial, reflector.reflect(attempt, delta)))
    return max_trials, "budget exhausted"


def ex3_stuck_heuristic() -> None:
    baseline_trials = len(run_reflexion(max_trials=4, use_memory=False))
    no_memory = run_with_stuck_check(use_memory=False)
    with_memory = run_with_stuck_check(use_memory=True)
    print(f"  no memory, main loop     : {baseline_trials} trials, never notices")
    print(f"  no memory, stuck check   : trial {no_memory[0]} -> {no_memory[1]}")
    print(f"  with memory, stuck check : trial {with_memory[0]} -> {with_memory[1]}")
    assert baseline_trials == 4
    assert no_memory[0] == 2 and no_memory[1].startswith("stuck")
    assert with_memory == (3, "success")        # a working reflexion loop never trips it


# ---------------------------------------------------------------------------
# Exercise 4 - an actor that ignores reflections
#
# SIMULATED: a scripted actor cannot be persuaded, so "ignores reflections" is
# modelled as an actor that reads only the first line of its prompt.
#
# Minimum that works: stop appending reflections as history at the bottom and
# hoist the newest one to the top as a single imperative the actor can execute.
# Nothing in a prompt can force an actor that reads none of it; that case needs
# the stuck check from exercise 3, enforced outside the model.
# ---------------------------------------------------------------------------

class SkimmingActor:
    def __init__(self) -> None:
        self.attempt = [1, 2, 3]

    def act(self, prompt: str) -> list[int]:
        rule = re.match(r"RULE: move the sum by ([+-]\d+)", prompt)
        if rule:
            self.attempt = nudge(self.attempt, int(rule.group(1)))
        return self.attempt


def run_skimmer(hoist: bool, max_trials: int = 6) -> int | None:
    actor, reflector, memory = SkimmingActor(), SelfReflector(), EpisodicMemory()
    rule = ""
    for trial in range(1, max_trials + 1):
        prompt = rule + f"Pick three ints in 1..9 summing to {TARGET}.\nPast reflections:\n" + memory.as_prompt()
        attempt = actor.act(prompt)
        delta = sum(attempt) - TARGET
        if delta == 0:
            return trial
        memory.add(Reflection(trial, reflector.reflect(attempt, delta)))
        if hoist:
            rule = f"RULE: move the sum by {-delta:+d}\n"
    return None


def ex4_adversarial_actor() -> None:
    appended, hoisted = run_skimmer(hoist=False), run_skimmer(hoist=True)
    print(f"  reflections appended as history : solved at trial {appended}")
    print(f"  newest reflection hoisted       : solved at trial {hoisted}")
    assert appended is None and hoisted == 2


# ---------------------------------------------------------------------------
# Exercise 5 - the AlfWorld result (Reflexion paper, Section 4.1)
#
# The paper's number is 130 of 134 tasks solved by ReAct + Reflexion within 12
# trials, an absolute gain of 22% over the baseline; ReAct alone stops improving
# between trials 6 and 7. (The "130%" in the exercise text is that 130/134.)
#
# Key delta versus vanilla ReAct: AlfWorld failures are systematic, not random.
# The agent believes it holds an item it does not, or searches the same places
# again. A fresh ReAct retry repeats the same mistake. Reflexion carries a short
# written lesson into the next trial, so each failure mode is removed once.
#
# The simulation is a toy with made-up rates; only the shape is the point.
# ---------------------------------------------------------------------------

def solved_by_trial(use_memory: bool, tasks: int = 134, trials: int = 12, seed: int = 0) -> list[int]:
    task_rng, luck = random.Random(seed), random.Random(seed + 1)    # same tasks for both agents
    solved = [0] * trials
    for _ in range(tasks):
        traps = task_rng.choice([0, 0, 0, 1, 1, 2, 2, 3])   # systematic failure modes in this task
        for trial in range(trials):
            lucky_retry = not use_memory and trial > 0 and luck.random() < 0.03
            if traps == 0 or lucky_retry:
                for later in range(trial, trials):
                    solved[later] += 1
                break
            if use_memory and luck.random() < 0.7:          # a useful reflection removes one trap
                traps -= 1
    return solved


def ex5_alfworld_shape() -> None:
    react, reflexion = solved_by_trial(use_memory=False), solved_by_trial(use_memory=True)
    for trial in (1, 3, 6, 12):
        print(f"  after trial {trial:>2}: ReAct {react[trial - 1]:>3}/134   Reflexion {reflexion[trial - 1]:>3}/134")
    assert react[0] == reflexion[0]             # same first attempt
    assert reflexion[-1] > react[-1]


if __name__ == "__main__":
    print("Phase 14 - Lesson 03: Reflexion - exercises")
    for exercise in (ex1_scalar_evaluator, ex2_reflection_ttl, ex3_stuck_heuristic,
                     ex4_adversarial_actor, ex5_alfworld_shape):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
