"""Phase 14 - Lesson 05: Self-Refine and CRITIC - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import random
from typing import Callable

from main import Attempt, feedback_self, generate, run_loop, verify_external

Verifier = Callable[[str], tuple[str, bool]]


def run_with(verify: Verifier, max_iters: int = 4) -> list[Attempt]:
    """main.run_loop with the verifier passed in."""
    history: list[Attempt] = []
    output = generate("world facts", history)
    for i in range(1, max_iters + 1):
        critique, ok = verify(output)
        history.append(Attempt(i, output, critique, ok))
        if ok:
            break
        output = generate("world facts", history)
    return history


def wrong_facts(output: str) -> int:
    """Ground truth for scoring: known-wrong claims still present in the output."""
    text = output.lower()
    return ("paris" in text and "germany" in text) + ("everest" in text and "europe" in text)


# ---------------------------------------------------------------------------
# Exercise 1 - max_iterations=1
#
# With one iteration CRITIC cannot correct anything: generate once, verify
# once, stop. It still helps, as a guardrail rather than a refiner: the caller
# gets a grounded "do not ship this" that names the wrong claim. Fixing takes
# one pass per error plus a final pass that verifies clean - three here.
# ---------------------------------------------------------------------------

def ex1_single_iteration() -> None:
    one = run_loop("world facts", use_critic=True, max_iters=1)
    full = run_loop("world facts", use_critic=True, max_iters=4)
    print(f"  max_iters=1 : verified={one[-1].verified}, wrong facts left={wrong_facts(one[-1].output)}")
    print(f"                critique: {one[-1].critique}")
    print(f"  max_iters=4 : verified={full[-1].verified} after {len(full)} iterations")
    assert len(one) == 1 and not one[-1].verified
    assert "contradicts reference data" in one[-1].critique
    assert full[-1].verified and len(full) == 3


# ---------------------------------------------------------------------------
# Exercise 2 - a verifier with 30% false positives
#
# What the loop does: it reaches the correct answer at iteration 3 as before,
# then a random flag sends it round again. The generator has nothing to fix, so
# the extra iterations are pure cost, and about 0.3 * 0.3 = 9% of runs hit the
# cap and report failure on an answer that is in fact correct. (A real LLM
# generator can do worse and "fix" a correct bullet to satisfy the critique.)
#
# Mitigation: only believe a failure the verifier repeats. A real error fails
# every time; a random flag fails twice in a row only 9% of the time.
# ---------------------------------------------------------------------------

def noisy(rng: random.Random, false_positive: float = 0.3) -> Verifier:
    def verify(output: str) -> tuple[str, bool]:
        critique, ok = verify_external(output)
        if ok and rng.random() < false_positive:
            return "verifier: bullet 3 looks unsupported", False
        return critique, ok
    return verify


def confirmed(verify: Verifier, times: int = 2) -> Verifier:
    def wrapper(output: str) -> tuple[str, bool]:
        for _ in range(times):
            critique, ok = verify(output)
            if ok:
                break
        return critique, ok
    return wrapper


def loop_stats(make_verifier: Callable[[random.Random], Verifier], runs: int = 2000) -> tuple[float, float, int]:
    histories = [run_with(make_verifier(random.Random(seed))) for seed in range(runs)]
    gave_up = sum(not h[-1].verified for h in histories) / runs
    mean_iters = sum(len(h) for h in histories) / runs
    still_wrong = sum(wrong_facts(h[-1].output) for h in histories)
    return gave_up, mean_iters, still_wrong


def ex2_noisy_verifier() -> None:
    stats = {
        "clean verifier          ": loop_stats(lambda rng: verify_external),
        "30% false positives     ": loop_stats(noisy),
        "same, failures confirmed": loop_stats(lambda rng: confirmed(noisy(rng))),
    }
    for label, (gave_up, mean_iters, still_wrong) in stats.items():
        print(f"  {label}: gave up {gave_up:>5.1%}   mean iterations {mean_iters:.2f}   wrong facts left {still_wrong}")
    clean, noisy_stats, confirmed_stats = stats.values()
    assert clean[0] == 0 and clean[1] == 3
    assert 0.06 < noisy_stats[0] < 0.12
    assert confirmed_stats[0] < noisy_stats[0] / 3
    assert noisy_stats[2] == 0          # every "failure" was a correct answer


# ---------------------------------------------------------------------------
# Exercise 3 - generator and critic on different models
#
# SIMULATED: no models on this machine, so the critics are scripts and the
# ranking below is true by construction. It illustrates the mechanism only.
#
# A critic from a different model breaks the rubber stamp, because its blind
# spots differ and its critique is specific enough to act on. It does not beat
# a grounded verifier: a small critic approves whatever it does not know, and
# approves it confidently, which is worse than a loop that fails to converge.
# ---------------------------------------------------------------------------

def small_model_critic(output: str) -> tuple[str, bool]:
    """A cheaper critic: knows capitals, has never heard of Everest."""
    if "Paris" in output and "Germany" in output:
        return "critic: Paris is not the capital of germany", False
    return "no issues", True


def ex3_different_model_critic() -> None:
    runs = {
        "same model critiques itself  ": run_with(feedback_self),
        "small model critiques big one": run_with(small_model_critic),
        "external verifier (CRITIC)   ": run_with(verify_external),
    }
    for label, history in runs.items():
        last = history[-1]
        print(f"  {label}: says {'ok' if last.verified else 'not ok'} after {len(history)} iterations, "
              f"wrong facts left {wrong_facts(last.output)}")
    left = [wrong_facts(h[-1].output) for h in runs.values()]
    assert left == [2, 1, 0]
    assert runs["small model critiques big one"][-1].verified        # confident and still wrong


# ---------------------------------------------------------------------------
# Exercise 4 - CRITIC Section 3: the verification tools
#
# Section 3 describes verify-then-correct: generate, let the model call a tool
# to check its own output, turn the tool result into a critique, correct, and
# repeat until the critique passes or the iteration cap is hit. The paper uses
# one tool per task rather than naming categories; grouped by kind, they are:
# ---------------------------------------------------------------------------

VERIFICATION_TOOLS = {
    "knowledge lookup": ("Google search, for free-form question answering",
                         "does the claim match what a retrieved page says?"),
    "code execution": ("Python interpreter, for mathematical program synthesis",
                       "does the program run, and is its result sensible?"),
    "scoring API": ("Perspective API, for toxicity reduction",
                    "is the toxicity score of the text under the threshold?"),
}


def ex4_verification_tools() -> None:
    for kind, (example, check) in VERIFICATION_TOOLS.items():
        print(f"  {kind:<16}: {example}\n  {'':<16}  {check}")
    # main.verify_external is the first kind: a lookup against reference facts.
    assert verify_external("- Paris is the capital of Germany\n- b\n- c")[1] is False
    assert len(VERIFICATION_TOOLS) == 3


# ---------------------------------------------------------------------------
# Exercise 5 - OpenAI Agents SDK output_guardrails as CRITIC's verifier
#
# What it gets right: the check runs outside the generator, on the final
# output, can call tools, and blocks a bad answer instead of trusting it.
# What it gets wrong, measured against CRITIC: a tripwire is verify-then-stop.
# It raises OutputGuardrailTripwireTriggered and ends the run; nothing feeds
# the finding back to the generator. The correct half of the loop - catch the
# tripwire, pass its info on as the critique, try again - is left to the
# caller. A guardrail that is only another LLM prompt is also Self-Refine, not
# CRITIC: it needs a tool behind it to be grounded.
#
# The stand-in below has the SDK's shape, not its package.
# ---------------------------------------------------------------------------

class OutputGuardrailTripwireTriggered(Exception):
    pass


def fact_guardrail(output: str) -> None:
    critique, ok = verify_external(output)
    if not ok:
        raise OutputGuardrailTripwireTriggered(critique)


def run_guarded(max_retries: int) -> tuple[str, int]:
    history: list[Attempt] = []
    for attempt in range(1, max_retries + 2):
        output = generate("world facts", history)
        try:
            fact_guardrail(output)
            return output, attempt
        except OutputGuardrailTripwireTriggered as tripwire:
            if attempt > max_retries:
                raise
            history.append(Attempt(attempt, output, str(tripwire), False))   # the part the SDK leaves to you
    raise AssertionError("unreachable")


def ex5_output_guardrails() -> None:
    try:
        run_guarded(max_retries=0)
        raise AssertionError("the guardrail should have tripped")
    except OutputGuardrailTripwireTriggered as tripwire:
        print(f"  guardrail alone      : run ends with tripwire -> {tripwire}")
    output, attempts = run_guarded(max_retries=3)
    print(f"  tripwire fed back in : clean output on attempt {attempts}")
    assert attempts == 3 and wrong_facts(output) == 0


if __name__ == "__main__":
    print("Phase 14 - Lesson 05: Self-Refine and CRITIC - exercises")
    for exercise in (ex1_single_iteration, ex2_noisy_verifier, ex3_different_model_critic,
                     ex4_verification_tools, ex5_output_guardrails):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
