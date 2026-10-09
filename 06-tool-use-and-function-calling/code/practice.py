"""Phase 14 - Lesson 06: Tool Use and Function Calling - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any, Callable, Literal

from main import ToolCall, ToolDef, ToolRegistry, ToolResult, _coerce, add, classify, multiply, validate

INT_PAIR = {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]}
STATUS = {"type": "object",
          "properties": {"status": {"type": "string", "enum": ["open", "closed", "pending"]}},
          "required": ["status"]}
SCORE = {"type": "object",
         "properties": {"p": {"type": "number", "minimum": 0, "maximum": 1}, "tags": {"type": "array"}},
         "required": ["p"]}
QUERY = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}


def lesson_registry(registry: ToolRegistry | None = None) -> ToolRegistry:
    """The three tools main.py registers inside main()."""
    registry = registry or ToolRegistry()
    registry.register(ToolDef("add", "Add two integers a and b. Use for any integer addition.", INT_PAIR, add))
    registry.register(ToolDef("multiply", "Multiply two integers a and b. Prefer multiplication over "
                              "looped addition.", INT_PAIR, multiply))
    registry.register(ToolDef("classify", "Classify a status as one of the allowed labels.", STATUS, classify))
    return registry


# ---------------------------------------------------------------------------
# Exercise 1 - a no-op tool, measured on a BFCL-style hallucination test
#
# SCRIPTED MODEL: forced_choice stands in for a model running under
# tool_choice="required". It must emit a call, so it takes the tool whose
# description shares the most content words with the prompt. The numbers
# describe this policy, not an LLM, and the prompts are mine, not BFCL's.
#
# With no escape hatch every irrelevant prompt becomes a hallucinated call.
# no_tool gives the model somewhere legal to go. It does not fix near-misses
# (a prompt that mentions "status" still looks like classify).
# ---------------------------------------------------------------------------

NO_TOOL = ToolDef("no_tool", "Call this when none of the other tools fits the request.",
                  {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]},
                  lambda reason: f"no tool used: {reason}")

RELEVANT = [("add the integers 4 and 9", "add"),
            ("do the integer addition of 12 and 30", "add"),
            ("multiply the integers 6 and 7", "multiply"),
            ("classify this status as one of the labels", "classify"),
            ("what is 3 times 5", "multiply")]
IRRELEVANT = ["what is the weather in Lisbon tomorrow",
              "translate good morning into French",
              "book a flight to Berlin for Friday",
              "summarise the status report for me",
              "who won the match last night"]


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def forced_choice(prompt: str, catalog: list[dict[str, Any]]) -> str:
    words = content_words(prompt)
    real = [t for t in catalog if t["name"] != "no_tool"]
    best = max(real, key=lambda t: len(words & content_words(t["description"])))
    has_escape = len(real) < len(catalog)
    if has_escape and not words & content_words(best["description"]):
        return "no_tool"
    return best["name"]


def ex1_no_op_tool() -> None:
    plain, with_escape = lesson_registry(), lesson_registry()
    with_escape.register(NO_TOOL)
    rows = {}
    for label, registry in (("without no_tool", plain), ("with no_tool   ", with_escape)):
        catalog = registry.catalog()
        hallucinated = sum(forced_choice(p, catalog) != "no_tool" for p in IRRELEVANT)
        correct = sum(forced_choice(p, catalog) == tool for p, tool in RELEVANT)
        rows[label] = (hallucinated, correct)
        print(f"  {label}: hallucinated calls {hallucinated}/{len(IRRELEVANT)}   "
              f"correct tool on relevant prompts {correct}/{len(RELEVANT)}")
    refusal = with_escape.dispatch(ToolCall("u1", "no_tool", {"reason": "no weather tool"}))
    print(f"  dispatching it       : {refusal.content}")
    assert rows["without no_tool"] == (5, 4)
    assert rows["with no_tool   "] == (1, 4)       # the "status report" near-miss survives
    assert refusal.ok


# ---------------------------------------------------------------------------
# Exercise 2 - int-as-string and float-as-string coercion
#
# main._coerce already does both, with int() and float(). Coercion starts to
# hide real bugs where it stops being lossless and unambiguous:
#   - int() accepts padding, underscores and non-ASCII digits: output drift you
#     would want to see, not absorb;
#   - "007" becomes 7: that value was an identifier and the schema is wrong;
#   - float() accepts "nan" and "inf", and nan slips through minimum/maximum
#     because every comparison with nan is False.
# Rule: coerce only the canonical form, and reject the rest loudly.
# ---------------------------------------------------------------------------

PLAIN_INT = re.compile(r"-?(0|[1-9][0-9]*)")


def strict_coerce(value: Any, schema: dict[str, Any]) -> tuple[Any, str | None]:
    """main._coerce minus the conversions that are lossy or ambiguous."""
    kind = schema.get("type")
    if kind == "integer" and isinstance(value, str) and not PLAIN_INT.fullmatch(value):
        return value, f"refusing to coerce {value!r} to integer: not a plain decimal"
    coerced, err = _coerce(value, schema)
    if err is None and kind == "number" and not math.isfinite(coerced):
        return value, f"refusing non-finite number {value!r}"
    return coerced, err


def ex2_coercion() -> None:
    cases = [("integer", "4"), ("integer", " 12 "), ("integer", "1_000"), ("integer", "007"),
             ("integer", "\u0663"), ("number", "0.5"), ("number", "nan"), ("number", "1e999")]
    accepted_by_main, accepted_by_strict = [], []
    for kind, raw in cases:
        loose, loose_err = _coerce(raw, {"type": kind})
        _, strict_err = strict_coerce(raw, {"type": kind})
        if loose_err is None:
            accepted_by_main.append(raw)
        if strict_err is None:
            accepted_by_strict.append(raw)
        print(f"  {kind:<7} {ascii(raw):<10} main -> {loose!r:<6} strict -> "
              f"{'ok' if strict_err is None else 'rejected'}")
    validated, errors = validate({"p": "nan"}, SCORE)
    print(f"  main.validate lets p='nan' past minimum=0/maximum=1: errors={errors}")
    assert len(accepted_by_main) == len(cases)
    assert accepted_by_strict == ["4", "0.5"]
    assert errors == [] and math.isnan(validated["p"])


# ---------------------------------------------------------------------------
# Exercise 3 - per-tool timeout and a circuit breaker
#
# How it changes recovery: an error stops being about one call and becomes
# information about the tool. "circuit open, 60s" tells the model that retrying
# is pointless, so it switches tools or tells the user instead of spending its
# turn budget on a dead endpoint. The timeout guarantees that every call
# produces an observation at all; a hung tool otherwise stalls the loop.
# Validation errors never trip the breaker: those are the model's mistake.
# ---------------------------------------------------------------------------

class GuardedRegistry(ToolRegistry):
    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 max_failures: int = 3, cooldown_s: float = 60.0) -> None:
        super().__init__()
        self.clock = clock
        self.max_failures = max_failures
        self.cooldown_s = cooldown_s
        self.failures: dict[str, int] = {}
        self.open_until: dict[str, float] = {}
        self._pool = ThreadPoolExecutor(max_workers=8)

    def dispatch(self, call: ToolCall) -> ToolResult:
        remaining = self.open_until.get(call.name, 0.0) - self.clock()
        if remaining > 0:
            return ToolResult(call.tool_use_id, False,
                              f"circuit open: {call.name} failed {self.max_failures} times in a row and is "
                              f"disabled for another {remaining:.0f}s; use a different tool or tell the user")
        tool = self._tools.get(call.name)
        future = self._pool.submit(super().dispatch, call)
        try:
            result = future.result(timeout=tool.timeout_s if tool else None)
        except FutureTimeout:
            # ponytail: the worker thread keeps running after we stop waiting for it;
            # run tools in a subprocess if they have to be killed.
            result = ToolResult(call.tool_use_id, False,
                                f"timeout: {call.name} gave no result within {tool.timeout_s}s")
        self._record(call.name, result)
        return result

    def _record(self, name: str, result: ToolResult) -> None:
        if result.ok:
            self.failures[name] = 0
        elif result.content.startswith(("timeout", "execution error")):
            self.failures[name] = self.failures.get(name, 0) + 1
            if self.failures[name] >= self.max_failures:
                self.open_until[name] = self.clock() + self.cooldown_s
                self.failures[name] = self.max_failures - 1     # half-open: one more failure re-opens


def ex3_timeout_and_breaker() -> None:
    now = [0.0]
    registry = GuardedRegistry(clock=lambda: now[0])
    upstream = {"calls": 0, "healthy": False}

    def search(q: str) -> str:
        upstream["calls"] += 1
        if not upstream["healthy"]:
            raise ConnectionError("upstream returned 503")
        return f"3 results for {q}"

    def crawl(q: str) -> str:
        time.sleep(0.3)
        return "too late"

    registry.register(ToolDef("search", "Search the index.", QUERY, search))
    registry.register(ToolDef("crawl", "Fetch a page.", QUERY, crawl, timeout_s=0.05))

    timed_out = registry.dispatch(ToolCall("t0", "crawl", {"q": "docs"}))
    print(f"  slow tool    : {timed_out.content}")
    results = [registry.dispatch(ToolCall(f"t{i}", "search", {"q": "docs"})) for i in range(1, 5)]
    for result in results:
        print(f"  search {result.tool_use_id}    : {result.content}")
    assert timed_out.content.startswith("timeout")
    assert [r.content.split(":")[0] for r in results] == ["execution error"] * 3 + ["circuit open"]
    assert upstream["calls"] == 3                       # the fourth call never reached the tool

    now[0] += 61
    upstream["healthy"] = True
    recovered = registry.dispatch(ToolCall("t5", "search", {"q": "docs"}))
    print(f"  after 61s    : {recovered.content}")
    assert recovered.ok and registry.failures["search"] == 0

    bad_args = [registry.dispatch(ToolCall(f"v{i}", "search", {})) for i in range(5)]
    assert all(r.content.startswith("validation error") for r in bad_args)
    assert registry.dispatch(ToolCall("t6", "search", {"q": "docs"})).ok       # breaker stayed closed


# ---------------------------------------------------------------------------
# Exercise 4 - a BFCL-style multi-turn run: 10 conversations, pass rate
#
# SYNTHETIC CONVERSATIONS and a SCRIPTED AGENT, in BFCL's style: the ten
# conversations are mine and RuleAgent is a regex policy. What carries over
# is the scoring idea from BFCL V3+: judge the state the conversation ends in,
# not the syntax of each call.
#
# The three failures are all the multi-turn kind the lesson describes: a
# reference to an earlier turn, an edit of the previous call, and "that"
# pointing past an unrelated call. Single-turn calls never fail here.
# ---------------------------------------------------------------------------

class RuleAgent:
    """One regex per tool; `that` means the previous observation."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self.last = ""

    def turn(self, text: str) -> ToolResult:
        text = text.lower().replace("that", self.last)
        if m := re.search(r"add (\S+) (?:and|to) (\S+)", text):
            call = ToolCall("u", "add", {"a": m[1], "b": m[2]})
        elif m := re.search(r"multiply (\S+) by (\S+)", text):
            call = ToolCall("u", "multiply", {"a": m[1], "b": m[2]})
        elif m := re.search(r"classify .* as (\w+)", text):
            call = ToolCall("u", "classify", {"status": m[1]})
        else:
            return ToolResult("u", False, "no tool call")
        result = self.registry.dispatch(call)
        self.last = result.content
        return result


CONVERSATIONS: list[tuple[list[str], str]] = [
    (["add 2 and 3", "multiply that by 4"], "20"),
    (["multiply 6 by 7", "add 8 to that"], "50"),
    (["add 10 and 5", "multiply that by 2", "add 1 to that"], "31"),
    (["multiply 3 by 3", "multiply that by 3"], "27"),
    (["add 1 and 1", "classify the ticket as open"], "classified as open"),
    (["multiply 2 by 5", "add that and that"], "20"),
    (["add 100 and 200", "multiply that by 0"], "0"),
    (["add 2 and 2", "multiply the first result by itself"], "16"),
    (["add 4 and 4", "actually make it 5 and 5"], "10"),
    (["multiply 7 by 8", "classify the ticket as closed", "add 1 to that"], "57"),
]


def ex4_multi_turn_eval() -> None:
    passed = 0
    for turns, expected in CONVERSATIONS:
        agent = RuleAgent(lesson_registry())
        final = [agent.turn(t) for t in turns][-1]
        ok = final.ok and final.content == expected
        passed += ok
        if not ok:
            print(f"  FAIL {turns[-1]!r}: wanted {expected!r}, got {final.content[:48]!r}")
    print(f"  pass rate: {passed}/{len(CONVERSATIONS)}")
    assert passed == 7


# ---------------------------------------------------------------------------
# Exercise 5 - port the validator to Pydantic
#
# Needs pydantic (pip install pydantic).
# Measured with pydantic 2.13 on the eight cases below:
#   - Pydantic caught two things the toy let through: "nan" against a
#     minimum/maximum, and wrong item types inside an array (the toy only
#     checks "is a list").
#   - The toy caught one thing Pydantic let through: True for a number field.
#     Pydantic's default mode converts it to 1.0; strict mode is needed to
#     refuse it.
#   - Both accept "1_000" as an integer, so the coercion worry from exercise 2
#     survives the port.
# Pydantic also reports errors per field with a location, not as one string.
# ---------------------------------------------------------------------------

def ex5_pydantic_port() -> str | None:
    try:
        from pydantic import BaseModel, ConfigDict, Field, ValidationError
    except ImportError:
        print("  needs pydantic: pip install pydantic")
        return "pydantic"

    class Args(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class AddArgs(Args):
        a: int
        b: int

    class ClassifyArgs(Args):
        status: Literal["open", "closed", "pending"]

    class ScoreArgs(Args):
        p: float = Field(ge=0, le=1)
        tags: list[str] = []

    cases = [
        ("string for int", INT_PAIR, AddArgs, {"a": "4", "b": 5}),
        ("underscore int", INT_PAIR, AddArgs, {"a": "1_000", "b": 1}),
        ("float for int", INT_PAIR, AddArgs, {"a": 4.7, "b": 1}),
        ("unknown field", INT_PAIR, AddArgs, {"a": 1, "b": 2, "c": 3}),
        ("bad enum", STATUS, ClassifyArgs, {"status": "in_progress"}),
        ("nan against bounds", SCORE, ScoreArgs, {"p": "nan"}),
        ("array item types", SCORE, ScoreArgs, {"p": 0.5, "tags": [1, None]}),
        ("bool for number", SCORE, ScoreArgs, {"p": True}),
    ]
    verdicts = {}
    for label, schema, model, args in cases:
        toy_ok = not validate(args, schema)[1]
        try:
            model.model_validate(args)
            pydantic_ok = True
        except ValidationError:
            pydantic_ok = False
        verdicts[label] = (toy_ok, pydantic_ok)
        note = "" if toy_ok == pydantic_ok else "   <- they disagree"
        print(f"  {label:<19} toy: {'accept' if toy_ok else 'reject'}   "
              f"pydantic: {'accept' if pydantic_ok else 'reject'}{note}")
    disagreements = {label: verdict for label, verdict in verdicts.items() if verdict[0] != verdict[1]}
    assert disagreements == {"nan against bounds": (True, False), "array item types": (True, False),
                             "bool for number": (False, True)}
    assert verdicts["underscore int"] == (True, True)
    return None


if __name__ == "__main__":
    print("Phase 14 - Lesson 06: Tool Use and Function Calling - exercises")
    outcomes = []
    for exercise in (ex1_no_op_tool, ex2_coercion, ex3_timeout_and_breaker,
                     ex4_multi_turn_eval, ex5_pydantic_port):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
