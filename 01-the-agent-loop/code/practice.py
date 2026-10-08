"""Phase 14 - Lesson 01: The Agent Loop - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from main import AgentLoop, ToolCall, ToolRegistry, ToyLLM, Turn, build_demo_agent, calculator


# ---------------------------------------------------------------------------
# Exercises 1, 2 and 5 share one loop: several tool calls per assistant turn.
#
# Reply shape:  {"content": str, "calls": [{"id", "name", "args"}, ...]}
# ---------------------------------------------------------------------------

@dataclass
class MultiCallLoop:
    llm: ToyLLM
    tools: ToolRegistry
    max_turns: int = 8
    max_tool_calls_per_turn: int = 2                              # Ex 1
    stop_on: str = "no_tool_calls"                                # Ex 2: or "finish_tool"
    turns: list[dict[str, str]] = field(default_factory=list)     # per turn: call id -> observation
    completion_order: list[str] = field(default_factory=list)     # Ex 5

    def run(self, user_message: str) -> str:
        history = [Turn(kind="user", content=user_message)]
        for _ in range(self.max_turns):
            reply = self.llm.respond(history)
            calls = reply.get("calls", [])
            if not calls:
                if self.stop_on == "no_tool_calls":
                    return reply.get("content", "")
                continue                    # explicit-finish mode: plain text never ends the run
            for call in calls:
                if call["name"] == "finish":
                    return call["args"]["answer"]
            self.turns.append(self._execute(calls))
        return "budget exhausted"

    def _execute(self, calls: list[dict[str, Any]]) -> dict[str, str]:
        cap = self.max_tool_calls_per_turn
        # Every call id gets a result, including the ones we refuse to run.
        results = {c["id"]: f"error: skipped, over the cap of {cap} tool calls per turn; re-issue it"
                   for c in calls[cap:]}
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(self.tools.dispatch, ToolCall(c["name"], c["args"])): c["id"]
                       for c in calls[:cap]}
            for future in as_completed(futures):        # finish order, not issue order
                call_id = futures[future]
                self.completion_order.append(call_id)
                results[call_id] = future.result()      # correlated by id, never by position
        return results


# ---------------------------------------------------------------------------
# Exercise 1 - max_tool_calls_per_turn
#
# What breaks if the model issues three calls and only two run: the third call
# has no result. Providers reject the next request (every tool call needs a
# matching result), and a lenient loop is worse - the model assumes the call
# happened. So the cap must answer the extra calls with an error observation.
# ---------------------------------------------------------------------------

def ex1_cap() -> None:
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    three = [{"id": f"c{i}", "name": "calculator", "args": {"expr": f"{i} * 10"}} for i in (1, 2, 3)]
    script = [{"calls": three}, {"calls": [three[2]]}, {"content": "10, 20 and 30"}]
    loop = MultiCallLoop(ToyLLM(script), tools, max_tool_calls_per_turn=2)
    final = loop.run("multiply 1, 2 and 3 by ten")

    for i, turn in enumerate(loop.turns, 1):
        print(f"  turn {i}: {dict(sorted(turn.items()))}")
    assert set(loop.turns[0]) == {"c1", "c2", "c3"}             # nothing left unanswered
    assert loop.turns[0]["c3"].startswith("error: skipped")
    assert loop.turns[1] == {"c3": "30"}                        # re-issued and executed
    assert final == "10, 20 and 30"


# ---------------------------------------------------------------------------
# Exercise 2 - "no tool calls -> done" versus an explicit finish tool
#
# The implicit rule ends the run on any text-only turn, including narration
# like "let me check first". An explicit finish tool is safer against that
# early-termination bug because stopping becomes a deliberate action; its own
# risk is never stopping, which the turn budget covers either way.
# ---------------------------------------------------------------------------

def ex2_stop_paths() -> None:
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    script = [
        {"content": "Let me work that out first."},                                  # narration only
        {"calls": [{"id": "c1", "name": "calculator", "args": {"expr": "120 + 18"}}]},
        {"calls": [{"id": "c2", "name": "finish", "args": {"answer": "138"}}]},
    ]
    question = "what is 120 plus 18?"
    implicit = MultiCallLoop(ToyLLM(script), tools).run(question)
    explicit = MultiCallLoop(ToyLLM(script), tools, stop_on="finish_tool").run(question)

    print(f"  no_tool_calls -> done : {implicit!r}")
    print(f"  finish tool           : {explicit!r}")
    assert implicit == "Let me work that out first."    # stopped before doing any work
    assert explicit == "138"


# ---------------------------------------------------------------------------
# Exercise 3 - malformed arguments, recovered through an error observation
#
# main.ToolRegistry.dispatch already turns bad arguments into an "error: ..."
# observation instead of raising. What was missing is a model that makes the
# mistake and reads the observation. FlakyLLM is that model.
# ---------------------------------------------------------------------------

class FlakyLLM(ToyLLM):
    """Corrupts every second action; resends it correctly after an error observation."""

    def __init__(self, script: list[dict[str, Any]]) -> None:
        super().__init__(script)
        self.actions_sent = 0

    def respond(self, history: list[Turn]) -> dict[str, Any]:
        last = history[-1]
        if last.kind == "action" and (last.observation or "").startswith("error:"):
            self.cursor -= 1                        # retry the entry that just failed
            return super().respond(history)
        entry = super().respond(history)
        if entry["kind"] != "action":
            return entry
        self.actions_sent += 1
        if self.actions_sent % 4 == 2:              # misspelled argument name
            return {**entry, "args": {k + "_": v for k, v in entry["args"].items()}}
        if self.actions_sent % 4 == 0:              # a JSON string where a dict belongs
            return {**entry, "args": json.dumps(entry["args"])}
        return entry


def ex3_malformed_args() -> None:
    agent = build_demo_agent()
    agent.llm = FlakyLLM(agent.llm.script)
    final = agent.run("What is 120 plus 15% tax, stored in kv?")
    errors = [t.observation for t in agent.history if t.kind == "action" and t.observation.startswith("error:")]

    for obs in errors:
        print(f"  fed back: {obs[:70]}")
    print(f"  final   : {final}")
    assert len(errors) == 2
    assert final == "the total including 15% tax is 138.0"      # same answer as the clean run


# ---------------------------------------------------------------------------
# Exercise 4 - a real Responses API call, thoughts on the reasoning channel
#
# What changes in the transcript: the thought turns disappear. Reasoning comes
# back as separate opaque items that the loop cannot read or edit; its only job
# is to replay them, together with the tool call they belong to, on the next
# request. The control flow - observe, think, act - is untouched.
#
# ResponsesLLM needs the openai package and an OPENAI_API_KEY. Without a key,
# ex4 checks the transcript shape with the toy model only.
# ---------------------------------------------------------------------------

CALCULATOR_SCHEMA = {
    "type": "function", "name": "calculator",
    "description": "Evaluate an arithmetic expression such as '120 * 0.15'.",
    "parameters": {"type": "object", "properties": {"expr": {"type": "string"}},
                   "required": ["expr"], "additionalProperties": False},
}


class ResponsesLLM:
    """Drop-in for ToyLLM backed by the OpenAI Responses API."""

    def __init__(self, model: str, tool_schemas: list[dict[str, Any]]) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.tool_schemas = tool_schemas
        self.items: list[Any] = []              # provider transcript: messages, reasoning, calls, outputs
        self.pending_call_id: str | None = None
        self.reasoning: list[str] = []          # summaries only; the raw reasoning stays opaque

    def respond(self, history: list[Turn]) -> dict[str, Any]:
        last = history[-1]
        if last.kind == "user":
            self.items.append({"role": "user", "content": last.content})
        elif last.kind == "action":
            self.items.append({"type": "function_call_output", "call_id": self.pending_call_id,
                               "output": last.observation or ""})
        response = self.client.responses.create(
            model=self.model, input=self.items, tools=self.tool_schemas,
            reasoning={"summary": "auto"},
            parallel_tool_calls=False,          # main.AgentLoop runs one action per turn
        )
        self.items += response.output           # reasoning items travel back with their tool call
        for item in response.output:
            if item.type == "reasoning":
                self.reasoning.append(" ".join(part.text for part in item.summary))
        for item in response.output:
            if item.type == "function_call":
                self.pending_call_id = item.call_id
                return {"kind": "action", "action": item.name, "args": json.loads(item.arguments)}
        return {"kind": "finish", "content": response.output_text}


def ex4_reasoning_channel() -> None:
    # Offline: same demo, thoughts moved out of the reply and into a side channel.
    agent = build_demo_agent()
    side_channel = [entry.pop("thought") for entry in agent.llm.script if "thought" in entry]
    final = agent.run("What is 120 plus 15% tax, stored in kv?")
    thoughts = [t.content for t in agent.history if t.kind == "thought"]

    print(f"  transcript thought turns : {thoughts}")
    print(f"  reasoning items held     : {len(side_channel)}")
    assert all(t == "" for t in thoughts) and len(side_channel) == 5
    assert final == "the total including 15% tax is 138.0"

    if not os.environ.get("OPENAI_API_KEY"):
        print("  live Responses API call  : set OPENAI_API_KEY to enable")
        return
    tools = ToolRegistry()
    tools.register("calculator", calculator)
    llm = ResponsesLLM(os.environ.get("OPENAI_MODEL", "gpt-5"), [CALCULATOR_SCHEMA])
    live = AgentLoop(llm=llm, tools=tools, max_turns=6)     # type: ignore[arg-type]
    print(f"  live answer              : {live.run('What is 120 plus 15% tax?')}")
    print(f"  live reasoning summaries : {llm.reasoning}")


# ---------------------------------------------------------------------------
# Exercise 5 - tool_use_id correlation
#
# Why every provider requires it: parallel calls finish in any order, and one
# turn can call the same tool twice. Position cannot tell those results apart;
# an id can, and it lets the API verify each call got exactly one result.
# ---------------------------------------------------------------------------

def ex5_tool_use_id() -> None:
    def lookup(key: str, delay: float) -> str:
        time.sleep(delay)
        return f"value-of-{key}"

    tools = ToolRegistry()
    tools.register("lookup", lookup)
    calls = [{"id": "toolu_slow", "name": "lookup", "args": {"key": "a", "delay": 0.2}},
             {"id": "toolu_fast", "name": "lookup", "args": {"key": "b", "delay": 0.0}}]
    loop = MultiCallLoop(ToyLLM([{"calls": calls}, {"content": "done"}]), tools)
    loop.run("look up a and b")

    print(f"  issued    : {[c['id'] for c in calls]}")
    print(f"  completed : {loop.completion_order}")
    print(f"  results   : {loop.turns[0]}")
    assert loop.completion_order == ["toolu_fast", "toolu_slow"]        # out of order
    assert loop.turns[0] == {"toolu_slow": "value-of-a", "toolu_fast": "value-of-b"}


if __name__ == "__main__":
    print("Phase 14 - Lesson 01: The Agent Loop - exercises")
    for exercise in (ex1_cap, ex2_stop_paths, ex3_malformed_args, ex4_reasoning_channel, ex5_tool_use_id):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall checks passed")
