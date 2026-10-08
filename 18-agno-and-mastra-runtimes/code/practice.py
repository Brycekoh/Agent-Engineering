"""Phase 14 - Lesson 18: Production Agent Runtimes (Agno and Mastra) - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
import timeit
from pathlib import Path
from typing import Any

from main import AgnoAgent, MastraAgent, MastraTool


def load_lesson_01() -> Any:
    """The lesson 01 loop, loaded from its own folder, for the two port exercises."""
    path = Path(__file__).resolve().parents[2] / "01-the-agent-loop" / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("lesson01_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Exercise 1 - the lesson 01 ReAct loop on Agno
#
# agno_port() needs the agno package and a model key. It follows Agno's
# "first agent" page; `model` is a "provider:model" string.
#
# What disappeared: everything that was plumbing. The turn loop, the message
# buffer, the tool registry and its dispatch, the stop condition and the
# scripted model are all inside Agent now. What stayed: the tool functions,
# unchanged, and the instructions. Two things stayed and got more important:
# docstrings, which are now the tool descriptions the model reads, and tools
# returning error strings instead of raising, because you no longer own the
# loop that would have caught the exception. The count below is measured on
# lesson 01's source.
# ---------------------------------------------------------------------------

def agno_port(model: str) -> Any:
    from agno.agent import Agent

    lesson_01 = load_lesson_01()
    store = lesson_01.KVStore()

    def calculator(expr: str) -> str:
        """Evaluate an arithmetic expression such as '120 * 0.15'."""
        return lesson_01.calculator(expr)

    def kv_set(key: str, value: str) -> str:
        """Store a value under a key for later steps."""
        return store.set(key, value)

    def kv_get(key: str) -> str:
        """Read back a value stored with kv_set."""
        return store.get(key)

    return Agent(name="tax-helper", model=model, tools=[calculator, kv_set, kv_get],
                 instructions=["Use the calculator for all arithmetic.",
                               "Store intermediate values with kv_set before the final answer."])
    # then: agent.print_response("What is 120 plus 15% tax, stored in kv?")


def ex1_agno_port() -> None:
    lesson_01 = load_lesson_01()
    lines = lambda *objects: sum(len(inspect.getsource(obj).splitlines()) for obj in objects)
    plumbing = lines(lesson_01.ToolRegistry, lesson_01.ToyLLM, lesson_01.AgentLoop, lesson_01.Turn, lesson_01.ToolCall)
    kept = lines(lesson_01.calculator, lesson_01.KVStore)
    print(f"  lesson 01 lines that become the framework's job : {plumbing}")
    print(f"  lesson 01 lines that survive the port           : {kept}")
    assert lesson_01.calculator("120 * 0.15") == "18.0"         # the surviving tools still stand alone
    assert plumbing > 2 * kept
    try:
        import agno  # noqa: F401
        print("  agno is installed: call agno_port('provider:model') with a model key to run it")
    except ImportError:
        print("  Agno port: pip install agno, then call agno_port('provider:model')")


# ---------------------------------------------------------------------------
# Exercise 2 - the same loop on Mastra, and what Zod changes
#
# The Mastra port is TypeScript: see practice.ts next to this file. It needs
# Node, @mastra/core, zod and a model key.
#
# What changed in tool typing: in lesson 01, and in main.MastraTool here, the
# input schema is decoration. main.MastraAgent.run calls tool.fn(**args)
# without looking at it, so a wrong type flows straight through and a missing
# argument crashes the run. With Zod the schema is the one source of truth:
# it validates the model's arguments at runtime, it is what gets sent to the
# model as the tool's JSON schema, and it gives execute() its argument type at
# compile time. TypedMastraAgent adds the runtime third of that.
# ---------------------------------------------------------------------------

JSON_TYPES = {"string": str, "number": (int, float), "boolean": bool}


class TypedMastraAgent(MastraAgent):
    def run(self, prompt: str, tool_calls: list[tuple[str, dict[str, Any]]]) -> tuple[str, list[tuple[str, str]]]:
        checked = []
        errors: list[tuple[str, str]] = []
        for name, args in tool_calls:
            tool = next((t for t in self.tools if t.name == name), None)
            problem = self._problem(tool, args) if tool else None
            if problem:
                errors.append((name, f"validation error: {problem}"))
            else:
                checked.append((name, args))
        output, trace = super().run(prompt, checked)
        return output, errors + trace

    @staticmethod
    def _problem(tool: MastraTool, args: dict[str, Any]) -> str | None:
        schema = tool.input_schema
        for field in schema.get("required", []):
            if field not in args:
                return f"{field} is required"
        for field, value in args.items():
            expected = schema["properties"].get(field, {}).get("type")
            if expected is None:
                return f"unknown field {field}"
            if isinstance(value, bool) != (expected == "boolean") or not isinstance(value, JSON_TYPES[expected]):
                return f"{field} must be a {expected}, got {type(value).__name__}"
        return None


def ex2_typed_tools() -> None:
    search = MastraTool("search", {"type": "object", "properties": {"query": {"type": "string"}},
                                   "required": ["query"]}, lambda query: f"results for {query!r}")
    untyped = MastraAgent("untyped", "search and cite", tools=[search])
    typed = TypedMastraAgent("typed", "search and cite", tools=[search])

    _, loose = untyped.run("research", [("search", {"query": 42})])
    print(f"  no validation, wrong type  : {loose[0][1]}   <- accepted")
    try:
        untyped.run("research", [("search", {})])
        raise AssertionError("a missing argument should have crashed the untyped agent")
    except TypeError as crash:
        print(f"  no validation, missing arg : run crashes with TypeError ({str(crash)[:44]}...)")
    _, strict = typed.run("research", [("search", {"query": 42}), ("search", {}), ("search", {"query": "zod"})])
    for name, result in strict:
        print(f"  schema enforced            : {result}")
    assert loose[0][1] == "results for 42"
    assert [r.startswith("validation error") for _, r in strict] == [True, True, False]
    assert (Path(__file__).parent / "practice.ts").exists()


# ---------------------------------------------------------------------------
# Exercise 3 - agent instantiation latency
#
# Measured on this machine with main's stand-in classes, not on Agno itself.
# Does a 2 microsecond constructor matter? Only when construction is a real
# share of the request. With one model call per request it is a millionth of
# the latency. It starts to matter when you build thousands of agents per
# second with no model call behind each (evaluation fan-out, simulations), or
# when a framework's constructor does real work, like the heavy one below.
# ---------------------------------------------------------------------------

MODEL_CALL_S = 0.8          # a modest single LLM round trip


def ex3_instantiation_latency() -> None:
    schema = {"type": "object", "properties": {f"arg{i}": {"type": "string"} for i in range(8)}}
    light = lambda: AgnoAgent(name="a", fn=str)
    with_tools = lambda: MastraAgent("a", "search, summarise, cite",
                                     tools=[MastraTool(f"tool{i}", schema, str) for i in range(10)])
    heavy = lambda: MastraAgent("a", "x", tools=[MastraTool(f"tool{i}", {**schema, "copy": [dict(schema) for _ in range(50)]}, str)
                                                 for i in range(100)])
    results = {}
    for label, build, number in (("bare agent           ", light, 20000), ("agent + 10 tools     ", with_tools, 2000),
                                 ("heavy constructor    ", heavy, 50)):
        seconds = min(timeit.repeat(build, number=number, repeat=3)) / number
        results[label] = seconds
        print(f"  {label}: {seconds * 1e6:>9.2f} us   = {seconds / (seconds + MODEL_CALL_S):.5%} of a request "
              f"with one {MODEL_CALL_S}s model call")
    assert results["bare agent           "] < results["heavy constructor    "]
    assert results["bare agent           "] / MODEL_CALL_S < 1e-4


# ---------------------------------------------------------------------------
# Exercise 4 - migrating a CrewAI project to Agno: what breaks
#
# A DESIGN, from the two frameworks' docs. No real project was migrated, so
# check each row against the Agno version you install. The mechanical half
# (roles to instructions, tasks to ordered steps) is the function below.
# ---------------------------------------------------------------------------

MIGRATION = [
    ("Agent(role, goal, backstory)", "Agent(name, instructions)",
     "No backstory field. Its text has to be folded into instructions, and tone changes with it."),
    ("Task(description, expected_output, context)", "a step in a workflow, or a prompt",
     "There is no Task object. Output contracts and the context= wiring between tasks must be rebuilt by hand."),
    ("Task(output_pydantic=Model)", "a schema on the agent's output",
     "Validation moves from the task to the agent; per-task retries on bad JSON are not the same mechanism."),
    ("Crew(process=Process.sequential)", "ordered workflow steps",
     "Straightforward, but the implicit passing of the previous output becomes explicit."),
    ("Crew(process=Process.hierarchical)", "a team with a leader",
     "Routing is done by a different prompt and loop. Same roles, different decisions: re-run your evals."),
    ("crew.kickoff(inputs={'topic': ...})", "a plain call with a formatted string",
     "No {placeholder} interpolation into agent and task text."),
    ("Crew(memory=True)", "sessions and memory on a database",
     "Stored memories do not migrate, and entity memory has no one-line switch."),
    ("BaseTool subclass / @tool", "plain functions or toolkits",
     "args_schema classes and _run methods are rewritten; tool state moves into closures or toolkit objects."),
    ("a script that calls kickoff()", "a stateless FastAPI service",
     "Anything kept in process memory between kickoffs is gone on the next request. State must live in the db."),
]


def crewai_agent_to_agno(agent: dict[str, str], tasks: list[dict[str, str]]) -> dict[str, Any]:
    return {"name": agent["role"].lower().replace(" ", "-"),
            "instructions": [f"You are the {agent['role']}. Goal: {agent['goal']}.", agent["backstory"]],
            "steps": [f"{t['description']} Expected output: {t['expected_output']}" for t in tasks]}


def ex4_crewai_to_agno() -> None:
    for old, new, breaks in MIGRATION:
        print(f"  {old:<44} -> {new}\n      breaks: {breaks}")
    converted = crewai_agent_to_agno(
        {"role": "Researcher", "goal": "find 3 credible sources", "backstory": "Former librarian. Terse."},
        [{"description": "Research the topic.", "expected_output": "3 sources"}])
    assert converted["name"] == "researcher" and "Former librarian" in converted["instructions"][1]
    assert converted["steps"] == ["Research the topic. Expected output: 3 sources"]
    assert len(MIGRATION) == 9


# ---------------------------------------------------------------------------
# Exercise 5 - Mastra's ee/ license and an open-source fork
#
# Read from LICENSE.md and ee/LICENSE on Mastra's main branch on 2026-10-07.
# This is a reading of the text, not legal advice; licences change.
# ---------------------------------------------------------------------------

EE_LICENSE_NOTES = [
    "The repository is Apache 2.0 except for directories named ee/, which the top-level licence carves out.",
    "Code under ee/ may be used and modified for development and testing only.",
    "Production use of ee/ code needs a paid agreement and a licence key.",
    "ee/ code may not be copied, redistributed, sublicensed, sold, or offered to others as a hosted service.",
    "Rights in modifications to ee/ code stay with the licensor.",
]
FORK_CONSEQUENCES = [
    "A fork can redistribute the Apache 2.0 parts, keeping the licence and notices.",
    "It cannot ship the ee/ directories, so they have to be stripped, and anything importing from them breaks.",
    "It cannot offer the enterprise features to others as a hosted service.",
    "Before forking, list which features you rely on live under ee/ (auth, the agent builder, the editor).",
]


def ex5_ee_license() -> None:
    print("  what the licences say:")
    for note in EE_LICENSE_NOTES:
        print(f"    - {note}")
    print("  what that means for an open-source fork:")
    for consequence in FORK_CONSEQUENCES:
        print(f"    - {consequence}")


if __name__ == "__main__":
    print("Phase 14 - Lesson 18: Agno and Mastra - exercises")
    for exercise in (ex1_agno_port, ex2_typed_tools, ex3_instantiation_latency, ex4_crewai_to_agno, ex5_ee_license):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall checks passed")
