"""Phase 14 - Lesson 15: Role-Based Agent Teams (CrewAI) - exercises.

Solves the seven exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
main.py imports numpy, so this file needs it too. Exercise 5 also uses pydantic.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace
from typing import Any, Callable

try:
    from main import (Agent, Flow, HierarchicalCrew, Memory, SequentialCrew, Task, _editor, _researcher,
                      _writer, build_agents, search)
except ModuleNotFoundError as missing:
    raise SystemExit(f"this lesson's main.py needs {missing.name}: pip install {missing.name}")

TOPIC = "agent engineering 2026"


def sequential_crew(agents: tuple[Agent, Agent, Agent], memory: Memory | None = None) -> SequentialCrew:
    researcher, writer, editor = agents
    return SequentialCrew(agents=list(agents), memory=memory, tasks=[
        Task("research the topic", "3 sources", researcher),
        Task("write a draft", "3 paragraphs", writer),
        Task("edit to final brief", "800 words", editor),
    ])


# ---------------------------------------------------------------------------
# Exercise 1 - the Sequential crew as a Flow
#
# Touchpoints where variability drops, counted from how CrewAI behaves (the
# toy is deterministic either way): in a Crew the framework and the model
# settle the order of work, what each task is handed, and whether the
# researcher calls its tool. In a Flow those four are lines of code; only the
# three generations inside the steps are left to the model.
#
# Where readability dropped: the order is no longer one list you can read. It
# is implied by topic strings matched across decorators, and a mismatch is
# not an error. The second flow below has one typo and stops early in silence.
# ---------------------------------------------------------------------------

TOUCHPOINTS = [   # (decision, who makes it in a Crew, who makes it in a Flow)
    ("which step runs next", "process", "code"),
    ("what the writer is handed", "framework", "code"),
    ("what the editor is handed", "framework", "code"),
    ("whether the researcher searches", "model", "code"),
    ("the research text", "model", "model"),
    ("the draft text", "model", "model"),
    ("the final text", "model", "model"),
]


def build_flow(edit_listens_to: str = "drafted") -> Flow:
    flow = Flow()

    @flow.start
    def research(topic: str) -> tuple[str, str]:
        return "researched", _researcher(topic, [search], None)

    @flow.listen("researched")
    def draft(prior: str) -> tuple[str, str]:
        return "drafted", _writer(prior, [], None)

    @flow.listen(edit_listens_to)
    def edit(prior: str) -> tuple[str, str]:
        return "edited", _editor(prior, [], None)

    return flow


def ex1_crew_to_flow() -> None:
    crew_final = sequential_crew(build_agents()).kickoff({"topic": TOPIC})[-1]
    trace = build_flow().kickoff(TOPIC)
    fixed = sum(flow == "code" for _, _, flow in TOUCHPOINTS)
    print(f"  flow steps        : {[step for step, _, _ in trace]}")
    print(f"  same final output : {crew_final == '[editor] ' + trace[-1][2]}")
    print(f"  touchpoints       : {len(TOUCHPOINTS)} in the crew, {fixed} of them fixed in code by the flow")
    assert fixed == 4
    assert crew_final == "[editor] " + trace[-1][2]

    broken = build_flow(edit_listens_to="draftd").kickoff(TOPIC)
    print(f"  one typo in a topic: flow ends after {[step for step, _, _ in broken]}, no error raised")
    assert [step for step, _, _ in broken] == ["start", "draft"] and broken[-1][2].startswith("draft")


# ---------------------------------------------------------------------------
# Exercise 2 - entity memory that persists across kickoffs
#
# main.Memory has an entity store and a write method but nothing reads it.
# recall_entities is the read: facts are fetched by the entity's name, not by
# similarity. That matters here in particular, because main's long-term store
# embeds text as seeded random vectors, so its similarity ranking means
# nothing. The same Memory object is handed to each new crew, which is what
# makes the facts outlive a kickoff.
# ---------------------------------------------------------------------------

def recall_entities(memory: Memory, text: str) -> dict[str, dict[str, str]]:
    return {entity: facts for entity, facts in memory.entity.items() if entity.lower() in text.lower()}


def _intake(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    """Scripted entity extraction: '<Customer> is on the <plan> plan and prefers <channel>'."""
    found = re.search(r"(\w+) is on the (\w+) plan and prefers (\w+)", str(prior))
    if found and memory is not None:
        memory.write_entity(found[1], "plan", found[2])
        memory.write_entity(found[1], "contact", found[3])
    return f"noted: {prior}"


def _account_manager(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    facts = recall_entities(memory, str(prior)) if memory is not None else {}
    known = "; ".join(f"{entity}: " + ", ".join(f"{k}={v}" for k, v in sorted(f.items()))
                      for entity, f in facts.items())
    return f"reply to '{prior}' using [{known or 'no stored facts'}]"


def ex2_entity_memory() -> None:
    memory = Memory()
    intake = Agent("intake", "record customer facts", "careful note taker", _intake)
    manager = Agent("account manager", "answer with the right account context", "knows every account", _account_manager)

    for note in ("Acme is on the enterprise plan and prefers email", "Globex is on the starter plan and prefers phone"):
        SequentialCrew([intake], [Task("record the note", "a note", intake)], memory).kickoff({"topic": note})
    memory.reset_short_term()                       # a new run: only what persists is left

    reply = SequentialCrew([manager], [Task("answer the customer", "a reply", manager)], memory).kickoff(
        {"topic": "Acme asks for an SLA report"})[0]
    print(f"  entity store : {memory.entity}")
    print(f"  second kickoff: {reply}")
    assert "plan=enterprise" in reply and "contact=email" in reply
    assert "starter" not in reply and "Globex" not in reply.split("using")[1]      # the other customer stays out


# ---------------------------------------------------------------------------
# Exercise 3 - a manager that refuses to route to the editor
#
# main's manager only knows which roles have run. This one is shown the
# outputs, counts the paragraphs in the writer's draft, and sends it back to
# the writer with the reason until there are three. Each refusal is a line in
# the trace.
# ---------------------------------------------------------------------------

class GatedHierarchicalCrew(HierarchicalCrew):
    def kickoff(self, topic: str) -> list[str]:
        trace: list[str] = []
        results: dict[str, str] = {}
        current = topic
        for _ in range(self.max_steps):
            pick, reason = self.manager.fn(results, [], None)
            if reason:
                trace.append(f"[manager] {reason}")
            if pick == "done":
                trace.append("[manager] done")
                break
            specialist = self.specialists[pick]
            handed = f"{current}\n\nMANAGER FEEDBACK: {reason}" if reason else current
            current = results[pick] = specialist.fn(handed, specialist.tools, self.memory)
            trace.append(f"[manager -> {pick}] {current.count(chr(10) + chr(10)) + 1} paragraph(s): {current[:44]!r}")
        return trace


def paragraph_gate_manager(results: Any, tools: list[Callable[..., str]], memory: Memory | None) -> tuple[str, str]:
    for role in ("researcher", "writer"):
        if role not in results:
            return role, ""
    paragraphs = len([p for p in results["writer"].split("\n\n") if p.strip()])
    if paragraphs < 3:
        return "writer", f"not routing to the editor: the draft has {paragraphs} paragraph(s), it needs 3"
    return ("editor", "") if "editor" not in results else ("done", "")


def one_paragraph_then_three(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
    if "MANAGER FEEDBACK" in str(prior):
        return "Agents are loops.\n\nTools make the loop useful.\n\nMemory makes it last."
    return "Agents are loops with tools and memory."


def ex3_gated_hierarchy() -> None:
    researcher, writer, editor = build_agents()
    crew = GatedHierarchicalCrew(
        manager=Agent("manager", "route work", "PM background", paragraph_gate_manager),
        specialists={"researcher": researcher, "writer": replace(writer, fn=one_paragraph_then_three), "editor": editor},
    )
    trace = crew.kickoff(TOPIC)
    for line in trace:
        print(f"  {line}")
    routed = [line.split("]")[0] for line in trace if "->" in line]
    assert routed == ["[manager -> researcher", "[manager -> writer", "[manager -> writer", "[manager -> editor"]
    assert sum("not routing to the editor" in line for line in trace) == 1 and trace[-1] == "[manager] done"


# ---------------------------------------------------------------------------
# Exercise 4 - a BaseTool subclass against the @tool decorator
#
# STAND-IN: crewai is not installed, so BaseTool here is a small class with
# the shape of crewai.tools.BaseTool (name, description, an args schema, a
# _run method). The decorator version is main's `search`.
#
# The trace shapes differ because a class has somewhere to keep things. The
# decorated function leaves nothing behind except its return value. The class
# validates its arguments against the schema before running, remembers
# results, and records each of those as an event you can inspect.
# ---------------------------------------------------------------------------

class BaseTool:
    name = ""
    description = ""
    args_schema: dict[str, type] = {}
    is_tool = True

    def __init__(self) -> None:
        self.events: list[str] = []
        self._cache: dict[tuple, str] = {}

    @property
    def tool_name(self) -> str:
        return self.name

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        bound = dict(zip(self.args_schema, args)) | kwargs
        self.events.append("tool_started")
        for field, kind in self.args_schema.items():
            if not isinstance(bound.get(field), kind):
                self.events.append("validation_failed")
                return f"error: {field} must be a {kind.__name__}"
        key = tuple(sorted(bound.items()))
        if key in self._cache:
            self.events.append("cache_hit")
            return self._cache[key]
        self._cache[key] = self._run(**bound)
        self.events.append("tool_finished")
        return self._cache[key]


class MockWebSearch(BaseTool):
    name = "Search the web"
    description = "Search the web and return the top results."
    args_schema = {"query": str}

    def _run(self, query: str) -> str:              # type: ignore[override]
        return search(query)


def ex4_base_tool() -> None:
    researcher, writer, editor = build_agents()
    class_tool = MockWebSearch()
    with_decorator = sequential_crew((researcher, writer, editor)).kickoff({"topic": TOPIC})
    class_agents = (replace(researcher, tools=[class_tool]), writer, editor)
    with_class = sequential_crew(class_agents).kickoff({"topic": TOPIC})
    sequential_crew(class_agents).kickoff({"topic": TOPIC})             # same query again
    class_tool(42)                                                      # the model sends a bad argument
    print(f"  @tool decorator : output only, no events ({with_decorator[0][:52]}...)")
    print(f"  BaseTool class  : {class_tool.events}")
    assert with_class == with_decorator                                 # same work
    assert class_tool.events == ["tool_started", "tool_finished", "tool_started", "cache_hit",
                                 "tool_started", "validation_failed"]


# ---------------------------------------------------------------------------
# Exercise 5 - output_pydantic=Brief, with one malformed output
#
# STAND-IN for CrewAI's behaviour: validate the task output against the
# model, and on failure hand the error back to the agent and retry, up to a
# limit (CrewAI's guardrail_max_retries defaults to 3). Real pydantic does the
# validating.
#
# The exercise has the writer emit malformed JSON once. Validation sits on the
# task that declares the model, the editor's, so that is where it is caught:
# the editor passes the broken draft through, is told why it was rejected,
# and repairs it on the retry.
# ---------------------------------------------------------------------------

def run_structured(agent_fn: Callable[[str], str], task_input: str, model: Any, max_retries: int = 3) -> tuple[Any, list[str]]:
    from pydantic import ValidationError

    trace, feedback = [], ""
    for attempt in range(1, max_retries + 2):
        raw = agent_fn(task_input + feedback)
        try:
            parsed = model.model_validate_json(raw)
            trace.append(f"attempt {attempt}: valid {model.__name__}")
            return parsed, trace
        except ValidationError as error:
            problem = error.errors()[0]["msg"]
            trace.append(f"attempt {attempt}: rejected, {problem}")
            feedback = f"\n\nREJECTED: output was not valid {model.__name__} JSON ({problem}). Return only JSON."
    raise ValueError(f"no valid {model.__name__} after {max_retries} retries")


def ex5_output_pydantic() -> str | None:
    try:
        from pydantic import BaseModel
    except ImportError:
        print("  needs pydantic: pip install pydantic")
        return "pydantic"

    class Brief(BaseModel):
        title: str
        summary: str
        sections: list[str]

    def writer_json(prior: str) -> str:             # malformed: a trailing comma
        return '{"title": "Agent engineering 2026", "summary": "Loops, tools and memory.", "sections": ["Loop", "Tools", "Memory"],}'

    def editor_json(prior: str) -> str:
        draft = prior.split("\n\nREJECTED")[0]
        return re.sub(r",\s*}", "}", draft) if "REJECTED" in prior else draft

    brief, trace = run_structured(editor_json, writer_json(TOPIC), Brief)
    for line in trace:
        print(f"  {line}")
    print(f"  result.pydantic : {brief!r}")
    assert len(trace) == 2 and trace[0].startswith("attempt 1: rejected") and trace[1] == "attempt 2: valid Brief"
    assert brief.sections == ["Loop", "Tools", "Memory"]
    return None


# ---------------------------------------------------------------------------
# Exercise 6 - port to the real crewai API
#
# build_real_crew() needs the crewai package, and a kickoff needs a model key.
# It follows the CrewAI docs for Agent, Task, Crew and @tool.
#
# Guarantees the stdlib version skipped:
#   - each agent is a model in a loop that decides whether and how to use its
#     tools; main's researcher calls the first search tool unconditionally;
#   - structured task output with validation and retry (exercise 5);
#   - {topic} interpolation from kickoff(inputs=...) into task text;
#   - a manager model and delegation between agents in hierarchical runs;
#   - memory backed by real embeddings and storage; main's long-term memory
#     ranks by seeded random vectors;
#   - iteration and rate limits, async tasks, callbacks and telemetry.
# ---------------------------------------------------------------------------

def build_real_crew() -> Any:
    from crewai import Agent as CrewAgent
    from crewai import Crew, Process
    from crewai import Task as CrewTask
    from crewai.tools import tool as crew_tool

    @crew_tool("Search the web")
    def web_search(query: str) -> str:
        """Return top results for the query."""
        return search(query)

    researcher = CrewAgent(role="researcher", goal="find 3 credible sources",
                           backstory="former librarian. terse. cites primaries.", tools=[web_search])
    writer = CrewAgent(role="writer", goal="turn sources into a draft", backstory="editorial voice. paragraphs of three.")
    editor = CrewAgent(role="editor", goal="tighten draft to final brief", backstory="cuts adjectives. enforces house style.")
    research = CrewTask(description="Research {topic}.", expected_output="3 sources", agent=researcher)
    draft = CrewTask(description="Write a draft.", expected_output="3 paragraphs", agent=writer, context=[research])
    final = CrewTask(description="Edit to a final brief.", expected_output="800 words", agent=editor, context=[draft])
    return Crew(agents=[researcher, writer, editor], tasks=[research, draft, final], process=Process.sequential)
    # then: build_real_crew().kickoff(inputs={"topic": "agent engineering 2026"})


def ex6_real_crewai() -> str | None:
    try:
        crew = build_real_crew()
    except ImportError:
        print("  needs crewai: pip install crewai")
        return "crewai"
    print(f"  built a real Crew with {len(crew.tasks)} tasks; kickoff(inputs=...) needs a model key")
    return None


# ---------------------------------------------------------------------------
# Exercise 7 - AgentOps or Langfuse on a real run
#
# Sending traces to AgentOps or Langfuse needs an account and a real model
# run. Shown here: the span tree the toy can produce, next to what a real
# trace carries that it cannot.
# ---------------------------------------------------------------------------

MISSING_FROM_THE_TOY = [
    "one generation span per model call, with the model name, the prompt and the completion",
    "token counts and cost per call and per run",
    "the manager's routing calls in a hierarchical run",
    "retries: a rejected structured output and the attempt that replaced it",
    "memory reads and writes as their own spans",
    "errors as events with a stack, instead of an error string inside the output",
]


def traced_agents(agents: tuple[Agent, ...], spans: list[dict[str, Any]]) -> tuple[Agent, ...]:
    def trace_tool(fn: Callable[..., str], parent: str) -> Callable[..., str]:
        def call(*args: Any, **kwargs: Any) -> str:
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            spans.append({"name": f"tool:{fn.tool_name}", "parent": parent,
                          "ms": round((time.perf_counter() - start) * 1000, 3), "output_chars": len(out)})
            return out
        call.is_tool, call.tool_name = True, fn.tool_name      # type: ignore[attr-defined]
        return call

    def trace_agent(agent: Agent) -> Agent:
        def fn(prior: Any, tools: list[Callable[..., str]], memory: Memory | None) -> str:
            name = f"task:{agent.role}"
            start = time.perf_counter()
            out = agent.fn(prior, [trace_tool(t, name) for t in tools], memory)
            spans.append({"name": name, "parent": "crew", "ms": round((time.perf_counter() - start) * 1000, 3),
                          "input_chars": len(str(prior)), "output_chars": len(out)})
            return out
        return replace(agent, fn=fn)

    return tuple(trace_agent(agent) for agent in agents)


def ex7_observability() -> None:
    spans: list[dict[str, Any]] = []
    sequential_crew(traced_agents(build_agents(), spans)).kickoff({"topic": TOPIC})
    for span in spans:
        print(f"  {span['parent']:<16} > {span['name']:<20} {span['ms']} ms, {span['output_chars']} chars out")
    print("  a real AgentOps or Langfuse trace would also carry:")
    for item in MISSING_FROM_THE_TOY:
        print(f"    - {item}")
    assert [s["name"] for s in spans] == ["tool:Search the web", "task:researcher", "task:writer", "task:editor"]
    assert spans[0]["parent"] == "task:researcher"


if __name__ == "__main__":
    print("Phase 14 - Lesson 15: Role-Based Agent Teams (CrewAI) - exercises")
    outcomes = {}
    for exercise in (ex1_crew_to_flow, ex2_entity_memory, ex3_gated_hierarchy, ex4_base_tool, ex5_output_pydantic,
                     ex6_real_crewai, ex7_observability):
        print(f"\n{exercise.__name__}")
        outcomes[exercise.__name__] = exercise()
    missing = [outcome for outcome in outcomes.values() if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
