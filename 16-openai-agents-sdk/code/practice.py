"""Phase 14 - Lesson 16: OpenAI Agents SDK - Handoffs, Guardrails, Tracing - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import itertools
import json
import time
from typing import Any, Callable

from main import (Agent, GuardrailTripped, Handoff, InputGuardrail, OutputGuardrail, Runner, Span,
                  _billing_policy, _pii_check, _triage_policy)

Policy = Callable[[str], dict[str, Any]]


def ping_pong_agents() -> tuple[Agent, Agent]:
    """Triage and billing that keep handing the same request back to each other."""
    billing = Agent("billing", "handle refunds and invoices",
                    policy=lambda text: {"kind": "handoff", "to": "triage", "input": text})
    triage = Agent("triage", "route queries to the right specialist", policy=_triage_policy,
                   handoffs=[Handoff(target=billing)])
    billing.handoffs.append(Handoff(target=triage))
    return triage, billing


# ---------------------------------------------------------------------------
# Exercise 1 - a handoff hop counter
#
# main.Runner's max_hops bounds the loop, but when the bound is hit it falls
# out with an empty string: the caller cannot tell a refusal from an agent
# that had nothing to say. The counter below counts transfers specifically,
# and the transfer past the limit is refused in words and recorded as a span.
# ---------------------------------------------------------------------------

def limit_handoffs(agents: list[Agent], runner: Runner, max_transfers: int) -> None:
    transfers = itertools.count(1)

    def guarded(agent: Agent, policy: Policy) -> Policy:
        def wrapped(text: str) -> dict[str, Any]:
            decision = policy(text)
            if decision["kind"] != "handoff":
                return decision
            n = next(transfers)
            if n <= max_transfers:
                return decision
            runner.trace.children.append(Span("handoff_refused", {
                "from": agent.name, "to": decision["to"], "transfer": n, "limit": max_transfers}))
            return {"kind": "final", "text": f"transfer {n} refused: limit of {max_transfers} handoffs reached, "
                                             "escalating to a human"}
        return wrapped

    for agent in agents:
        agent.policy = guarded(agent, agent.policy)


def ex1_hop_counter() -> None:
    request = "I need a refund for invoice 4711"
    triage, _ = ping_pong_agents()
    silent = Runner(max_hops=6).run(triage, request)
    print(f"  main, max_hops=6     : returns {silent!r}")

    triage, billing = ping_pong_agents()
    runner = Runner(max_hops=6)
    limit_handoffs([triage, billing], runner, max_transfers=2)
    refused = runner.run(triage, request)
    print(f"  limit of 2 transfers : {refused}")
    print(f"  trace                : {[span.name for span in runner.trace.children]}")
    assert silent == ""
    assert refused.startswith("transfer 3 refused")
    assert [s.name for s in runner.trace.children] == ["agent.triage", "agent.billing", "agent.triage", "handoff_refused"]


# ---------------------------------------------------------------------------
# Exercise 2 - nest_handoff_history as an option
#
# In the SDK this is RunConfig.nest_handoff_history (or a per-handoff
# override), off by default while it stabilises. With it on, the receiving
# agent does not get the earlier turns as separate items; it gets them folded
# into one summary inside a <CONVERSATION HISTORY> block. main passes only a
# single string between agents, so the transcript is kept here.
# ---------------------------------------------------------------------------

def handoff_input(transcript: list[tuple[str, str]], request: str, nest: bool) -> list[str]:
    """The items the next agent starts with."""
    if not nest:
        return [f"{role}: {text}" for role, text in transcript] + [f"user: {request}"]
    history = "\n".join(f"{role}: {text}" for role, text in transcript)
    return [f"assistant: <CONVERSATION HISTORY>\n{history}\n</CONVERSATION HISTORY>", f"user: {request}"]


def ex2_nested_history() -> None:
    transcript = [
        ("user", "hi, I was charged twice for invoice 4711"),
        ("triage", "sorry about that, let me look"),
        ("tool lookup_invoice", "invoice 4711: two charges of 49.00 on 3 March"),
        ("triage", "I can see both charges; transferring you to billing"),
    ]
    request = "please refund the duplicate"
    received: dict[bool, list[str]] = {}
    for nest in (False, True):
        seen: list[str] = []
        billing = Agent("billing", "handle refunds", policy=lambda text, seen=seen: (
            seen.append(text), _billing_policy(text))[1])
        triage = Agent("triage", "route", handoffs=[Handoff(target=billing)],
                       policy=lambda text, nest=nest: {"kind": "handoff", "to": "billing",
                                                       "input": "\n".join(handoff_input(transcript, text, nest))})
        Runner().run(triage, request)
        received[nest] = handoff_input(transcript, request, nest)
        assert seen[0] == "\n".join(received[nest])          # billing really started from these items
    print(f"  nest_handoff_history=False: billing receives {len(received[False])} items")
    print(f"  nest_handoff_history=True : billing receives {len(received[True])} items")
    print("    " + received[True][0].replace("\n", "\n    "))
    assert len(received[False]) == 5 and len(received[True]) == 2
    assert received[True][0].count("CONVERSATION HISTORY") == 2 and "two charges of 49.00" in received[True][0]


# ---------------------------------------------------------------------------
# Exercise 3 - a blocking output guardrail, and what it costs
#
# An output guardrail runs after the agent has produced its answer and must
# finish before the answer is released. So a prompt that trips it takes as
# long as one that passes: the full generation plus the check, and the result
# is then thrown away. Compare an input guardrail tripping on the same bad
# request, which stops before any generation is paid for.
# Latencies are simulated with sleep: 80 ms generation, 40 ms check.
# ---------------------------------------------------------------------------

GENERATION_S, CHECK_S = 0.08, 0.04


def slow_agent() -> Agent:
    def policy(text: str) -> dict[str, Any]:
        time.sleep(GENERATION_S)
        leaked = " card 4111 1111 1111 1111" if "card" in text else ""
        return {"kind": "final", "text": f"your order is on its way{leaked}"}
    return Agent("support", "answer order questions", policy=policy)


def blocking_card_check(text: str) -> tuple[bool, str]:
    time.sleep(CHECK_S)
    return "4111" not in text, "card number in output"


def timed(runner: Runner, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()
    try:
        outcome = runner.run(slow_agent(), prompt)
    except GuardrailTripped as tripped:
        outcome = f"TRIPPED ({tripped.which})"
    return outcome, time.perf_counter() - start


def ex3_blocking_output_guardrail() -> None:
    output_guarded = lambda: Runner(output_guardrails=[OutputGuardrail("card_check", blocking_card_check)])
    input_guarded = lambda: Runner(input_guardrails=[InputGuardrail(
        "card_request", lambda text: (time.sleep(CHECK_S), ("card" not in text, "asks for card data"))[1])])
    passed, t_pass = timed(output_guarded(), "where is my order?")
    tripped, t_trip = timed(output_guarded(), "where is my order? include my card")
    early, t_early = timed(input_guarded(), "where is my order? include my card")
    print(f"  output guardrail, passes : {t_pass * 1000:>4.0f} ms  {passed}")
    print(f"  output guardrail, trips  : {t_trip * 1000:>4.0f} ms  {tripped}")
    print(f"  input guardrail, trips   : {t_early * 1000:>4.0f} ms  {early}")
    assert tripped == "TRIPPED (output)" and early == "TRIPPED (input)"
    assert abs(t_trip - t_pass) < 0.04                  # tripping saves nothing
    assert t_early < t_trip - 0.05                      # stopping at the input skips the generation


# ---------------------------------------------------------------------------
# Exercise 4 - add_trace_processor wired to a JSON logger
#
# STAND-IN for agents.add_trace_processor. The processor has the method names
# of the SDK's TracingProcessor (on_trace_start/end, on_span_start/end,
# shutdown, force_flush). The SDK documents which fields a span carries
# (trace_id, parent_id, started_at, ended_at, span_data) but not the exported
# JSON, so the exact key layout below is mine, built from those fields.
#
# Shape per span: one flat JSON object - its own id, the trace it belongs to,
# its parent, two timestamps, and a span_data object whose `type` says what
# kind of step it was and whose other keys depend on that type.
# ---------------------------------------------------------------------------

class JsonLogProcessor:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def on_trace_start(self, trace: dict[str, Any]) -> None:
        self.lines.append(json.dumps({"object": "trace", **trace}))

    def on_span_start(self, span: dict[str, Any]) -> None:
        pass

    def on_span_end(self, span: dict[str, Any]) -> None:
        self.lines.append(json.dumps(span))

    def on_trace_end(self, trace: dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


PROCESSORS: list[JsonLogProcessor] = []


def add_trace_processor(processor: JsonLogProcessor) -> None:
    PROCESSORS.append(processor)


def emit_trace(root: Span, workflow_name: str) -> None:
    """Walk main's span tree and feed it to every processor, children before parents (end order)."""
    ids = itertools.count(1)
    trace = {"id": "trace_1", "workflow_name": workflow_name}
    for processor in PROCESSORS:
        processor.on_trace_start(trace)

    def walk(span: Span, parent_id: str | None) -> None:
        span_id = f"span_{next(ids)}"
        kind = span.name.split(".")[0]
        record = {"object": "trace.span", "id": span_id, "trace_id": trace["id"], "parent_id": parent_id,
                  "started_at": time.time(), "span_data": {"type": kind, "name": span.name, **span.attributes}}
        for child in span.children:
            walk(child, span_id)
        record["ended_at"] = time.time()
        for processor in PROCESSORS:
            processor.on_span_end(record)

    for child in root.children:
        walk(child, None)
    for processor in PROCESSORS:
        processor.on_trace_end(trace)


def ex4_trace_processor() -> None:
    billing = Agent("billing", "handle refunds and invoices", policy=_billing_policy)
    triage = Agent("triage", "route queries", policy=_triage_policy, handoffs=[Handoff(target=billing)])
    runner = Runner(input_guardrails=[InputGuardrail("pii_block", _pii_check)])
    runner.run(triage, "I need a refund for invoice 4711")

    logger = JsonLogProcessor()
    add_trace_processor(logger)
    emit_trace(runner.trace, workflow_name="support triage")
    spans = [json.loads(line) for line in logger.lines[1:]]
    print(f"  {logger.lines[0]}")
    print(f"  keys of every span line: {sorted(spans[0])}")
    for span in spans:
        print(f"  {span['id']} parent={span['parent_id']} span_data={json.dumps(span['span_data'])[:84]}")
    assert all({"id", "trace_id", "parent_id", "started_at", "ended_at", "span_data"} <= set(s) for s in spans)
    assert [s["span_data"]["type"] for s in spans] == ["input_guardrail", "handoff", "agent", "llm_generation", "agent"]
    assert spans[1]["parent_id"] == spans[2]["id"]          # the handoff happened inside the triage agent span


# ---------------------------------------------------------------------------
# Exercise 5 - port to openai-agents-python
#
# build_sdk_team() needs the openai-agents package, and running it needs an
# OpenAI key. It follows the SDK's handoffs page.
#
# What the toy modelled wrong:
#   - a handoff carries the conversation, not a string. The next agent gets
#     the prior items (or the nested summary from exercise 2);
#   - input guardrails run only for the first agent and output guardrails only
#     for the last; main runs its output guardrails on whatever came out last,
#     which happens to match, but it has no notion of which agent that was;
#   - running out of turns is an error in the SDK, not an empty string;
#   - tools have a JSON schema derived from the function signature; main's
#     FunctionTool has a name and a description and nothing to validate with;
#   - tracing is a tree of spans with ids, parents and timestamps, pushed to
#     processors as the run happens; main builds its tree in memory afterwards.
# ---------------------------------------------------------------------------

def build_sdk_team() -> Any:
    from agents import Agent as SdkAgent
    from agents import handoff

    billing = SdkAgent(name="billing", instructions="Handle refunds and invoices.")
    support = SdkAgent(name="support", instructions="Handle bugs and errors.")
    return SdkAgent(
        name="triage",
        instructions="Route each request to billing or support.",
        handoffs=[billing, handoff(support, tool_description_override="Bugs, crashes and error messages.")],
    )       # run with: Runner.run_sync(triage, "I need a refund for invoice 4711")


def ex5_sdk_port() -> str | None:
    try:
        triage = build_sdk_team()
    except ImportError:
        print("  needs openai-agents: pip install openai-agents")
        return "openai-agents"
    print(f"  built SDK agent {triage.name!r} with {len(triage.handoffs)} handoffs; Runner.run_sync needs an OpenAI key")
    return None


if __name__ == "__main__":
    print("Phase 14 - Lesson 16: OpenAI Agents SDK - exercises")
    outcomes = []
    for exercise in (ex1_hop_counter, ex2_nested_history, ex3_blocking_output_guardrail,
                     ex4_trace_processor, ex5_sdk_port):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
