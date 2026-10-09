"""Phase 14 - Lesson 23: OpenTelemetry GenAI Semantic Conventions - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import os
import random
import sqlite3
import sys
import threading
import time
import urllib.request
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from main import Span, Tracer

LESSONS = Path(__file__).resolve().parents[2]


def load_lesson(folder: str, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, LESSONS / folder / "code" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Exercise 1 - the lesson 01 loop with invoke_agent and per-tool spans,
# exported over OTLP
#
# The agent run is one INTERNAL span named "invoke_agent {agent name}" and
# every tool call is a child span. main.py labels tool spans tool_call; the
# published conventions list that operation as execute_tool, which is what is
# used here.
#
# Jaeger takes OTLP over HTTP (JSON on port 4318, path /v1/traces). The
# payload is posted to a small local collector so the exchange can be
# checked, and to OTEL_EXPORTER_OTLP_ENDPOINT as well when that is set, which
# is how it reaches a running Jaeger.
# ---------------------------------------------------------------------------

def traced_react_run(tracer: Tracer, question: str) -> str:
    agent = load_lesson("01-the-agent-loop", "lesson01_main").build_demo_agent()
    dispatch = agent.tools.dispatch

    def traced_dispatch(call: Any) -> str:
        span = tracer.start_span(f"execute_tool {call.name}", attributes={
            "gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": call.name})
        try:
            result = dispatch(call)
            if result.startswith("error"):
                span.attributes["error.type"] = "tool_error"
            return result
        finally:
            tracer.end_span()

    agent.tools.dispatch = traced_dispatch
    tracer.start_span("invoke_agent toy_react", kind="INTERNAL", attributes={
        "gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "toy_react",
        "gen_ai.provider.name": "scripted"})
    try:
        return agent.run(question)
    finally:
        tracer.end_span()


def to_otlp(root: Span, service: str) -> dict[str, Any]:
    """main's span tree as an OTLP/JSON trace export request."""
    epoch_offset = time.time_ns() - time.perf_counter_ns()
    trace_id, ids, spans = f"{random.getrandbits(128):032x}", itertools.count(1), []

    def walk(span: Span, parent_id: str | None) -> None:
        span_id = f"{next(ids):016x}"
        record = {"traceId": trace_id, "spanId": span_id, "name": span.name,
                  "kind": {"INTERNAL": 1, "CLIENT": 3}[span.kind],
                  "startTimeUnixNano": str(span.start_ns + epoch_offset),
                  "endTimeUnixNano": str(span.end_ns + epoch_offset),
                  "attributes": [{"key": key, "value": {"stringValue": str(value)}}
                                 for key, value in sorted(span.attributes.items())]}
        if parent_id:
            record["parentSpanId"] = parent_id
        spans.append(record)
        for child in span.children:
            walk(child, span_id)

    for child in root.children:
        walk(child, None)
    return {"resourceSpans": [{
        "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": service}}]},
        "scopeSpans": [{"scope": {"name": "lesson23.practice"}, "spans": spans}]}]}


def send_otlp(payload: dict[str, Any], endpoint: str) -> int:
    request = urllib.request.Request(f"{endpoint.rstrip('/')}/v1/traces", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status


def local_collector(received: list[dict[str, Any]]) -> HTTPServer:
    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            received.append({"path": self.path,
                             "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def ex1_instrument_and_export() -> None:
    tracer = Tracer()
    answer = traced_react_run(tracer, "What is 120 plus 15% tax, stored in kv?")
    payload = to_otlp(tracer.root, service="toy-react-agent")

    received: list[dict[str, Any]] = []
    server = local_collector(received)
    status = send_otlp(payload, f"http://127.0.0.1:{server.server_port}")
    server.shutdown()

    spans = received[0]["body"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    agent_span, tool_spans = spans[0], spans[1:]
    print(f"  agent answered: {answer}")
    print(f"  POST {received[0]['path']} -> {status}; collector received {len(spans)} spans")
    for span in spans:
        print(f"    {'  ' if 'parentSpanId' in span else ''}{span['name']}")
    assert status == 200 and received[0]["path"] == "/v1/traces"
    assert agent_span["name"] == "invoke_agent toy_react" and agent_span["kind"] == 1
    assert len(tool_spans) == 5 and all(s["parentSpanId"] == agent_span["spanId"] for s in tool_spans)

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        print(f"  also sent to {endpoint}: HTTP {send_otlp(payload, endpoint)}")


# ---------------------------------------------------------------------------
# Exercise 2 - content capture in "references only" mode
#
# Prompts and outputs go into SQLite. The span gets a row id and nothing
# else, so anyone who can read traces cannot read customer text, and the
# content can be deleted or access-controlled in one place.
# ---------------------------------------------------------------------------

class SqliteContentStore:
    """Same three methods as main.ExternalContentStore, on disk."""

    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS content (id INTEGER PRIMARY KEY, body TEXT)")

    def put(self, content: str) -> str:
        cursor = self.db.execute("INSERT INTO content (body) VALUES (?)", (content,))
        self.db.commit()
        return f"row:{cursor.lastrowid}"

    def get(self, cid: str) -> str:
        row = self.db.execute("SELECT body FROM content WHERE id = ?", (int(cid.split(":")[1]),)).fetchone()
        return row[0] if row else ""

    def items(self) -> list[tuple[str, str]]:
        return [(f"row:{rid}", body) for rid, body in self.db.execute("SELECT id, body FROM content ORDER BY id")]


def chat_turn(tracer: Tracer, prompt: str, output: str) -> Span:
    span = tracer.start_span("chat", attributes={"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic"})
    tracer.add_content(span, "gen_ai.input.messages", prompt)
    tracer.add_content(span, "gen_ai.output.messages", output)
    tracer.end_span()
    return span


def ex2_references_only() -> None:
    prompt = "customer ava@example.com asks why invoice 4711 was charged twice"
    store = SqliteContentStore()
    referenced = chat_turn(Tracer(capture_inline=False, content_store=store), prompt, "refund issued")
    inline = chat_turn(Tracer(capture_inline=True), prompt, "refund issued")

    print(f"  inline capture   : {inline.attributes['gen_ai.input.messages']!r}")
    print(f"  references only  : {referenced.attributes}")
    print(f"  sqlite rows      : {store.items()}")
    assert "ava@example.com" in str(inline.attributes)                      # the leak
    assert "ava@example.com" not in str(referenced.attributes)
    assert referenced.attributes["gen_ai.input.messages.reference_id"] == "row:1"
    assert store.get("row:1") == prompt


# ---------------------------------------------------------------------------
# Exercise 3 - gen_ai.data_source.id on the lesson 09 Mem0 search
#
# The conventions define gen_ai.data_source.id as the identifier of the data
# source a retrieval consulted, and put it on the retrieval span. It should be
# the id the system itself uses for that store. With it on every search span
# you can ask a trace backend which store a bad answer was grounded on.
# ---------------------------------------------------------------------------

def traced_search(tracer: Tracer, memory: Any, data_source_id: str, query: str, user_id: str) -> list[Any]:
    span = tracer.start_span(f"retrieval {data_source_id}", attributes={
        "gen_ai.operation.name": "retrieval", "gen_ai.data_source.id": data_source_id})
    try:
        hits = memory.search(query, user_id=user_id)
        span.attributes["retrieval.hits"] = len(hits)
        return hits
    finally:
        tracer.end_span()


def ex3_data_source_id() -> None:
    lesson_09 = load_lesson("09-hybrid-memory-mem0", "lesson09_main")
    stores = {"mem0-prod-profiles": lesson_09.Mem0(), "mem0-prod-billing": lesson_09.Mem0()}
    stores["mem0-prod-profiles"].add("ava lives in Lisbon", user_id="ava")
    stores["mem0-prod-billing"].add("ava was refunded for invoice 4711", user_id="ava")

    tracer = Tracer()
    for source, memory in stores.items():
        hits = traced_search(tracer, memory, source, "where does ava live", user_id="ava")
        print(f"  {source:<19} -> {[record.text for _, record in hits]}")
    spans = tracer.root.children
    assert [s.attributes["gen_ai.data_source.id"] for s in spans] == list(stores)
    assert all(s.attributes["gen_ai.operation.name"] == "retrieval" for s in spans)


# ---------------------------------------------------------------------------
# Exercise 4 - OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
#
# SIMULATED COLLECTOR: the "collector upgrade" below is a function, with three
# renames the GenAI conventions really went through. The check is the one the
# exercise asks for. An emitter that honours the opt-in sends current names,
# which the upgrade leaves alone. One that does not sends legacy names, and
# the upgrade renames them underneath every dashboard that queries them.
# ---------------------------------------------------------------------------

LEGACY_TO_CURRENT = {
    "gen_ai.system": "gen_ai.provider.name",
    "gen_ai.usage.prompt_tokens": "gen_ai.usage.input_tokens",
    "gen_ai.usage.completion_tokens": "gen_ai.usage.output_tokens",
}


def chat_attributes(environ: dict[str, str]) -> dict[str, Any]:
    current = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic",
               "gen_ai.request.model": "claude-opus-4-6",
               "gen_ai.usage.input_tokens": 812, "gen_ai.usage.output_tokens": 164}
    if "gen_ai_latest_experimental" in environ.get("OTEL_SEMCONV_STABILITY_OPT_IN", "").split(","):
        return current
    to_legacy = {new: old for old, new in LEGACY_TO_CURRENT.items()}
    return {to_legacy.get(key, key): value for key, value in current.items()}


def collector_upgrade(attributes: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    renamed = [key for key in attributes if key in LEGACY_TO_CURRENT]
    return {LEGACY_TO_CURRENT.get(key, key): value for key, value in attributes.items()}, renamed


def ex4_stability_opt_in() -> None:
    opted_in = chat_attributes({"OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental"})
    default = chat_attributes({})
    _, renamed_opted_in = collector_upgrade(opted_in)
    upgraded, renamed_default = collector_upgrade(default)
    print(f"  opt-in set   : {len(renamed_opted_in)} attributes renamed by the collector")
    print(f"  opt-in unset : {len(renamed_default)} renamed -> {renamed_default}")
    print(f"  this process : OTEL_SEMCONV_STABILITY_OPT_IN={os.environ.get('OTEL_SEMCONV_STABILITY_OPT_IN', '(unset)')}")
    assert renamed_opted_in == [] and collector_upgrade(opted_in)[0] == opted_in
    assert len(renamed_default) == 3 and upgraded == opted_in


# ---------------------------------------------------------------------------
# Exercise 5 - a dashboard: which tool errors go with which models
#
# SYNTHETIC TRACES, with one effect planted: write_file fails far more often
# under model-b. The dashboard has to find it from GenAI attributes alone.
# Tool spans do not carry the model. The model is on the agent span, so the
# join runs through the parent link, which is why orphaned tool spans make
# this question unanswerable.
#
# A table of rates invites reading noise as a pattern, so each cell is also
# scored against all the other calls (a two-proportion z-score). The planted
# cell is the only one more than three standard errors above the rest.
# ---------------------------------------------------------------------------

MODELS, TOOLS = ("model-a", "model-b", "model-c"), ("search", "read_file", "write_file")


def synthetic_traces(count: int = 900, seed: int = 0) -> list[Span]:
    rng = random.Random(seed)
    traces = []
    for _ in range(count):
        model = rng.choice(MODELS)
        agent = Span("invoke_agent support_bot", attributes={
            "gen_ai.operation.name": "invoke_agent", "gen_ai.request.model": model})
        for tool in TOOLS:
            failure_rate = 0.30 if (model, tool) == ("model-b", "write_file") else 0.04
            span = Span(f"execute_tool {tool}", attributes={"gen_ai.operation.name": "execute_tool",
                                                           "gen_ai.tool.name": tool})
            if rng.random() < failure_rate:
                span.attributes["error.type"] = "tool_error"
            agent.children.append(span)
        traces.append(agent)
    return traces


def z_against_rest(cell: tuple[str, str], calls: Counter, errors: Counter) -> float:
    rest_calls, rest_errors = sum(calls.values()) - calls[cell], sum(errors.values()) - errors[cell]
    pooled = sum(errors.values()) / sum(calls.values())
    standard_error = math.sqrt(pooled * (1 - pooled) * (1 / calls[cell] + 1 / rest_calls))
    return (errors[cell] / calls[cell] - rest_errors / rest_calls) / standard_error


def ex5_error_dashboard() -> None:
    calls: Counter[tuple[str, str]] = Counter()
    errors: Counter[tuple[str, str]] = Counter()
    for agent in synthetic_traces():
        model = agent.attributes["gen_ai.request.model"]
        for child in agent.children:
            if child.attributes.get("gen_ai.operation.name") == "execute_tool":
                key = (model, child.attributes["gen_ai.tool.name"])
                calls[key] += 1
                errors[key] += "error.type" in child.attributes
    print(f"  {'error rate':<12}" + "".join(f"{tool:>12}" for tool in TOOLS))
    for model in MODELS:
        print(f"  {model:<12}" + "".join(f"{errors[model, tool] / calls[model, tool]:>12.1%}" for tool in TOOLS))
    overall = sum(errors.values()) / sum(calls.values())
    worst = max(calls, key=lambda key: errors[key] / calls[key])
    print(f"  overall {overall:.1%}; worst cell {worst} at {errors[worst] / calls[worst]:.1%}, "
          f"{errors[worst] / calls[worst] / overall:.1f}x the average")
    scores = {cell: z_against_rest(cell, calls, errors) for cell in calls}
    outliers = [cell for cell, z in scores.items() if z > 3]
    print(f"  z-score of {worst} against the rest: {scores[worst]:.1f}; cells above 3: {len(outliers)} of {len(scores)}")
    assert worst == ("model-b", "write_file") and outliers == [worst] and scores[worst] > 8


if __name__ == "__main__":
    print("Phase 14 - Lesson 23: OpenTelemetry GenAI Semantic Conventions - exercises")
    for exercise in (ex1_instrument_and_export, ex2_references_only, ex3_data_source_id, ex4_stability_opt_in,
                     ex5_error_dashboard):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
