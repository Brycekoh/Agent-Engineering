"""Phase 14 - Lesson 14: The Actor Model for Agents - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from main import Actor, ChecklistAgent, Message, ReviewerAgent, Runtime

try:        # exercise 5 only; nothing else in this file needs it
    from autogen_core import AgentId, MessageContext, RoutedAgent, SingleThreadedAgentRuntime, message_handler
except ImportError:
    RoutedAgent = None

SNIPPETS = [
    "def add(a, b): return a + b",
    "def hazard(): eval('1+1')",
    "def silent(): \n    try:\n        f()\n    except:\n        pass",
]


def run_review_scenario(runtime: Runtime) -> ChecklistAgent:
    """main.py's demo: three reviews plus one message that crashes the reviewer."""
    checklist = ChecklistAgent("checklist", partner="reviewer")
    runtime.register(ReviewerAgent("reviewer"))
    runtime.register(checklist)
    runtime.send("__user__", "checklist", "start", SNIPPETS)
    runtime.send("__user__", "reviewer", "crash_me", {})
    runtime.run_until_idle()
    return checklist


# ---------------------------------------------------------------------------
# Exercise 1 - a dead-letter queue a human can work with
#
# main already parks failing messages in runtime.dead_letters. What it lacks
# is the other half: a way to look at them, put one back after fixing the
# cause, and a delivery limit so a poison message cannot loop forever.
#
# How often it is hit in the toy: 1 of 8 messages, 12.5%. That number is a
# property of the script (one message is built to crash). In a real system
# the rate is the thing to alert on.
# ---------------------------------------------------------------------------

@dataclass
class DLQRuntime(Runtime):
    max_deliveries: int = 3
    deliveries: dict[int, int] = field(default_factory=dict)

    def dlq_rate(self) -> float:
        return len(self.dead_letters) / self.counter if self.counter else 0.0

    def inspect(self) -> list[str]:
        return [f"m{m.mid:03d} {m.sender} -> {m.recipient} topic={m.topic}: {reason}"
                for m, reason in self.dead_letters]

    def replay(self, mid: int) -> str:
        for i, (message, _) in enumerate(self.dead_letters):
            if message.mid == mid:
                delivered = self.deliveries.get(mid, 1)
                if delivered >= self.max_deliveries:
                    return "quarantined: delivery limit reached"
                self.deliveries[mid] = delivered + 1
                del self.dead_letters[i]
                self.queue.append(message)
                return "requeued"
        return "not in the dead-letter queue"


class PatchedReviewer(ReviewerAgent):
    def receive(self, message: Message, runtime: Runtime) -> None:
        if message.topic != "crash_me":
            super().receive(message, runtime)


def ex1_dead_letter_queue() -> None:
    runtime = DLQRuntime()
    run_review_scenario(runtime)
    parked = runtime.dead_letters[0][0].mid
    print(f"  parked   : {runtime.inspect()}")
    print(f"  DLQ rate : {len(runtime.dead_letters)} of {runtime.counter} messages ({runtime.dlq_rate():.1%})")
    assert runtime.dlq_rate() == 1 / 8

    runtime.replay(parked)                      # replayed without fixing anything
    runtime.run_until_idle()
    assert len(runtime.dead_letters) == 1       # straight back in the queue
    runtime.register(PatchedReviewer("reviewer"))
    print(f"  after the handler is fixed: {runtime.replay(parked)}")
    runtime.run_until_idle()
    assert runtime.dead_letters == []

    strict = DLQRuntime(max_deliveries=1)
    run_review_scenario(strict)
    print(f"  poison message, limit 1   : {strict.replay(strict.dead_letters[0][0].mid)}")
    assert strict.replay(strict.dead_letters[0][0].mid).startswith("quarantined")


# ---------------------------------------------------------------------------
# Exercise 2 - SelectorGroupChat
#
# A selector actor owns the transcript and, after every reply, picks who
# speaks next from the state of the conversation. `choose` is a scripted
# stand-in for the selector's LLM call. Round-robin would hand the turn back
# to the planner after the reviewer's rejection; the selector sends the work
# to the coder, who is the one who can act on it.
# ---------------------------------------------------------------------------

class Selector(Actor):
    def __init__(self, name: str, choose: Callable[[list[tuple[str, str]]], str | None], max_turns: int = 8) -> None:
        super().__init__(name)
        self.choose = choose
        self.max_turns = max_turns
        self.transcript: list[tuple[str, str]] = []
        self.finished = False

    def receive(self, message: Message, runtime: Runtime) -> None:
        self.transcript.append((message.sender, str(message.body)))
        speaker = self.choose(self.transcript)
        if speaker is None or len(self.transcript) > self.max_turns:
            self.finished = True
            return
        runtime.send(self.name, speaker, "speak", [list(turn) for turn in self.transcript])


class ScriptedSpeaker(Actor):
    def __init__(self, name: str, lines: list[str]) -> None:
        super().__init__(name)
        self.lines = iter(lines)

    def receive(self, message: Message, runtime: Runtime) -> None:
        runtime.send(self.name, message.sender, "reply", next(self.lines))


def choose_next(transcript: list[tuple[str, str]]) -> str | None:
    speaker, text = transcript[-1]
    if speaker == "reviewer":
        return None if "approve" in text else "coder"
    return {"__user__": "planner", "planner": "coder", "coder": "reviewer"}[speaker]


def ex2_selector_group_chat() -> None:
    runtime = Runtime()
    selector = Selector("selector", choose_next)
    runtime.register(selector)
    runtime.register(ScriptedSpeaker("planner", ["plan: retry the upload with backoff"]))
    runtime.register(ScriptedSpeaker("coder", ["v1: retry in a tight loop", "v2: retry with exponential backoff"]))
    runtime.register(ScriptedSpeaker("reviewer", ["reject: no backoff, this will hammer the server", "approve"]))
    runtime.send("__user__", "selector", "task", "make uploads survive a flaky network")
    runtime.run_until_idle()

    speakers = [speaker for speaker, _ in selector.transcript[1:]]
    for speaker, text in selector.transcript:
        print(f"  {speaker:>8}: {text}")
    assert speakers == ["planner", "coder", "reviewer", "coder", "reviewer"]
    assert selector.finished and not runtime.dead_letters


# ---------------------------------------------------------------------------
# Exercise 3 - JSON-over-HTTP transport, actors in separate processes
#
# send() checks an address book. A local recipient goes on the in-process
# queue as before; a remote one gets the message as JSON in an HTTP POST. The
# actors do not change at all - that is the point of the model. The exercise
# starts a second Python process for the reviewer and exchanges real messages
# with it over localhost.
#
# ponytail: no auth, no retries, at-most-once delivery, loopback only. Enough
# to show the seam; a real transport is a broker or gRPC.
# ---------------------------------------------------------------------------

@dataclass
class HttpRuntime(Runtime):
    remote: dict[str, str] = field(default_factory=dict)        # actor name -> base URL of its process

    def send(self, sender: str, recipient: str, topic: str, body: Any) -> None:
        url = self.remote.get(recipient)
        if url is None:
            return super().send(sender, recipient, topic, body)
        self.counter += 1
        payload = {"sender": sender, "recipient": recipient, "topic": topic, "body": body, "mid": self.counter}
        self.trace.append(f"[http m{self.counter:03d}] {sender} -> {recipient} at {url} topic={topic}")
        try:
            request = urllib.request.Request(f"{url}/message", json.dumps(payload).encode(),
                                             {"Content-Type": "application/json"})
            urllib.request.urlopen(request, timeout=5).read()
        except OSError as error:
            self.dead_letters.append((Message(**payload), f"transport: {error}"))


def serve(runtime: Runtime, port: int, handle_inline: bool) -> HTTPServer:
    class Inbox(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            runtime.queue.append(Message(**payload))
            if handle_inline:
                runtime.run_until_idle()
            self.send_response(204)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    return HTTPServer(("127.0.0.1", port), Inbox)


def reviewer_process(parent_url: str) -> None:
    """Entry point of the second process: hosts the reviewer, replies to the parent over HTTP."""
    runtime = HttpRuntime(remote={"checklist": parent_url})
    runtime.register(ReviewerAgent("reviewer"))
    server = serve(runtime, 0, handle_inline=True)
    print(server.server_port, flush=True)       # tells the parent where to send
    server.serve_forever()


def ex3_http_transport() -> None:
    runtime = HttpRuntime()
    server = serve(runtime, 0, handle_inline=False)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    child = subprocess.Popen([sys.executable, __file__, "--reviewer-process",
                              f"http://127.0.0.1:{server.server_port}"], stdout=subprocess.PIPE, text=True)
    try:
        runtime.remote["reviewer"] = f"http://127.0.0.1:{int(child.stdout.readline())}"
        checklist = ChecklistAgent("checklist", partner="reviewer")
        runtime.register(checklist)
        runtime.send("__user__", "checklist", "start", SNIPPETS)
        deadline = time.monotonic() + 10
        while len(checklist.results) < 3 and time.monotonic() < deadline:
            runtime.run_until_idle()            # replies arrive on the server thread; drain them here
            time.sleep(0.01)
    finally:
        child.terminate()
        child.wait(timeout=5)
        server.shutdown()

    print(f"  parent pid {os.getpid()}, reviewer pid {child.pid}")
    for line in runtime.trace:
        print(f"  {line[:96]}")
    assert child.pid != os.getpid()
    assert [r["ok"] for r in checklist.results] == [True, False, False]
    assert checklist.consensus is False and not runtime.dead_letters


# ---------------------------------------------------------------------------
# Exercise 4 - one span per message, with the GenAI attribute names
#
# NO-OP STAND-IN, as the exercise allows: a span is a small dataclass
# collected in a list. The names follow lesson 23: the span is
# called "invoke_agent {agent name}" and carries gen_ai.operation.name and
# gen_ai.agent.name. A handler that raises marks its span as an error and the
# exception still reaches the runtime, so the dead-letter queue keeps working.
# ---------------------------------------------------------------------------

@dataclass
class Span:
    name: str
    attributes: dict[str, Any]
    status: str = "ok"
    duration_ms: float = 0.0


@dataclass
class TracedRuntime(Runtime):
    spans: list[Span] = field(default_factory=list)

    def register(self, actor: Actor) -> None:
        handler = actor.receive

        def traced(message: Message, runtime: Runtime) -> None:
            span = Span(f"invoke_agent {actor.name}", {
                "gen_ai.operation.name": "invoke_agent",
                "gen_ai.agent.name": actor.name,
                "actor.topic": message.topic,
                "actor.message_id": message.mid,
            })
            start = time.perf_counter()
            try:
                handler(message, runtime)
            except Exception as error:
                span.status = "error"
                span.attributes["error.type"] = type(error).__name__
                raise
            finally:
                span.duration_ms = (time.perf_counter() - start) * 1000
                self.spans.append(span)

        actor.receive = traced                  # type: ignore[method-assign]
        super().register(actor)


def ex4_span_per_message() -> None:
    runtime = TracedRuntime()
    run_review_scenario(runtime)
    for span in runtime.spans[:3]:
        print(f"  {span.name:<24} {span.status:<5} {span.attributes}")
    print(f"  {len(runtime.spans)} spans for {runtime.counter} messages")
    assert len(runtime.spans) == runtime.counter == 8
    assert all(s.attributes["gen_ai.operation.name"] == "invoke_agent" and
               s.name == f"invoke_agent {s.attributes['gen_ai.agent.name']}" for s in runtime.spans)
    failed = [s for s in runtime.spans if s.status == "error"]
    assert len(failed) == 1 and failed[0].attributes["error.type"] == "RuntimeError"
    assert len(runtime.dead_letters) == 1


# ---------------------------------------------------------------------------
# Exercise 5 - the same reviewer on the real autogen_core API
#
# Needs autogen-core (pip install autogen-core). The port follows the "Agent
# and Agent Runtime" page of the AutoGen core guide.
#
# What the toy skipped that matters in production:
#   - handlers are async, so a slow agent does not hold up every other inbox;
#   - messages are typed (dataclasses), and the handler is chosen by type;
#     that same typing is what lets a message be serialised between processes;
#   - you do not construct agents. You register a factory for an agent type
#     and the runtime creates an instance per AgentId (type, key) when the
#     first message for it arrives, and owns its lifecycle from then on;
#   - topics and subscriptions for publish/subscribe, a distributed runtime,
#     cancellation, and built-in tracing - none of which the toy has.
# ---------------------------------------------------------------------------

@dataclass
class ReviewRequest:
    code: str


@dataclass
class ReviewResult:
    ok: bool
    issues: list[str]


if RoutedAgent is not None:
    class AutogenReviewer(RoutedAgent):
        def __init__(self) -> None:
            super().__init__("Reviews code snippets")

        @message_handler
        async def review(self, message: ReviewRequest, ctx: MessageContext) -> ReviewResult:
            issues = [label for needle, label in (("eval(", "uses eval"), ("except:", "bare except"))
                      if needle in message.code]
            return ReviewResult(ok=not issues, issues=issues)

    async def review_on_autogen_core() -> list[ReviewResult]:
        runtime = SingleThreadedAgentRuntime()
        await AutogenReviewer.register(runtime, "reviewer", lambda: AutogenReviewer())
        runtime.start()
        results = [await runtime.send_message(ReviewRequest(code), AgentId("reviewer", "default"))
                   for code in SNIPPETS]
        await runtime.stop_when_idle()
        return results


def ex5_autogen_core_port() -> str | None:
    if RoutedAgent is None:
        print("  needs autogen-core: pip install autogen-core")
        return "autogen-core"
    results = asyncio.run(review_on_autogen_core())
    for code, result in zip(SNIPPETS, results):
        print(f"  {code[:28]!r:<32} ok={result.ok} issues={result.issues}")
    assert [r.ok for r in results] == [True, False, False]
    return None


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reviewer-process"]:
        reviewer_process(sys.argv[2])
        sys.exit()
    print("Phase 14 - Lesson 14: The Actor Model for Agents - exercises")
    outcomes = []
    for exercise in (ex1_dead_letter_queue, ex2_selector_group_chat, ex3_http_transport,
                     ex4_span_per_message, ex5_autogen_core_port):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
