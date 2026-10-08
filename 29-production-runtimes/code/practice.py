"""Phase 14 - Lesson 29: Production Runtimes - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import importlib.util
import json
import queue
import random
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import main as lesson
from main import EventBus, Job, QueueRuntime

QUESTION = "What is 120 plus 15% tax, stored in kv?"


def load_lesson_01() -> Any:
    path = Path(__file__).resolve().parents[2] / "01-the-agent-loop" / "code" / "main.py"
    spec = importlib.util.spec_from_file_location("lesson01_main", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def agent_steps(question: str) -> Iterator[str]:
    """The lesson 01 ReAct loop, yielding one line per turn. Same control flow as AgentLoop.run."""
    lesson_01 = load_lesson_01()
    agent = lesson_01.build_demo_agent()
    history = [lesson_01.Turn("user", question)]
    for _ in range(agent.max_turns):
        reply = agent.llm.respond(history)
        if reply["kind"] == "finish":
            yield f"final: {reply['content']}"
            return
        call = lesson_01.ToolCall(reply["action"], reply.get("args", {}))
        observation = agent.tools.dispatch(call)
        history.append(lesson_01.Turn("action", call.name, call, observation))
        yield f"{call.name} -> {observation}"
    yield "final: budget exhausted"


# ---------------------------------------------------------------------------
# Exercise 1 - the lesson 01 loop in all six runtime shapes
#
# One agent, six shells. Every shell must return the same answer; what
# differs is who waits, what survives a crash and what starts the run.
# ---------------------------------------------------------------------------

FITS = {
    "request-response": "chat turn or API call that finishes in seconds",
    "streaming": "anything a person watches: show each step as it happens",
    "durable": "long runs you cannot afford to restart from zero",
    "queue": "background jobs, batch work, anything that can wait for a worker",
    "event-driven": "reactions: a PR opened, an alert fired, an email arrived",
    "scheduled": "nightly evals, weekly reports, memory consolidation",
}


def request_response(question: str) -> str:
    return list(agent_steps(question))[-1]


def durable(question: str, checkpoint: Path, crash_after: int | None = None) -> str:
    """Checkpoint after every step; a restarted run picks up where the record ends.

    ponytail: recovery replays the earlier tool calls to rebuild the agent's in-memory state,
    which is right for these idempotent toy tools. With real side effects, store each result
    and skip the call on replay.
    """
    done: list[str] = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    for index, step in enumerate(agent_steps(question)):
        if index < len(done):
            continue
        done.append(step)
        checkpoint.write_text(json.dumps(done))
        if crash_after is not None and len(done) == crash_after:
            raise RuntimeError("worker died")
    return done[-1]


def due(schedule: list[tuple[str, str]], now: datetime, fired: set[tuple[str, str]]) -> list[str]:
    """Cron stand-in: events whose HH:MM has passed today and that have not fired today."""
    ready = [event for hhmm, event in schedule
             if now.strftime("%H:%M") >= hhmm and (now.date().isoformat(), event) not in fired]
    fired.update((now.date().isoformat(), event) for event in ready)
    return ready


def ex1_six_shapes() -> None:
    answers: dict[str, str] = {"request-response": request_response(QUESTION)}

    chunks = list(agent_steps(QUESTION))                        # streaming: the generator itself
    answers["streaming"] = chunks[-1]

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "run.json"
        try:
            durable(QUESTION, checkpoint, crash_after=3)
        except RuntimeError:
            saved = len(json.loads(checkpoint.read_text()))
        answers["durable"] = durable(QUESTION, checkpoint)
        recorded = json.loads(checkpoint.read_text())

    lesson._agent_fn = lambda payload: list(agent_steps(payload))  # QueueRuntime.worker calls main._agent_fn
    try:
        runtime = QueueRuntime()
        runtime.enqueue(QUESTION)
        answers["queue"] = runtime.worker(fail_policy=lambda job: False)[0][1]
    finally:
        lesson._agent_fn = _original_agent_fn

    bus = EventBus()
    bus.subscribe("question.asked", request_response)
    answers["event-driven"] = bus.publish("question.asked", QUESTION)[0][1]

    fired: set[tuple[str, str]] = set()
    clock, scheduled_runs = datetime(2026, 10, 8, 0, 0), []
    for _ in range(96):                                         # two days in half-hour ticks
        for event in due([("02:00", "question.asked")], clock, fired):
            scheduled_runs += bus.publish(event, QUESTION)
        clock += timedelta(minutes=30)
    answers["scheduled"] = scheduled_runs[-1][1]

    for shape, answer in answers.items():
        print(f"  {shape:<17} {answer:<45} fits: {FITS[shape]}")
    print(f"  streaming delivered {len(chunks)} chunks; durable crashed after {saved} steps and resumed to {len(recorded)}; "
          f"the schedule fired {len(scheduled_runs)} times in two days")
    assert set(answers.values()) == {"final: the total including 15% tax is 138.0"}
    assert saved == 3 and len(recorded) == len(chunks) == 6 and len(scheduled_runs) == 2


_original_agent_fn = lesson._agent_fn


# ---------------------------------------------------------------------------
# Exercise 2 - a dead-letter queue under 10% failure
#
# main.QueueRuntime already retries twice and then dead-letters. What decides
# the DLQ's size is the kind of failure, not the rate. If each attempt fails
# 10% of the time on its own, three in a row is a 1-in-1000 event and the DLQ
# stays nearly empty. If 10% of jobs are poison and fail every time, every
# one of them lands there after burning three attempts.
# (worker() asks fail_policy twice per attempt, so the policy remembers its
# answer; a fresh coin flip each time would dead-letter jobs on attempt one.)
# ---------------------------------------------------------------------------

def failing(rate: float, seed: int, poison: bool) -> Callable[[Job], bool]:
    rng, decided = random.Random(seed), {}

    def policy(job: Job) -> bool:
        key = job.jid if poison else (job.jid, job.attempt)
        if key not in decided:
            decided[key] = rng.random() < rate
        return decided[key]

    return policy


def ex2_dlq_under_failure() -> None:
    sizes = {}
    for label, poison in (("transient, 10% per attempt", False), ("poison, 10% of jobs", True)):
        runtime = QueueRuntime()
        for n in range(200):
            runtime.enqueue(f"job {n}")
        results = runtime.worker(fail_policy=failing(0.10, seed=0, poison=poison))
        retries = sum(status == "retry" for _, status in results)
        completed = sum(status.startswith("final") for _, status in results)
        sizes[label] = (len(runtime.dlq), completed)
        print(f"  {label:<27} completed {completed:>3}, retries {retries:>2}, DLQ size {len(runtime.dlq):>2} "
              f"{[job.jid for job in runtime.dlq][:4]}")
    assert sizes["transient, 10% per attempt"][0] <= 2
    assert 10 <= sizes["poison, 10% of jobs"][0] <= 30
    assert all(dlq + completed == 200 for dlq, completed in sizes.values())     # nothing vanished


# ---------------------------------------------------------------------------
# Exercise 3 - a cron-triggered eval agent over the day's top 20 traces
#
# SYNTHETIC TRACES and a scripted judge. The schedule is the stand-in from
# exercise 1 driving main's EventBus; in production the same handler hangs
# off a real cron entry (`0 2 * * *`). Each night it takes the 20 most
# expensive traces of the previous day and reports how many pass.
# ---------------------------------------------------------------------------

def traces_for(day: str, count: int = 60) -> list[dict[str, Any]]:
    rng = random.Random(day)
    return [{"id": f"{day}-{n:02d}", "tokens": rng.randint(200, 6000), "steps": rng.randint(2, 14),
             "error": rng.random() < 0.12} for n in range(count)]


def nightly_eval(day: str) -> str:
    top = sorted(traces_for(day), key=lambda trace: -trace["tokens"])[:20]
    failed = [trace["id"] for trace in top if trace["error"] or trace["steps"] > 10]
    return f"{day}: {20 - len(failed)}/20 of the costliest traces pass; failing: {failed[:3]}"


def ex3_nightly_eval() -> None:
    bus = EventBus()
    bus.subscribe("eval.nightly", nightly_eval)
    fired: set[tuple[str, str]] = set()
    clock, reports = datetime(2026, 10, 8, 0, 0), []
    for _ in range(144):                                        # three days in half-hour ticks
        for event in due([("02:00", "eval.nightly")], clock, fired):
            yesterday = (clock - timedelta(days=1)).date().isoformat()
            reports += [report for _, report in bus.publish(event, yesterday)]
        clock += timedelta(minutes=30)
    for report in reports:
        print(f"  {report}")
    assert len(reports) == 3 and [r[:10] for r in reports] == ["2026-10-07", "2026-10-08", "2026-10-09"]


# ---------------------------------------------------------------------------
# Exercise 4 - streaming with backpressure
#
# The agent runs in its own thread and pushes chunks to the client through a
# buffer. With an unbounded buffer the agent never waits and the buffer grows
# as fast as the client is slow. With a bounded one the agent blocks when the
# client falls behind: it is paused.
#
# How that meets a turn budget: a budget counted in turns or tokens does not
# notice the pause. A wall-clock budget does. Measured naively it expires
# while the agent is only waiting, and a slow reader gets a truncated answer.
# Time spent blocked has to be taken off the clock.
# (A plain generator, like main.streaming, has backpressure built in: it only
# advances when the client asks for the next chunk.)
# ---------------------------------------------------------------------------

def stream(buffer_size: int, deadline_s: float | None = None, count_blocked_time: bool = True,
           steps: int = 8, step_s: float = 0.005, client_s: float = 0.02) -> dict[str, Any]:
    chunks: queue.Queue[str | None] = queue.Queue(maxsize=buffer_size)
    stats: dict[str, Any] = {"blocked_s": 0.0, "max_buffered": 0, "sent": 0}

    def agent() -> None:
        start = time.perf_counter()
        for i in range(steps):
            time.sleep(step_s)                                  # the model working on this step
            elapsed = time.perf_counter() - start
            if deadline_s is not None and elapsed - (0 if count_blocked_time else stats["blocked_s"]) > deadline_s:
                break
            before = time.perf_counter()
            chunks.put(f"chunk {i}")                            # blocks while the buffer is full
            stats["blocked_s"] += time.perf_counter() - before
            stats["sent"] += 1
            stats["max_buffered"] = max(stats["max_buffered"], chunks.qsize())
        stats["agent_wall_s"] = time.perf_counter() - start
        chunks.put(None)

    worker = threading.Thread(target=agent)
    worker.start()
    received = 0
    while chunks.get() is not None:
        received += 1
        time.sleep(client_s)                                    # a slow client
    worker.join()
    return {**stats, "received": received}


def ex4_backpressure() -> None:
    runs = {
        "unbounded buffer": stream(buffer_size=0),
        "buffer of 1": stream(buffer_size=1),
        "buffer of 1, 80 ms wall-clock budget": stream(buffer_size=1, deadline_s=0.08),
        "same budget, blocked time excluded": stream(buffer_size=1, deadline_s=0.08, count_blocked_time=False),
    }
    for label, r in runs.items():
        print(f"  {label:<37} sent {r['sent']}/8, agent ran {r['agent_wall_s'] * 1000:>4.0f} ms "
              f"(blocked {r['blocked_s'] * 1000:>4.0f} ms), most buffered {r['max_buffered']}")
    assert runs["unbounded buffer"]["max_buffered"] > runs["buffer of 1"]["max_buffered"] == 1
    assert runs["buffer of 1"]["agent_wall_s"] > 2 * runs["unbounded buffer"]["agent_wall_s"]      # it was paused
    assert runs["buffer of 1, 80 ms wall-clock budget"]["sent"] < 8                                # cut off while waiting
    assert runs["same budget, blocked time excluded"]["sent"] == 8


# ---------------------------------------------------------------------------
# Exercise 5 - when to move a self-hosted long-horizon agent to managed
#
# From the Claude Managed Agents documentation: an agent there is a stored,
# versioned configuration, each session runs in a hosted container where its
# tools execute, events stream back, and sessions can be started on a
# schedule. It is a beta, and it is not offered through Bedrock, Vertex AI or
# Foundry. The rule below is that reading turned into a check: move when at
# least one thing pulls you there and nothing rules it out.
# ---------------------------------------------------------------------------

REASONS_TO_MOVE = {
    "runs_outlive_workers": "runs last longer than your worker processes or deploys do",
    "you_run_the_sandbox": "you are operating the container that the agent's shell and file edits run in",
    "needs_schedule": "runs should start on a schedule without a scheduler of your own",
    "needs_versioned_config": "prompt and tool changes must not disturb sessions already running",
}
REASONS_TO_STAY = {
    "third_party_cloud_only": "the model is reached through Bedrock, Vertex AI or Foundry",
    "beta_not_acceptable": "a beta surface is not acceptable for this system",
    "lives_in_your_process": "the agent is part of a CLI or desktop tool working on local files",
}


def move_to_managed(agent: dict[str, bool]) -> tuple[bool, list[str]]:
    stay = [reason for key, reason in REASONS_TO_STAY.items() if agent.get(key)]
    move = [reason for key, reason in REASONS_TO_MOVE.items() if agent.get(key)]
    return (bool(move) and not stay), stay or move


def ex5_self_hosted_or_managed() -> None:
    agents = {
        "nightly repo-maintenance agent": {"runs_outlive_workers": True, "needs_schedule": True, "you_run_the_sandbox": True},
        "same agent, model on Bedrock": {"runs_outlive_workers": True, "needs_schedule": True, "third_party_cloud_only": True},
        "local coding assistant": {"lives_in_your_process": True},
        "ten-second support reply": {},
    }
    decisions = {name: move_to_managed(profile) for name, profile in agents.items()}
    for name, (move, reasons) in decisions.items():
        print(f"  {name:<31} {'move to managed' if move else 'stay self-hosted':<17} {reasons[0] if reasons else 'nothing pulls it there'}")
    assert [move for move, _ in decisions.values()] == [True, False, False, False]


if __name__ == "__main__":
    print("Phase 14 - Lesson 29: Production Runtimes - exercises")
    for exercise in (ex1_six_shapes, ex2_dlq_under_failure, ex3_nightly_eval, ex4_backpressure,
                     ex5_self_hosted_or_managed):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
