"""Phase 14 - Lesson 13: Stateful Graph Orchestration - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import copy
import inspect
import json
import operator
import sqlite3
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator

from main import END, InMemoryCheckpointer, PausedAtNode, Runner, State, StateGraph, Update, _classify, build_graph

# ---------------------------------------------------------------------------
# Exercise 1 - end the run when classification confidence is low, then resume
#
# main._classify silently falls back to "sales" when no keyword matches. Here
# that fallback carries a low confidence, a conditional edge sends low
# confidence to END, and nothing downstream runs on a guess. A human then sets
# `route` on the checkpointed state and the run resumes at that node.
# ---------------------------------------------------------------------------

KEYWORDS = ("refund", "money back", "crash", "bug", "error", "pricing", "quote")


def classify_with_confidence(state: State) -> Update:
    matched = any(word in state["input"].lower() for word in KEYWORDS)
    return {**_classify(state), "confidence": 0.9 if matched else 0.3}


def build_gated_graph(threshold: float = 0.6) -> StateGraph:
    graph = build_graph()
    graph.nodes["classify"] = classify_with_confidence
    graph.edges["classify"] = []
    graph.add_conditional_edges(
        "classify",
        router=lambda s: "unsure" if s["confidence"] < threshold else s["route"],
        targets={"unsure": END, "refund": "refund", "bug": "bug", "sales": "sales"},
    )
    return graph


def ex1_low_confidence_exit() -> None:
    checkpoints = InMemoryCheckpointer()
    runner = Runner(build_gated_graph(), checkpoints)
    initial: State = {"input": "the export thing is acting weird again", "step": 0, "human_approval": True}

    stopped = runner.run("s1", initial)
    print(f"  first run   : stopped after {[n for n, _ in checkpoints.history('s1')]}, "
          f"guess={stopped['route']!r} at confidence {stopped['confidence']}, ticket={stopped.get('ticket')}")
    assert "ticket" not in stopped and "output" not in stopped

    _, saved = checkpoints.load_latest("s1")
    corrected = {**saved, "route": "bug", "confidence": 1.0}       # the human's decision
    final = runner.run("s1", initial, resume_from=corrected["route"], state_override=corrected)
    print(f"  after human : {[n for n, _ in checkpoints.history('s1')]} -> {final['output']}")
    assert final["output"].startswith("sent BUG-")
    assert [n for n, _ in checkpoints.history("s1")] == ["classify", "bug", "human_gate", "send"]


# ---------------------------------------------------------------------------
# Exercise 2 - a real SQLite checkpointer, and what it costs per step
#
# Same three methods as main.InMemoryCheckpointer, so Runner takes either.
# The in-memory one pays a deepcopy per step; SQLite pays JSON encoding plus an
# insert plus a commit to disk, and the commit dominates. What that buys is the
# thing the lesson is about: the checkpoint survives the process.
# State must now be JSON-serialisable, which deepcopy never required.
# ---------------------------------------------------------------------------

class SqliteCheckpointer:
    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS checkpoints ("
                        "seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, step_name TEXT, state TEXT)")

    def save(self, session_id: str, step_name: str, state: State) -> None:
        self.db.execute("INSERT INTO checkpoints (session_id, step_name, state) VALUES (?, ?, ?)",
                        (session_id, step_name, json.dumps(state)))
        self.db.commit()

    def history(self, session_id: str) -> list[tuple[str, State]]:
        rows = self.db.execute("SELECT step_name, state FROM checkpoints WHERE session_id = ? ORDER BY seq",
                               (session_id,))
        return [(name, json.loads(state)) for name, state in rows]

    def load_latest(self, session_id: str) -> tuple[str, State] | None:
        history = self.history(session_id)
        return history[-1] if history else None

    def close(self) -> None:
        self.db.close()


def microseconds_per_save(checkpointer: Any, state: State, saves: int = 100) -> float:
    start = time.perf_counter()
    for i in range(saves):
        checkpointer.save("bench", f"step{i}", state)
    return (time.perf_counter() - start) / saves * 1e6


def ex2_sqlite_checkpointer() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoints.db"
        initial: State = {"input": "the CLI crashes on ctrl-c, please fix", "step": 0, "human_approval": False}

        first_process = SqliteCheckpointer(path)
        try:
            Runner(build_graph(), first_process).run("s1", initial)
        except PausedAtNode as paused:
            print(f"  process 1: paused at {paused.node}, then exits")
        first_process.close()

        second_process = SqliteCheckpointer(path)           # a new connection: nothing in memory carried over
        node, saved = second_process.load_latest("s1")
        approved = {k: v for k, v in saved.items() if k != "_pause_reason"} | {"human_approval": True}
        final = Runner(build_graph(), second_process).run("s1", initial, resume_from="send", state_override=approved)
        print(f"  process 2: loaded checkpoint {node!r} from disk -> {final['output']}")
        assert node == "human_gate" and final["output"].startswith("sent BUG-")

        for label, state in (("small state (4 fields)", saved), ("large state (+50 KB)  ", {**saved, "doc": "x" * 50_000})):
            in_memory = microseconds_per_save(InMemoryCheckpointer(), state)
            on_disk = microseconds_per_save(second_process, state)
            print(f"  {label}: in-memory {in_memory:>7.1f} us/step   sqlite {on_disk:>8.1f} us/step")
            assert on_disk > 0
        second_process.close()


# ---------------------------------------------------------------------------
# Exercise 3 - parallel edges merged by a reducer
#
# What immutable state buys: both nodes read the same snapshot, so neither can
# see or corrupt the other's work, and the result does not depend on which one
# finishes first. Merging happens once, at the end of the step, in declaration
# order, through a reducer you chose. Two writers to a key with no reducer is
# an error rather than a silent last-write-wins.
# ---------------------------------------------------------------------------

Reducer = Callable[[Any, Any], Any]


class ConflictingUpdate(Exception):
    pass


def run_parallel(state: State, nodes: dict[str, Callable[[State], Update]],
                 reducers: dict[str, Reducer]) -> State:
    with ThreadPoolExecutor() as pool:
        futures = {name: pool.submit(fn, copy.deepcopy(state)) for name, fn in nodes.items()}
        updates = {name: future.result() or {} for name, future in futures.items()}     # declaration order
    merged, written_by = dict(state), {}
    for name, update in updates.items():
        for key, value in update.items():
            if key in reducers:
                merged[key] = reducers[key](merged[key], value)
            elif key in written_by:
                raise ConflictingUpdate(f"{written_by[key]} and {name} both wrote {key!r} and it has no reducer")
            else:
                merged[key] = value
            written_by[key] = name
    return merged


def ex3_parallel_edges() -> None:
    def make_nodes(lint_delay: float, scan_delay: float) -> dict[str, Callable[[State], Update]]:
        def lint(state: State) -> Update:
            time.sleep(lint_delay)
            state["findings"].append("scribbled on my own copy")        # a badly behaved node
            return {"findings": ["lint: unused import"], "step": state["step"] + 1}

        def security_scan(state: State) -> Update:
            time.sleep(scan_delay)
            return {"findings": ["security: eval() call"], "step": state["step"] + 1}

        return {"lint": lint, "security_scan": security_scan}

    start: State = {"input": "diff #42", "findings": [], "step": 0}
    try:
        run_parallel(start, make_nodes(0, 0), {"findings": operator.add})
        raise AssertionError("two writers to 'step' should conflict")
    except ConflictingUpdate as conflict:
        print(f"  no reducer for step : {conflict}")

    reducers = {"findings": operator.add, "step": max}
    lint_slow = run_parallel(start, make_nodes(0.05, 0.0), reducers)
    scan_slow = run_parallel(start, make_nodes(0.0, 0.05), reducers)
    print(f"  merged              : {lint_slow}")
    assert lint_slow == scan_slow                                       # finish order does not matter
    assert lint_slow["findings"] == ["lint: unused import", "security: eval() call"] and lint_slow["step"] == 1
    assert start["findings"] == []                                      # the snapshot was never touched


# ---------------------------------------------------------------------------
# Exercise 4 - create_supervisor from langgraph-supervisor
#
# real_supervisor() needs langgraph, langgraph-supervisor and a chat model; it
# follows the package README. The comparison of trace shapes below is made on
# main's StateGraph, wired the way a supervisor is.
#
# The shapes differ in three ways. The static graph visits each node once,
# along edges fixed in code; the supervisor graph is a star, and control comes
# back to the supervisor after every worker, so it runs once per hop plus once
# to finish. Each of those supervisor turns is an LLM call that picks the next
# worker through a handoff tool (transfer_to_<agent>). And the state is a
# growing message list rather than named fields, so what a worker "returned"
# is whatever it appended.
# ---------------------------------------------------------------------------

WORKERS = {"triage_agent": "this is a bug in the CLI", "ticket_agent": "opened BUG-1042",
           "reply_agent": "told the user it is logged"}


def build_supervisor_graph() -> StateGraph:
    graph = StateGraph()

    def supervisor(state: State) -> Update:
        pending = [name for name in WORKERS if name not in state["done"]]
        choice = pending[0] if pending else "FINISH"                    # scripted stand-in for the LLM's handoff
        note = f"supervisor: transfer_to_{choice}" if pending else "supervisor: finished"
        return {"next": choice, "messages": state["messages"] + [note]}

    def worker(name: str) -> Callable[[State], Update]:
        return lambda state: {"messages": state["messages"] + [f"{name}: {WORKERS[name]}"],
                              "done": state["done"] + [name]}

    graph.add_node("supervisor", supervisor)
    for name in WORKERS:
        graph.add_node(name, worker(name))
        graph.add_edge(name, "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: s["next"], {**{n: n for n in WORKERS}, "FINISH": END})
    graph.set_entry("supervisor")
    return graph


def real_supervisor(model: Any, tools: dict[str, list[Callable[..., Any]]]) -> Any:
    """The same team on the real package. `model` is any LangChain chat model."""
    from langgraph.prebuilt import create_react_agent
    from langgraph_supervisor import create_supervisor

    agents = [create_react_agent(model=model, tools=tools[name], name=name) for name in WORKERS]
    workflow = create_supervisor(agents, model=model, prompt="You are a support supervisor managing three agents.")
    return workflow.compile()       # .invoke({"messages": [{"role": "user", "content": "..."}]})


def ex4_supervisor_shape() -> None:
    static_trace, star_trace = InMemoryCheckpointer(), InMemoryCheckpointer()
    Runner(build_graph(), static_trace).run(
        "s", {"input": "the CLI crashes on ctrl-c", "step": 0, "human_approval": True})
    final = Runner(build_supervisor_graph(), star_trace).run(
        "s", {"messages": ["user: the CLI crashes on ctrl-c"], "done": []})
    static_nodes = [node for node, _ in static_trace.history("s")]
    star_nodes = [node for node, _ in star_trace.history("s")]
    print(f"  static graph     : {static_nodes}")
    print(f"  supervisor graph : {star_nodes}")
    print(f"  supervisor turns : {star_nodes.count('supervisor')} for {len(WORKERS)} workers; "
          f"state is {len(final['messages'])} messages")
    assert star_nodes.count("supervisor") == len(WORKERS) + 1
    assert len(static_nodes) == len(set(static_nodes))              # every node exactly once
    try:
        import langgraph_supervisor  # noqa: F401
        print("  langgraph-supervisor is installed: call real_supervisor(model, tools) with a chat model")
    except ImportError:
        print("  real create_supervisor: pip install langgraph-supervisor, then call real_supervisor(model, tools)")


# ---------------------------------------------------------------------------
# Exercise 5 - streaming partial state
#
# A node may be a generator. Each value it yields is a partial update, merged
# into state and handed to the caller straight away. Checkpoints stay at node
# boundaries: a half-written reply is not a state anyone should resume from.
# ---------------------------------------------------------------------------

class StreamingRunner(Runner):
    def stream(self, session_id: str, initial_state: State) -> Iterator[tuple[str, Update]]:
        state = copy.deepcopy(initial_state)
        current = self.graph.entry
        while current is not None and current != END:
            result = self.graph.nodes[current](state)
            for delta in (result if inspect.isgenerator(result) else [result or {}]):
                state = {**state, **delta}
                yield current, delta
            self.checkpointer.save(session_id, current, state)
            current = self.graph._next(current, state)


def draft_reply(state: State) -> Iterator[Update]:
    reply = ""
    for chunk in ("Thanks for the report.", " We logged it as a", f" {state['route']} ticket", " and will follow up."):
        reply += chunk
        yield {"reply": reply}


def ex5_streaming() -> None:
    graph = StateGraph()
    graph.add_node("classify", _classify)
    graph.add_node("draft_reply", draft_reply)
    graph.set_entry("classify")
    graph.add_edge("classify", "draft_reply")
    graph.add_edge("draft_reply", END)

    checkpoints = InMemoryCheckpointer()
    deltas = []
    for node, delta in StreamingRunner(graph, checkpoints).stream("s", {"input": "it crashes on start", "step": 0}):
        deltas.append((node, delta))
        print(f"  [{node}] {delta}")
    history = checkpoints.history("s")
    assert [node for node, _ in deltas] == ["classify"] + ["draft_reply"] * 4
    assert [node for node, _ in history] == ["classify", "draft_reply"]      # one checkpoint per node
    assert history[-1][1]["reply"].endswith("will follow up.")


if __name__ == "__main__":
    print("Phase 14 - Lesson 13: Stateful Graph Orchestration - exercises")
    for exercise in (ex1_low_confidence_exit, ex2_sqlite_checkpointer, ex3_parallel_edges,
                     ex4_supervisor_shape, ex5_streaming):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall checks passed")
