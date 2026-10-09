"""Phase 14 - Lesson 17: The Harness as a Library - Subagents and Session Store - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from main import AgentRun, Harness, Hooks, SessionStore, Tool, ToolRegistry

ToolCalls = list[tuple[str, dict[str, Any]]]


def demo_harness(harness_class: type[Harness] = Harness, hooks: Hooks | None = None) -> Harness:
    tools = ToolRegistry()
    tools.register(Tool("read_file", "read a file", lambda path: f"[content of {path}: 42 lines]"))
    tools.register(Tool("write_file", "write a file", lambda path, content: f"wrote {len(content)} chars to {path}"))
    return harness_class(tools, hooks or Hooks(), SessionStore())


# ---------------------------------------------------------------------------
# Exercise 1 - 20 tasks in groups of 5, against one subagent per task
#
# Only a subagent's final output comes back to the orchestrator, so the
# orchestrator's context grows by one result per subagent. One subagent per
# task returns 20 results; batching five tasks into each returns 4. The price
# is paid inside the subagents: each batched one carries five tasks' worth of
# tool output, and a failure now costs five tasks instead of one.
# The four batched subagents run concurrently.
# ---------------------------------------------------------------------------

def spawn_batched(harness: Harness, parent: str, tasks: list[tuple[str, ToolCalls]],
                  batch_size: int) -> list[AgentRun]:
    batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    def run_batch(numbered: tuple[int, list[tuple[str, ToolCalls]]]) -> AgentRun:
        number, batch = numbered
        prompt = "; ".join(prompt for prompt, _ in batch)
        calls = [call for _, tool_calls in batch for call in tool_calls]
        return harness.run_agent(f"{parent}.batch{number:02d}", prompt, calls, parent_session=parent)

    with ThreadPoolExecutor(max_workers=5) as pool:
        return list(pool.map(run_batch, enumerate(batches, 1)))


def ex1_batched_subagents() -> None:
    tasks = [(f"find the test file for module_{i:02d}", [("read_file", {"path": f"module_{i:02d}.py"})])
             for i in range(20)]
    results = {}
    for label, batch_size in (("one subagent per task", 1), ("five tasks per subagent", 5)):
        harness = demo_harness()
        orchestrator = harness.run_agent("main", "find the test file for each of these 20 modules", [])
        runs = spawn_batched(harness, "main", tasks, batch_size)
        returned = sum(len(run.output.split()) for run in runs)          # what lands in the orchestrator
        results[label] = (len(runs), orchestrator.context_tokens + returned, max(r.context_tokens for r in runs))
        print(f"  {label:<24}: {len(runs):>2} subagents, orchestrator context {results[label][1]:>3} tokens, "
              f"largest subagent context {results[label][2]:>3} tokens")
    per_task, batched = results.values()
    assert (per_task[0], batched[0]) == (20, 4)
    assert batched[1] < per_task[1] and batched[2] > per_task[2]


# ---------------------------------------------------------------------------
# Exercise 2 - a PreToolUse hook that rate-limits write_file
#
# main's pre_tool_use hooks can observe a call but not stop it: a hook that
# raises takes the whole run down with it. GatedHarness lets a hook raise
# ToolDenied, and the denial becomes the tool's result, so the model reads
# "denied: ..." and can wait or do something else. That mirrors the real SDK,
# where a PreToolUse hook answers with permissionDecision "deny" and a reason.
# The limit is a sliding window per session with an injectable clock.
# ---------------------------------------------------------------------------

class ToolDenied(Exception):
    pass


class GatedHarness(Harness):
    current_session = ""

    def run_agent(self, session_id: str, prompt: str, tool_calls: ToolCalls,
                  parent_session: str | None = None) -> AgentRun:
        self.current_session = session_id       # ponytail: one run at a time; pass it explicitly for threads
        return super().run_agent(session_id, prompt, tool_calls, parent_session)

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> str:
        try:
            return super()._dispatch(tool_name, args)
        except ToolDenied as denied:
            return f"denied: {denied}"


def write_rate_limit(harness: GatedHarness, clock: Callable[[], float], limit: int = 5,
                     window_s: float = 60.0) -> Callable[[str, dict[str, Any]], None]:
    history: dict[str, deque[float]] = {}

    def hook(tool_name: str, args: dict[str, Any]) -> None:
        if tool_name != "write_file":
            return
        now = clock()
        recent = history.setdefault(harness.current_session, deque())
        while recent and now - recent[0] >= window_s:
            recent.popleft()
        if len(recent) >= limit:
            raise ToolDenied(f"write_file limit of {limit} per {window_s:.0f}s reached; "
                             f"retry in {window_s - (now - recent[0]):.0f}s")
        recent.append(now)

    return hook


def ex2_rate_limit_hook() -> None:
    now = [0.0]
    harness = demo_harness(GatedHarness)
    harness.hooks.pre_tool_use.append(write_rate_limit(harness, clock=lambda: now[0]))
    write = ("write_file", {"path": "notes.md", "content": "hello"})

    first = harness.run_agent("session_a", "write seven notes", [write] * 7 + [("read_file", {"path": "notes.md"})])
    for _, _, result in first.tool_calls:
        print(f"  session_a: {result}")
    other = harness.run_agent("session_b", "write one note", [write])
    now[0] += 61
    later = harness.run_agent("session_a", "one more", [write])
    print(f"  session_b: {other.tool_calls[0][2]}")
    print(f"  session_a, 61s later: {later.tool_calls[0][2]}")
    results = [result for _, _, result in first.tool_calls]
    assert [r.startswith("wrote") for r in results[:7]] == [True] * 5 + [False] * 2
    assert results[5].startswith("denied: write_file limit") and results[7].startswith("[content")
    assert other.tool_calls[0][2].startswith("wrote") and later.tool_calls[0][2].startswith("wrote")


# ---------------------------------------------------------------------------
# Exercise 3 - list_subkeys rendered as a subagent tree
#
# What deep nesting looks like: every level is a full agent with its own
# context, and a result has to be summarised once per level on its way up, so
# detail is lost and latency is added at each hop. It also exposed a bug in
# main: SessionStore.delete removes a session's children but not their
# children, so deleting the root of a deep tree leaves orphans behind.
# (The real SDK lets subagents spawn subagents, three levels deep by default.)
# ---------------------------------------------------------------------------

def render_tree(store: SessionStore, session_id: str, indent: str = "") -> list[str]:
    lines = [f"{indent}{session_id}  ({len(store.load(session_id))} turns)"]
    for child in store.list_subkeys(session_id):
        lines += render_tree(store, child, indent + "    ")
    return lines


def delete_tree(store: SessionStore, session_id: str) -> None:
    for child in store.list_subkeys(session_id):
        delete_tree(store, child)
    store.delete(session_id)


def build_nested_sessions(harness: Harness) -> None:
    read = [("read_file", {"path": "a.py"})]
    harness.run_agent("main", "audit the repo", [])
    backend, frontend = harness.spawn_subagents("main", [("audit backend", read), ("audit frontend", read)])
    api, _ = harness.spawn_subagents(backend.session_id, [("audit api", read), ("audit db", read)])
    harness.spawn_subagents(api.session_id, [("audit auth routes", read)])


def ex3_subagent_tree() -> None:
    harness = demo_harness()
    build_nested_sessions(harness)
    tree = render_tree(harness.store, "main")
    print("  " + "\n  ".join(tree))
    assert len(tree) == 6 and tree[3].startswith(" " * 12)             # the deepest session is three levels down

    harness.store.delete("main")
    orphans = harness.store.list_sessions()
    print(f"  after main's delete('main'): {len(orphans)} orphaned sessions {orphans}")
    assert len(orphans) == 3

    harness = demo_harness()
    build_nested_sessions(harness)
    delete_tree(harness.store, "main")
    assert harness.store.list_sessions() == []


# ---------------------------------------------------------------------------
# Exercise 4 - port to the real claude-agent-sdk package
#
# sdk_options() needs the claude-agent-sdk package. It follows the SDK's
# custom tools, hooks and subagents pages.
#
# What changes about tool registration:
#   - built-in tools (Read, Write, Bash, ...) are not registered at all. They
#     exist; allowed_tools decides which run without a permission prompt;
#   - a custom tool is an async function decorated with
#     @tool(name, description, input_schema). It receives one dict of
#     validated arguments and returns {"content": [{"type": "text", ...}]}
#     rather than a bare string;
#   - tools are not handed to the agent directly. They are wrapped in an
#     in-process MCP server and the model sees them as mcp__<server>__<tool>;
#   - hooks return decisions instead of raising, and are scoped by a matcher
#     on the tool name;
#   - subagents are declared up front (AgentDefinition) and the model decides
#     when to call one through the Agent tool; main spawns them imperatively.
# ---------------------------------------------------------------------------

def sdk_options() -> Any:
    from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, HookMatcher, create_sdk_mcp_server, tool

    @tool("find_test_file", "Find the test file for a module", {"module": str})
    async def find_test_file(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": f"tests/test_{args['module']}.py"}]}

    async def protect_env_files(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        if input_data["tool_input"].get("file_path", "").endswith(".env"):
            return {"hookSpecificOutput": {"hookEventName": input_data["hook_event_name"],
                                           "permissionDecision": "deny",
                                           "permissionDecisionReason": "Cannot modify .env files"}}
        return {}

    return ClaudeAgentOptions(
        mcp_servers={"repo": create_sdk_mcp_server(name="repo", version="1.0.0", tools=[find_test_file])},
        allowed_tools=["Read", "Grep", "Glob", "Agent", "mcp__repo__find_test_file"],
        hooks={"PreToolUse": [HookMatcher(matcher="Write|Edit", hooks=[protect_env_files])]},
        agents={"module-reviewer": AgentDefinition(
            description="Reviews one module. Use for per-module review work.",
            prompt="You review a single Python module and report problems concisely.",
            tools=["Read", "Grep", "Glob"])},
    )       # run with: async for message in query(prompt="review these three modules", options=sdk_options())


def ex4_sdk_port() -> str | None:
    try:
        options = sdk_options()
    except ImportError:
        print("  needs claude-agent-sdk: pip install claude-agent-sdk")
        return "claude-agent-sdk"
    print(f"  built ClaudeAgentOptions with tools {options.allowed_tools}; pass it to query() to run")
    return None


# ---------------------------------------------------------------------------
# Exercise 5 - self-hosted Agent SDK or Claude Managed Agents
#
# The Agent SDK gives you the harness (the loop, the tools, hooks, subagents)
# and nothing else: you run the process, keep it alive, store its sessions and
# sandbox its shell. Managed Agents gives you the harness and the deployment:
# an agent is a stored, versioned config, and each session gets a hosted
# container where its tools execute, with events streamed back to you.
#
# The answer is written as a rule so it can be checked: switch when at least
# one thing pulls the agent there and nothing rules it out.
# ---------------------------------------------------------------------------

SWITCH_TO_MANAGED_WHEN = {
    "outlives_process": "runs outlive your process: hours-long or asynchronous work you would otherwise babysit with a job queue",
    "scheduled": "the agent should run on a schedule; deployments fire sessions without a scheduler of your own",
    "sandbox_is_a_burden": "you do not want to own the sandbox; bash and file edits happen in a hosted per-session container",
    "versioned_config": "you need versioned agent configs: update the prompt without breaking running sessions, pin, roll back",
    "graded_outcomes": "work must meet a rubric; outcomes re-run the agent against a grader until it passes",
}
STAY_SELF_HOSTED_WHEN = {
    "in_your_program": "the agent lives inside your own program: a CLI, a CI job or a desktop tool working on local files "
                       "(if only tool execution must stay in your network, Managed Agents has self-hosted sandboxes for that)",
    "third_party_cloud": "you deploy through Bedrock, Vertex AI or Foundry, where Managed Agents is not available",
    "in_process_hooks": "you need in-process hooks and custom control over every tool call",
    "no_beta": "a beta surface is not acceptable for this system yet",
}


def where_to_run(traits: set[str]) -> str:
    pulled, held = traits & set(SWITCH_TO_MANAGED_WHEN), traits & set(STAY_SELF_HOSTED_WHEN)
    return "managed" if pulled and not held else "self-hosted"


def ex5_managed_or_self_hosted() -> None:
    print("  switch to Managed Agents when:")
    for reason in SWITCH_TO_MANAGED_WHEN.values():
        print(f"    - {reason}")
    print("  stay on the self-hosted Agent SDK when:")
    for reason in STAY_SELF_HOSTED_WHEN.values():
        print(f"    - {reason}")
    agents = {"nightly repo-maintenance agent": {"outlives_process", "scheduled", "sandbox_is_a_burden"},
              "the same agent, model on Bedrock": {"outlives_process", "scheduled", "third_party_cloud"},
              "coding assistant in a CLI": {"in_your_program", "in_process_hooks"},
              "ten-second support reply": set()}
    decisions = {name: where_to_run(traits) for name, traits in agents.items()}
    for name, decision in decisions.items():
        print(f"  {name:<33} -> {decision}")
    assert list(decisions.values()) == ["managed", "self-hosted", "self-hosted", "self-hosted"]
    assert all(trait in {**SWITCH_TO_MANAGED_WHEN, **STAY_SELF_HOSTED_WHEN} for traits in agents.values() for trait in traits)


if __name__ == "__main__":
    print("Phase 14 - Lesson 17: Subagents and Session Store - exercises")
    outcomes = []
    for exercise in (ex1_batched_subagents, ex2_rate_limit_hook, ex3_subagent_tree, ex4_sdk_port,
                     ex5_managed_or_self_hosted):
        print(f"\n{exercise.__name__}")
        outcomes.append(exercise())
    missing = [outcome for outcome in outcomes if outcome]
    print("\nall exercises passed" if not missing else f"\nall checks passed; install {', '.join(missing)} for the rest")
