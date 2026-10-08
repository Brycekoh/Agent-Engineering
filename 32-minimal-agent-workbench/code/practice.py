"""Phase 14 - Lesson 32: The Minimal Agent Workbench - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --lint <workbench dir>   (exercise 4 as a command)
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from main import AGENTS_MD, AgentState, Task, load_board, load_state, run_one_turn, save_board, save_state, write_initial

REPO = Path(__file__).resolve().parents[2]


def new_workbench(root: Path) -> tuple[Path, Path, Path]:
    """The three files, written by main.py into a directory of our choosing."""
    paths = (root / "agent_state.json", root / "task_board.json", root / "AGENTS.md")
    write_initial(*paths)
    return paths


# ---------------------------------------------------------------------------
# Exercise 1 - a last_run timestamp, and a refusal when the state is stale
#
# The timestamp lives inside the file. The file's modification time would not
# do: a git checkout rewrites it. A state file that has never run has no
# timestamp and is allowed to start.
# ---------------------------------------------------------------------------

MAX_AGE = timedelta(hours=24)


@dataclass
class StampedState(AgentState):
    last_run: str | None = None         # ISO 8601, UTC


class StaleState(RuntimeError):
    pass


def guarded_turn(state_path: Path, board_path: Path, now: datetime, operator_confirms: bool = False) -> StampedState:
    state = StampedState(**json.loads(state_path.read_text()))
    if state.last_run and not operator_confirms:
        age = now - datetime.fromisoformat(state.last_run)
        if age > MAX_AGE:
            raise StaleState(f"state last ran {age.total_seconds() / 3600:.0f}h ago; an operator must confirm it still holds")
    board = load_board(board_path)
    run_one_turn(state, board)
    state.last_run = now.isoformat()
    save_state(state_path, state)
    save_board(board_path, board)
    return state


def ex1_last_run_timestamp() -> None:
    start = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        state_path, board_path, _ = new_workbench(Path(tmp))
        first = guarded_turn(state_path, board_path, start)
        print(f"  first turn       : ran, stamped {first.last_run}")
        second = guarded_turn(state_path, board_path, start + timedelta(hours=23))
        print(f"  23h later        : ran, touched {second.touched_files}")
        before = state_path.read_bytes()
        late = start + timedelta(hours=23 + 25)
        try:
            guarded_turn(state_path, board_path, late)
            refused = ""
        except StaleState as error:
            refused = str(error)
        print(f"  25h after that   : refused - {refused}")
        assert refused and state_path.read_bytes() == before        # a refusal writes nothing
        confirmed = guarded_turn(state_path, board_path, late, operator_confirms=True)
        print(f"  operator confirms: ran, touched {confirmed.touched_files}, restamped")
        assert confirmed.last_run == late.isoformat() and confirmed.touched_files == ["app.py", "test_app.py"]


# ---------------------------------------------------------------------------
# Exercise 2 - a priority field, and a puller that takes the highest first
#
# main.py's puller takes the first todo on the board. Handing it the board
# sorted by priority is enough: the sort is stable, so equal priorities still
# go in board order. The board on disk keeps its own order.
# ---------------------------------------------------------------------------

@dataclass
class PriorityTask(Task):
    priority: int = 0


def pull_by_priority(state: AgentState, board: list[PriorityTask]) -> None:
    run_one_turn(state, sorted(board, key=lambda task: -task.priority))


def ex2_priority_puller() -> None:
    def board() -> list[PriorityTask]:
        return [PriorityTask("T-001", "rename a variable", "builder", ["lint passes"], priority=1),
                PriorityTask("T-002", "fix the login outage", "builder", ["login test passes"], priority=9),
                PriorityTask("T-003", "fix the signup outage", "builder", ["signup test passes"], priority=9),
                PriorityTask("T-004", "ship the big feature", "builder", ["feature test passes"], status="done", priority=10)]

    in_order, by_priority = AgentState(active_task_id=None), AgentState(active_task_id=None)
    run_one_turn(in_order, board())
    tasks = board()
    pull_by_priority(by_priority, tasks)
    print(f"  main.py's puller takes {in_order.active_task_id}; the priority puller takes {by_priority.active_task_id}")
    assert (in_order.active_task_id, by_priority.active_task_id) == ("T-001", "T-002")     # tie goes to board order
    assert [task.id for task in tasks] == ["T-001", "T-002", "T-003", "T-004"]             # order on disk untouched
    assert [task.status for task in tasks] == ["todo", "in_progress", "todo", "done"]      # done is never pulled


# ---------------------------------------------------------------------------
# Exercise 3 - the task board as JSON Lines
#
# One task per line, keys sorted so a line never changes shape. Measured below
# on three edits: adding a task is one added line with no neighbour touched,
# where the indented array also rewrites the line before it to add a comma.
# Two more things follow from one-task-per-line: a new task is an append to
# the file instead of a rewrite, and a line-union merge can combine two
# branches that each added a task.
# ---------------------------------------------------------------------------

def save_board_jsonl(path: Path, board: list[Task]) -> None:
    path.write_text("".join(json.dumps(asdict(task), sort_keys=True) + "\n" for task in board), newline="\n")


def load_board_jsonl(path: Path) -> list[Task]:
    return [Task(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def migrate_board(json_path: Path) -> Path:
    jsonl_path = json_path.with_suffix(".jsonl")
    save_board_jsonl(jsonl_path, load_board(json_path))
    return jsonl_path


def diff_lines(before: str, after: str) -> int:
    diff = difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="")
    return sum(line.startswith(("+", "-")) and not line.startswith(("+++", "---")) for line in diff)


def ex3_board_as_json_lines() -> None:
    def add_task(board: list[Task]) -> None:
        board.append(Task("T-003", "rate-limit /signup", "builder", ["pytest test_app.py::test_rate_limit"]))

    def change_status(board: list[Task]) -> None:
        board[0].status = "in_progress"

    def add_acceptance(board: list[Task]) -> None:
        board[1].acceptance.append("docs/api.md has an example request")

    with tempfile.TemporaryDirectory() as tmp:
        _, json_path, _ = new_workbench(Path(tmp))
        jsonl_path = migrate_board(json_path)
        assert load_board_jsonl(jsonl_path) == load_board(json_path)                # nothing lost in the move
        print(f"  migrated {len(load_board(json_path))} tasks to {jsonl_path.name}, one per line")
        for edit in (add_task, change_status, add_acceptance):
            board = load_board(json_path)
            edit(board)
            as_json = json.dumps([asdict(task) for task in board], indent=2) + "\n"
            as_jsonl = "".join(json.dumps(asdict(task), sort_keys=True) + "\n" for task in board)
            counts = diff_lines(json_path.read_text(), as_json), diff_lines(jsonl_path.read_text(), as_jsonl)
            print(f"  {edit.__name__:<15} diff lines: json {counts[0]:>2}, jsonl {counts[1]}")
            assert counts[1] <= counts[0] and counts[1] == (1 if edit is add_task else 2)


# ---------------------------------------------------------------------------
# Exercise 4 - lint_workbench: AGENTS.md stays short and points at real files
#
# Run on the workbench main.py lays down, the linter finds something real:
# the seed AGENTS.md routes the agent to docs/agent-rules.md, and nothing
# creates that file.
#
# It is also run on the capstone pack as the course ships it (lesson 42).
# There every document AGENTS.md points to exists, and the two files it
# reports are the state file and the board, which the pack's installer
# creates in the target repository. So the linter belongs on an installed
# workbench; the pack on its own shelf needs a check of its own.
# ---------------------------------------------------------------------------

MAX_LINES = 80
# ponytail: a file reference is a backticked path with a known extension; add markdown links when AGENTS.md uses them
FILE_REFERENCE = re.compile(r"`([\w./-]+\.(?:md|json|jsonl|py|sh|ya?ml|toml|txt))`")


def lint_workbench(root: Path) -> list[str]:
    agents = root / "AGENTS.md"
    if not agents.exists():
        return ["AGENTS.md is missing"]
    text = agents.read_text()
    problems = []
    if len(text.splitlines()) > MAX_LINES:
        problems.append(f"AGENTS.md is {len(text.splitlines())} lines (limit {MAX_LINES})")
    problems += [f"AGENTS.md references {name}, which does not exist"
                 for name in sorted(set(FILE_REFERENCE.findall(text))) if not (root / name).exists()]
    return problems


def ex4_lint_workbench() -> None:
    def run_lint(root: Path) -> tuple[int, str]:
        done = subprocess.run([sys.executable, __file__, "--lint", str(root)], capture_output=True, text=True)
        return done.returncode, done.stdout.strip()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, _, agents_path = new_workbench(root)
        as_written = run_lint(root)
        print(f"  as main.py writes it: exit {as_written[0]} - {as_written[1]}")
        (root / "docs").mkdir()
        (root / "docs" / "agent-rules.md").write_text("# Agent Rules\n")
        fixed = run_lint(root)
        print(f"  rules file added    : exit {fixed[0]} - {fixed[1]}")
        agents_path.write_text(AGENTS_MD + "\n".join(f"- note {number}" for number in range(MAX_LINES)) + "\n")
        long = run_lint(root)
        print(f"  padded with notes   : exit {long[0]} - {long[1]}")
    assert as_written == (1, "AGENTS.md references docs/agent-rules.md, which does not exist")
    assert fixed == (0, "workbench ok") and long[0] == 1 and "limit 80" in long[1]

    shipped = lint_workbench(REPO / "42-agent-workbench-capstone" / "outputs" / "agent-workbench-pack")
    print(f"  the capstone pack as shipped: {len(shipped)} problems, both files the installer creates")
    assert shipped == ["AGENTS.md references agent_state.json, which does not exist",
                       "AGENTS.md references task_board.json, which does not exist"]


# ---------------------------------------------------------------------------
# Exercise 5 - which of the three files hurts the most to lose
#
# agent_state.json. Measured below by deleting each file in the middle of the
# second task and letting main.py recover.
#
#   AGENTS.md        comes back byte for byte. It is a constant, it changes
#                    rarely, and a copy sits in version control.
#   task_board.json  comes back with every status reset. The finished task
#                    will be done again, which wastes work but is visible, and
#                    the state file still knows which task is active.
#   agent_state.json comes back empty. The touched files, the assumption and
#                    the blocker existed nowhere else. Worse, the board still
#                    says the task is in progress, the puller only takes todo
#                    tasks, and the agent reports that there is no work.
#
# The board is the plan and the plan is recoverable from people. The state is
# the only record of what the last session knew.
# ---------------------------------------------------------------------------

def lose(file_name: str) -> tuple[list[str], str]:
    """Delete one file mid-task, recover, and return (fields lost, what the next turn does)."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = new_workbench(Path(tmp))
        state_path, board_path, agents_path = paths
        state, board = load_state(state_path), load_board(board_path)
        for _ in range(6):                      # finish T-001, then pick up T-002 and edit app.py
            run_one_turn(state, board)
        state.assumptions.append("password minimum is 8 characters")
        state.blockers.append("docs/api.md does not exist yet")
        save_state(state_path, state)
        save_board(board_path, board)

        before = {"agent_state.json": asdict(state), "AGENTS.md": {"text": agents_path.read_text()},
                  "task_board.json": {task.id + ".status": task.status for task in board}}[file_name]
        (Path(tmp) / file_name).unlink()
        write_initial(*paths)
        state, board = load_state(state_path), load_board(board_path)
        after = {"agent_state.json": asdict(state), "AGENTS.md": {"text": agents_path.read_text()},
                 "task_board.json": {task.id + ".status": task.status for task in board}}[file_name]
        run_one_turn(state, board)
        return [key for key in before if before[key] != after[key]], state.next_action


def ex5_which_file_hurts_most() -> None:
    lost = {name: lose(name) for name in ("AGENTS.md", "task_board.json", "agent_state.json")}
    for name, (fields, next_action) in lost.items():
        print(f"  lose {name:<16} -> {len(fields)} fields lost {fields}")
        print(f"  {'':<21}    next turn: {next_action}")
    assert lost["AGENTS.md"][0] == []
    assert lost["task_board.json"] == (["T-001.status", "T-002.status"], "run verification command for T-002")
    assert len(lost["agent_state.json"][0]) == 5 and lost["agent_state.json"][1] == "no work on the board, idle"


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run twice in a temp copy of this folder.
#   exits zero on the first and second run   yes
#   the second run picks up where the first  yes: it starts with T-001 active
#   left off                                 and edits app.py
#   the script prints a diff of the one      it prints the list of touched
#   file the turn touched                    files, ['app.py'], not a diff
# ---------------------------------------------------------------------------

def run_main(times: int = 1) -> list[tuple[int, str, dict[str, str]]]:
    """Run this folder's main.py in a temp copy. One (exit code, stdout, files left behind) per run."""
    runs = []
    with tempfile.TemporaryDirectory() as tmp:
        code = Path(tmp) / "code"
        code.mkdir()
        (Path(tmp) / "outputs").mkdir()             # every lesson folder in the repo has one
        shutil.copy(Path(__file__).with_name("main.py"), code)
        for _ in range(times):
            done = subprocess.run([sys.executable, "main.py"], cwd=code, capture_output=True, text=True, timeout=120)
            files = {path.relative_to(tmp).as_posix(): path.read_text(errors="replace") for path in sorted(Path(tmp).rglob("*"), key=str)
                     if path.is_file() and path.name != "main.py" and "__pycache__" not in path.parts}
            runs.append((done.returncode, done.stdout, files))
    return runs


def mission_acceptance() -> None:
    first, second = run_main(times=2)
    state = json.loads(second[2]["code/workdir/agent_state.json"])
    print(f"  exit codes {first[0]} and {second[0]}; after the second run the state has {state['active_task_id']} active "
          f"and touched {state['touched_files']}")
    assert first[0] == second[0] == 0
    assert "active task : None" in first[1].split("after turn:")[0] and "active task : T-001" in second[1].split("after turn:")[0]
    assert state["active_task_id"] == "T-001" and state["touched_files"] == ["app.py"] and "touched     : ['app.py']" in second[1]
    assert list(second[2]) == ["code/workdir/AGENTS.md", "code/workdir/agent_state.json", "code/workdir/task_board.json"]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--lint"]:
        found = lint_workbench(Path(sys.argv[2]))
        print("; ".join(found) or "workbench ok")
        sys.exit(1 if found else 0)
    print("Phase 14 - Lesson 32: The Minimal Agent Workbench - exercises")
    for exercise in (ex1_last_run_timestamp, ex2_priority_puller, ex3_board_as_json_lines, ex4_lint_workbench,
                     ex5_which_file_hurts_most, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
