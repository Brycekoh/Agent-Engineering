"""Phase 14 - Lesson 34: Repo Memory and Durable State - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable
from unittest import mock

import main as lesson
from main import BOARD_SCHEMA, STATE_SCHEMA, SchemaError, StateManager


def base_state(**changes: Any) -> dict[str, Any]:
    return {"schema_version": 1, "active_task_id": "T-001", "touched_files": [], "assumptions": [], "blockers": [],
            "next_action": "read the /signup handler", **changes}


# ---------------------------------------------------------------------------
# Exercise 1 - last_human_touch, and no agent write within five seconds of it
#
# The timestamp that counts is the one on disk. An agent write carries it
# forward unchanged, so an agent cannot clear the field to get past the guard.
# main.py's validator has no "number" type, so the field is whole milliseconds.
# ---------------------------------------------------------------------------

HUMAN_QUIET_MS = 5000
TOUCH_SCHEMA = {**STATE_SCHEMA, "properties": {**STATE_SCHEMA["properties"], "last_human_touch": {"type": ["integer", "null"]}}}


class HumanEditing(RuntimeError):
    pass


class TouchAwareManager(StateManager):
    # ponytail: only writes made through commit(actor="human") are stamped; if people edit the file in an
    # editor, compare its st_mtime with the time of the agent's own last write as well
    def __init__(self, state_path: Path, clock: Callable[[], int] = lambda: int(time.time() * 1000)):
        super().__init__(state_path, TOUCH_SCHEMA)
        self.clock = clock

    def commit(self, state: Any, actor: str = "agent") -> None:
        now = self.clock()
        touched = self.load().get("last_human_touch") if self.state_path.exists() else None
        if actor == "human":
            touched = now
        elif touched is not None and now - touched < HUMAN_QUIET_MS:
            raise HumanEditing(f"a human edited this {now - touched} ms ago; agent writes wait {HUMAN_QUIET_MS} ms")
        super().commit({**state, "last_human_touch": touched})


def ex1_last_human_touch() -> None:
    now = [0]
    with tempfile.TemporaryDirectory() as tmp:
        manager = TouchAwareManager(Path(tmp) / "agent_state.json", clock=lambda: now[0])
        manager.commit(base_state(), actor="human")
        outcomes = {}
        for label, at_ms, state in (("agent at 3.0 s", 3000, base_state(next_action="overwrite the human")),
                                    ("agent clears the stamp, 3.0 s", 3000, base_state(last_human_touch=None)),
                                    ("agent at 5.0 s", 5000, base_state(next_action="add the test"))):
            now[0] = at_ms
            try:
                manager.commit(state)
                outcomes[label] = "written"
            except HumanEditing as refusal:
                outcomes[label] = f"refused ({refusal})"
            print(f"  {label:<30} {outcomes[label]}")
        final = manager.load()
    assert [outcome.split()[0] for outcome in outcomes.values()] == ["refused", "refused", "written"]
    assert final["next_action"] == "add the test" and final["last_human_touch"] == 0       # the stamp survives the agent


# ---------------------------------------------------------------------------
# Exercise 2 - oneOf in the validator: a build task or a review task
#
# Exactly one option must accept the value. The new validate replaces the one
# in main.py's namespace, so main.py's own recursion reaches a oneOf nested
# under items and StateManager uses it without a change.
#
# The two task shapes differ in more than their required fields: a review
# task must name the task it reviews, and a builder cannot own one.
# ---------------------------------------------------------------------------

_base_validate = lesson.validate


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    # ponytail: keywords written next to oneOf are not checked; put them inside each option
    if "oneOf" not in schema:
        return _base_validate(value, schema, path)
    errors = []
    for number, option in enumerate(schema["oneOf"], 1):
        try:
            validate(value, option, path)
        except SchemaError as error:
            errors.append(f"option {number}: {error}")
    matched = len(schema["oneOf"]) - len(errors)
    if matched != 1:
        raise SchemaError(f"{path}: matches {matched} of {len(schema['oneOf'])} options, needs exactly 1. " + " | ".join(errors))


lesson.validate = validate

TASK = BOARD_SCHEMA["items"]
BUILD_TASK = {**TASK, "required": [*TASK["required"], "kind"], "properties": {**TASK["properties"], "kind": {"enum": ["build"]}}}
REVIEW_TASK = {
    "type": "object",
    "required": ["id", "kind", "owner", "status", "reviews"],
    "properties": {"id": TASK["properties"]["id"], "kind": {"enum": ["review"]}, "status": TASK["properties"]["status"],
                   "owner": {"type": "string", "enum": ["reviewer", "human"]}, "reviews": TASK["properties"]["id"],
                   "findings": {"type": "array", "items": {"type": "string"}}},
}
BOARD_SCHEMA_V2 = {"type": "array", "items": {"oneOf": [BUILD_TASK, REVIEW_TASK]}}


def ex2_one_of() -> None:
    build = {"id": "T-001", "kind": "build", "goal": "validate /signup payloads", "owner": "builder",
             "acceptance": ["pytest -x test_app.py"], "status": "done"}
    review = {"id": "T-002", "kind": "review", "owner": "reviewer", "status": "todo", "reviews": "T-001"}
    boards = {
        "one build task, one review task": [build, review],
        "review task with no 'reviews'": [{key: value for key, value in review.items() if key != "reviews"}],
        "review task owned by the builder": [{**review, "owner": "builder"}],
        "build task with a 'reviews' field": [{**build, "reviews": "T-001"}],
    }
    with tempfile.TemporaryDirectory() as tmp:
        manager = StateManager(Path(tmp) / "task_board.json", BOARD_SCHEMA_V2)       # main.py's class, unchanged
        accepted = {}
        for label, board in boards.items():
            try:
                manager.commit(board)
                accepted[label] = True
            except SchemaError as error:
                accepted[label] = False
                print(f"  {label:<34} rejected: {str(error).split('.')[0]}")
            else:
                print(f"  {label:<34} accepted")
    assert list(accepted.values()) == [True, False, False, False]
    try:        # the difference from anyOf: a value that fits two options is rejected
        validate("T-001", {"oneOf": [{"type": "string"}, {"type": "string", "pattern": "^T-"}]})
        ambiguous = False
    except SchemaError:
        ambiguous = True
    print(f"  a value that fits two options      rejected: {ambiguous}")
    assert ambiguous


# ---------------------------------------------------------------------------
# Exercise 3 - schema_version 2: blockers becomes risks
#
# The migration runs on load, one version step at a time, and the migrated
# state is written back through commit, so it is validated against the new
# schema and written atomically. A file from a newer version than this code
# knows is refused; there is no path downward. The v1 file stays recoverable
# from version control.
# ---------------------------------------------------------------------------

V2_PROPERTIES = {**STATE_SCHEMA["properties"], "schema_version": {"type": "integer", "enum": [2]},
                 "risks": STATE_SCHEMA["properties"]["blockers"]}
del V2_PROPERTIES["blockers"]
STATE_SCHEMA_V2 = {**STATE_SCHEMA, "properties": V2_PROPERTIES}


def v1_to_v2(state: dict[str, Any]) -> dict[str, Any]:
    renamed = {("risks" if key == "blockers" else key): value for key, value in state.items()}
    return {**renamed, "schema_version": 2}


MIGRATIONS = {1: v1_to_v2}


class MigratingManager(StateManager):
    def load(self) -> Any:
        raw = json.loads(self.state_path.read_text())
        state = raw
        while isinstance(state, dict) and state.get("schema_version") in MIGRATIONS:
            state = MIGRATIONS[state["schema_version"]](state)
        if state != raw:
            self.commit(state)
        else:
            validate(state, self.schema)
        return state


def ex3_migration() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_state.json"
        StateManager(path, STATE_SCHEMA).commit(base_state(blockers=["docs/api.md does not exist yet"]))
        try:
            StateManager(path, STATE_SCHEMA_V2).load()
            unmigrated = ""
        except SchemaError as error:
            unmigrated = str(error)
        print(f"  v2 code reading the v1 file as is: {unmigrated}")
        manager = MigratingManager(path, STATE_SCHEMA_V2)
        migrated = manager.load()
        on_disk = json.loads(path.read_text())
        print(f"  after migration: version {migrated['schema_version']}, risks {migrated['risks']}, 'blockers' in file: {'blockers' in on_disk}")
        assert unmigrated and migrated == on_disk and migrated["risks"] == ["docs/api.md does not exist yet"]
        assert list(migrated) == [("risks" if key == "blockers" else key) for key in base_state()]       # field order kept
        before = path.read_bytes()
        assert manager.load() == migrated and path.read_bytes() == before        # second load changes nothing

        path.write_text(json.dumps({**migrated, "schema_version": 3}))
        try:
            manager.load()
            newer = ""
        except SchemaError as error:
            newer = str(error)
        print(f"  a file from version 3: refused - {newer}")
        assert newer


# ---------------------------------------------------------------------------
# Exercise 4 - the same StateManager API on SQLite
#
# Same constructor, same load(), same commit(); the path is a database file.
# What it buys: writers queue on SQLite's lock, and a read-modify-write can
# sit inside one transaction, which is the fix for exercise 5. What it costs:
# the state is no longer a text file, so a state change stops showing up in a
# diff or a pull request.
# ---------------------------------------------------------------------------

class SqliteStateManager(StateManager):
    def _database(self) -> Any:
        database = sqlite3.connect(self.state_path)
        database.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK (id = 1), body TEXT NOT NULL)")
        return closing(database)

    def load(self) -> Any:
        with self._database() as database:
            row = database.execute("SELECT body FROM state").fetchone()
        if row is None:
            raise FileNotFoundError(self.state_path)
        raw = json.loads(row[0])
        validate(raw, self.schema)
        return raw

    def commit(self, state: Any) -> None:
        validate(state, self.schema)
        with self._database() as database, database:
            database.execute("INSERT OR REPLACE INTO state (id, body) VALUES (1, ?)", (json.dumps(state),))


def behaviour(manager: StateManager) -> list[str]:
    log = []
    try:
        manager.load()
    except FileNotFoundError:
        log.append("load before any commit raises FileNotFoundError")
    manager.commit(base_state())
    log.append(f"commit then load round-trips: {manager.load() == base_state()}")
    manager.commit(base_state(touched_files=["app.py"]))
    log.append(f"second commit replaces the first: {manager.load()['touched_files']}")
    try:
        manager.commit(base_state(active_task_id="T-bogus"))
    except SchemaError:
        log.append("write that breaks the schema is rejected")
    log.append(f"state after the rejected write: {manager.load()['active_task_id']}")
    return log


def ex4_sqlite_backend() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        on_file = behaviour(StateManager(Path(tmp) / "agent_state.json", STATE_SCHEMA))
        on_sqlite = behaviour(SqliteStateManager(Path(tmp) / "agent_state.db", STATE_SCHEMA))
    for line in on_sqlite:
        print(f"  sqlite: {line}")
    print(f"  the JSON file backend gives the same {len(on_file)} results: {on_file == on_sqlite}")
    assert on_file == on_sqlite and len(on_sqlite) == 5


# ---------------------------------------------------------------------------
# Exercise 5 - two agents, one state file, 50 ms apart
#
# What goes wrong: a lost update. Both agents load the same state, each adds
# its own file, and the second write replaces the first. One agent's work
# disappears from the record and nothing reports an error.
#
# What the atomic rename saves: the file itself. A write that dies halfway
# leaves the previous state in place and no temp file behind, where an
# in-place write leaves half a document that no session can load.
#
# What it does not save: the lost update. That needs the read, the change and
# the write to happen under one lock, or inside one transaction.
#
# Measured on Windows: the rename is refused while another handle has the
# file open, and the write raises PermissionError. The old state is intact
# and the writer has to retry. POSIX lets the rename through.
# ---------------------------------------------------------------------------

def ex5_write_race() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "agent_state.json"
        manager = StateManager(path, STATE_SCHEMA)
        manager.commit(base_state())
        both_loaded = threading.Barrier(2)

        def agent(file_name: str, delay_s: float) -> None:
            state = manager.load()
            both_loaded.wait(timeout=5)
            time.sleep(delay_s)
            state["touched_files"].append(file_name)
            manager.commit(state)

        agents = [threading.Thread(target=agent, args=("a.py", 0.0)), threading.Thread(target=agent, args=("b.py", 0.05))]
        for thread in agents:
            thread.start()
        for thread in agents:
            thread.join()
        survived = manager.load()["touched_files"]
        print(f"  two agents wrote a.py and b.py 50 ms apart; the file records {survived}")
        assert survived == ["b.py"]

        good = path.read_text()
        update = json.dumps(base_state(touched_files=["a.py", "b.py"]), indent=2) + "\n"
        in_place = Path(tmp) / "in_place.json"
        in_place.write_text(update[: len(update) // 2])         # what an in-place write leaves when the process dies halfway
        try:
            json.loads(in_place.read_text())
            torn = False
        except json.JSONDecodeError:
            torn = True
        with mock.patch.object(os, "fsync", side_effect=OSError("process died mid-write")):
            try:
                manager.commit(base_state(touched_files=["a.py", "b.py"]))
                died = False
            except OSError:
                died = True
        leftovers = sorted(entry.name for entry in Path(tmp).iterdir())
        print(f"  write dies halfway, in place : file loads afterwards: {not torn}")
        print(f"  write dies halfway, atomic   : file loads afterwards: {path.read_text() == good}, temp files left: "
              f"{len(leftovers) - 2}")
        assert torn and died and path.read_text() == good and leftovers == ["agent_state.json", "in_place.json"]

        with open(path) as reader:
            try:
                manager.commit(base_state(touched_files=["a.py", "b.py"]))
                outcome = "rename went through, the reader keeps the old file"
            except PermissionError:
                outcome = "rename refused with PermissionError, old state intact, the writer must retry"
            assert json.loads(reader.read()) == json.loads(good)
        print(f"  write while a reader holds the file ({sys.platform}): {outcome}")
        assert manager.load()["touched_files"] in (["b.py"], ["a.py", "b.py"])
        assert sorted(entry.name for entry in Path(tmp).iterdir()) == leftovers


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run twice in a temp copy of this folder.
#   exits zero                               yes
#   a bad write (missing required field,     yes: main.py shows a bad id; the
#   bad enum) is refused, not persisted      missing field and the bad enum
#                                            are checked here
#   workdir/agent_state.json validates       yes
# The mission lists StateManager.load, update and commit. main.py has load and
# commit. An update is a load, a change and a commit, which is the sequence
# exercise 5 shows losing a write when two agents do it at once.
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
    validate(state, STATE_SCHEMA)
    refused = []
    with tempfile.TemporaryDirectory() as tmp:
        manager = StateManager(Path(tmp) / "agent_state.json", STATE_SCHEMA)
        manager.commit(base_state())
        bad_writes = {"missing required field": {key: value for key, value in base_state().items() if key != "next_action"},
                      "bad enum": base_state(schema_version=7)}
        for label, bad in bad_writes.items():
            try:
                manager.commit(bad)
            except SchemaError:
                refused.append(label)
            assert manager.load() == base_state()
    print(f"  exit codes {first[0]} and {second[0]}; the state file validates; refused and not persisted: {refused}")
    assert first[0] == second[0] == 0 and "rejected bad write" in second[1] and refused == list(bad_writes)
    assert json.loads(second[2]["code/workdir/schemas/agent_state.schema.json"]) == STATE_SCHEMA
    assert not hasattr(StateManager, "update")


if __name__ == "__main__":
    print("Phase 14 - Lesson 34: Repo Memory and Durable State - exercises")
    for exercise in (ex1_last_human_touch, ex2_one_of, ex3_migration, ex4_sqlite_backend, ex5_write_race, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
