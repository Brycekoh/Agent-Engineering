"""Phase 14 - Lesson 37: Runtime Feedback Loops - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --tui < feedback_record.jsonl   (exercise 5 as a command)
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import main as lesson
from main import FeedbackRecord

PYTHON = sys.executable         # main.py's demo calls "python3", which not every machine has


@contextlib.contextmanager
def temp_record() -> Iterator[Path]:
    """Point main.py's record file at a temp directory, and put it back afterwards."""
    saved = lesson.HERE, lesson.RECORD
    with tempfile.TemporaryDirectory() as tmp:
        lesson.HERE = Path(tmp)
        lesson.RECORD = lesson.HERE / saved[1].name
        try:
            yield lesson.HERE
        finally:
            lesson.HERE, lesson.RECORD = saved


# ---------------------------------------------------------------------------
# Exercise 1 - a cwd field on every record
#
# The record class gains a cwd that fills itself in when the record is made,
# and takes the place of main.py's class, so run_with_feedback and load_all
# write and read it without a change.
#
# One thing to know before adding a field to a log that already has readers:
# main.py's loader skips any line it cannot build a record from, and says
# nothing. A reader from before the field existed loads none of the new
# records. Measured below.
# ---------------------------------------------------------------------------

@dataclass
class CwdRecord(FeedbackRecord):
    cwd: str = field(default_factory=os.getcwd)


lesson.FeedbackRecord = CwdRecord


def run_in(cwd: Path, command: list[str], **kwargs: Any) -> CwdRecord:
    # ponytail: chdir changes the whole process; pass cwd to subprocess.run once commands run on threads
    with contextlib.chdir(cwd):
        return lesson.run_with_feedback(command, **kwargs)


def ex1_cwd_field() -> None:
    count_files = [PYTHON, "-c", "import os; print(len(os.listdir('.')))"]
    with temp_record() as root:
        for name, files in (("api", 1), ("web", 3)):
            (root / name).mkdir()
            for number in range(files):
                (root / name / f"module_{number}.py").write_text("")
        records = [run_in(root / "api", count_files), run_in(root / "web", count_files)]
        for record in records:
            print(f"  cwd .../{Path(record.cwd).name:<4} command identical: {record.command == count_files}  stdout {record.stdout_tail.strip()}")
        loaded = lesson.load_all()
        lesson.FeedbackRecord = FeedbackRecord
        try:
            old_reader = lesson.load_all()
        finally:
            lesson.FeedbackRecord = CwdRecord
    print(f"  reader with the cwd field loads {len(loaded)} of 2 records; a reader from before the field loads {len(old_reader)}")
    assert [Path(record.cwd).name for record in loaded] == ["api", "web"]
    assert [record.stdout_tail.strip() for record in records] == ["1", "3"] and old_reader == []


# ---------------------------------------------------------------------------
# Exercise 2 - a redaction step that strips whole lines
#
# strip_secret_lines replaces any line that matches ^Bearer or password=
# with a marker, so a reader can see a line was there. Tested on a fixture
# record, then measured against main.py's redact on seven lines:
#   - main.py's \b before "password" and "token" does not match after an
#     underscore, so the usual env-var shapes (DB_PASSWORD=, GITHUB_TOKEN=,
#     AWS_SECRET_ACCESS_KEY=) pass through it untouched.
#   - the two patterns the exercise names miss those too, and miss a Bearer
#     token that follows "Authorization:".
#   - one more pattern for env-var shaped names closes the gap; the line that
#     merely mentions a password in a test name is left alone by all three.
#
# main.py truncates before it redacts. A private key that begins in the head
# and ends in the part that is cut loses its END line, the pattern no longer
# matches, and the first lines of the key are written to the log. safe_capture
# redacts the whole output first, then truncates, and replaces main.py's
# capture step for every command this file runs. The price is running the
# patterns over all of the output instead of 35 lines of it.
# ---------------------------------------------------------------------------

SECRET_LINES = [re.compile(r"^Bearer "), re.compile(r"password=")]                   # the two the exercise names
ENV_SECRET = re.compile(r"\b[A-Z0-9_]*(SECRET|TOKEN|PASSWORD|KEY)[A-Z0-9_]*=")        # DB_PASSWORD=, GITHUB_TOKEN=
MAX_LINE_CHARS = 200
_main_capture = lesson._process_capture


def strip_secret_lines(text: str, patterns: list[re.Pattern[str]] = SECRET_LINES) -> tuple[str, int]:
    lines = text.splitlines()
    hits = [any(pattern.search(line) for pattern in patterns) for line in lines]
    return "\n".join("[REDACTED LINE]" if hit else line for line, hit in zip(lines, hits)), sum(hits)


def scrub(text: str) -> tuple[str, int]:
    stripped, lines = strip_secret_lines(text, [*SECRET_LINES, ENV_SECRET])
    redacted, tokens = lesson.redact(stripped)
    return redacted, lines + tokens


def safe_capture(text: str) -> tuple[str, int, int]:
    """Redact, then truncate, then clip long lines. Same return shape as main.py's _process_capture."""
    redacted, hits = scrub(text)
    tailed, cut = lesson.deterministic_tail(redacted)
    clipped = [line if len(line) <= MAX_LINE_CHARS else f"{line[:MAX_LINE_CHARS]}...[{len(line) - MAX_LINE_CHARS} more chars]"
               for line in tailed.splitlines()]
    return "\n".join(clipped), cut, hits


lesson._process_capture = safe_capture


def ex2_redaction() -> None:
    fixture = CwdRecord(command_id="fixture00001", parent_command_id=None, command=["curl", "-v", "https://api.example.test"],
                        stdout_tail="Bearer abc.def.ghi\nHTTP/1.1 200 OK\npassword=hunter2\n{\"ok\": true}",
                        stderr_tail="* Connected to api.example.test", exit_code=0, duration_ms=80, started_at=0.0, agent_note="")
    fixture.stdout_tail, removed = strip_secret_lines(fixture.stdout_tail)
    print(f"  fixture record: {removed} lines stripped, stdout now {fixture.stdout_tail.splitlines()}")
    assert removed == 2 and "hunter2" not in fixture.stdout_tail and "abc.def.ghi" not in fixture.stdout_tail
    assert "HTTP/1.1 200 OK" in fixture.stdout_tail and len(fixture.stdout_tail.splitlines()) == 4

    samples = [("Bearer abc.def.ghi", "abc.def.ghi"), ("Authorization: Bearer abc.def.ghi", "abc.def.ghi"),
               ("password=hunter2", "hunter2"), ("DB_PASSWORD=hunter2", "hunter2"), ("GITHUB_TOKEN=ghp_abc123", "ghp_abc123"),
               ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI", "wJalrXUtnFEMI"), ("test_password_reset passed", "")]
    steps = {"main.py redact": lesson.redact, "the two line rules": strip_secret_lines, "all of them": scrub}
    leaks = {name: [line for line, secret in samples if secret and secret in step(line)[0]] for name, step in steps.items()}
    for name, leaked in leaks.items():
        print(f"  {name:<19} lets {len(leaked)} of 6 secrets through {leaked}")
    assert leaks["main.py redact"] == [line for line, _ in samples[3:6]]
    assert leaks["the two line rules"] == [line for line, _ in samples[1:2] + samples[3:6]]
    assert leaks["all of them"] == [] and scrub(samples[6][0]) == (samples[6][0], 0)

    marker = "-----BEGIN " + "RSA PRIVATE KEY-----"
    key = [marker, *[f"MIIEfake{number:02d}" for number in range(10)], marker.replace("BEGIN", "END")]
    output = "\n".join(key + [f"log line {number}" for number in range(40)])
    truncate_first, redact_first = _main_capture(output)[0], safe_capture(output)[0]
    print(f"  private key across the cut: truncate first leaves {truncate_first.count('MIIEfake')} key lines in the log, "
          f"redact first leaves {redact_first.count('MIIEfake')}")
    assert truncate_first.count("MIIEfake") == 4 and "MIIEfake" not in redact_first


# ---------------------------------------------------------------------------
# Exercise 3 - a 1 MB cap on everything the record keeps
#
# main.py rotates the active file at 1 MB and keeps five older files, so what
# it bounds is about 6 MB, not 1. Capping the total takes two changes:
#   - the rotation point becomes 1 MB / 6, less room for one record, because
#     the size check runs before the append;
#   - a record needs a largest size. main.py truncates by line count, and one
#     very long line passes through whole. safe_capture clips lines at 200
#     characters, which bounds a record at about 90 KB even if every character
#     takes six bytes in JSON.
#
# The policy, defended. Rotate by size, because the cost that matters is the
# disk and the loader, which reads every file on each lookup. Keep whole
# numbered files and drop the oldest, because a rename never splits a record
# and the active file stays append-only. Lose the oldest sixth in one step,
# because this log feeds the next turn and the current review; long-term
# history belongs in telemetry. The cost is real and measured: a retry chain
# that reaches past the cap ends at the oldest surviving record.
# ---------------------------------------------------------------------------

TOTAL_CAP_BYTES = 1024 * 1024
# ponytail: command and agent_note are assumed short; clip them too if agents start writing essays there
MAX_RECORD_BYTES = 2 * (lesson.HEAD_LINES + lesson.TAIL_LINES + 1) * (MAX_LINE_CHARS * 6 + 40) + 1024
CAPPED_ROTATE_BYTES = TOTAL_CAP_BYTES // (lesson.MAX_ROTATIONS + 1) - MAX_RECORD_BYTES


def chain_is_complete(chain: list[FeedbackRecord]) -> bool:
    """False when the oldest record in a chain names a parent that is no longer on disk."""
    return bool(chain) and chain[0].parent_command_id is None


def ex3_rotation_cap() -> None:
    noisy = "\n".join("x" * MAX_LINE_CHARS for _ in range(35))            # SCRIPTED: 35 full lines on each stream
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=noisy, stderr=noisy)

    def write_records(count: int) -> tuple[int, int, list[FeedbackRecord]]:
        largest_total, previous = 0, None
        with mock.patch.object(lesson.subprocess, "run", return_value=completed):
            for _ in range(count):
                previous = lesson.run_with_feedback(["make", "build"], parent_command_id=previous and previous.command_id)
                largest_total = max(largest_total, sum(path.stat().st_size for path in lesson.HERE.iterdir()))
        return largest_total, len(list(lesson.HERE.iterdir())), lesson.retry_chain(previous.command_id)

    with temp_record():
        as_shipped = write_records(200)
    saved, lesson.ROTATE_BYTES = lesson.ROTATE_BYTES, CAPPED_ROTATE_BYTES
    try:
        with temp_record():
            capped = write_records(200)
    finally:
        lesson.ROTATE_BYTES = saved
    for label, (largest, files, chain) in (("main.py as shipped", as_shipped), ("rotation point lowered", capped)):
        print(f"  {label:<22} 200 records: at most {largest / 1024:>5.0f} KB on disk in {files} files; "
              f"chain reaches back {len(chain)} records, complete: {chain_is_complete(chain)}")
    assert as_shipped[0] > TOTAL_CAP_BYTES and chain_is_complete(as_shipped[2]) and len(as_shipped[2]) == 200
    assert capped[0] <= TOTAL_CAP_BYTES and capped[1] == lesson.MAX_ROTATIONS + 1
    assert 0 < len(capped[2]) < 200 and not chain_is_complete(capped[2])

    one_long_line = "y" * 5_000_000
    print(f"  one 5 MB line: main.py's truncation keeps {len(lesson.deterministic_tail(one_long_line)[0]) / 1e6:.0f} MB, "
          f"safe_capture keeps {len(safe_capture(one_long_line)[0])} characters")
    assert len(safe_capture(one_long_line)[0]) < MAX_LINE_CHARS + 40


# ---------------------------------------------------------------------------
# Exercise 4 - parent_command_id, so chains are visible
#
# main.py has the field and retry_chain walks it. Here the chain is not only
# retries: a parent is whichever command produced what this one consumed. The
# build writes a file, the test reads it and fails, the fix rewrites it and
# the test runs again.
#
# A parent id that matches nothing (mistyped, or rotated away as in exercise
# 3) gives a chain that simply starts at the child, which looks the same as a
# command with no parent. chain_is_complete tells the two apart.
# ---------------------------------------------------------------------------

def ex4_parent_command_id() -> None:
    with temp_record() as root:
        build = run_in(root, [PYTHON, "-c", "open('artifact.txt', 'w').write('v1')"], agent_note="build the artifact")
        check = [PYTHON, "-c", "import sys; sys.exit(0 if open('artifact.txt').read() == 'v2' else 1)"]
        test = run_in(root, check, agent_note="expect v2", parent_command_id=build.command_id)
        fix = run_in(root, [PYTHON, "-c", "open('artifact.txt', 'w').write('v2')"], agent_note="rebuild as v2",
                     parent_command_id=test.command_id)
        retest = run_in(root, check, agent_note="retry after the rebuild", parent_command_id=fix.command_id)
        unrelated = run_in(root, [PYTHON, "-c", "print('lint ok')"])
        orphan = run_in(root, check, parent_command_id="0123456789ab")
        chain, orphan_chain = lesson.retry_chain(retest.command_id), lesson.retry_chain(orphan.command_id)
    for record in chain:
        print(f"  {record.command_id} <- {record.parent_command_id or '-':<12} exit {record.exit_code}  {record.agent_note}")
    print(f"  chain complete: {chain_is_complete(chain)}; a child of an unknown parent: {len(orphan_chain)} record, "
          f"complete: {chain_is_complete(orphan_chain)}")
    assert [record.command_id for record in chain] == [build.command_id, test.command_id, fix.command_id, retest.command_id]
    assert [record.exit_code for record in chain] == [0, 1, 0, 0] and unrelated.command_id not in {r.command_id for r in chain}
    assert chain_is_complete(chain) and len(orphan_chain) == 1 and not chain_is_complete(orphan_chain)
    assert all(lesson.loop_can_advance(record) for record in chain)


# ---------------------------------------------------------------------------
# Exercise 5 - a tiny TUI over the JSONL, highlighting the latest failure
#
# The eight things it has to show to be useful in a review:
#   1. the latest command that did not exit zero, marked, with its stderr
#   2. the exit code of every command, with "no exit code" (a timeout, a
#      missing binary) kept apart from zero
#   3. the command as it was run, and the directory it ran in
#   4. order and time: when each command started and how long it took
#   5. lineage: each command's parent, and whether a failure was later
#      followed by a child that passed
#   6. the agent's note, so what it expected sits beside what happened
#   7. how much of the output was cut or redacted, so nobody reads a tail as
#      the whole output
#   8. a summary line: how many passed, failed, had no exit code, and how
#      many failures are still unresolved
# The mark is plain text so it survives a pipe; reverse video is added on a
# terminal.
# ---------------------------------------------------------------------------

def render(records: list[CwdRecord], colour: bool = False) -> str:
    records = sorted(records, key=lambda record: record.started_at)
    children: dict[str, list[CwdRecord]] = {}
    for record in records:
        children.setdefault(record.parent_command_id or "", []).append(record)

    def resolved(record: CwdRecord) -> bool:
        return any(child.exit_code == 0 or resolved(child) for child in children.get(record.command_id, []))

    failed = [record for record in records if record.exit_code != 0]
    latest = failed[-1] if failed else None
    lines = [f"   # {'started':<8} {'took':>9}  {'exit':<4}  {'id':<12}  {'parent':<12}  {'command':<24} cwd"]
    for number, record in enumerate(records, 1):
        notes = [f"cut {sum(record.truncations.values())}"] if sum(record.truncations.values()) else []
        notes += [f"redacted {sum(record.redactions.values())}"] if sum(record.redactions.values()) else []
        notes += ["retried ok"] if record.exit_code != 0 and resolved(record) else []
        row = (f"{'>>' if record is latest else '  '}{number:>2} {time.strftime('%H:%M:%S', time.localtime(record.started_at))}"
               f" {record.duration_ms:>7}ms  {'none' if record.exit_code is None else record.exit_code:<4}  {record.command_id:<12}"
               f"  {record.parent_command_id or '-':<12}  {shlex.join(record.command)[:24]:<24} {record.cwd}"
               f"{'  [' + ', '.join(notes) + ']' if notes else ''}")
        lines.append(chr(27) + "[7m" + row + chr(27) + "[0m" if colour and record is latest else row)
        if record is latest:
            lines += [f"       note: {record.agent_note or '-'}"]
            lines += [f"       {name}: {line}" for name, text in (("error", record.error or ""), ("stderr", record.stderr_tail))
                      for line in text.splitlines()[-5:]]
    unresolved = [record for record in failed if not resolved(record)]
    lines.append(f"{len(records)} commands: {len(records) - len(failed)} ok, {sum(r.exit_code is not None for r in failed)} failed, "
                 f"{sum(r.exit_code is None for r in failed)} with no exit code; unresolved failures: {len(unresolved)}")
    return "\n".join(lines)


def ex5_tui() -> None:
    def record(command_id: str, parent: str | None, command: list[str], exit_code: int | None, minute: int, **more: Any) -> CwdRecord:
        base = dict(stdout_tail="", stderr_tail="", duration_ms=120, agent_note="", cwd="/repo")
        return CwdRecord(command_id=command_id, parent_command_id=parent, command=command, exit_code=exit_code,
                         started_at=1_700_000_000 + 60 * minute, **{**base, **more})

    records = [     # a review's worth of records, written by hand
        record("aaa111aaa111", None, ["pytest", "-x"], 1, 0, stderr_tail="E   assert 400 == 200", agent_note="expect green",
               truncations={"stdout": 212, "stderr": 0}),
        record("bbb222bbb222", "aaa111aaa111", ["pytest", "-x"], 0, 1, agent_note="retry after fixing the handler"),
        record("ccc333ccc333", None, ["pytest", "-x"], 0, 2, cwd="/repo/services/billing"),
        record("ddd444ddd444", None, ["npm", "run", "build"], None, 3, error="timeout after 30.0s", duration_ms=30000),
        record("eee555eee555", None, ["ruff", "check", "."], 1, 4, stderr_tail="app.py:3:1: F401 `os` imported but unused",
               agent_note="expect clean", redactions={"stdout": 0, "stderr": 2}),
    ]
    jsonl = "".join(json.dumps(asdict(entry)) + "\n" for entry in records)
    shown = subprocess.run([PYTHON, __file__, "--tui"], input=jsonl, capture_output=True, text=True).stdout
    for line in shown.splitlines():
        print(f"  {line}")
    rows = shown.splitlines()
    marked = [row for row in rows if row.startswith(">>")]
    features = {
        "1 latest failure marked, with stderr": len(marked) == 1 and "eee555eee555" in marked[0] and "stderr: app.py:3:1: F401" in shown,
        "2 no exit code kept apart from zero": "none  ddd444ddd444" in shown and "0     bbb222bbb222" in shown,
        "3 command and directory": "ruff check ." in shown and "/repo/services/billing" in shown,
        "4 start time and duration": "30000ms" in shown and rows[0].split()[1:3] == ["started", "took"],
        "5 parent, and failure later resolved": "bbb222bbb222  aaa111aaa111" in shown and "retried ok" in rows[1],
        "6 the agent's note": "note: expect clean" in shown,
        "7 cut and redacted counts": "cut 212" in shown and "redacted 2" in shown,
        "8 summary line": rows[-1] == "5 commands: 2 ok, 2 failed, 1 with no exit code; unresolved failures: 2",
    }
    missing = [name for name, present in features.items() if not present]
    print(f"  features shown: {len(features) - len(missing)} of 8")
    assert not missing, missing
    assert chr(27) not in shown and chr(27) + "[7m" in render(records, colour=True)


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run twice in a temp copy of this folder.
#   exits zero                               yes
#   feedback_record.jsonl accumulates one    no: main.py deletes the log when
#   record per command across re-runs        it starts, so after two runs it
#                                            holds five records, not ten
#   a null exit code cannot be marked        yes
#   successful by the loop
# The mission's demo is one success, one failure and one slow command. main.py
# runs a success, a leak, a failure, a retry and a missing binary; none is slow.
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
    records = [json.loads(line) for line in second[2]["code/feedback_record.jsonl"].splitlines()]
    no_exit = [record for record in records if record["exit_code"] is None]
    print(f"  exit codes {first[0]} and {second[0]}; after two runs the log holds {len(records)} records")
    print(f"  records with no exit code: {len(no_exit)}; the loop may advance on them: "
          f"{any(lesson.loop_can_advance(CwdRecord(**record)) for record in no_exit)}")
    assert first[0] == second[0] == 0 and len(records) == 5 and "5 records persisted" in second[1]
    assert no_exit and not any(lesson.loop_can_advance(CwdRecord(**record)) for record in no_exit)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--tui"]:
        print(render([CwdRecord(**json.loads(line)) for line in sys.stdin if line.strip()], colour=sys.stdout.isatty()))
        sys.exit(0)
    print("Phase 14 - Lesson 37: Runtime Feedback Loops - exercises")
    for exercise in (ex1_cwd_field, ex2_redaction, ex3_rotation_cap, ex4_parent_command_id, ex5_tui, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
