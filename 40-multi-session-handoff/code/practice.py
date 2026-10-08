"""Phase 14 - Lesson 40: Multi-Session Handoff - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
           python practice.py --packet <workbench dir>   (write handoff.md and handoff.json there)
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from main import HandoffPayload, WorkbenchSnapshot, generate_handoff, trim_feedback

REPO = Path(__file__).resolve().parents[2]


def snapshot(**changes: Any) -> WorkbenchSnapshot:
    """main.py's demo snapshot, with the given fields changed."""
    demo = dict(task_id="T-001",
                state={"active_task_id": None, "blockers": ["awaiting decision on rate-limit window"],
                       "next_action": "open PR with current diff and request review"},
                verdict={"passed": True, "findings": [{"severity": "warn", "detail": "off-scope: README.md"}]},
                review={"verdict": "pass", "total": 8},
                feedback=[{"command": "pytest", "exit_code": 0}, {"command": "ruff check .", "exit_code": 0},
                          {"command": "pytest test_signup.py", "exit_code": 1}, {"command": "pytest test_signup.py", "exit_code": 0}],
                diff_summary={"touched": ["app/signup.py", "tests/test_signup.py", "README.md"]})
    return WorkbenchSnapshot(**{**demo, **changes})


# ---------------------------------------------------------------------------
# Exercise 1 - assumptions_to_validate
#
# An assumption counts as validated only when the reviewer scored that
# assumption above 1. One it scored 0 or 1, and one it never scored, both go
# into the packet: silence from the reviewer is not agreement.
# ASSUMPTION: the review report carries assumption_scores, one per assumption.
#
# Lesson 39's own "assumptions" dimension cannot stand in for that. Checked
# below: its scorer gives 2 as soon as anything is written down, whatever it
# says. It scores the act of recording, not the assumption.
# ---------------------------------------------------------------------------

def assumptions_to_validate(snap: WorkbenchSnapshot) -> list[str]:
    scores = snap.review.get("assumption_scores") or {}
    return [text for text in snap.state.get("assumptions") or [] if scores.get(text, 0) <= 1]       # type: ignore[union-attr]


def ex1_assumptions_to_validate() -> None:
    logged = ["users sign up with email and password only", "passwords under 8 characters are rejected upstream too",
              "the tests are probably fine"]
    snap = snapshot(state={**snapshot().state, "assumptions": logged},
                    review={"verdict": "pass", "total": 8, "assumption_scores": {logged[0]: 2, logged[1]: 1}})
    pending = assumptions_to_validate(snap)
    for text in logged:
        score = snap.review["assumption_scores"].get(text, "not scored")        # type: ignore[union-attr]
        print(f"  reviewer: {score!s:<10} {'VALIDATE' if text in pending else 'ok':<8} {text}")
    assert pending == logged[1:] and assumptions_to_validate(snapshot()) == []

    spec = importlib.util.spec_from_file_location("lesson39", REPO / "39-reviewer-agent" / "code" / "main.py")
    lesson39 = importlib.util.module_from_spec(spec)
    sys.modules["lesson39"] = lesson39
    spec.loader.exec_module(lesson39)
    dimension = lesson39.score_assumptions(lesson39.ReviewerInputs("T-001", "", {}, {"assumptions": logged[2:]}, [], {}))
    print(f"  lesson 39's dimension score for {logged[2]!r} alone: {dimension.score}/2")
    assert dimension.score == 2


# ---------------------------------------------------------------------------
# Exercise 2 - trimming failures and passes differently
#
# A pass carries one fact: this command is green. The packet keeps the latest
# pass of each command, as command and exit code, without the output.
# A failure carries what the next session needs so it does not repeat it. The
# packet keeps the latest failure of each kind with its error text, how many
# times it happened, and whether the same command passed afterwards. Both
# sides are bounded.
#
# The asymmetry is in the cost of losing each. Lose a pass and the next
# session re-runs a green command. Lose a failure and it spends its first
# half hour rediscovering it.
#
# Measured against main.py's trim on a log of 56 records:
#   - main.py does not count a missing exit code as a failure, so a timeout
#     outside the last five records is dropped from the packet;
#   - it keeps every non-zero record, so twelve retries are twelve entries;
#   - it lists the tail first and older failures after, so the list ends on a
#     failure that came before the pass which fixed it.
# ---------------------------------------------------------------------------

PASS_KEEP, FAIL_KEEP = 5, 10


def trim_asymmetric(records: list[dict[str, object]]) -> list[dict[str, object]]:
    latest: dict[tuple[str, object], int] = {}
    times: dict[tuple[str, object], int] = {}
    for index, record in enumerate(records):
        kind = (str(record.get("command")), record.get("exit_code"))
        latest[kind], times[kind] = index, times.get(kind, 0) + 1
    passes = sorted(index for (_, code), index in latest.items() if code == 0)[-PASS_KEEP:]
    failures = sorted(index for (_, code), index in latest.items() if code != 0)[-FAIL_KEEP:]
    last_pass = {command: index for (command, code), index in latest.items() if code == 0}
    trimmed: list[dict[str, object]] = []
    for index in sorted(passes + failures):
        record, command = records[index], str(records[index].get("command"))
        if record.get("exit_code") == 0:
            trimmed.append({"command": command, "exit_code": 0})
        else:
            trimmed.append({**record, "times": times[(command, record.get("exit_code"))],
                            "resolved_later": last_pass.get(command, -1) > index})
    return trimmed


def ex2_asymmetric_trim() -> None:
    noisy = "collected 212 items\n" + "." * 2000
    log: list[dict[str, object]] = [{"command": "pytest", "exit_code": 0, "stdout_tail": noisy} for _ in range(3)]
    log.append({"command": "npm run build", "exit_code": None, "error": "timeout after 30.0s"})
    log += [{"command": "pytest", "exit_code": 0, "stdout_tail": noisy} for _ in range(36)]
    log += [{"command": "pytest tests/test_signup.py", "exit_code": 1, "stderr_tail": f"E   assert 400 == 200 (attempt {attempt})"}
            for attempt in range(1, 13)]
    log.append({"command": "pytest tests/test_signup.py", "exit_code": 0, "stdout_tail": noisy})
    log += [{"command": "ruff check .", "exit_code": 0, "stdout_tail": noisy} for _ in range(3)]

    shipped, mine = trim_feedback(log), trim_asymmetric(log)
    for label, kept in (("main.py", shipped), ("asymmetric", mine)):
        timeout = any(record.get("exit_code") is None for record in kept)
        print(f"  {label:<11} keeps {len(kept):>2} of {len(log)} records, {len(json.dumps(kept)):>5} bytes, timeout kept: {timeout}, "
              f"last entry: {kept[-1]['command']} exit {kept[-1]['exit_code']}")
    for record in mine:
        extra = "" if record["exit_code"] == 0 else f"  x{record['times']}, resolved later: {record['resolved_later']}"
        print(f"    {record['command']:<28} exit {record['exit_code']!s:<4}{extra}")
    assert len(shipped) == 16 and not any(record.get("exit_code") is None for record in shipped) and shipped[-1]["exit_code"] == 1
    assert [(record["command"], record["exit_code"]) for record in mine] == [
        ("npm run build", None), ("pytest", 0), ("pytest tests/test_signup.py", 1), ("pytest tests/test_signup.py", 0), ("ruff check .", 0)]
    assert (mine[0]["resolved_later"], mine[2]["resolved_later"], mine[2]["times"]) == (False, True, 12)
    assert "attempt 12" in str(mine[2]["stderr_tail"]) and len(json.dumps(mine)) < len(json.dumps(shipped)) / 10


# ---------------------------------------------------------------------------
# Exercise 3 - questions for the human
#
# The threshold for the packet: the question is still unanswered when the
# session ends, and the next session would act differently depending on the
# answer. In practice that means it blocks something named.
#   answered during the session      -> chat; the answer goes into state
#   unanswered, blocks nothing, and
#   the agent took a default         -> not a question: an assumption to validate
#   unanswered, blocks nothing       -> chat, or nowhere
#   unanswered, blocks something     -> the packet
# A packet holds at most three, the ones blocking the next action first.
# A list of fifteen questions is a list nobody answers; the rest are counted
# and left in state.
# ASSUMPTION: the builder keeps open questions in state["open_questions"],
# as lesson 33's uncertainty rule asks.
# ---------------------------------------------------------------------------

MAX_QUESTIONS = 3


def triage(question: dict[str, str]) -> str:
    if question.get("answer"):
        return "chat"
    if question.get("blocks"):
        return "packet"
    return "assumption" if question.get("default_taken") else "chat"


def questions_for_packet(snap: WorkbenchSnapshot) -> tuple[list[str], list[str]]:
    """(lines for the packet, defaults the agent took that need validating)."""
    asked: list[dict[str, str]] = list(snap.state.get("open_questions") or [])       # type: ignore[call-overload]
    blocking = sorted((question for question in asked if triage(question) == "packet"),
                      key=lambda question: question["blocks"] != "next_action")
    lines = [f"{question['question']} (blocks: {question['blocks']})" for question in blocking[:MAX_QUESTIONS]]
    if len(blocking) > MAX_QUESTIONS:
        lines.append(f"{len(blocking) - MAX_QUESTIONS} more blocking questions are in agent_state.json")
    assumed = [f"{question['question']} - assumed: {question['default_taken']}" for question in asked if triage(question) == "assumption"]
    return lines, assumed


def ex3_questions_for_human() -> None:
    asked = [
        {"question": "Which CSS class does the error banner use?", "answer": "form-error"},
        {"question": "Should the minimum password length be 8 or 12?", "default_taken": "8, as the current docs say"},
        {"question": "Is it fine that I renamed the helper?"},
        {"question": "May the migration drop the legacy_email column?", "blocks": "T-003"},
        {"question": "What is the rate-limit window?", "blocks": "next_action"},
    ]
    kinds = [triage(question) for question in asked]
    for question, kind in zip(asked, kinds):
        print(f"  {kind:<10} {question['question']}")
    packet, assumed = questions_for_packet(snapshot(state={**snapshot().state, "open_questions": asked}))
    assert kinds == ["chat", "assumption", "chat", "packet", "packet"]
    assert packet == ["What is the rate-limit window? (blocks: next_action)", "May the migration drop the legacy_email column? (blocks: T-003)"]
    assert assumed == ["Should the minimum password length be 8 or 12? - assumed: 8, as the current docs say"]

    many = [{"question": f"Question {number}?", "blocks": "T-009"} for number in range(1, 6)]
    capped, _ = questions_for_packet(snapshot(state={"open_questions": many}))
    print(f"  five blocking questions -> {len(capped) - 1} in the packet, then: {capped[-1]!r}")
    assert len(capped) == MAX_QUESTIONS + 1 and capped[-1].startswith("2 more")


# ---------------------------------------------------------------------------
# Exercise 5 (used by the generator below) - next session prereqs
#
# Exactly what the next session must load before it acts, and why. The state
# file is always on the list. Everything else is there only when this
# session's outcome makes it necessary, so a clean close-out asks for one
# file and a blocked one asks for five. A prereq that does not exist is a
# broken handoff, so the list can be checked against the disk.
# ---------------------------------------------------------------------------

def prereqs(snap: WorkbenchSnapshot) -> list[dict[str, str]]:
    needed = [{"path": "agent_state.json", "why": "where the work stands and the next action"}]
    if snap.state.get("active_task_id"):
        needed.append({"path": f"scope_contract/{snap.task_id}.json", "why": "the task is still open; read its scope before any write"})
    flagged = [finding for finding in snap.verdict.get("findings") or []            # type: ignore[union-attr]
               if isinstance(finding, dict) and finding.get("severity") in ("warn", "block")]
    if not snap.verdict.get("passed") or flagged:
        gate = "passed with warnings" if snap.verdict.get("passed") else "did not pass"
        needed.append({"path": f"outputs/verification/{snap.task_id}.json", "why": f"the gate {gate}"})
    if snap.review.get("verdict") != "pass":
        needed.append({"path": f"outputs/review/{snap.task_id}.json", "why": f"the review verdict is {snap.review.get('verdict')}"})
    if any(record.get("resolved_later") is False for record in trim_asymmetric(snap.feedback)):
        needed.append({"path": "feedback_record.jsonl", "why": "a failure was never followed by a pass; the packet holds only its tail"})
    return needed


def missing_prereqs(root: Path, needed: list[dict[str, str]]) -> list[str]:
    return [entry["path"] for entry in needed if not (root / entry["path"]).exists()]


# ---------------------------------------------------------------------------
# The generator: main.py's packet plus the fields of exercises 1, 2, 3 and 5.
# The Markdown is written from the payload, so the two forms cannot diverge.
# ---------------------------------------------------------------------------

@dataclass
class HandoffPlus(HandoffPayload):
    assumptions_to_validate: list[str] = field(default_factory=list)
    questions_for_human: list[str] = field(default_factory=list)
    prereqs: list[dict[str, str]] = field(default_factory=list)


def generate_plus(snap: WorkbenchSnapshot) -> tuple[str, HandoffPlus]:
    markdown, base = generate_handoff(snap)
    questions, assumed = questions_for_packet(snap)
    payload = HandoffPlus(**{**asdict(base), "feedback_tail": trim_asymmetric(snap.feedback)},
                          assumptions_to_validate=assumptions_to_validate(snap) + assumed,
                          questions_for_human=questions, prereqs=prereqs(snap))
    sections = [("Assumptions to validate", payload.assumptions_to_validate), ("Questions for the human", payload.questions_for_human),
                ("Load before acting", [f"`{entry['path']}` - {entry['why']}" for entry in payload.prereqs])]
    for title, items in sections:
        markdown += "\n".join(["", f"## {title}", *([f"- {item}" for item in items] or ["- none"])]) + "\n"
    return markdown, payload


# ---------------------------------------------------------------------------
# Exercise 4 - an idempotent generator
#
# main.py's generate_handoff is a pure function, so the same snapshot always
# gives the same packet. What breaks in practice is the snapshot. Four things
# have to be stable for a second run to write the same bytes:
#   - the inputs: the generator must not read what it wrote. A loader that
#     lists changed files picks up handoff.md and handoff.json on the second
#     run. It must not append to the log it reads, either.
#   - the order: anything that came from a set or a directory listing is
#     sorted. Set order of strings changes from one Python process to the next.
#   - the content: no clock and no random id in the packet body. When it was
#     written is the file's own timestamp, or the commit's.
#   - the text: every sentence is built from the inputs, with no model in
#     the path.
# Measured below: three runs in three processes with different hash seeds.
# ---------------------------------------------------------------------------

INPUTS = {"agent_state.json", "verification.json", "review.json", "feedback_record.jsonl"}
OUTPUTS = {"handoff.md", "handoff.json"}


def load_snapshot(root: Path, stable: bool = True) -> WorkbenchSnapshot:
    def read(name: str) -> Any:
        return json.loads((root / name).read_text())

    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} - INPUTS
    touched = sorted(files - OUTPUTS) if stable else list(files)        # the unstable loader is the naive one
    feedback = [json.loads(line) for line in (root / "feedback_record.jsonl").read_text().splitlines() if line.strip()]
    return WorkbenchSnapshot(task_id="T-001", state=read("agent_state.json"), verdict=read("verification.json"),
                             review=read("review.json"), feedback=feedback, diff_summary={"touched": touched})


def write_packet(root: Path, stable: bool = True) -> None:
    markdown, payload = generate_plus(load_snapshot(root, stable))
    (root / "handoff.md").write_text(markdown, newline="\n")
    (root / "handoff.json").write_text(json.dumps(asdict(payload), indent=2, sort_keys=True) + "\n", newline="\n")


def ex4_idempotent_generator() -> None:
    assert generate_plus(snapshot()) == generate_plus(snapshot())           # the pure part

    def three_runs(stable: bool) -> list[bytes]:
        demo, written = snapshot(), []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, content in (("agent_state.json", demo.state), ("verification.json", demo.verdict), ("review.json", demo.review)):
                (root / name).write_text(json.dumps(content))
            (root / "feedback_record.jsonl").write_text("".join(json.dumps(record) + "\n" for record in demo.feedback))
            for name in demo.diff_summary["touched"]:
                (root / name).parent.mkdir(parents=True, exist_ok=True)
                (root / name).write_text("")
            for seed in ("1", "2", "3"):
                subprocess.run([sys.executable, __file__, "--packet", str(root), *([] if stable else ["--naive"])],
                               env={**os.environ, "PYTHONHASHSEED": seed}, check=True)
                written.append((root / "handoff.json").read_bytes() + (root / "handoff.md").read_bytes())
        return written

    naive, stable = three_runs(stable=False), three_runs(stable=True)
    own_output = b'"handoff.json"'          # the generator's own file, listed as a changed file
    print(f"  naive loader : {len(set(naive))} different packets from 3 runs; the second run lists its own output: "
          f"{own_output in naive[1]}")
    print(f"  stable loader: {len(set(stable))} packet from 3 runs, {len(stable[0])} bytes each time")
    assert len(set(naive)) > 1 and len(set(stable)) == 1
    assert own_output not in naive[0] and own_output in naive[1] and own_output not in stable[2]


def ex5_next_session_prereqs() -> None:
    blocked = snapshot(state={"active_task_id": "T-001", "next_action": "fix the failing signup test"},
                       verdict={"passed": False, "findings": [{"severity": "block", "detail": "acceptance exit 1"}]},
                       review={"verdict": "soft_fail", "total": 6},
                       feedback=[{"command": "pytest test_signup.py", "exit_code": 1, "stderr_tail": "E   assert 400 == 200"}])
    finished = snapshot(state={"active_task_id": None, "next_action": "pick next task from board"},
                        verdict={"passed": True, "findings": []})
    lists = {"clean close-out": prereqs(finished), "main.py's demo": prereqs(snapshot()), "blocked mid-task": prereqs(blocked)}
    for label, needed in lists.items():
        print(f"  {label:<17} {len(needed)} to load: {[entry['path'] for entry in needed]}")
    assert [entry["path"] for entry in lists["clean close-out"]] == ["agent_state.json"]
    assert [entry["path"] for entry in lists["main.py's demo"]] == ["agent_state.json", "outputs/verification/T-001.json"]
    assert [entry["path"] for entry in lists["blocked mid-task"]] == [
        "agent_state.json", "scope_contract/T-001.json", "outputs/verification/T-001.json", "outputs/review/T-001.json",
        "feedback_record.jsonl"]

    markdown, payload = generate_plus(blocked)
    section = markdown.split("## Load before acting\n")[1].splitlines()
    for line in section:
        print(f"    | {line}")
    assert len(section) == len(payload.prereqs) == 5 and all(entry["path"] in markdown for entry in payload.prereqs)
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "agent_state.json").write_text("{}")
        absent = missing_prereqs(Path(tmp), payload.prereqs)
    print(f"  checked against a workbench that only has the state file: {len(absent)} of 5 missing")
    assert len(absent) == 4 and "agent_state.json" not in absent


# ---------------------------------------------------------------------------
# Mission (mission.md) - the acceptance criteria, checked
#
# main.py is run twice in a temp copy of this folder.
#   exits zero                               yes
#   both files carry all seven fields and    yes
#   a non-empty next_action
#   re-running with the same inputs gives    yes, byte for byte
#   an identical packet
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


SEVEN_FIELDS = ("summary", "changed_files", "commands_run", "failed_attempts", "open_risks", "next_action", "verdict_pointer")


def mission_acceptance() -> None:
    first, second = run_main(times=2)
    packet, markdown = json.loads(second[2]["code/handoff.json"]), second[2]["code/handoff.md"]
    headings = ("Summary", "Changed files", "Commands run", "Failed attempts", "Open risks", "Next action", "Receipts")
    print(f"  exit codes {first[0]} and {second[0]}; fields filled in handoff.json: "
          f"{sum(bool(packet[name]) for name in SEVEN_FIELDS)} of 7; next action {packet['next_action']!r}")
    print(f"  second run wrote the same bytes as the first: {first[2] == second[2]}")
    assert first[0] == second[0] == 0 and all(packet[name] for name in SEVEN_FIELDS)
    assert all(heading in markdown for heading in headings) and first[2] == second[2]


if __name__ == "__main__":
    if sys.argv[1:2] == ["--packet"]:
        write_packet(Path(sys.argv[2]), stable="--naive" not in sys.argv)
        sys.exit(0)
    print("Phase 14 - Lesson 40: Multi-Session Handoff - exercises")
    for exercise in (ex1_assumptions_to_validate, ex2_asymmetric_trim, ex3_questions_for_human, ex4_idempotent_generator,
                     ex5_next_session_prereqs, mission_acceptance):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
