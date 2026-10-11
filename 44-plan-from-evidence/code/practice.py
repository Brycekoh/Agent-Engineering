"""Phase 14 - Lesson 44: Build an Evidence-Backed Execution Plan - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)

The plan is for the bug lesson 43 framed: lesson 16's Runner.run returns an empty string when it
runs out of hops. Every piece of evidence is a line that exists in this repository.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

from main import WorkItem, execution_waves, plan_document, validate

REPO = Path(__file__).resolve().parents[2]
RUNNER = "16-openai-agents-sdk/code/main.py"
PINNED = [  # the three checks that assert the current behaviour (lesson 43, exercise 2): (file, the line that does it)
    ("16-openai-agents-sdk/code/practice.py", 'assert silent == ""'),
    ("19-benchmarks-swebench-gaia/code/practice.py", "def t3_hop_limit_reported"),
    ("30-eval-driven-agent-development/code/practice.py", "assert not now.passed and after_fix.passed"),
]


def receipt(path: str, text: str) -> str:
    """'path:line' for the line of that file that contains the text, or '' when no line does."""
    file = REPO / path
    if file.is_file():
        for number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
            if text in line:
                return f"{path}:{number}"
    return ""


@dataclass(frozen=True)
class Gated(WorkItem):
    needs_approval: bool = False


def first_plan() -> list[WorkItem]:
    """The plan as first written: decide, fix, run everything."""
    return [
        WorkItem("contract", "Decide what a used-up run gives back: an exception or a message",
                 (receipt(RUNNER, 'final_output = ""'), receipt(RUNNER, "for hop in range(self.max_hops):")), (),
                 "the answer is recorded in the task frame of lesson 43"),
        WorkItem("runner", "Make Runner.run report a used-up hop limit",
                 (receipt(RUNNER, "for hop in range(self.max_hops):"),), ("contract",),
                 "python 43-frame-the-task-before-code/code/practice.py --acceptance"),
        WorkItem("suite", "Run every lesson's practice file",
                 (receipt("30-eval-driven-agent-development/code/practice.py", "def run_lesson_cases"),), ("runner",),
                 "python 30-eval-driven-agent-development/code/practice.py --run-suite"),
    ]


# ---------------------------------------------------------------------------
# Exercise 1 - a migration item that needs explicit human approval
#
# The first plan is missing a step, and the evidence says which: three checks
# in this repository assert the empty string, so the suite at the end cannot
# pass until they are brought over to the new contract. That migration is the
# new item.
#
# It needs a person's approval because of what it edits. It rewrites
# assertions, and an agent that changes a failing check until it passes has
# changed what "done" means. A person approves the new expectation; the agent
# only types it.
#
# main.py has no notion of approval. Here an unapproved item holds itself and
# everything after it, and nothing before it: the fix can go ahead while the
# approval is pending.
# ---------------------------------------------------------------------------

MIGRATION = Gated("migrate-pins", "Bring the three checks that pin the empty string over to the new contract",
                  tuple(receipt(path, text) for path, text in PINNED), ("runner",),
                  " && ".join(f"python {path}" for path, _ in PINNED), needs_approval=True)


def with_migration(items: list[WorkItem]) -> list[WorkItem]:
    return [*items[:-1], MIGRATION, replace(items[-1], depends_on=(*items[-1].depends_on, MIGRATION.id))]


def held(items: list[WorkItem], approved: set[str]) -> dict[str, str]:
    """The items that may not start yet, with the reason."""
    by_id, reasons = {item.id: item for item in items}, {}
    for wave in execution_waves(items):
        for item_id in wave:
            waits_on = [dep for dep in by_id[item_id].depends_on if dep in reasons]
            if getattr(by_id[item_id], "needs_approval", False) and item_id not in approved:
                reasons[item_id] = "needs explicit human approval"
            elif waits_on:
                reasons[item_id] = f"waits on {waits_on[0]}"
    return reasons


def ex1_migration_with_approval() -> None:
    items = with_migration(first_plan())
    waiting = held(items, approved=set())
    startable = [[item for item in wave if item not in waiting] for wave in execution_waves(items)]
    print(f"  waves: {execution_waves(items)}")
    print(f"  with no approval: may start {[wave for wave in startable if wave]}; held {waiting}")
    print(f"  once a person approves: held {held(items, approved={'migrate-pins'})}")
    assert validate(items) == [] and all(MIGRATION.evidence)
    assert execution_waves(items) == [["contract"], ["runner"], ["migrate-pins"], ["suite"]]
    assert waiting == {"migrate-pins": "needs explicit human approval", "suite": "waits on migrate-pins"}
    assert held(items, approved={"migrate-pins"}) == {} and plan_document(items)["items"][2]["needs_approval"] is True


# ---------------------------------------------------------------------------
# Exercise 2 - a cycle, and the disagreement behind it
#
# The cycle: the fix waits for the checks ("write the expectation first"),
# and the checks wait for the fix ("we need the real wording to assert on").
# Both are reasonable, and each is waiting for the other to say what the run
# should give back. That is the disagreement, and it is about the product,
# not about ordering: nobody has decided between an exception and a message.
#
# The cycle goes away when that decision becomes an item of its own that both
# depend on. Nothing about either task changed.
#
# main.py reports "dependency cycle among: migrate-pins, runner, suite". The
# suite is not in the cycle; it is only behind it. cycle_members() names the
# items that actually depend on themselves.
# ---------------------------------------------------------------------------

def cycle_members(items: list[WorkItem]) -> list[str]:
    needs = {item.id: set(item.depends_on) for item in items}

    def reaches(start: str, goal: str, seen: set[str]) -> bool:
        return any(dep == goal or (dep not in seen and reaches(dep, goal, seen | {dep})) for dep in needs.get(start, ()))

    return sorted(item_id for item_id in needs if reaches(item_id, item_id, set()))


def ex2_cycle() -> None:
    items = with_migration(first_plan())
    tests_first = [replace(item, depends_on=("migrate-pins",)) if item.id == "runner" else item for item in items[1:]]
    issues = validate(tests_first)
    print(f"  main.py: {issues[0]}")
    print(f"  items that depend on themselves: {cycle_members(tests_first)}")
    print(f"  with the decision as its own item: {execution_waves(items)[:3]}")
    assert issues == ["dependency cycle among: migrate-pins, runner, suite"]
    assert cycle_members(tests_first) == ["migrate-pins", "runner"] and cycle_members(items) == []


# ---------------------------------------------------------------------------
# Exercise 3 - one item with several proof commands, split
#
# The migration item's proof is three commands joined with &&, one per file.
# When that proof fails it does not say which file, and the three edits
# cannot be handed out or checked separately. One item should carry one claim
# and one proof, so it becomes three, each with the receipt for its own file.
#
# The split has to carry two things across: the items that depended on the
# whole now depend on every part, and each part still needs the approval the
# whole needed.
# ---------------------------------------------------------------------------

def split_by_proof(items: list[WorkItem], item_id: str) -> list[WorkItem]:
    whole = next(item for item in items if item.id == item_id)
    proofs = whole.proof.split(" && ")
    parts = [replace(whole, id=f"{whole.id}-{evidence[:2]}", evidence=(evidence,), proof=proof)
             for evidence, proof in zip(whole.evidence, proofs, strict=True)]
    out: list[WorkItem] = []
    for item in items:
        if item.id == item_id:
            out += parts
        elif item_id in item.depends_on:
            out.append(replace(item, depends_on=tuple(dep for dep in item.depends_on if dep != item_id) + tuple(p.id for p in parts)))
        else:
            out.append(item)
    return out


def ex3_split_by_proof() -> None:
    items = split_by_proof(with_migration(first_plan()), "migrate-pins")
    parts = [item for item in items if item.id.startswith("migrate-pins-")]
    for part in parts:
        print(f"  {part.id:<16} proof: {part.proof:<58} evidence: {part.evidence[0]}")
    print(f"  waves: {execution_waves(items)}")
    assert validate(items) == [] and [part.id for part in parts] == ["migrate-pins-16", "migrate-pins-19", "migrate-pins-30"]
    assert all(" && " not in item.proof for item in items) and all(part.needs_approval for part in parts)      # type: ignore[attr-defined]
    assert execution_waves(items)[2] == [part.id for part in parts]                        # the three can go in parallel
    assert set(items[-1].depends_on) == {"runner", *(part.id for part in parts)}
    assert held(items, approved={"migrate-pins-16"})["suite"] == "waits on migrate-pins-19"  # approval is per part now


# ---------------------------------------------------------------------------
# Exercise 4 - a second-wave item that touches neither branch
#
# The plan has two branches after the decision: the fix (one file in lesson
# 16) and the migration (the three practice files). The new item closes the
# open question in lesson 43's task frame once the decision is made. It
# depends on the decision only and its evidence is in a file neither branch
# goes near, so it runs in the second wave beside the fix.
#
# "Touches neither" is checked, not asserted by eye: the files an item's
# evidence points at are compared across the plan.
# ---------------------------------------------------------------------------

FRAME_ITEM = WorkItem("frame", "Close the human unknown in the lesson 43 task frame with the decision",
                      (receipt("43-frame-the-task-before-code/code/practice.py", "CONTRACT_QUESTION = ("),), ("contract",),
                      "python 43-frame-the-task-before-code/code/practice.py")


def files_of(item: WorkItem) -> set[str]:
    return {evidence.rsplit(":", 1)[0] for evidence in item.evidence}


def final_plan() -> list[WorkItem]:
    items = split_by_proof(with_migration(first_plan()), "migrate-pins")
    return [*items[:-1], FRAME_ITEM, items[-1]]


def ex4_second_wave_item() -> None:
    items = final_plan()
    waves = execution_waves(items)
    others = set().union(*(files_of(item) for item in items if item.id not in ("frame", "contract")))
    print(f"  waves: {waves}")
    print(f"  the frame item's file: {sorted(files_of(FRAME_ITEM))}; shared with any other branch: {sorted(files_of(FRAME_ITEM) & others)}")
    assert validate(items) == [] and waves[1] == ["frame", "runner"] and len(waves) == 4
    assert all(FRAME_ITEM.evidence) and not files_of(FRAME_ITEM) & others
    assert "frame" not in held(items, approved=set())                 # it does not wait for the migration's approval


# ---------------------------------------------------------------------------
# Exercise 5 - Markdown for people, JSON as the source of truth
#
# to_markdown() takes the JSON document and nothing else, so the Markdown
# cannot say anything the JSON does not. The other direction is checked too:
# in_step() reads the waves back out of a Markdown file and compares them
# with the JSON. A plan edited by hand in the Markdown fails that check, and
# the fix is to change the JSON and render again.
# ---------------------------------------------------------------------------

def to_markdown(document: dict) -> str:
    by_id = {item["id"]: item for item in document["items"]}
    lines = [f"# Plan ({document['status']})", ""]
    lines += [f"- BLOCKED: {issue}" for issue in document["issues"]]
    for number, wave in enumerate(document["waves"], 1):
        lines += [f"## Wave {number}", ""]
        for item_id in wave:
            item = by_id[item_id]
            gate = "  (needs human approval)" if item.get("needs_approval") else ""
            lines += [f"- **{item_id}**: {item['change']}{gate}", f"  - proof: `{item['proof']}`",
                      f"  - evidence: {', '.join(f'`{evidence}`' for evidence in item['evidence'])}"]
        lines.append("")
    return "\n".join(lines)


def in_step(document: dict, markdown: str) -> bool:
    waves, current = [], None
    for line in markdown.splitlines():
        if line.startswith("## Wave "):
            current = []
            waves.append(current)
        elif (found := re.match(r"- \*\*([\w-]+)\*\*", line)) and current is not None:
            current.append(found[1])
    return waves == document["waves"]


def ex5_markdown_from_json() -> None:
    document = json.loads(json.dumps(plan_document(final_plan())))        # as it would be read back from the file
    markdown = to_markdown(document)
    for line in markdown.splitlines()[:9]:
        print(f"  | {line[:150]}")
    print(f"  | ... {len(markdown.splitlines())} lines in all")
    edited = markdown.replace("## Wave 3\n\n", "")                        # someone merges two waves in the Markdown to go faster
    print(f"  Markdown in step with the JSON: {in_step(document, markdown)}; after a hand edit: {in_step(document, edited)}")
    assert document["status"] == "ready" and in_step(document, markdown) and not in_step(document, edited)
    assert markdown.count("needs human approval") == 3 and all(f"**{item['id']}**" in markdown for item in document["items"])
    blocked = to_markdown(plan_document([replace(final_plan()[0], evidence=())]))
    assert blocked.startswith("# Plan (blocked)") and "BLOCKED: contract has no repository evidence" in blocked


if __name__ == "__main__":
    print("Phase 14 - Lesson 44: Build an Evidence-Backed Execution Plan - exercises")
    for exercise in (ex1_migration_with_approval, ex2_cycle, ex3_split_by_proof, ex4_second_wave_item, ex5_markdown_from_json):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
