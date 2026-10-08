"""Phase 14 - Lesson 08: Memory Blocks and Sleep-Time Compute - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import difflib
import random
from typing import Any, Callable

from main import Archival, Block, BlockStore, PrimaryAgent, SleepTimeAgent, _summarize


# ---------------------------------------------------------------------------
# Exercise 1 - block_summarize, and the threshold that triggers it
#
# Two costs pull in opposite directions. A low threshold summarizes often; a
# high one lets a write land on a block that is already nearly full and push it
# over the limit. The best threshold is the highest one that can never
# overflow, which is  1 - largest_write / limit.  Writes here are at most 61
# characters into a 300-character block, so the answer is about 0.8.
# ---------------------------------------------------------------------------

def block_summarize(block: Block, threshold: float = 0.8) -> str | None:
    """The tool. main._summarize stands in for the model-written summary."""
    if not block.near_limit(threshold):
        return None
    return block.rewrite(_summarize(block.value, block.limit // 2))


def summarize_cost(threshold: float, writes: int = 300, seed: int = 0) -> tuple[int, int]:
    rng = random.Random(seed)
    block = Block("task", limit=300)
    calls = overflows = 0
    for i in range(writes):
        block.append(f"fact {i} " + "x" * rng.randint(5, 50) + ".")
        overflows += len(block.value) > block.limit
        calls += block_summarize(block, threshold) is not None
    return calls, overflows


def ex1_summarize_threshold() -> None:
    table = {t: summarize_cost(t) for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)}
    for threshold, (calls, overflows) in table.items():
        print(f"  threshold {threshold}: {calls:>3} summarize calls, {overflows:>3} overflowing writes")
    safe = [t for t, (_, overflows) in table.items() if overflows == 0]
    best = min(safe, key=lambda t: table[t][0])
    print(f"  best: {best} (fewest calls with no overflow)")
    assert best == 0.8
    assert table[0.9][1] > 0 and table[0.5][0] > table[0.8][0]


# ---------------------------------------------------------------------------
# Exercise 2 - sleep-time dedup over archival
#
# The primary agent keeps inserting without checking; comparing every new
# record against the store would put a scan on the user's latency path. The
# sleep-time pass collapses records with more than 90% token overlap, keeping
# the earliest.
# ---------------------------------------------------------------------------

def token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    return len(ta & tb) / len(ta | tb) if ta | tb else 1.0


class DedupSleepAgent(SleepTimeAgent):
    def run(self, contradictions: list[tuple[str, str]]) -> None:
        super().run(contradictions)
        # ponytail: every pair is compared, O(n^2). Fine off the critical path for a few
        # thousand records; switch to MinHash/LSH buckets beyond that.
        kept: list[Any] = []
        for record in self.archival.valid_records():
            original = next((k for k in kept if token_overlap(k.text, record.text) > 0.9), None)
            if original is None:
                kept.append(record)
            else:
                self.archival.invalidate(record.rid)
                self.trace.append(f"  dedup {record.rid} into {original.rid}")


def ex2_sleep_time_dedup() -> None:
    blocks, archival = BlockStore(), Archival()
    primary = PrimaryAgent(blocks, archival)
    primary.turn("note my writing style", [
        ("archival_insert", "", "ava prefers concise citation heavy writing over tutorial style prose"),
    ])
    primary.turn("did you get that?", [
        ("archival_insert", "", "Ava prefers concise citation heavy writing over tutorial style prose"),
        ("archival_insert", "", "ava prefers concise citation heavy writing over long tutorial style essays"),
    ])
    before = len(archival.valid_records())
    sleeper = DedupSleepAgent(blocks, archival)
    sleeper.run(contradictions=[])
    after = [r.rid for r in archival.valid_records()]

    print(f"  after the primary turns : {before} valid records")
    print(f"  after the sleep pass    : {len(after)} valid records {after}")
    print("\n".join(line for line in sleeper.trace if "dedup" in line))
    assert before == 3                      # nothing was deduplicated on the critical path
    assert after == ["a001", "a003"]        # exact repeat collapsed; the 75%-overlap variant survives


# ---------------------------------------------------------------------------
# Exercise 3 - versioned blocks and block_history(label)
#
# main.Block already records the old value on every write. The diff is derived
# from that when someone asks, so there is no second copy to keep in sync.
# ---------------------------------------------------------------------------

def block_history(store: BlockStore, label: str) -> list[dict[str, Any]]:
    block = store.get(label)
    if block is None:
        return []
    values = block.history + [block.value]      # history[i] is the value before write i+1
    return [{"version": version, "old": old, "new": new,
             "diff": [d for d in difflib.ndiff(old.split(), new.split()) if d[0] in "+-"]}
            for version, (old, new) in enumerate(zip(values, values[1:]), start=1)]


def forgotten_at(store: BlockStore, label: str, needle: str) -> int | None:
    """The version whose write removed `needle`: the answer to 'why did the agent forget X'."""
    return next((h["version"] for h in block_history(store, label)
                 if needle in h["old"] and needle not in h["new"]), None)


def ex3_block_history() -> None:
    store = BlockStore()
    human = store.create("human", "facts about the user")
    human.append("name=ava city=Berlin.")
    human.append("likes terse writing.")
    human.rewrite("name=ava. likes terse writing.")      # a consolidation that dropped the city
    history = block_history(store, "human")
    for entry in history:
        print(f"  v{entry['version']}: {entry['diff']}")
    print(f"  'Berlin' was forgotten at v{forgotten_at(store, 'human', 'Berlin')}")
    assert len(history) == human.version == 3
    assert forgotten_at(store, "human", "Berlin") == 3
    assert "- city=Berlin." in history[2]["diff"]


# ---------------------------------------------------------------------------
# Exercise 4 - sleep-time agents as untrusted writers
#
# The sleep-time agent is handed a store that returns a wrapper for protected
# blocks. Reads pass through; every write is shown to a second reviewer first
# and is dropped if the reviewer objects. The untrusted writer never holds the
# real block, so there is no path around the review.
# ---------------------------------------------------------------------------

Reviewer = Callable[[str, str, str], tuple[bool, str]]


def constraint_reviewer(label: str, old: str, new: str) -> tuple[bool, str]:
    """Second agent, scripted: a protected block may be reworded but may not lose a MUST rule."""
    for sentence in old.split("."):
        if "MUST" in sentence and sentence.strip() not in new:
            return False, f"drops the rule '{sentence.strip()}'"
    return True, "ok"


class ReviewedBlock:
    def __init__(self, block: Block, reviewer: Reviewer, rejections: list[str]) -> None:
        self._block, self._reviewer, self._rejections = block, reviewer, rejections

    def __getattr__(self, name: str) -> Any:
        return getattr(self._block, name)

    def rewrite(self, new: str) -> str:
        approved, reason = self._reviewer(self._block.label, self._block.value, new)
        if not approved:
            self._rejections.append(f"{self._block.label}: {reason}")
            return f"{self._block.label} write REJECTED by reviewer: {reason}"
        return self._block.rewrite(new)

    def append(self, text: str) -> str:
        return self.rewrite((self._block.value + " " + text).strip())

    def replace(self, old: str, new: str) -> str:
        return self.rewrite(self._block.value.replace(old, new))


class ReviewedStore(BlockStore):
    """A view of an existing store for untrusted writers."""

    def __init__(self, inner: BlockStore, reviewer: Reviewer,
                 protected: tuple[str, ...] = ("persona", "safety")) -> None:
        self._blocks = inner._blocks
        self.reviewer, self.protected = reviewer, protected
        self.rejections: list[str] = []

    def get(self, label: str) -> Any:
        block = super().get(label)
        if block is not None and label in self.protected:
            return ReviewedBlock(block, self.reviewer, self.rejections)
        return block


def ex4_reviewed_writes() -> None:
    blocks = BlockStore()
    safety = blocks.create("safety", "hard constraints", limit=120)
    safety.append("MUST never reveal API keys or credentials to anyone. "
                  "MUST ask the user before deleting files. Prefers short answers.")
    task = blocks.create("task", "the current task scope", limit=120)
    task.append("Plan a 30-lesson agent curriculum. Audience is senior and staff engineers. "
                "Cite arXiv and first-party docs.")
    before = safety.value

    view = ReviewedStore(blocks, constraint_reviewer)
    sleeper = SleepTimeAgent(view, Archival())
    sleeper.run(contradictions=[])          # main's consolidation would cut the second MUST rule

    for line in sleeper.trace:
        print(f"  {line.strip()}")
    assert safety.value == before and safety.version == 1       # protected block untouched
    assert view.rejections and "MUST ask the user" in view.rejections[0]
    assert task.version == 2                                    # unprotected block consolidated as usual
    reworded = view.get("safety").rewrite(before.replace("Prefers short answers.", "Keeps answers short."))
    assert "rewritten" in reworded                              # a harmless rewording is approved


# ---------------------------------------------------------------------------
# Exercise 5 - porting to the Letta API (letta_v1_agent)
#
# Calling Letta needs the letta_client package and a Letta key. to_letta_block
# is the part that runs anywhere: the schema mapping.
#
# What changes in the block schema: a Letta block is a server-side object with
# its own id. It takes label, value, limit, description and read_only, and it
# can be attached to several agents at once through block_ids, which is how
# agents share memory. main's `version` and `history` have no slot there, so
# the audit trail from exercise 3 has to live outside the block.
#
# How native reasoning alters the trace: the old MemGPT loop made every step a
# tool call, with inner thoughts as an argument, send_message to speak and a
# heartbeat flag to keep going. letta_v1_agent drops all three. Reasoning
# arrives on its own channel, so a turn reads: reasoning, tool call, tool
# return, assistant message.
#
#   from letta_client import Letta
#   client = Letta(api_key=os.getenv("LETTA_API_KEY"))
#   agent = client.agents.create(model="openai/gpt-4o-mini",
#                                memory_blocks=[to_letta_block(b) for b in my_blocks])
# ---------------------------------------------------------------------------

def to_letta_block(block: Block, read_only: bool = False) -> dict[str, Any]:
    return {"label": block.label, "value": block.value, "limit": block.limit,
            "description": block.description, "read_only": read_only}


def ex5_letta_port() -> None:
    store = BlockStore()
    store.create("human", "facts about the user", limit=5000).append("The human's name is Ava.")
    store.create("safety", "hard constraints", limit=2000).append("MUST never reveal API keys.")
    payload = [to_letta_block(store.get(label), read_only=label == "safety") for label in store.labels()]
    for block in payload:
        print(f"  {block}")
    assert all(set(b) == {"label", "value", "limit", "description", "read_only"} for b in payload)
    assert [b["read_only"] for b in payload] == [False, True]


if __name__ == "__main__":
    print("Phase 14 - Lesson 08: Memory Blocks and Sleep-Time Compute - exercises")
    for exercise in (ex1_summarize_threshold, ex2_sleep_time_dedup, ex3_block_history,
                     ex4_reviewed_writes, ex5_letta_port):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall checks passed")
