"""Phase 14 - Lesson 09: Hybrid Memory (Vector + Graph + KV) - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import math
import random
import re
import time
import zlib
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from main import Edge, GraphStore, Mem0, Record, VectorStore

# (stored fact, a question that words it differently, a keyword only the right fact contains)
FACTS = [
    ("ava prefers terse citation-heavy writing", "ava's writing preference", "terse"),
    ("ava lives in lisbon since last month", "where is ava living", "lisbon"),
    ("ava is building a retrieval bot for the sales org", "what did ava build", "retrieval"),
    ("the launch deadline moved to the ninth of june", "when are we launching", "june"),
    ("the team chose postgres with pgvector for storage", "which database stores the vectors", "postgres"),
    ("finance approved four thousand dollars a month for model calls", "monthly approval from finance", "thousand"),
    ("priya manages the agent rollout and wants weekly notes", "who is the manager of the rollout", "priya"),
    ("the browser extension is written in typescript", "what language are the extensions written in", "typescript"),
    ("langgraph was picked for orchestration because of checkpointing", "which orchestrator has checkpoints", "langgraph"),
    ("the first customer is the enterprise renewals group", "who are our first customers", "renewals"),
    ("ava is allergic to peanuts and orders vegetarian", "does ava have an allergy", "peanuts"),
    ("the new laptop has sixteen inches and runs embeddings locally", "what laptops run local embedding models", "sixteen"),
    ("the team is on holiday in the second week of august", "when are the holidays", "august"),
    ("the vendor refunded invoice 4711 after the outage", "invoices that were refunded", "4711"),
    ("the staging cluster runs in frankfurt on three nodes", "which region hosts the staging nodes", "frankfurt"),
    ("evaluations run nightly against two hundred golden conversations", "how often do we evaluate", "nightly"),
    ("the on-call rotation changes every monday morning", "when does on-call rotate", "monday"),
    ("tracing uses opentelemetry with the genai conventions", "what do we trace with", "opentelemetry"),
    ("the voice agent pilot uses livekit for transport", "which transport do voice agents pilot on", "livekit"),
    ("prompt injection tests are required before every release", "what testing is required to release", "injection"),
]


# ---------------------------------------------------------------------------
# Exercise 1 - an embedding store, and recall@10 across 1000 writes
#
# STAND-IN EMBEDDING: sentence-transformers is not installed here, so
# hash_embed is a stdlib substitute - hashed character trigrams, normalised.
# It captures spelling similarity ("prefers" ~ "preference"), not meaning
# ("city" ~ "lisbon"). EmbeddingStore takes any `embed` function, so a trained
# model drops in:  EmbeddingStore(lambda t: dict(enumerate(model.encode(t)))).
#
# Does ranking drift over 1000 writes? The stored vectors never change. What
# drifts is rank: every write adds possible near neighbours, and the right
# record gets crowded out of the top 10. Filters (user, scope) and the
# importance/recency terms of the fusion score are what hold it back.
# ---------------------------------------------------------------------------

def hash_embed(text: str, dim: int = 512) -> dict[int, float]:
    vec: Counter[int] = Counter()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        padded = f"#{word}#"
        for i in range(len(padded) - 2):
            vec[zlib.crc32(padded[i:i + 3].encode()) % dim] += 1
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


class EmbeddingStore(VectorStore):
    def __init__(self, embed: Callable[[str], dict[int, float]] = hash_embed) -> None:
        super().__init__()
        self.embed = embed
        self._vectors: dict[str, dict[int, float]] = {}

    def add(self, record: Record) -> None:
        super().add(record)
        self._vectors[record.rid] = self.embed(record.text)

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Record]]:
        q = self.embed(query)
        scored = [(sum(w * vec.get(k, 0.0) for k, w in q.items()), self._records[rid])
                  for rid, vec in self._vectors.items()]
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]


def recall_curve(store: VectorStore, checkpoints: tuple[int, ...] = (20, 100, 300, 1000)) -> dict[int, float]:
    """recall@10 for the 20 facts as unrelated chatter is written on top of them."""
    rng = random.Random(0)
    vocabulary = sorted({w for fact, query, _ in FACTS for w in (fact + " " + query).split()})
    for i, (fact, _, _) in enumerate(FACTS):
        store.add(Record(f"g{i}", fact, "user", "ava", "s0"))
    written, curve = len(FACTS), {}
    for target in checkpoints:
        while written < target:
            chatter = " ".join(rng.choices(vocabulary, k=rng.randint(5, 9)))
            store.add(Record(f"c{written}", chatter, "user", "ava", "s0"))
            written += 1
        hits = sum(f"g{i}" in [r.rid for _, r in store.search(query, top_k=10)]
                   for i, (_, query, _) in enumerate(FACTS))
        curve[target] = hits / len(FACTS)
    return curve


def ex1_embedding_recall() -> None:
    overlap, embedded = recall_curve(VectorStore()), recall_curve(EmbeddingStore())
    for writes in overlap:
        print(f"  after {writes:>4} writes: recall@10  token overlap {overlap[writes]:.2f}   "
              f"trigram embedding {embedded[writes]:.2f}")
    assert embedded[20] == 1.0 and embedded[20] > overlap[20]
    assert embedded[1000] < embedded[20]            # rank drift from crowding
    assert all(embedded[w] >= overlap[w] for w in overlap)


# ---------------------------------------------------------------------------
# Exercise 2 - temporal queries: search(query, as_of=timestamp)
#
# Which store needs the most work: the graph. Vector and KV records already
# carry a timestamp, so "at or before t" is one filter. A graph edge only has
# `valid`, a boolean about now. Answering "was this true at t" needs the time
# the edge stopped being true, so the edge gets a new field, the write path
# has to stamp it, and reads need a new query.
# ---------------------------------------------------------------------------

@dataclass
class TemporalEdge(Edge):
    invalid_at: float | None = None


class TemporalGraph(GraphStore):
    def add_edge(self, subject: str, relation: str, obj: str, ts: float | None = None) -> None:
        ts = time.time() if ts is None else ts
        for edge in self._edges:
            if edge.valid and edge.subject == subject and edge.relation == relation:
                edge.valid = False
                edge.invalid_at = ts
        self._edges.append(TemporalEdge(subject=subject, relation=relation, obj=obj, ts=ts))

    def neighbors_as_of(self, subject: str, as_of: float) -> list[Edge]:
        return [e for e in self._edges
                if e.subject == subject and e.ts <= as_of
                and (e.invalid_at is None or e.invalid_at > as_of)]


class TemporalMem0(Mem0):
    def __init__(self) -> None:
        super().__init__()
        self.graph = TemporalGraph()

    def add(self, text: str, *, ts: float | None = None,
            graph_triples: tuple[tuple[str, str, str], ...] = (), **kwargs) -> str:
        ts = time.time() if ts is None else ts
        rid = super().add(text, **kwargs)           # edges are written below so they share ts
        self.vector._records[rid].ts = ts           # main.Mem0.add has no ts parameter
        for subject, relation, obj in graph_triples:
            self.graph.add_edge(subject, relation, obj, ts=ts)
        return rid

    def search(self, query: str, *, user_id: str, as_of: float | None = None,
               top_k: int = 5, **kwargs) -> list[tuple[float, Record]]:
        if as_of is None:
            return super().search(query, user_id=user_id, top_k=top_k, **kwargs)
        # ponytail: recency is still measured from now rather than from as_of. Order among
        # the survivors is unaffected; pass `now` into the scorer if absolute scores matter.
        hits = super().search(query, user_id=user_id, top_k=top_k * 4, **kwargs)
        return [(score, r) for score, r in hits if r.ts <= as_of][:top_k]


def ex2_temporal_search() -> None:
    mem = TemporalMem0()
    mem.add("ava lives in Berlin", user_id="ava", ts=1000.0,
            kv_triples=(("city", "Berlin"),), graph_triples=(("ava", "lives_in", "Berlin"),))
    mem.add("ava moved to Lisbon and lives there now", user_id="ava", ts=2000.0,
            kv_triples=(("city", "Lisbon"),), graph_triples=(("ava", "lives_in", "Lisbon"),))

    for as_of in (1500.0, 2500.0):
        records = [r.text for _, r in mem.search("where does ava live", user_id="ava", as_of=as_of)]
        cities = [e.obj for e in mem.graph.neighbors_as_of("ava", as_of)]
        print(f"  as_of={as_of:.0f}: graph says {cities}, records {records}")
    assert [e.obj for e in mem.graph.neighbors_as_of("ava", 1500.0)] == ["Berlin"]
    assert [e.obj for e in mem.graph.neighbors_as_of("ava", 2500.0)] == ["Lisbon"]
    assert [r.text for _, r in mem.search("where does ava live", user_id="ava", as_of=1500.0)] == ["ava lives in Berlin"]
    assert len(mem.search("where does ava live", user_id="ava", as_of=2500.0)) == 2


# ---------------------------------------------------------------------------
# Exercise 3 - conflict detector on the graph
#
# main.GraphStore invalidates any earlier edge with the same subject and
# relation, silently, and it does so for every relation - so a second project
# wipes out the first. A contradiction only exists for single-valued relations.
# The detector knows which those are, invalidates the old edge and logs both.
# ---------------------------------------------------------------------------

class ConflictGraph(GraphStore):
    def __init__(self, single_valued: tuple[str, ...] = ("lives_in",)) -> None:
        super().__init__()
        self.single_valued = set(single_valued)
        self.conflicts: list[tuple[Edge, Edge]] = []

    def add_edge(self, subject: str, relation: str, obj: str) -> None:
        same = [e for e in self._edges if e.valid and e.subject == subject and e.relation == relation]
        if any(e.obj == obj for e in same):
            return                                  # restating a known fact changes nothing
        new = Edge(subject=subject, relation=relation, obj=obj)
        if relation in self.single_valued:
            for old in same:
                old.valid = False
                self.conflicts.append((old, new))
        self._edges.append(new)


def ex3_conflict_detector() -> None:
    def load(mem: Mem0) -> Mem0:
        mem.add("ava lives in Berlin", user_id="ava", graph_triples=(("ava", "lives_in", "Berlin"),))
        mem.add("ava owns the curriculum project", user_id="ava", graph_triples=(("ava", "owns_project", "curriculum"),))
        mem.add("ava lives in Lisbon", user_id="ava", graph_triples=(("ava", "lives_in", "Lisbon"),))
        mem.add("ava owns the retrieval bot", user_id="ava", graph_triples=(("ava", "owns_project", "retrieval_bot"),))
        return mem

    baseline, detecting = load(Mem0()), Mem0()
    detecting.graph = ConflictGraph()
    load(detecting)

    for old, new in detecting.graph.conflicts:
        print(f"  conflict: {old.subject} {old.relation} {old.obj} (now invalid)  vs  {new.obj} (now valid)")
    print(f"  valid edges with main.GraphStore : {sorted(e.obj for e in baseline.graph.neighbors('ava'))}")
    print(f"  valid edges with the detector    : {sorted(e.obj for e in detecting.graph.neighbors('ava'))}")
    assert [(old.obj, new.obj) for old, new in detecting.graph.conflicts] == [("Berlin", "Lisbon")]
    assert sorted(e.obj for e in detecting.graph.neighbors("ava")) == ["Lisbon", "curriculum", "retrieval_bot"]
    assert "curriculum" not in [e.obj for e in baseline.graph.neighbors("ava")]     # main loses it


# ---------------------------------------------------------------------------
# Exercise 4 - a user_feedback term in the fusion score, hard to game
#
# The failure to prevent is a loop: the agent returns what was liked, it gets
# liked again, and it is all the agent returns. Four rules break the loop:
#   1. only the human's thumbs count; the agent cannot vote for its own picks;
#   2. one vote per user per record, so showing a record often adds nothing;
#   3. the weight is capped well below relevance, so feedback reorders
#      candidates of similar relevance and cannot rescue an irrelevant one;
#   4. unseen records score a neutral 0.5, not 0, so new facts are not starved.
# ---------------------------------------------------------------------------

class FeedbackMem0(Mem0):
    W_FEEDBACK = 0.15

    def __init__(self) -> None:
        super().__init__()
        self.votes: dict[str, dict[str, int]] = {}      # rid -> {user_id: +1 or -1}

    def feedback(self, rid: str, user_id: str, thumbs_up: bool, *, from_agent: bool = False) -> None:
        if not from_agent:
            self.votes.setdefault(rid, {})[user_id] = 1 if thumbs_up else -1

    def _feedback_score(self, rid: str) -> float:
        votes = self.votes.get(rid, {}).values()
        return (sum(v > 0 for v in votes) + 1) / (len(votes) + 2)

    def search(self, query: str, *, user_id: str, scope: str | None = None,
               top_k: int = 5) -> list[tuple[float, Record]]:
        base = super().search(query, user_id=user_id, scope=scope, top_k=top_k * 3)
        rescored = [(score + self.W_FEEDBACK * self._feedback_score(r.rid), r) for score, r in base]
        rescored.sort(key=lambda x: -x[0])
        return rescored[:top_k]


def ex4_feedback_fusion() -> None:
    mem = FeedbackMem0()
    liked = mem.add("ava prefers terse writing", user_id="ava")
    relevant = mem.add("ava lives in lisbon", user_id="ava")
    twin_a = mem.add("the deploy runbook lives in the wiki", user_id="ava")
    twin_b = mem.add("the deploy runbook lives in the repo", user_id="ava")

    for _ in range(50):
        mem.feedback(liked, "ava", thumbs_up=True)                      # 50 clicks, one vote
        mem.feedback(liked, "agent", thumbs_up=True, from_agent=True)   # ignored
    mem.feedback(twin_b, "ava", thumbs_up=True)

    top = mem.search("ava lives in which city", user_id="ava", top_k=1)[0][1]
    order = [r.rid for _, r in mem.search("where is the deploy runbook", user_id="ava", top_k=2)]
    print(f"  liked record, 50 clicks : feedback score {mem._feedback_score(liked):.2f} from {len(mem.votes[liked])} vote")
    print(f"  'ava lives in which city' -> {top.text!r}")
    print(f"  equally relevant twins   -> liked one first: {order == [twin_b, twin_a]}")
    assert mem.votes[liked] == {"ava": 1}
    assert top.rid == relevant                  # a liked but irrelevant record does not win
    assert order == [twin_b, twin_a]            # between equals, feedback decides


# ---------------------------------------------------------------------------
# Exercise 5 - port to the mem0 client, compare on the same 20 queries
#
# NOT RUN against mem0: the package is not installed here, and Memory() needs
# an OpenAI key for fact extraction and embeddings. Only the toy column was
# measured. Mem0Backend follows the calls in the mem0 open-source quickstart;
# with it installed and configured the second column fills in.
# ---------------------------------------------------------------------------

class ToyBackend:
    def __init__(self) -> None:
        self.memory = Mem0()

    def add(self, text: str, user_id: str) -> None:
        self.memory.add(text, user_id=user_id)

    def search(self, query: str, user_id: str, k: int = 3) -> list[str]:
        return [r.text for _, r in self.memory.search(query, user_id=user_id, top_k=k)]


class Mem0Backend:
    def __init__(self) -> None:
        from mem0 import Memory
        self.memory = Memory()

    def add(self, text: str, user_id: str) -> None:
        self.memory.add([{"role": "user", "content": text}], user_id=user_id)

    def search(self, query: str, user_id: str, k: int = 3) -> list[str]:
        results = self.memory.search(query, filters={"user_id": user_id})["results"]
        return [hit["memory"] for hit in results[:k]]


def hit_rate_at_3(backend: ToyBackend | Mem0Backend) -> float:
    for fact, _, _ in FACTS:
        backend.add(fact, "ava")
    hits = sum(any(keyword in text.lower() for text in backend.search(query, "ava"))
               for _, query, keyword in FACTS)
    return hits / len(FACTS)


def ex5_mem0_port() -> None:
    toy = hit_rate_at_3(ToyBackend())
    print(f"  toy Mem0 (token overlap) : hit@3 {toy:.2f} on {len(FACTS)} queries")
    try:
        print(f"  mem0 client              : hit@3 {hit_rate_at_3(Mem0Backend()):.2f}")
    except Exception as error:      # not installed, or no model key configured
        print(f"  mem0 client              : not run ({type(error).__name__}: {error})")
    assert 0 < toy < 1


if __name__ == "__main__":
    print("Phase 14 - Lesson 09: Hybrid Memory - exercises")
    for exercise in (ex1_embedding_recall, ex2_temporal_search, ex3_conflict_detector,
                     ex4_feedback_fusion, ex5_mem0_port):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall offline checks passed; exercise 5 compares against mem0 only where it is installed")
