"""Phase 14 - Lesson 24: Agent Observability (Langfuse, Phoenix, Opik) - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import urllib.request
from collections import Counter
from typing import Any

from main import SpanEvent, TraceCollector, scripted_llm_judge, summarize


def week_of_traces(seed: int = 0, per_day: int = 30, new_prompt_from_day: int | None = None,
                   upstream_outage_from_day: int | None = None) -> list[SpanEvent]:
    """SYNTHETIC: seven days of support-agent sessions with a baseline failure rate."""
    rng = random.Random(seed)
    spans: list[SpanEvent] = []
    for day in range(7):
        version = "v8" if new_prompt_from_day is not None and day >= new_prompt_from_day else "v7"
        outage = upstream_outage_from_day is not None and day >= upstream_outage_from_day
        for n in range(per_day):
            sid = f"d{day}-s{n:02d}"
            common = {"day": day, "prompt.version": version, "session.id": sid}
            spans.append(SpanEvent(f"t-{sid}", sid, "invoke_agent", attributes={**common, "gen_ai.provider.name": "anthropic"}))
            roll = rng.random()
            if roll < 0.05:
                spans.append(SpanEvent(f"t-{sid}", sid, "chat", "error", {**common, "error.reason": "rate_limited"}))
            elif roll < 0.05 + (0.25 if version == "v8" else 0.03):
                spans.append(SpanEvent(f"t-{sid}", sid, "tool_call search_tool", "error", {**common, "error.reason": "tool_misuse"}))
            elif outage and roll < 0.45:
                spans.append(SpanEvent(f"t-{sid}", sid, "tool_call search_tool", "error", {**common, "error.reason": "timeout"}))
            else:
                spans.append(SpanEvent(f"t-{sid}", sid, "tool_call search_tool", attributes=common))
                spans.append(SpanEvent(f"t-{sid}", sid, "chat", attributes={**common, "gen_ai.output.reference_id": f"c-{sid}", "tokens": 700}))
    return spans


# ---------------------------------------------------------------------------
# Exercise 1 - a week of traces: which sessions failed, and why
#
# The week is synthetic. The question is answered locally with main's
# collector and judge. langfuse_request() builds the real export: Langfuse
# takes OTLP over HTTP at /api/public/otel/v1/traces with Basic auth made from
# the project's public and secret key, and groups spans by the session.id
# attribute. It is sent when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are
# set.
# ---------------------------------------------------------------------------

def to_otlp(spans: list[SpanEvent]) -> dict[str, Any]:
    records = []
    for i, span in enumerate(spans):
        records.append({
            "traceId": hashlib.md5(span.trace_id.encode()).hexdigest(),
            "spanId": f"{i + 1:016x}", "name": span.name,
            "status": {"code": 2 if span.status == "error" else 1},
            "attributes": [{"key": key, "value": {"stringValue": str(value)}} for key, value in span.attributes.items()],
        })
    return {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "support-agent"}}]},
                               "scopeSpans": [{"scope": {"name": "lesson24.practice"}, "spans": records}]}]}


def langfuse_request(spans: list[SpanEvent], public_key: str, secret_key: str,
                     host: str = "https://cloud.langfuse.com") -> urllib.request.Request:
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return urllib.request.Request(f"{host}/api/public/otel/v1/traces", json.dumps(to_otlp(spans)).encode(),
                                  {"Content-Type": "application/json", "Authorization": f"Basic {auth}"})


def ex1_week_of_traces() -> None:
    spans = week_of_traces()
    collector = TraceCollector()
    for span in spans:
        collector.ingest(span)
    failed = [s for s in summarize(collector) if scripted_llm_judge(collector.by_session()[s.session_id])[1] == "FAIL"]
    reasons = Counter(reason for summary in failed for reason in summary.failure_reasons.elements())
    by_day = Counter(summary.session_id[:2] for summary in failed)
    print(f"  {len(collector.by_session())} sessions, {len(failed)} failed: {dict(reasons.most_common())}")
    print(f"  failures per day: {dict(sorted(by_day.items()))}")
    print(f"  first failed sessions: {sorted(s.session_id for s in failed)[:5]}")
    assert len(collector.by_session()) == 210 and 0 < len(failed) < 40
    assert set(reasons) == {"rate_limited", "tool_misuse"}

    request = langfuse_request(spans, os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-example"),
                               os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-example"))
    assert request.full_url.endswith("/api/public/otel/v1/traces") and request.get_header("Authorization").startswith("Basic ")
    if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"  exported {len(spans)} spans to Langfuse: HTTP {response.status}")
    else:
        print(f"  Langfuse export ready ({len(spans)} spans); set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to send")


# ---------------------------------------------------------------------------
# Exercise 2 - a judge rubric for support answers, tried on 50 traces
#
# SCRIPTED JUDGE: each rubric line is a rule here; the text of the rubric is
# what an LLM judge would be given. The 50 sessions are generated with known
# defects, so the judge itself can be scored.
#
# The result worth keeping: the judge is exact on the two dimensions it can
# ground (figures against tool results, tools against an allowlist) and misses
# part of the third. Tone has nothing to be checked against, so a rule list,
# like an ungrounded model, lets sarcasm through.
# ---------------------------------------------------------------------------

RUBRIC = {
    "factual": "Every figure in the answer appears in a tool result from the same session.",
    "tone": "No blame or dismissal of the customer.",
    "scope": "Only support tools are used, and no legal or medical advice is given.",
}
SUPPORT_TOOLS = {"lookup_invoice", "issue_refund", "search_docs"}
RUDE = ("as i already said", "you should have", "calm down")


def judge(session: dict[str, Any]) -> dict[str, bool]:
    answer = session["answer"].lower()
    grounded = set(re.findall(r"\d+(?:\.\d+)?", " ".join(session["tool_results"])))
    return {
        "factual": all(figure in grounded for figure in re.findall(r"\d+(?:\.\d+)?", answer)),
        "tone": not any(phrase in answer for phrase in RUDE),
        "scope": set(session["tools"]) <= SUPPORT_TOOLS and "legal" not in answer and "lawsuit" not in answer,
    }


def fifty_sessions(seed: int = 1) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sessions = []
    for i in range(50):
        amount = f"{rng.randint(10, 200)}.00"
        session = {"id": f"s{i:02d}", "tools": ["lookup_invoice", "issue_refund"],
                   "tool_results": [f"invoice {4000 + i}: charged {amount} twice"],
                   "answer": f"Sorry about that. I have refunded {amount} for invoice {4000 + i}.",
                   "truth": {"factual": True, "tone": True, "scope": True}}
        defect = rng.choice(["none"] * 5 + ["wrong figure", "rude", "sarcastic", "legal advice", "wrong tool"])
        if defect == "wrong figure":
            session["answer"] = session["answer"].replace(amount, "999.00")
            session["truth"]["factual"] = False
        elif defect == "rude":
            session["answer"] = "As I already said, " + session["answer"]
            session["truth"]["tone"] = False
        elif defect == "sarcastic":
            session["answer"] = "Great job reading your own invoice. " + session["answer"]
            session["truth"]["tone"] = False
        elif defect == "legal advice":
            session["answer"] += " You could also file a lawsuit against your bank."
            session["truth"]["scope"] = False
        elif defect == "wrong tool":
            session["tools"] = session["tools"] + ["delete_account"]
            session["truth"]["scope"] = False
        sessions.append(session)
    return sessions


def ex2_judge_rubric() -> None:
    sessions = fifty_sessions()
    verdicts = [judge(session) for session in sessions]
    passed = sum(all(verdict.values()) for verdict in verdicts)
    print(f"  judged {len(sessions)} sessions: {passed} pass all three lines")
    agreement = {}
    for dimension, text in RUBRIC.items():
        agreement[dimension] = sum(v[dimension] == s["truth"][dimension] for v, s in zip(verdicts, sessions)) / len(sessions)
        failing = sum(not v[dimension] for v in verdicts)
        print(f"  {dimension:<8} flagged {failing:>2}, agrees with the known defects {agreement[dimension]:.0%}  ({text})")
    assert agreement["factual"] == 1.0 and agreement["scope"] == 1.0
    assert 0.8 < agreement["tone"] < 1.0                # the sarcastic answers get through


# ---------------------------------------------------------------------------
# Exercise 3 - prompt versioning against trace clustering
#
# Two synthetic incidents, each looked at both ways. The Langfuse-style view
# is the failure rate grouped by the prompt version on each trace. The
# Phoenix-style view groups failing traces by what they look like and
# compares before with after.
#
# Which tells you what broke faster depends on what broke. When the cause is
# a prompt change, the version view names it in one query; clustering shows
# what the failures look like but not which change caused them. When nothing
# you versioned changed (an upstream outage), the version view is flat and
# clustering is the only one with a signal.
# ---------------------------------------------------------------------------

def failed_sessions(spans: list[SpanEvent]) -> dict[str, SpanEvent]:
    return {span.session_id: span for span in spans if span.status == "error"}


def failure_rate_by_version(spans: list[SpanEvent]) -> dict[str, float]:
    sessions = {span.session_id: span.attributes["prompt.version"] for span in spans}
    failed = failed_sessions(spans)
    totals, bad = Counter(sessions.values()), Counter(sessions[sid] for sid in failed)
    return {version: bad[version] / totals[version] for version in sorted(totals)}


def cluster_growth(spans: list[SpanEvent], split_day: int) -> dict[str, tuple[int, int]]:
    """Failing sessions grouped by signature, counted before and after a day."""
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    for span in failed_sessions(spans).values():
        signature = f"{span.attributes['error.reason']} in {span.name}"
        (after if span.attributes["day"] >= split_day else before)[signature] += 1
    return {signature: (before[signature], after[signature]) for signature in sorted(set(before) | set(after))}


def ex3_versioning_vs_clustering() -> None:
    incidents = {
        "prompt v8 ships on day 4": week_of_traces(new_prompt_from_day=4),
        "upstream search times out from day 4": week_of_traces(upstream_outage_from_day=4),
    }
    views = {}
    for label, spans in incidents.items():
        views[label] = (failure_rate_by_version(spans), cluster_growth(spans, split_day=4))
        print(f"  incident: {label}")
        print(f"    by prompt version : {{{', '.join(f'{v}: {rate:.0%}' for v, rate in views[label][0].items())}}}")
        print(f"    clusters (days 0-3, days 4-6): {views[label][1]}")
    prompt_rates, prompt_clusters = views["prompt v8 ships on day 4"]
    outage_rates, outage_clusters = views["upstream search times out from day 4"]
    assert prompt_rates["v8"] > 2 * prompt_rates["v7"]                      # the version view names the cause
    assert list(outage_rates) == ["v7"]                                     # ... and has nothing to compare here
    assert outage_clusters["timeout in tool_call search_tool"][0] == 0      # a cluster that did not exist before
    assert prompt_clusters["tool_misuse in tool_call search_tool"][1] > prompt_clusters["tool_misuse in tool_call search_tool"][0]


# ---------------------------------------------------------------------------
# Exercise 4 - a PII redaction guardrail on an agent run
#
# Opik's guardrails are validators: Guardrail(guards=[PII(blocked_entities=
# [...])]).validate(text) raises GuardrailValidationFailed, and the result is
# attached to the trace. They need the opik package and its guardrails
# backend. The exercise asks for redaction, so the guard here rewrites
# instead of raising: it runs on every span as it is ingested, replaces what
# it finds, and records what it did on the span, the way Opik records a
# guardrail result. Card numbers are checked with the Luhn sum so that an
# order number is not mistaken for one.
# ---------------------------------------------------------------------------

def luhn_valid(digits: str) -> bool:
    total = 0
    for i, digit in enumerate(reversed(digits)):
        value = int(digit) * (2 if i % 2 else 1)
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def redact(text: str) -> tuple[str, Counter]:
    found: Counter[str] = Counter()

    def card(match: re.Match) -> str:
        if not luhn_valid(re.sub(r"\D", "", match.group())):
            return match.group()
        found["CREDIT_CARD"] += 1
        return "[CREDIT_CARD]"

    def count(kind: str) -> Any:
        def replace(match: re.Match) -> str:
            found[kind] += 1
            return f"[{kind}]"
        return replace

    text = re.sub(r"\b(?:\d[ -]?){15}\d\b", card, text)
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", count("EMAIL_ADDRESS"), text)
    text = re.sub(r"\+\d{1,3}[ -]?\d{2,4}[ -]?\d{3,4}[ -]?\d{3,4}\b", count("PHONE_NUMBER"), text)
    return text, found


class RedactingCollector(TraceCollector):
    def ingest(self, span: SpanEvent) -> None:
        redactions: Counter[str] = Counter()
        for key, value in list(span.attributes.items()):
            if isinstance(value, str):
                span.attributes[key], found = redact(value)
                redactions += found
        if redactions:
            span.attributes["guardrail.pii.redacted"] = dict(redactions)
        super().ingest(span)


def ex4_pii_guardrail() -> None:
    collector = RedactingCollector()
    collector.ingest(SpanEvent("t1", "s1", "chat", attributes={
        "input": "I am ava@example.com, call me on +351 912 345 678 about order 4711",
        "output": "charged card 4111 1111 1111 1111; reference 1234 5678 9012 3456"}))
    stored = collector.spans[0].attributes
    for key, value in stored.items():
        print(f"  {key:<22}: {value}")
    assert "ava@example.com" not in str(stored) and "4111" not in str(stored) and "912 345 678" not in str(stored)
    assert "order 4711" in stored["input"] and "1234 5678 9012 3456" in stored["output"]       # not PII, left alone
    assert stored["guardrail.pii.redacted"] == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1, "CREDIT_CARD": 1}


# ---------------------------------------------------------------------------
# Exercise 5 - benchmark on your own corpus, not on vendor numbers
#
# LOCAL BACKENDS: the two timed here run in this process, an in-memory
# collector and a SQLite one. The harness takes anything with ingest() and
# by_session(), which is the seam where a Langfuse, Phoenix or Opik client
# goes; their numbers depend on your network and plan, which is the reason to
# measure them yourself.
# ---------------------------------------------------------------------------

class SqliteCollector:
    def __init__(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.execute("CREATE TABLE spans (session_id TEXT, body TEXT)")

    def ingest(self, span: SpanEvent) -> None:
        self.db.execute("INSERT INTO spans VALUES (?, ?)", (span.session_id, json.dumps(vars(span))))

    def by_session(self) -> dict[str, list[SpanEvent]]:
        sessions: dict[str, list[SpanEvent]] = {}
        for session_id, body in self.db.execute("SELECT session_id, body FROM spans"):
            sessions.setdefault(session_id, []).append(SpanEvent(**json.loads(body)))
        return sessions


def benchmark(backend: Any, corpus: list[SpanEvent]) -> dict[str, float]:
    start = time.perf_counter()
    for span in corpus:
        backend.ingest(span)
    ingested = time.perf_counter()
    verdicts = Counter(scripted_llm_judge(spans)[1] for spans in backend.by_session().values())
    done = time.perf_counter()
    return {"ingest_ms": (ingested - start) * 1000, "eval_ms": (done - ingested) * 1000,
            "spans_per_s": len(corpus) / (done - start), "failed": verdicts["FAIL"]}


def ex5_benchmark_backends() -> None:
    corpus = week_of_traces(per_day=300)
    results = {name: benchmark(backend, corpus) for name, backend in
               (("in-memory collector", TraceCollector()), ("sqlite collector", SqliteCollector()))}
    for name, r in results.items():
        print(f"  {name:<20} ingest {r['ingest_ms']:>7.1f} ms   eval {r['eval_ms']:>7.1f} ms   "
              f"{r['spans_per_s']:>9,.0f} spans/s   {r['failed']} sessions failed")
    print(f"  corpus: {len(corpus)} spans")
    assert results["in-memory collector"]["failed"] == results["sqlite collector"]["failed"] > 0
    assert all(r["spans_per_s"] > 0 for r in results.values())


if __name__ == "__main__":
    print("Phase 14 - Lesson 24: Agent Observability - exercises")
    for exercise in (ex1_week_of_traces, ex2_judge_rubric, ex3_versioning_vs_clustering, ex4_pii_guardrail,
                     ex5_benchmark_backends):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
