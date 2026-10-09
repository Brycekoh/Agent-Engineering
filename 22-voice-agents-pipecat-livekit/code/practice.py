"""Phase 14 - Lesson 22: Voice Agents (Pipecat and LiveKit) - exercises.

Solves the five exercises from docs/en.md on top of main.py in this folder.
Run with:  python practice.py   (every exercise asserts its own result)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from main import LLM, STT, TTS, VAD, Frame, Processor, Transport, link

REPLIES = {
    "hello": "hi there, how can I help today?",
    "refund please": "sure, what order number should I look up?",
    "can you check my order status?": "your order shipped yesterday.",
}
# The lesson's typical 2026 latencies per stage, in milliseconds (low, high).
STAGE_MS = {"vad": (20, 60), "stt": (100, 250), "llm": (150, 400), "tts": (100, 200), "transport": (30, 80)}


def midpoint_s(stage: str) -> float:
    low, high = STAGE_MS[stage]
    return (low + high) / 2 / 1000


class Clock:
    """Virtual time. A stage's latency advances it, so nothing here sleeps and no result depends on the machine."""

    def __init__(self) -> None:
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Exercise 1 - a metrics observer: frames per stage, and where time goes
#
# The observer wraps each processor, counts the frames it sees and measures
# its own time (time inside the stage minus time spent in the stages after
# it). SIMULATED latencies: each stage advances a virtual clock by the
# midpoint of the lesson's range for it, so the shares are those ranges
# measured back. What the observer adds is the mechanism and the ranking: the
# LLM's first token and then STT are where a turn's latency accumulates. VAD
# and transport together are about a seventh.
# ---------------------------------------------------------------------------

class MetricsObserver:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.frames: Counter[str] = Counter()
        self.self_seconds: defaultdict[str, float] = defaultdict(float)
        self._children: list[float] = []

    def attach(self, *processors: Processor) -> None:
        for processor in processors:
            self._wrap(processor)

    def _wrap(self, processor: Processor) -> None:
        original = processor.process

        def observed(frame: Frame) -> None:
            self.frames[processor.name] += 1
            self._children.append(0.0)
            start = self.clock.now
            if processor.name in STAGE_MS and frame.direction == "downstream":
                self.clock.advance(midpoint_s(processor.name))      # the stage doing its work
            original(frame)
            elapsed = self.clock.now - start
            downstream = self._children.pop()
            self.self_seconds[processor.name] += elapsed - downstream
            if self._children:
                self._children[-1] += elapsed

        processor.process = observed            # type: ignore[method-assign]


def standard_pipeline() -> tuple[VAD, list[Processor], Transport]:
    stages: list[Processor] = [VAD("vad"), STT("stt"), LLM("llm", REPLIES), TTS("tts"), Transport("transport")]
    link(*stages)
    return stages[0], stages, stages[-1]            # type: ignore[return-value]


def ex1_metrics_observer() -> None:
    clock = Clock()
    source, stages, transport = standard_pipeline()
    observer = MetricsObserver(clock)
    observer.attach(*stages)
    for utterance in ("hello", "refund please", "hello", "refund please", "hello"):
        source.process(Frame("audio_chunk", utterance))
    total = sum(observer.self_seconds.values())
    for stage in stages:
        frames, seconds = observer.frames[stage.name], observer.self_seconds[stage.name]
        print(f"  {stage.name:<9} {frames} frames, {frames / clock.now:.2f} frames/s, {seconds / frames * 1000:>4.0f} ms each, "
              f"{seconds / total:>4.0%} of pipeline time")
    ranked = sorted(observer.self_seconds, key=observer.self_seconds.get, reverse=True)
    print(f"  five turns took {clock.now:.3f} s; latency accumulates in: {ranked[0]}, then {ranked[1]}")
    assert ranked == ["llm", "stt", "tts", "transport", "vad"] and len(transport.delivered) == 5
    assert all(observer.frames[stage.name] == 5 for stage in stages) and abs(total - clock.now) < 1e-9
    assert round((observer.self_seconds["vad"] + observer.self_seconds["transport"]) / total, 2) == 0.14


# ---------------------------------------------------------------------------
# Exercise 2 - confidence-gated STT
#
# A transcript below the threshold never reaches the LLM. The STT stage sends
# "could you repeat that?" down the pipeline as text instead, so TTS speaks it
# and the model is not asked to answer something that was not said.
# ---------------------------------------------------------------------------

class ConfidenceSTT(STT):
    def __init__(self, name: str, threshold: float = 0.6) -> None:
        super().__init__(name)
        self.threshold = threshold

    def process(self, frame: Frame) -> None:
        if frame.kind != "vad_speech":
            Processor.process(self, frame)
            return
        text, confidence = frame.payload["text"], frame.payload["confidence"]
        if confidence < self.threshold:
            self.trace.append(f"STT: {text!r} at {confidence:.2f} is below {self.threshold}; asking again")
            Processor.process(self, Frame("text", "could you repeat that?"))
        else:
            self.trace.append(f"STT: -> {text!r} ({confidence:.2f})")
            Processor.process(self, Frame("transcript", text))


def ex2_confidence_gate() -> None:
    vad, stt, llm, tts, transport = VAD("vad"), ConfidenceSTT("stt"), LLM("llm", REPLIES), TTS("tts"), Transport("transport")
    link(vad, stt, llm, tts, transport)
    vad.process(Frame("audio_chunk", {"text": "refund please", "confidence": 0.92}))
    vad.process(Frame("audio_chunk", {"text": "refun pleez", "confidence": 0.41}))
    for words in transport.delivered:
        print(f"  agent says: {' '.join(words)}")
    generated = [line for line in llm.trace if line.startswith("LLM:")]
    assert " ".join(transport.delivered[1]) == "could you repeat that?"
    assert len(generated) == 1 and "refun pleez" not in " ".join(llm.trace)     # the model never saw it


# ---------------------------------------------------------------------------
# Exercise 3 - semantic turn detection by rule
#
# The rule from the exercise: a transcript ending in "?" ends the turn. The
# detector sits between STT and the LLM, holds partial transcripts, and
# releases them as one turn. A rule this simple needs a fallback for turns
# that are not questions, so a silence frame also ends the turn.
# ---------------------------------------------------------------------------

class TurnDetector(Processor):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.partials: list[str] = []

    def process(self, frame: Frame) -> None:
        if frame.kind == "transcript":
            self.partials.append(str(frame.payload))
            if str(frame.payload).rstrip().endswith("?"):
                self._end_turn("question mark")
        elif frame.kind == "silence":
            if self.partials:
                self._end_turn("silence")
        else:
            super().process(frame)

    def _end_turn(self, why: str) -> None:
        text, self.partials = " ".join(self.partials), []
        self.trace.append(f"turn ended by {why}: {text!r}")
        super().process(Frame("transcript", text))


def ex3_turn_detection() -> None:
    chunks = ["can you check", "my order status?"]

    source, _, plain_transport = standard_pipeline()
    for chunk in chunks:
        source.process(Frame("audio_chunk", chunk))

    vad, stt, turns, llm, tts, transport = (VAD("vad"), STT("stt"), TurnDetector("turns"), LLM("llm", REPLIES),
                                            TTS("tts"), Transport("transport"))
    link(vad, stt, turns, llm, tts, transport)
    for chunk in chunks:
        vad.process(Frame("audio_chunk", chunk))
    vad.process(Frame("audio_chunk", "refund please"))
    vad.process(Frame("silence", None))

    print(f"  without a detector: {len(plain_transport.delivered)} replies, first is {' '.join(plain_transport.delivered[0])!r}")
    for line in turns.trace:
        if line.startswith("turn ended"):
            print(f"  {line}")
    print(f"  with the detector : {[' '.join(words) for words in transport.delivered]}")
    assert len(plain_transport.delivered) == 2                  # it answered half a sentence
    assert [" ".join(words) for words in transport.delivered] == [REPLIES["can you check my order status?"],
                                                                 REPLIES["refund please"]]


# ---------------------------------------------------------------------------
# Exercise 4 - swap the transport for a SmallWebRTCTransport config (stub)
#
# In Pipecat a transport is one object with two ends: transport.input() goes
# first in the pipeline and transport.output() goes last, and it is built as
# SmallWebRTCTransport(webrtc_connection=..., params=TransportParams(...)).
# The stub has that construction and that wiring; a list stands in for the
# peer connection. pipecat_transport() is the same configuration on the real
# class, which needs pipecat-ai[webrtc] and a SmallWebRTCConnection.
# ---------------------------------------------------------------------------

@dataclass
class TransportParams:
    audio_in_enabled: bool = True
    audio_out_enabled: bool = True


class SmallWebRTCTransportStub:
    def __init__(self, webrtc_connection: list[Any], params: TransportParams) -> None:
        self.connection = webrtc_connection
        self.params = params
        self._input = Processor("webrtc_in")
        self._output = Transport("webrtc_out")

    def input(self) -> Processor:
        return self._input

    def output(self) -> Transport:
        return self._output

    def receive(self, audio: Any) -> None:
        """Audio arriving from the peer."""
        if not self.params.audio_in_enabled:
            return
        already = len(self._output.delivered)
        self._input.process(Frame("audio_chunk", audio))
        if self.params.audio_out_enabled:
            self.connection.extend(self._output.delivered[already:])


def pipecat_transport(connection: Any) -> Any:
    from pipecat.transports.base_transport import TransportParams as PipecatParams
    try:
        from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
    except ImportError:
        from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
    return SmallWebRTCTransport(webrtc_connection=connection,
                                params=PipecatParams(audio_in_enabled=True, audio_out_enabled=True))


def ex4_webrtc_transport_stub() -> None:
    sent: dict[str, list[Any]] = {}
    for label, params in (("audio out enabled", TransportParams()), ("audio out disabled", TransportParams(audio_out_enabled=False))):
        peer: list[Any] = []
        transport = SmallWebRTCTransportStub(webrtc_connection=peer, params=params)
        link(transport.input(), VAD("vad"), STT("stt"), LLM("llm", REPLIES), TTS("tts"), transport.output())
        transport.receive("hello")
        sent[label] = peer
        print(f"  {label:<18}: peer received {[' '.join(words) for words in peer]}")
    assert [" ".join(words) for words in sent["audio out enabled"]] == [REPLIES["hello"]]
    assert sent["audio out disabled"] == []


# ---------------------------------------------------------------------------
# Exercise 5 - a speech-to-speech model against the STT + LLM + TTS cascade
#
# MODELLED, not measured against the two APIs: the stage latencies are the
# lesson's ranges, and the speech-to-speech model is assumed to produce first
# audio in about the time an LLM produces a first token. Under that model the
# cascade pays for two extra hops, STT and TTS: 200 to 450 ms per turn.
#
# That is the latency cost of text-level control, and exercises 2 and 3 are
# what it buys. Confidence gating, turn rules, guardrails and logged tool
# arguments all act on a transcript, and the speech-to-speech path never
# produces one.
# ---------------------------------------------------------------------------

class SpeechToSpeech(Processor):
    """One model hop: speech in, speech out, no text in between."""

    def __init__(self, name: str, clock: Clock) -> None:
        super().__init__(name)
        self.clock = clock

    def process(self, frame: Frame) -> None:
        if frame.kind == "vad_speech":
            self.clock.advance(midpoint_s("llm"))
            super().process(Frame("tts_audio", REPLIES[str(frame.payload)].split()))
        else:
            super().process(frame)


def ex5_cascade_vs_speech_to_speech() -> None:
    cascade_ms = [sum(STAGE_MS[s][i] for s in STAGE_MS) for i in (0, 1)]
    direct_ms = [sum(STAGE_MS[s][i] for s in ("vad", "llm", "transport")) for i in (0, 1)]
    print(f"  cascade           : {cascade_ms[0]}-{cascade_ms[1]} ms to first audio")
    print(f"  speech-to-speech  : {direct_ms[0]}-{direct_ms[1]} ms (assumed model hop)")
    print(f"  text-level control: +{cascade_ms[0] - direct_ms[0]}-{cascade_ms[1] - direct_ms[1]} ms per turn")

    cascade_clock, direct_clock = Clock(), Clock()
    source, stages, _ = standard_pipeline()
    MetricsObserver(cascade_clock).attach(*stages)
    source.process(Frame("audio_chunk", "hello"))

    vad, model, transport = VAD("vad"), SpeechToSpeech("speech_model", direct_clock), Transport("transport")
    link(vad, model, transport)
    MetricsObserver(direct_clock).attach(vad, transport)
    vad.process(Frame("audio_chunk", "hello"))
    cascade_toy, direct_toy = round(cascade_clock.now * 1000), round(direct_clock.now * 1000)
    print(f"  one turn through the toy, at the midpoints: cascade {cascade_toy} ms, speech-to-speech {direct_toy} ms")
    assert cascade_ms == [400, 990] and [c - d for c, d in zip(cascade_ms, direct_ms)] == [200, 450]
    assert (cascade_toy, direct_toy) == (695, 370) and transport.delivered == [REPLIES["hello"].split()]


if __name__ == "__main__":
    print("Phase 14 - Lesson 22: Voice Agents - exercises")
    for exercise in (ex1_metrics_observer, ex2_confidence_gate, ex3_turn_detection, ex4_webrtc_transport_stub,
                     ex5_cascade_vs_speech_to_speech):
        print(f"\n{exercise.__name__}")
        exercise()
    print("\nall exercises passed")
