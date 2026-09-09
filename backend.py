"""P1 synthetic interview integrity demo — Python 3.10+.

Install: python -m pip install 'fastapi>=0.115,<1' 'pydantic>=2.9,<3' 'uvicorn>=0.30,<1'
Run:     python backend.py                 (127.0.0.1:8000)
Test:    python backend.py --test

Provisional API contract (design.md was not supplied):
POST /evaluate
{
  "synthetic_profile": true,
  "resume_skills": ["Python", "SQL"],
  "transcript": "I use generators to yield values and joins with indexes.",
  "metadata": {
    "gaze_diversion_ratio": 0.0,
    "synthetic_voice_confidence": 0.0,
    "lip_sync_variance_ms": 0.0,
    "network_latency_ms": 0.0,
    "network_jitter_ms": 0.0,
    "packet_loss_percent": 0.0
  }
}
Response: {"decision": {"risk_score": 0, "risk_tier": "LOW"},
           "reason": "Two or three explanatory sentences.",
           "next_action": "proceed normally"}

Only synthetic data is supported; synthetic_profile=true is a caller attestation,
not a way to verify provenance. No media is inspected, no model is trained, and
scores are deterministic demonstration heuristics, not calibrated probabilities
or evidence of deception. Skill vocabulary depends on interview coverage. Gaze
is ambiguous. Never use the score as an autonomous hiring decision. No payloads
are stored or logged. Bind privately; add authenticated TLS ingress before any
external deployment. Dependency installation is the only setup required.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import unittest
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints, model_validator

MAX_BODY_BYTES = 128_000
AADHAAR = re.compile(r"(?<!\w)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\w)")
PAN = re.compile(r"(?<!\w)[A-Z]{5}\d{4}[A-Z](?!\w)", re.IGNORECASE)


def sanitize(text: str) -> str:
    """Redact common contiguous, space-separated, or hyphenated ID formats."""
    return PAN.sub("[PAN Redacted]", AADHAAR.sub("[Aadhaar Redacted]", text))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


Unit = Annotated[float, Field(ge=0, le=1)]
Milliseconds = Annotated[float, Field(ge=0, le=60_000)]
Skill = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]


class Metadata(StrictModel):
    gaze_diversion_ratio: Unit
    synthetic_voice_confidence: Unit
    lip_sync_variance_ms: Milliseconds
    network_latency_ms: Milliseconds
    network_jitter_ms: Milliseconds
    packet_loss_percent: Annotated[float, Field(ge=0, le=100)]


class EvaluationRequest(StrictModel):
    synthetic_profile: StrictBool
    resume_skills: Annotated[list[Skill], Field(min_length=1, max_length=30)]
    transcript: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50_000)]
    metadata: Metadata

    @model_validator(mode="after")
    def synthetic_only(self) -> EvaluationRequest:
        if self.synthetic_profile is not True:
            raise ValueError("Only explicitly synthetic profiles are supported")
        return self


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Decision(StrictModel):
    risk_score: Annotated[int, Field(ge=0, le=100)]
    risk_tier: RiskTier


class EvaluationResponse(StrictModel):
    decision: Decision
    reason: str
    next_action: Literal["proceed normally", "ask targeted follow-up questions", "flag for human review"]


# Explicit, auditable vocabulary. Unknown skills use literal matching and are
# excluded from mismatch scoring because technical depth cannot be assessed.
VOCABULARY = {
    "python": ("generator", "yield", "decorator", "iterator", "comprehension", "async", "exception"),
    "sql": ("join", "index", "transaction", "query plan", "normalization", "group by", "foreign key"),
    "javascript": ("closure", "promise", "event loop", "prototype", "async", "callback"),
    "java": ("jvm", "generic", "interface", "garbage collection", "thread", "synchronization"),
    "machine learning": ("cross validation", "overfitting", "regularization", "gradient", "precision", "recall"),
    "fastapi": ("pydantic", "dependency injection", "asgi", "async", "validation", "middleware"),
    "docker": ("image", "container", "layer", "dockerfile", "volume", "multi stage"),
    "react": ("hook", "state", "effect", "component", "reconciliation", "render"),
}
ALIASES = {"js": "javascript", "postgresql": "sql", "mysql": "sql", "ml": "machine learning", "python3": "python"}


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def occurrences(text: str, term: str) -> int:
    # Word boundaries prevent e.g. 'index' matching 'indexedness'; simple plurals
    # support ordinary interview wording without stemming unrelated words.
    return len(re.findall(r"\b" + re.escape(term) + r"(?:s)?\b", text))


@dataclass(frozen=True)
class SemanticSignal:
    mismatch: float
    supported_skills: int
    total_skills: int
    sufficient_transcript: bool


def semantic_consistency(skills: list[str], transcript: str) -> SemanticSignal:
    text = normalize(transcript)
    word_count = len(text.split())
    canonical = sorted({ALIASES.get(normalize(s), normalize(s)) for s in skills})
    supported = [s for s in canonical if s in VOCABULARY]
    if word_count < 20 or not supported:
        return SemanticSignal(0.0, len(supported), len(canonical), word_count >= 20)
    consistencies = []
    for skill in supported:
        terms = VOCABULARY[skill]
        counts = [occurrences(text, t) for t in terms]
        breadth = min(sum(c > 0 for c in counts) / 3, 1.0)
        # Cap each term to limit repeated keyword stuffing. Technical breadth
        # dominates density; a bare skill name provides only weak support.
        density = min(sum(min(c, 2) for c in counts) / word_count / 0.04, 1.0)
        claimed_term = float(occurrences(text, skill) > 0)
        consistencies.append(0.7 * breadth + 0.2 * density + 0.1 * claimed_term)
    return SemanticSignal(1 - sum(consistencies) / len(consistencies), len(supported), len(canonical), True)


@dataclass(frozen=True)
class IntegritySignal:
    gaze: float
    voice: float
    lip_sync: float
    network_affected: bool


def analyze_integrity(metadata: Metadata) -> IntegritySignal:
    # Any reported network disturbance suppresses lip-sync evidence: without
    # synchronized capture clocks, raw timing variance cannot establish tampering.
    network = any((metadata.network_latency_ms > 0, metadata.network_jitter_ms > 0,
                   metadata.packet_loss_percent > 0))
    lip = 0.0 if network else min(max((metadata.lip_sync_variance_ms - 80) / 220, 0.0), 1.0)
    gaze = min(max((metadata.gaze_diversion_ratio - 0.4) / 0.6, 0.0), 1.0)
    return IntegritySignal(gaze, metadata.synthetic_voice_confidence, lip, network)


def evaluate_profile(payload: EvaluationRequest) -> EvaluationResponse:
    # Both free-text inputs are sanitized before either analyzer runs. Raw text
    # is never echoed, persisted, or passed to the scoring engine.
    skills = [sanitize(skill) for skill in payload.resume_skills]
    transcript = sanitize(payload.transcript)
    semantic = semantic_consistency(skills, transcript)
    integrity = analyze_integrity(payload.metadata)
    # Weak signals are limited to 30 points together; HIGH requires stronger
    # simulated integrity evidence. Network telemetry never adds points.
    score = int(25 * semantic.mismatch + 5 * integrity.gaze +
                40 * integrity.voice + 30 * integrity.lip_sync + 0.5)
    tier = RiskTier.HIGH if score >= 65 else RiskTier.MEDIUM if score >= 30 else RiskTier.LOW
    if not semantic.sufficient_transcript or not semantic.supported_skills:
        semantic_note = "Skill consistency is unassessed because transcript coverage or supported vocabulary is insufficient"
    else:
        semantic_note = (f"Technical vocabulary mismatch is {semantic.mismatch:.0%} across "
                         f"{semantic.supported_skills}/{semantic.total_skills} supported claimed skills")
    network_note = ("Network disturbance excludes lip-sync variance from risk" if integrity.network_affected
                    else "Lip-sync variance is assessed only because no network disturbance was reported")
    reason = (f"{semantic_note}. {network_note}; simulated voice evidence contributes "
              f"{40 * integrity.voice:.0f} points and lip-sync evidence {30 * integrity.lip_sync:.0f} points. "
              "This heuristic score is not proof of impersonation and requires human interpretation") + "."
    action = {RiskTier.LOW: "proceed normally", RiskTier.MEDIUM: "ask targeted follow-up questions",
              RiskTier.HIGH: "flag for human review"}[tier]
    if tier == RiskTier.LOW and (not semantic.sufficient_transcript or semantic.supported_skills < semantic.total_skills):
        action = "ask targeted follow-up questions"
    return EvaluationResponse(decision=Decision(risk_score=score, risk_tier=tier), reason=reason, next_action=action)


app = FastAPI(title="Synthetic Interview Integrity MVP", docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Pydantic errors normally contain raw rejected input; never echo it.
    return JSONResponse(status_code=422, content={"detail": "Invalid request; check required fields, types, bounds, and synthetic_profile=true"})


class BodyLimitMiddleware:
    """Bound the actual body, including chunked requests, before JSON parsing."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            limit = 2_900_000 if scope.get("path") == "/resume" else 400_000 if scope.get("path") == "/analyze/live" else MAX_BODY_BYTES
            if len(body) > limit:
                response = JSONResponse(status_code=413, content={"detail": "Request body too large"})
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def buffered_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, buffered_receive, send)


app.add_middleware(BodyLimitMiddleware)

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    return evaluate_profile(payload)


def demo_payload(case: str) -> dict:
    data = {
        "synthetic_profile": True,
        "resume_skills": ["Python", "SQL"],
        "transcript": ("In Python I use generators to yield values with an iterator and handle exceptions. "
                       "In SQL I inspect the query plan, add an index, and use joins within a transaction."),
        "metadata": {"gaze_diversion_ratio": 0.1, "synthetic_voice_confidence": 0.0,
                     "lip_sync_variance_ms": 20.0, "network_latency_ms": 0.0,
                     "network_jitter_ms": 0.0, "packet_loss_percent": 0.0},
    }
    if case == "edge":
        data["metadata"].update(network_latency_ms=500.0, network_jitter_ms=200.0,
                                packet_loss_percent=15.0, lip_sync_variance_ms=1500.0)
    elif case == "high":
        data["transcript"] = ("I usually coordinate meetings and discuss schedules with the team. "
                              "I cannot explain the technical implementation or describe how any of these systems work in practice.")
        data["metadata"].update(synthetic_voice_confidence=0.95, lip_sync_variance_ms=400.0,
                                gaze_diversion_ratio=0.9)
    return data


async def api_request(payload: dict | bytes, path: str = "/evaluate", method: str = "POST") -> tuple[int, dict]:
    """Exercise the full ASGI app without a third-party HTTP test client."""
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    messages = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)

    await app({"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
               "method": method, "scheme": "http", "path": path, "raw_path": path.encode(),
               "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json")],
               "server": ("test", 80), "client": ("test", 123)}, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    return status, json.loads(content)


class BackendTests(unittest.TestCase):
    def test_three_demo_cases(self):
        scores = {}
        for case, tier in (("normal", "LOW"), ("edge", "LOW"), ("high", "HIGH")):
            status, result = asyncio.run(api_request(demo_payload(case)))
            self.assertEqual(status, 200)
            self.assertEqual(set(result), {"decision", "reason", "next_action"})
            self.assertEqual(result["decision"]["risk_tier"], tier)
            self.assertIs(type(result["decision"]["risk_score"]), int)
            self.assertEqual(len(re.findall(r"\.(?: |$)", result["reason"])), 3)
            scores[case] = result["decision"]["risk_score"]
            print(f"\n{case}: {json.dumps(result)}")
        self.assertEqual(scores["normal"], scores["edge"])

    def test_network_suppression_and_independent_voice(self):
        for field in ("network_latency_ms", "network_jitter_ms", "packet_loss_percent"):
            data = demo_payload("normal")
            data["metadata"].update(lip_sync_variance_ms=60_000.0)
            data["metadata"][field] = 1.0
            _, result = asyncio.run(api_request(data))
            self.assertEqual(result["decision"]["risk_score"], 0)
            data["metadata"]["synthetic_voice_confidence"] = 1.0
            _, result = asyncio.run(api_request(data))
            self.assertEqual(result["decision"]["risk_score"], 40)

    def test_redaction_before_analysis(self):
        from unittest.mock import patch
        data = demo_payload("normal")
        secret = "1234 5678 9012 ABCDE1234F"
        data["transcript"] += " " + secret
        data["resume_skills"].append(secret)
        with patch(__name__ + ".semantic_consistency", wraps=semantic_consistency) as analyzer:
            status, response = asyncio.run(api_request(data))
            self.assertEqual(status, 200)
            args = analyzer.call_args.args
            self.assertNotIn(secret, str(args))
            self.assertIn("[Aadhaar Redacted] [PAN Redacted]", args[1])
            self.assertNotIn(secret, json.dumps(response))
        for number in ("123456789012", "1234-5678-9012", "1234 5678 9012"):
            self.assertEqual(sanitize(number), "[Aadhaar Redacted]")
        self.assertEqual(sanitize("abcde1234f"), "[PAN Redacted]")

    def test_invalid_requests_do_not_echo_inputs(self):
        for field, value in (("synthetic_profile", False), ("synthetic_profile", "true"),
                             ("transcript", " "), ("resume_skills", []), ("extra", "ABCDE1234F")):
            data = demo_payload("normal")
            data[field] = value
            status, response = asyncio.run(api_request(data))
            self.assertEqual(status, 422)
            self.assertNotIn("ABCDE1234F", json.dumps(response))
        for value in (-1, 1.1, "0.5", float("nan"), float("inf")):
            data = demo_payload("normal")
            data["metadata"]["synthetic_voice_confidence"] = value
            self.assertEqual(asyncio.run(api_request(data))[0], 422)
        self.assertEqual(asyncio.run(api_request(b'{invalid ABCDE1234F'))[0], 422)
        self.assertEqual(asyncio.run(api_request(b"x" * (MAX_BODY_BYTES + 1)))[0], 413)
        self.assertEqual(asyncio.run(api_request({}, method="GET"))[0], 405)
        self.assertEqual(asyncio.run(api_request({}, path="/docs"))[0], 404)

    def test_insufficient_evidence(self):
        for transcript, skills in (("Brief answer", ["Python"]), (demo_payload("normal")["transcript"], ["Unknown skill"])):
            data = demo_payload("normal")
            data.update(transcript=transcript, resume_skills=skills)
            _, result = asyncio.run(api_request(data))
            self.assertEqual(result["decision"]["risk_score"], 0)
            self.assertEqual(result["next_action"], "ask targeted follow-up questions")


from live import register_live
register_live(app, sanitize, VOCABULARY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Run synthetic demos and regression tests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["backend.py"], verbosity=2)
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
