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
import re
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
# Phone: Indian mobile (+91/0 prefix optional) and international formats
PHONE_IN = re.compile(r"(?<!\w)(?:\+91[\s.-]?|0)?[6-9]\d{4}[\s.-]?\d{5}(?!\w)")
PHONE_INTL = re.compile(r"(?<!\w)\+\d{1,3}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}[\s.-]?\d{3,4}(?!\w)")
EMAIL = re.compile(r"(?<!\w)[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}(?!\w)")
# 16-digit card numbers (spaces or dashes between groups)
CARD = re.compile(r"(?<!\w)\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}(?!\w)")
# UPI IDs: alphanumeric@bankcode
UPI = re.compile(r"(?<!\w)[A-Za-z0-9._%+\-]+@[a-z]{2,}(?!\w)")
# Safety net: any remaining long digit sequences (7+ digits, with optional separators)
LONG_DIGITS = re.compile(r"(?<!\w)\d(?:[\s.-]?\d){6,}(?!\w)")


def sanitize(text: str) -> str:
    """Redact PII: Aadhaar, PAN, phone, email, card numbers, UPI IDs, long digit sequences."""
    text = EMAIL.sub("[Email Redacted]", text)
    text = UPI.sub("[UPI Redacted]", text)
    text = CARD.sub("[Card Redacted]", text)
    text = AADHAAR.sub("[Aadhaar Redacted]", text)
    text = PAN.sub("[PAN Redacted]", text)
    text = PHONE_IN.sub("[Phone Redacted]", text)
    text = PHONE_INTL.sub("[Phone Redacted]", text)
    text = LONG_DIGITS.sub("[ID Redacted]", text)
    return text


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
            limits = {"/resume": 2_900_000, "/interview/summary": 2_200_000, "/analyze/live": 400_000}
            limit = limits.get(scope.get("path"), MAX_BODY_BYTES)
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




from live import register_live
register_live(app, sanitize, VOCABULARY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Run synthetic demos and regression tests")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.test:
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"), top_level_dir=str(Path(__file__).parent))
        raise SystemExit(not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful())
    else:
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)
