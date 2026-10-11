"""In-process API client and synthetic fixtures."""
import json
from backend import app

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
