import asyncio
import json
import re
import unittest

from backend import MAX_BODY_BYTES, sanitize, semantic_consistency
from tests.support import api_request, demo_payload

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
        with patch("backend.semantic_consistency", wraps=semantic_consistency) as analyzer:
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
