"""Evidence gates, large requests, socket protocol checks."""
import asyncio
import json
import unittest
from unittest.mock import patch

from backend import app, VOCABULARY
from live import claimed_skills
from tests.support import api_request
from tests.test_live import request


class SummaryRegressionTests(unittest.TestCase):
    def summary(self, answers):
        return asyncio.run(api_request(dict(synthetic_profile=True, resume_skills=['python', 'sql'], answers=answers), '/interview/summary'))

    def test_empty_session_has_no_invented_strengths_or_grade(self):
        status, result = self.summary([])
        self.assertEqual(status, 200)
        self.assertIsNone(result['total_points'])
        self.assertEqual(result['reviewed_answers'], 0)
        self.assertEqual(result['strengths'], ['No reliable technical vocabulary observations yet.'])

    def test_unreliable_and_empty_answers_do_not_contribute_terms(self):
        for confidence, transcript in [(None, request()['transcript']), (.3, request()['transcript']), (.9, '')]:
            with self.subTest(confidence=confidence, transcript=transcript):
                status, result = self.summary([dict(question='Explain Python', transcript=transcript, speech_confidence=confidence)])
                self.assertEqual(status, 200)
                self.assertEqual(result['reviewed_answers'], 0)
                self.assertEqual(result['answer_evaluations'][0]['matched_terms'], [])
                self.assertTrue(all(s['score'] is None for s in result['skill_breakdowns']))

    def test_unasked_domains_stay_unassessed(self):
        _, result = self.summary([dict(question='Tell me about your hobbies', transcript=request()['transcript'], speech_confidence=.9)])
        self.assertEqual(result['reviewed_answers'], 0)
        self.assertTrue(all(s['status'] == 'NOT_ASKED' for s in result['skill_breakdowns']))
        _, result = self.summary([dict(question='Explain Python', transcript=request()['transcript']+' SQL joins and indexes', speech_confidence=.9)])
        rows = {s['skill']: s for s in result['skill_breakdowns']}
        self.assertEqual(rows['sql']['status'], 'NOT_ASKED')
        self.assertEqual(rows['python']['status'], 'VOCABULARY_PRESENT')

    def test_retention_and_body_limits_agree(self):
        answer = dict(question='Python '+('é'*4993), transcript='é'*30000, speech_confidence=.9)
        self.assertEqual(self.summary([answer]*10)[0], 200)
        self.assertEqual(self.summary([dict(question='Python',transcript='')]*11)[0], 422)
        self.assertEqual(asyncio.run(api_request(b'x'*2_200_001, '/interview/summary'))[0], 413)

    def test_java_script_alias_does_not_claim_java(self):
        self.assertEqual(claimed_skills('java script', VOCABULARY), ['javascript'])


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def exchange(self, origin, messages):
        queue = asyncio.Queue()
        await queue.put({'type': 'websocket.connect'})
        for message in messages:
            await queue.put(message)
        sent = []
        scope = dict(type='websocket', asgi={'version':'3.0'}, scheme='ws', path='/interview/stream',
                     raw_path=b'/interview/stream', query_string=b'', root_path='',
                     headers=[(b'host', b'test'), (b'origin', origin)], server=('test',80), client=('test',1), subprotocols=[])
        async def send(message):
            sent.append(message)
        await app(scope, queue.get, send)
        return sent

    async def test_cross_origin_socket_is_rejected(self):
        sent = await self.exchange(b'http://elsewhere', [])
        self.assertEqual(sent[0], {'type':'websocket.close', 'code':1008, 'reason':''})

    async def test_invalid_attestation_returns_safe_error(self):
        sent = await self.exchange(b'http://test', [{'type':'websocket.receive','text':json.dumps({'synthetic_profile':False})}])
        self.assertTrue(any('error' in json.loads(m['text']) for m in sent if m['type']=='websocket.send'))

    async def test_pcm_finish_and_invalid_frame(self):
        class Recognizer:
            def __init__(self, *args): pass
            def SetWords(self, value): pass
            def AcceptWaveform(self, data): return False
            def PartialResult(self): return json.dumps({'partial':'Python'})
            def FinalResult(self): return json.dumps({'text':'Python generators','result':[{'conf':.9}]})
        with patch('live.load_speech_model', return_value=object()), patch('vosk.KaldiRecognizer', Recognizer):
            for data, expected_error in [(b'\0\0'*160, False),(b'\0',True)]:
                sent = await self.exchange(b'http://test', [
                    {'type':'websocket.receive','text':'{"synthetic_profile":true}'},
                    {'type':'websocket.receive','bytes':data},
                    {'type':'websocket.receive','text':'finish'}])
                replies = [json.loads(m['text']) for m in sent if m['type']=='websocket.send']
                self.assertTrue(replies[0]['ready'])
                if expected_error:
                    self.assertIn('error', replies[-1])
                else:
                    self.assertTrue(replies[-1]['final'])
                    self.assertEqual(replies[-1]['text'], 'Python generators')
                    self.assertEqual(replies[-1]['audio_seconds'], .01)
