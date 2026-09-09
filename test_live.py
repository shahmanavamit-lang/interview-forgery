"""Regression tests for the automatic workflow; no camera or microphone needed."""
import asyncio
import base64
import io
import math
import unittest
import zipfile
from unittest.mock import patch

from backend import api_request, sanitize, VOCABULARY
from live import LiveInput, Sample, analyze_answer, analyze_capture, extract_document


def request(**changes):
    data = dict(synthetic_profile=True, resume_skills=['python', 'sql'],
                question='Explain Python generators and how you use them.',
                transcript='In Python a generator uses yield to return one value at a time through an iterator without allocating the entire sequence in memory.',
                final=True, speech_confidence=.9, samples=[], capture_interrupted=False)
    data.update(changes)
    return data


class LiveWorkflowTests(unittest.TestCase):
    def test_resume_upload_redacts_and_extracts(self):
        raw = b'Synthetic demo: Python, SQL, FastAPI. Aadhaar 1234 5678 9012 PAN ABCDE1234F'
        status, result = asyncio.run(api_request(dict(synthetic_profile=True,filename='mock.txt',content_base64=base64.b64encode(raw).decode()), '/resume'))
        self.assertEqual(status, 200)
        self.assertEqual(result['skills'], ['python', 'sql', 'fastapi'])
        self.assertNotIn('ABCDE1234F', result['redacted_text'])
        self.assertIn('[Aadhaar Redacted]', result['redacted_text'])

    def test_document_formats_and_invalid_data(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('word/document.xml', '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:r><w:t>Python and SQL</w:t></w:r></w:p></w:document>')
        self.assertEqual(extract_document(archive.getvalue(), '.docx'), 'Python and SQL')
        for content in ('%%%invalid', base64.b64encode(b'not a PDF').decode()):
            status, _ = asyncio.run(api_request(dict(synthetic_profile=True,filename='mock.pdf',content_base64=content), '/resume'))
            self.assertEqual(status, 422)
        status, _ = asyncio.run(api_request(dict(synthetic_profile=False,filename='mock.txt',content_base64='UHl0aG9u'), '/resume'))
        self.assertEqual(status, 422)

    def test_only_question_skill_is_assessed(self):
        status, result = asyncio.run(api_request(request(), '/analyze/live'))
        self.assertEqual(status, 200)
        rows = {r['skill']: r for r in result['skill_assessments']}
        self.assertEqual(rows['python']['status'], 'VOCABULARY_PRESENT')
        self.assertEqual(rows['sql']['status'], 'NOT_ASKED')
        self.assertIsNone(result['decision']['risk_score'])
        self.assertEqual(result['detectors']['deepfake'], 'UNAVAILABLE')

    def test_weak_answer_and_confidence_gates(self):
        weak = 'I usually coordinate meetings and discuss schedules with the team but cannot explain how this system works in practice.'
        for changes, expected in [({'transcript':weak}, 'FOLLOW_UP'),
                                  ({'transcript':weak, 'final':False}, 'LISTENING'),
                                  ({'transcript':weak, 'speech_confidence':.3}, 'CHECK_TRANSCRIPT'),
                                  ({'transcript':'I do not know'}, 'INSUFFICIENT_ANSWER')]:
            _, result = asyncio.run(api_request(request(**changes), '/analyze/live'))
            self.assertEqual(result['skill_assessments'][0]['status'], expected)
        _, result = asyncio.run(api_request(request(question='Tell me about your hobbies'), '/analyze/live'))
        self.assertTrue(all(s['status'] == 'NOT_ASKED' for s in result['skill_assessments']))

    def test_redaction_happens_before_skill_analysis(self):
        payload = LiveInput(**request(question='Python ABCDE1234F', transcript='1234 5678 9012'))
        with patch('live.words', wraps=__import__('live').words) as normalizer:
            analyze_answer(payload, sanitize, VOCABULARY)
            inputs = ' '.join(c.args[0] for c in normalizer.call_args_list)
            self.assertNotIn('ABCDE1234F', inputs)
            self.assertNotIn('1234 5678 9012', inputs)

    def test_timing_observation_and_quality_exclusion(self):
        samples = [Sample(t=float(i*100),faces=1,blink=0.,mouth=.4+.3*math.sin((i-3)*.6),
                          rms=.05+.04*math.sin(i*.6),audio_fresh=True) for i in range(160)]
        result = analyze_capture(samples, False)
        self.assertEqual(result['lip_sync_status'], 'TIMING_REVIEW')
        self.assertEqual(result['estimated_offset_ms'], 300)
        self.assertEqual(analyze_capture(samples, True)['lip_sync_status'], 'EXCLUDED_CAPTURE_QUALITY')
        samples[40].audio_fresh = False
        self.assertEqual(analyze_capture(samples, False)['lip_sync_status'], 'EXCLUDED_CAPTURE_QUALITY')
        samples[40].audio_fresh = True
        for sample in samples[40:]: sample.t += 500
        self.assertEqual(analyze_capture(samples, False)['lip_sync_status'], 'EXCLUDED_CAPTURE_QUALITY')

    def test_blinks_are_observations_only(self):
        samples = [Sample(t=float(i*100),faces=1,blink=.8 if i % 10 in (2,3) else 0.,mouth=0.,rms=0.,audio_fresh=True) for i in range(310)]
        result = analyze_capture(samples, False)
        self.assertEqual(result['blink_count'], 31)
        self.assertIsNotNone(result['blink_rate_per_minute'])
        _, data = asyncio.run(api_request(request(samples=[s.model_dump() for s in samples]), '/analyze/live'))
        self.assertIsNone(data['decision']['risk_score'])
        self.assertEqual(data['capture']['lip_sync_status'], 'INSUFFICIENT_ACTIVITY')

    def test_validation_and_no_media(self):
        for changes in ({'synthetic_profile':False}, {'speech_confidence':1.5}, {'transcript':1}, {'samples':[dict(t=1.,faces=1,blink=0.,mouth=0.,rms=0.,audio_fresh=True)]*2}):
            self.assertEqual(asyncio.run(api_request(request(**changes), '/analyze/live'))[0], 422)
        self.assertEqual(analyze_capture([], False)['status'], 'INSUFFICIENT_CAPTURE')


if __name__ == '__main__':
    unittest.main(verbosity=2)
