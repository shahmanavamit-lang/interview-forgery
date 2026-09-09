"""Local mock-interview capture endpoints. Raw media and documents are not saved."""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import logging
import math
import re
import threading
import time
import zipfile
from pathlib import Path
from statistics import mean
from typing import Annotated, Literal
from xml.etree import ElementTree

from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / 'models' / 'vosk-model-small-en-us-0.15'
_model = None
_model_lock = threading.Lock()
# Document parser diagnostics may include rejected document bytes. Keep those
# diagnostics out of request logs; the endpoint returns a generic safe error.
logging.getLogger('pypdf').addHandler(logging.NullHandler())
logging.getLogger('pypdf').propagate = False


class Input(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    synthetic_profile: StrictBool

    @model_validator(mode='after')
    def require_synthetic(self):
        if not self.synthetic_profile:
            raise ValueError('Synthetic profiles only')
        return self


class ResumeInput(Input):
    filename: Annotated[str, Field(min_length=1, max_length=200)]
    content_base64: Annotated[str, Field(min_length=1, max_length=2_800_000)]


class Sample(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    t: Annotated[float, Field(ge=0, le=190_000)]
    faces: Annotated[int, Field(ge=0, le=2)]
    blink: Annotated[float, Field(ge=0, le=1)]
    mouth: Annotated[float, Field(ge=0, le=1)]
    rms: Annotated[float, Field(ge=0, le=1)]
    audio_fresh: bool


class LiveInput(Input):
    resume_skills: Annotated[list[Annotated[str, Field(min_length=1, max_length=100)]], Field(min_length=1, max_length=30)]
    question: Annotated[str, Field(max_length=5000)]
    transcript: Annotated[str, Field(max_length=30_000)]
    final: StrictBool
    speech_confidence: Annotated[float | None, Field(ge=0, le=1)] = None
    samples: Annotated[list[Sample], Field(max_length=1900)] = Field(default_factory=list)
    capture_interrupted: StrictBool = False

    @model_validator(mode='after')
    def ordered_samples(self):
        if any(b.t <= a.t for a, b in zip(self.samples, self.samples[1:])):
            raise ValueError('Samples must have increasing capture timestamps')
        return self


class AnswerRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    question: Annotated[str, Field(max_length=5000)]
    transcript: Annotated[str, Field(max_length=30_000)]
    speech_confidence: Annotated[float | None, Field(ge=0, le=1)] = None


class SummaryInput(Input):
    resume_skills: Annotated[list[Annotated[str, Field(min_length=1, max_length=100)]], Field(min_length=1, max_length=30)]
    answers: Annotated[list[AnswerRecord], Field(max_length=50)] = Field(default_factory=list)


def load_speech_model():
    global _model
    with _model_lock:
        if _model is None:
            from vosk import Model, SetLogLevel
            SetLogLevel(-1)
            if not MODEL_PATH.is_dir():
                raise RuntimeError('Speech model missing; run python setup_models.py')
            _model = Model(str(MODEL_PATH))
    return _model


def extract_document(data: bytes, extension: str) -> str:
    if not data or len(data) > 2_000_000:
        raise ValueError('Use a nonempty resume up to 2 MB')
    if extension == '.txt':
        text = data.decode('utf-8-sig')
    elif extension == '.pdf':
        if not data.startswith(b'%PDF-'):
            raise ValueError('Invalid PDF header')
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted or len(reader.pages) > 10:
            raise ValueError('Use an unlocked PDF with at most 10 pages')
        parts = []
        for page in reader.pages:
            # Bound decoded content before extraction, including compressed PDFs.
            content = page.get_contents()
            if content and len(content.get_data()) > 4_000_000:
                raise ValueError('PDF content is too large')
            parts.append(page.extract_text() or '')
        text = '\n'.join(parts)
    elif extension == '.docx':
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if sum(entry.file_size for entry in archive.infolist()) > 10_000_000:
                raise ValueError('Expanded DOCX is too large')
            raw = archive.read('word/document.xml')
            if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
                raise ValueError('Unsupported XML')
            root = ElementTree.fromstring(raw)
            text = ' '.join(node.text or '' for node in root.iter() if node.tag.endswith('}t'))
    else:
        raise ValueError('Supported files: TXT, text-based PDF, DOCX')
    if len(text) > 50_000:
        raise ValueError('Resume text exceeds 50,000 characters')
    if not text.strip():
        raise ValueError('No readable text; scanned PDFs need OCR before upload')
    return text


def words(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def contains(text, term):
    return bool(re.search(r'\b' + re.escape(term) + r'(?:s)?\b', text))


ALIASES = {'python': ['python', 'python3'], 'sql': ['sql', 'postgresql', 'mysql', 'sequel'],
           'javascript': ['javascript', 'java script', 'js'], 'java': ['java'],
           'machine learning': ['machine learning'], 'fastapi': ['fastapi', 'fast api'],
           'docker': ['docker'], 'react': ['react', 'reactjs', 'react js']}


def claimed_skills(text, vocabulary):
    normalized = words(text)
    return [skill for skill in vocabulary if any(contains(normalized, a) for a in ALIASES.get(skill, [skill]))]


def analyze_answer(payload, sanitize, vocabulary):
    question, transcript = sanitize(payload.question), sanitize(payload.transcript)
    question_words, answer_words = words(question), words(transcript)
    skills = sorted({words(sanitize(s)) for s in payload.resume_skills if words(sanitize(s)) in vocabulary})
    # Explicit skill mentions establish scope. A question with no skill name
    # can use domain-specific terms, excluding terms shared across domains.
    targeted = [s for s in skills if any(contains(question_words, a) for a in ALIASES.get(s, [s]))]
    if not targeted:
        for skill in skills:
            unique = [t for t in vocabulary[skill] if sum(t in values for values in vocabulary.values()) == 1]
            if any(contains(question_words, term) for term in unique):
                targeted.append(skill)
    assessments = []
    low_confidence = payload.speech_confidence is None or payload.speech_confidence < .65
    for skill in skills:
        matched = [term for term in vocabulary[skill] if contains(answer_words, term)]
        if skill not in targeted:
            status, message = 'NOT_ASKED', 'This skill has not been identified in the current question.'
        elif not payload.final:
            status, message = 'LISTENING', 'Answer in progress; no skill conclusion yet.'
        elif low_confidence:
            status, message = 'CHECK_TRANSCRIPT', 'Recognition confidence is insufficient; confirm what was said before assessing this skill.'
        elif len(answer_words.split()) < 12:
            status, message = 'INSUFFICIENT_ANSWER', 'Too little recognized speech to assess technical depth; ask a specific follow-up.'
        elif len(matched) < 2:
            status, message = 'FOLLOW_UP', f'Limited {skill} technical vocabulary in this answer; ask for an implementation example.'
        else:
            status, message = 'VOCABULARY_PRESENT', 'Relevant technical terms were observed; correctness and depth still need interviewer review.'
        assessments.append({'skill': skill, 'status': status, 'message': message, 'matched_terms': matched if skill in targeted else []})
    return question, transcript, assessments


def correlation(x, y):
    mx, my = mean(x), mean(y)
    denom = math.sqrt(sum((v - mx)**2 for v in x) * sum((v - my)**2 for v in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / denom if denom > 1e-8 else 0.0


def analyze_capture(samples, interrupted):
    observations = []
    result = {'status': 'INSUFFICIENT_CAPTURE', 'blink_count': None, 'blink_rate_per_minute': None,
              'lip_sync_status': 'UNASSESSED', 'estimated_offset_ms': None,
              'network_status': 'NOT_MEASURED_LOCAL_CAPTURE', 'observations': observations}
    if len(samples) < 20:
        return result
    duration = (samples[-1].t - samples[0].t) / 1000
    if duration < 5:
        return result
    valid = [s for s in samples if s.faces == 1]
    gaps = [b.t - a.t for a, b in zip(samples, samples[1:])]
    degraded = interrupted or max(gaps) > 350 or len(valid) / len(samples) < .9 or any(not s.audio_fresh for s in samples)
    result['status'] = 'CAPTURE_DEGRADED' if degraded else 'OBSERVATIONS_ONLY'
    if any(s.faces > 1 for s in samples):
        observations.append('More than one face was observed; verify that only the candidate is in the capture area.')
    if len(valid) / len(samples) < .9:
        observations.append('A single face was not consistently visible; improve framing or lighting.')
    blink_count, closed_at = 0, None
    for s in samples:
        if s.faces != 1:
            closed_at = None
        elif s.blink > .55 and closed_at is None:
            closed_at = s.t
        elif s.blink < .3 and closed_at is not None:
            if 60 <= s.t - closed_at <= 700:
                blink_count += 1
            closed_at = None
    result['blink_count'] = blink_count
    result['blink_rate_per_minute'] = round(blink_count * 60 / duration, 1) if duration >= 30 and not degraded else None
    observations.append(f'{blink_count} eye closures observed; blinking is not used to infer deception or raise risk.')
    if degraded:
        result['lip_sync_status'] = 'EXCLUDED_CAPTURE_QUALITY'
        observations.append('Dropped, delayed, missing-face or stale-audio samples exclude audio/mouth timing analysis.')
        return result
    if duration < 10:
        return result
    # Resample onto a common 100 ms capture grid. Use capture time, never network
    # arrival time. Mouth opening vs energy correlation is exploratory, not a
    # phoneme-alignment or biometric authenticity model.
    grid, j = [], 0
    for t in range(int(samples[0].t), int(samples[-1].t), 100):
        while j + 1 < len(samples) and abs(samples[j+1].t - t) < abs(samples[j].t - t):
            j += 1
        grid.append(samples[j])
    mouth, audio = [s.mouth for s in grid], [s.rms for s in grid]
    if max(mouth) - min(mouth) < .08 or max(audio) - min(audio) < .01 or sum(v > .015 for v in audio) < 20:
        result['lip_sync_status'] = 'INSUFFICIENT_ACTIVITY'
        return result
    correlations = {}
    for lag in range(-5, 6):
        x = mouth[max(lag, 0):len(mouth) + min(lag, 0)]
        y = audio[max(-lag, 0):len(audio) - max(lag, 0)]
        correlations[lag] = correlation(x, y)
    best = max(correlations, key=correlations.get)
    if abs(best) >= 2 and correlations[best] >= .6 and correlations[best] - correlations[0] > .2:
        result['lip_sync_status'] = 'TIMING_REVIEW'
        result['estimated_offset_ms'] = best * 100
        observations.append('An audio/mouth activity offset was observed; check device synchronization before drawing any integrity conclusion.')
    else:
        result['lip_sync_status'] = 'NO_CLEAR_OFFSET_OBSERVED'
    return result


def judge_candidate(payload: SummaryInput, sanitize, vocabulary):
    raw_skills = [words(sanitize(s)) for s in payload.resume_skills]
    skills = sorted({s for s in raw_skills if s in vocabulary})

    answer_evaluations = []
    skill_hits = {skill: set() for skill in skills}
    skill_targeted_counts = {skill: 0 for skill in skills}

    for idx, ans in enumerate(payload.answers, start=1):
        q_clean = sanitize(ans.question)
        t_clean = sanitize(ans.transcript)
        q_words = words(q_clean)
        t_words = words(t_clean)
        word_count = len(t_words.split())

        targeted = [s for s in skills if any(contains(q_words, a) for a in ALIASES.get(s, [s]))]
        if not targeted:
            for s in skills:
                unique = [term for term in vocabulary.get(s, ()) if sum(term in vals for vals in vocabulary.values()) == 1]
                if any(contains(q_words, term) for term in unique):
                    targeted.append(s)

        eval_skills = targeted if targeted else skills
        for s in targeted:
            skill_targeted_counts[s] += 1

        matched_terms_for_answer = {}
        for s in eval_skills:
            terms = [term for term in vocabulary.get(s, ()) if contains(t_words, term)]
            if terms:
                matched_terms_for_answer[s] = terms
                skill_hits[s].update(terms)

        all_matched = [term for terms in matched_terms_for_answer.values() for term in terms]

        # 1. Technical Vocabulary (up to 50 points)
        vocab_score = min(50, len(all_matched) * 25)

        # 2. Answer Depth & Substance (up to 35 points)
        if word_count >= 30:
            depth_score = 35
        elif word_count >= 20:
            depth_score = 25
        elif word_count >= 12:
            depth_score = 15
        elif word_count > 0:
            depth_score = 5
        else:
            depth_score = 0

        # 3. Speech Clarity & Confidence (up to 15 points)
        conf = ans.speech_confidence
        if conf is not None:
            if conf >= 0.80:
                clarity_score = 15
            elif conf >= 0.65:
                clarity_score = 10
            elif conf >= 0.50:
                clarity_score = 5
            else:
                clarity_score = 2
        else:
            clarity_score = 10

        answer_points = min(100, vocab_score + depth_score + clarity_score)

        if answer_points >= 80:
            rating = "Strong Answer"
            feedback = f"Strong technical response with relevant terminology ({', '.join(all_matched) if all_matched else 'good depth'})."
        elif answer_points >= 60:
            rating = "Good Answer"
            feedback = "Solid answer addressing the core concepts effectively."
        elif answer_points >= 40:
            rating = "Basic Answer"
            feedback = "Basic response; could benefit from more specific technical details and examples."
        else:
            rating = "Needs Improvement"
            feedback = "Brief or limited response; missing key technical terms for the tested domain."

        answer_evaluations.append({
            "index": idx,
            "question": q_clean,
            "transcript": t_clean,
            "points": answer_points,
            "rating": rating,
            "matched_terms": all_matched,
            "feedback": feedback
        })

    skill_breakdowns = []
    for s in skills:
        matched = sorted(skill_hits[s])
        targeted_count = skill_targeted_counts[s]
        if targeted_count == 0 and not matched:
            skill_breakdowns.append({
                "skill": s,
                "score": None,
                "status": "NOT_EVALUATED",
                "matched_terms": [],
                "feedback": "Skill was not covered in the interview questions."
            })
        else:
            term_score = min(60, len(matched) * 30)
            coverage_score = 40 if targeted_count > 0 and len(matched) >= 1 else 20
            skill_points = min(100, term_score + coverage_score)
            status = "STRONG" if skill_points >= 75 else "COMPETENT" if skill_points >= 50 else "NEEDS_IMPROVEMENT"
            feedback = f"Demonstrated terms: {', '.join(matched)}" if matched else "Limited domain terminology demonstrated."
            skill_breakdowns.append({
                "skill": s,
                "score": skill_points,
                "status": status,
                "matched_terms": matched,
                "feedback": feedback
            })

    if answer_evaluations:
        total_points = round(sum(a["points"] for a in answer_evaluations) / len(answer_evaluations))
    else:
        total_points = 0

    if total_points >= 80:
        verdict = "Strong Technical Performance"
        recommendation = "High Recommendation: Candidate exhibited strong technical vocabulary and thorough explanations."
    elif total_points >= 65:
        verdict = "Qualified Technical Performance"
        recommendation = "Recommendation: Candidate demonstrated good baseline technical competence across tested topics."
    elif total_points >= 50:
        verdict = "Moderate Technical Performance"
        recommendation = "Conditional Recommendation: Candidate grasped basic concepts but would benefit from further deep-dive validation."
    elif total_points > 0:
        verdict = "Needs Improvement"
        recommendation = "Follow-up Required: Answers showed limited technical depth and terminology."
    else:
        verdict = "No Answers Evaluated"
        recommendation = "No interview answers were provided for assessment."

    strengths = []
    improvements = []
    for sb in skill_breakdowns:
        if sb["status"] == "STRONG":
            strengths.append(f"Demonstrated command of {sb['skill']} concepts ({', '.join(sb['matched_terms'])}).")
        elif sb["status"] == "NEEDS_IMPROVEMENT":
            improvements.append(f"Follow up on {sb['skill']} implementation and architecture.")
        elif sb["status"] == "NOT_EVALUATED":
            improvements.append(f"Assess {sb['skill']} in future interview questions.")

    if not strengths:
        strengths.append("Engaged with interview questions and provided structured responses.")
    if not improvements:
        improvements.append("Candidate answered tested skill areas thoroughly.")

    summary_text = (
        f"Candidate completed {len(answer_evaluations)} question-and-answer turn(s), achieving an overall score of {total_points}/100 points. "
        f"{verdict}. {recommendation}"
    )

    return {
        "total_points": total_points,
        "verdict": verdict,
        "recommendation": recommendation,
        "summary": summary_text,
        "strengths": strengths,
        "areas_for_improvement": improvements,
        "skill_breakdowns": skill_breakdowns,
        "answer_evaluations": answer_evaluations
    }


def register_live(app, sanitize, vocabulary):
    slots = asyncio.Semaphore(4)

    @app.get('/capabilities')
    def capabilities():
        return {'speech': 'LOCAL_VOSK' if MODEL_PATH.is_dir() else 'MODEL_MISSING',
                'video': 'LOCAL_FACE_LANDMARKS', 'synthetic_only': True,
                'voice_clone_detection': 'UNAVAILABLE', 'deepfake_detection': 'UNAVAILABLE'}

    @app.post('/interview/summary')
    def interview_summary(payload: SummaryInput):
        return judge_candidate(payload, sanitize, vocabulary)

    @app.post('/resume')
    def resume(payload: ResumeInput):
        try:
            data = base64.b64decode(payload.content_base64, validate=True)
            text = sanitize(extract_document(data, Path(payload.filename).suffix.lower()))
        except (ValueError, UnicodeError, binascii.Error, KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise HTTPException(422, 'Unable to read resume. Use UTF-8 TXT, unlocked text PDF (up to 10 pages), or DOCX, up to 2 MB.') from exc
        except Exception as exc:
            raise HTTPException(422, 'Unable to extract document text; try a text resume.') from exc
        return {'skills': claimed_skills(text, vocabulary), 'redacted_text': text,
                'supported_skills': list(vocabulary), 'notice': 'Only recognized skills are extracted; unsupported skills are not assessed.'}

    @app.post('/analyze/live')
    def analyze(payload: LiveInput):
        question, transcript, assessments = analyze_answer(payload, sanitize, vocabulary)
        capture = analyze_capture(payload.samples, payload.capture_interrupted)
        follow_up = any(a['status'] in ('FOLLOW_UP', 'INSUFFICIENT_ANSWER', 'CHECK_TRANSCRIPT') for a in assessments)
        return {'question': question, 'transcript': transcript, 'skill_assessments': assessments,
                'capture': capture, 'decision': {'risk_score': None, 'risk_tier': 'UNASSESSED'},
                'reason': 'Live capture provides skill and timing observations, not calibrated impersonation evidence. Blinking and weak technical answers do not establish deception.',
                'next_action': 'ask targeted follow-up questions' if follow_up else 'continue interview and review observations',
                'detectors': {'voice_clone': 'UNAVAILABLE', 'deepfake': 'UNAVAILABLE'}}

    @app.websocket('/interview/stream')
    async def stream(socket: WebSocket):
        # Prevent another website from opening a browser socket to this local app.
        origin = socket.headers.get('origin')
        host = socket.headers.get('host', '')
        if origin not in (f'http://{host}', f'https://{host}'):
            await socket.close(code=1008)
            return
        await socket.accept()
        acquired = False
        try:
            try:
                await asyncio.wait_for(slots.acquire(), timeout=.1)
                acquired = True
            except asyncio.TimeoutError:
                await socket.send_json({'error': 'All four transcription slots are busy; try again shortly.'})
                return
            config = await asyncio.wait_for(socket.receive_json(), timeout=10)
            Input.model_validate(config)
            model = await asyncio.to_thread(load_speech_model)
            from vosk import KaldiRecognizer
            recognizer = KaldiRecognizer(model, 16000)
            recognizer.SetWords(True)
            await socket.send_json({'ready': True, 'sample_rate': 16000})
            committed, confidences, received_bytes = [], [], 0
            started = time.monotonic()
            while time.monotonic() - started < 190:
                message = await asyncio.wait_for(socket.receive(), timeout=20)
                if message['type'] == 'websocket.disconnect':
                    return
                data = message.get('bytes')
                if data is not None:
                    if len(data) > 32_000 or len(data) % 2:
                        raise ValueError('Invalid PCM frame')
                    received_bytes += len(data)
                    if received_bytes > 16000 * 2 * 180:
                        raise ValueError('Turn exceeds three minutes')
                    accepted = await asyncio.to_thread(recognizer.AcceptWaveform, data)
                    result = json.loads(recognizer.Result() if accepted else recognizer.PartialResult())
                    if accepted:
                        committed.append(result.get('text', ''))
                        confidences.extend(w['conf'] for w in result.get('result', []))
                    text = sanitize(' '.join(committed + ([] if accepted else [result.get('partial', '')])).strip())
                    await socket.send_json({'text': text, 'final': False,
                                            'confidence': mean(confidences) if confidences else None,
                                            'audio_seconds': received_bytes / 32000})
                elif message.get('text') == 'finish':
                    result = json.loads(await asyncio.to_thread(recognizer.FinalResult))
                    committed.append(result.get('text', ''))
                    confidences.extend(w['conf'] for w in result.get('result', []))
                    await socket.send_json({'text': sanitize(' '.join(committed).strip()), 'final': True,
                                            'confidence': mean(confidences) if confidences else None,
                                            'audio_seconds': received_bytes / 32000})
                    return
                else:
                    raise ValueError('Unknown stream message')
            raise ValueError('Turn time limit reached')
        except WebSocketDisconnect:
            pass
        except Exception:
            # Never echo exception strings that could contain input or raw speech.
            try:
                await socket.send_json({'error': 'Transcription stopped. Check microphone capture and the local speech model; then record the turn again.'})
            except Exception:
                pass
        finally:
            if acquired:
                slots.release()
            try:
                await socket.close()
            except Exception:
                pass
