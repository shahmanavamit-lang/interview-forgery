# P1 – Automatic mock interview workflow

Upload a synthetic resume, capture the interviewer’s question, then capture the candidate’s answer. The app automatically transcribes speech, identifies the question’s relevant resume skill, and displays skill follow-ups and measured camera observations.

This version uses **the camera and microphone on one device**, with separate interviewer and candidate turns. It does not join Zoom/Meet or provide remote interview rooms.

## Run

Python 3.10+; a recent desktop browser with camera, microphone, WebAssembly and AudioWorklet support. No cloud API key or frontend build is needed.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_models.py
python backend.py
```

Open http://127.0.0.1:8000. The speech model and browser vision assets are already installed in this workspace. `setup_models.py` is for a fresh checkout and downloads a checksum-verified 40 MB model. The Vosk model uses approximately 300 MB of RAM, so automatic recognition has a larger footprint than the original keyword-only simulator.

1. Confirm you are conducting a consented mock interview with a synthetic profile.
2. Upload a TXT, text-based PDF (at most 10 pages) or DOCX resume up to 2 MB. A downloadable mock resume is provided. Scanned PDFs need OCR first.
3. Select **Enable camera & microphone** and approve the browser’s device request.
4. Select **Record interviewer question**, ask e.g. “Explain Python generators and how you use them,” then **Finish turn**.
5. Select **Record candidate answer**, speak, then **Finish turn**. Questions and answers are automatically transcribed; no manual telemetry is required. Each turn is limited to 3 minutes.
6. Review the recognized text, skill follow-ups and camera observations. Repeat for another question. **Stop devices** releases capture; changing tabs also stops capture to avoid using interrupted video.

Use headphones or a quiet room. The system separates speakers through turn controls, not automatic speaker identification. Transcription is English-only and can misrecognize technical terms or accents. Low confidence and short answers produce explicit “check transcript” or “more detail needed” states. A transcription mistake should be resolved by recording the turn again before interpreting a flag.

## What is automatic

- Resume text extraction and regex Aadhaar/PAN redaction before skill extraction.
- Local streaming speech recognition using Vosk. The original audio is not saved.
- Identification of the skill asked about, vocabulary evidence in the answer, and targeted follow-up prompts. Skills not identified in the question remain unassessed.
- Browser-local face landmarks, face count, and eye-closure counts.
- Exploratory audio-energy/mouth-opening timing correlation using local capture timestamps. Dropped frames, missing faces, stale audio and interrupted capture exclude timing analysis.
- Session-only answer history, limited to the latest 10 entries. Reloading clears it.

## Limits that the interface makes explicit

There is **no trained deepfake or voice-clone classifier**. The live response uses `risk_score: null` and `risk_tier: "UNASSESSED"`; it does not substitute a fake score for missing evidence. Blinking is not an honesty measure. Vocabulary presence does not establish technical correctness; skill weakness does not establish impersonation.

The timing heuristic compares mouth activity with audio energy, not phonemes. It is not a validated lip-sync detector and can miss short blinks or timing offsets. Its output asks for a device-synchronization check, never an accusation of tampering. This local-device flow has no WebRTC network statistics, so remote-call latency, jitter and packet loss are explicitly **not measured**. Network arrival time is never used as lip-sync evidence.

This is a functional hackathon observation workflow, not a production biometric assessment or hiring system. Production/remote use needs authenticated HTTPS, a proper media transport, measured capture/network timestamps and independently validated detectors. Do not expose the local development server publicly.

## APIs

The existing `POST /evaluate` simulator contract is unchanged. It remains available through **Manual simulator** and its embedded synthetic demos; its simulated scores are not used by the live workflow. `design.md` was not provided, so these are documented provisional interfaces.

- `GET /capabilities`: local speech/vision availability and unavailable authenticity detectors.
- `POST /resume`: `{synthetic_profile: true, filename: "mock.txt", content_base64: "..."}` → `skills`, `redacted_text`, supported vocabulary and notice.
- `WS /interview/stream`: same-origin socket. Send `{ "synthetic_profile": true }`; wait for `ready`; stream mono signed 16-bit little-endian PCM at 16 kHz, at most 32 KB per frame. Send the literal text `finish` to flush. Results contain `text`, `final`, `confidence` (ASR only) and `audio_seconds`. At most four concurrent streams; 3-minute audio limit and 20-second idle timeout. Partial/final text is redacted before returning.
- `POST /analyze/live`: synthetic attestation, `resume_skills`, `question`, `transcript`, `final`, optional `speech_confidence`, `samples`, `capture_interrupted`. Samples are ordered browser-derived `{t, faces, blink, mouth, rms, audio_fresh}` measurements. Returns per-skill assessments, capture observations, unassessed authenticity decision and next action.

All documents, speech and samples are untrusted. Validation errors do not echo inputs. Resume text and transcriptions are sanitized before semantic analysis. No interview files are persisted and no external recognition service is called. Synthetic provenance remains the caller’s attestation.

## Verify

```sh
python backend.py --test
python test_live.py
node --check frontend/app.js
node --check frontend/pcm-worklet.js
```

The original five test groups and eight live-workflow groups cover scoring, redaction, upload validation, question scope, speech-confidence gates, timing-quality suppression and non-scoring blink observations. A real WebSocket transcription check with synthesized speech was also run locally. Physical camera/microphone capture requires the user’s browser permission and has not been hardware-tested by the agent.

Main files: `backend.py` (existing evaluator and serving), `live.py` (automatic workflow), `frontend/` (UI and local vision assets), `setup_models.py` (speech setup). No Node packages or UI build step are required.
