# Locally served model assets

MediaPipe Tasks Vision 0.10.32 JavaScript and WASM assets were downloaded from:
https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/

Source and license: https://github.com/google-ai-edge/mediapipe (Apache 2.0).

Face Landmarker float16 model, version 1:
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
Documentation: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

These assets run face landmark and blendshape inference in the browser. They do not classify deepfakes, cloned voices, emotions, or honesty. No face images are sent to Google or any other external service by this application.

The separate local speech model is Vosk small English US 0.15 (Apache 2.0), downloaded by setup_models.py. Source and model information: https://alphacephei.com/vosk/models
