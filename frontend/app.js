import {FaceLandmarker, FilesetResolver} from '/assets/vendor/vision_bundle.mjs';

const $ = id => document.getElementById(id);
let skills = [], stream = null, context = null, processor = null, face = null;
let socket = null, turn = null, preparing = false, stopping = false, epoch = 0;
let question = '', transcript = '', confidence = null, samples = [], interrupted = false;
let lastRms = 0, lastAudioAt = -Infinity, lastVideoTime = -1, frameBusy = false;
let visionTimer = null, clockTimer = null, analysisTimer = null, finishTimer = null;
let latestAnalysis = null, historyCount = 0, analysisChain = Promise.resolve();
let observedBlinks = 0, eyeClosedAt = null;
const labels = {NOT_ASKED:'Not asked',LISTENING:'Listening',CHECK_TRANSCRIPT:'Check transcript',INSUFFICIENT_ANSWER:'More detail needed',FOLLOW_UP:'Follow up',VOCABULARY_PRESENT:'Terms observed'};
function status(text) { $('status').textContent = text; }
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
function clearError() { $('error').hidden = true; }
function controls() {
  const connected = Boolean(stream), busy = Boolean(turn) || preparing || stopping;
  $('resume').disabled = !$('consent').checked || connected || busy;
  $('consent').disabled = connected || busy;
  $('connect').disabled = !skills.length || connected || busy || !$('consent').checked;
  $('disconnect').disabled = !connected || stopping;
  $('question-start').disabled = !connected || busy;
  $('answer-start').disabled = !connected || busy || !question;
  $('finish').disabled = !turn || turn.finishing;
  $('device-state').textContent = connected ? 'ON' : 'OFF';
}
async function post(path, payload) {
  const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 20000);
  try {
    const response = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload), signal:controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The server could not process this request.');
    return data;
  } finally { clearTimeout(timeout); }
}
function resetCapture() {
  samples = []; interrupted = false; observedBlinks = 0; eyeClosedAt = null; lastAudioAt = -Infinity;
  $('blink-count').textContent = '—'; $('face-count').textContent = '—'; $('sync-state').textContent = 'Unassessed';
  $('observations').replaceChildren(); const li = document.createElement('li'); li.textContent = 'Waiting for candidate capture.'; $('observations').append(li);
}
function renderSkills(assessments = []) {
  $('skill-results').replaceChildren();
  for (const skill of skills) {
    const item = assessments.find(a => a.skill === skill) || {status:'NOT_ASKED',message:'Waiting for a question about this skill.',matched_terms:[]};
    const card = document.createElement('div'); card.className = 'skill-result';
    const title = document.createElement('h3'); title.textContent = skill;
    const badge = document.createElement('span'); badge.className = 'tag' + (['FOLLOW_UP','INSUFFICIENT_ANSWER','CHECK_TRANSCRIPT'].includes(item.status) ? ' follow-up' : item.status === 'VOCABULARY_PRESENT' ? ' present' : '');
    badge.textContent = labels[item.status] || item.status; title.append(badge);
    const text = document.createElement('p'); text.textContent = item.message;
    card.append(title, text);
    if (item.matched_terms.length) { const terms = document.createElement('p'); terms.textContent = 'Observed: ' + item.matched_terms.join(', '); card.append(terms); }
    $('skill-results').append(card);
  }
}
$('consent').addEventListener('change', controls);
$('resume').addEventListener('change', async () => {
  const file = $('resume').files[0]; if (!file) return;
  clearError();
  skills = []; question = ''; transcript = ''; latestAnalysis = null;
  $('question-text').textContent = 'No question recorded.'; $('answer-text').textContent = 'No answer recorded.';
  $('skills').replaceChildren(); $('resume-preview').hidden = true; $('resume-text').textContent = '';
  $('history').replaceChildren(); historyCount = 0; $('history-count').textContent = '0'; $('history-empty').hidden = false;
  renderSkills(); resetCapture();
  preparing = true; controls(); status('Extracting resume skills…');
  try {
    if (file.size > 2000000 || !/\.(txt|pdf|docx)$/i.test(file.name)) throw new Error('Choose a TXT, PDF or DOCX file up to 2 MB.');
    const buffer = new Uint8Array(await file.arrayBuffer());
    let binary = ''; for (let i = 0; i < buffer.length; i += 8192) binary += String.fromCharCode(...buffer.subarray(i, i + 8192));
    const result = await post('/resume', {synthetic_profile:$('consent').checked,filename:file.name,content_base64:btoa(binary)});
    skills = result.skills;
    for (const skill of skills) { const chip = document.createElement('span'); chip.textContent = skill; $('skills').append(chip); }
    $('resume-text').textContent = result.redacted_text; $('resume-preview').hidden = false;
    if (!skills.length) throw new Error('No supported skills found. Supported: ' + result.supported_skills.join(', ') + '.');
    renderSkills(); status('Resume ready. Enable the camera and microphone for a mock interview.'); $('next-action').textContent = 'Enable camera & microphone';
  } catch (error) { showError(error.message); status('Resume could not be loaded.'); }
  finally { preparing = false; controls(); }
});

$('connect').addEventListener('click', async () => {
  clearError(); preparing = true; const attempt = ++epoch; controls(); status('Requesting camera and microphone access…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error('Camera access requires localhost or HTTPS in a supported browser.');
    const media = await navigator.mediaDevices.getUserMedia({video:{width:{ideal:640},height:{ideal:480},frameRate:{ideal:24}},audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});
    if (epoch !== attempt) { media.getTracks().forEach(track => track.stop()); return; }
    stream = media; $('camera').srcObject = media; await $('camera').play(); $('camera-empty').hidden = true;
    context = new AudioContext({sampleRate:16000});
    if (context.sampleRate !== 16000) throw new Error('This browser cannot capture at the required 16 kHz. Try a current desktop Chrome browser.');
    await context.audioWorklet.addModule('/assets/pcm-worklet.js');
    processor = new AudioWorkletNode(context, 'pcm-capture');
    context.createMediaStreamSource(stream).connect(processor); processor.connect(context.destination);
    processor.port.onmessage = ({data}) => {
      if (!turn || socket?.readyState !== WebSocket.OPEN) return;
      if (data.pcm) {
        if (socket.bufferedAmount > 128000) { interrupted = true; failTurn('Audio delivery fell behind. Record this turn again.'); return; }
        lastRms = data.rms; lastAudioAt = performance.now(); socket.send(data.pcm);
      }
      if (data.finished) socket.send('finish');
    };
    await context.resume();
    for (const track of stream.getTracks()) track.addEventListener('ended', () => { showError('A capture device disconnected. Reconnect to continue.'); stopDevices(); });
    status('Loading local face observations…'); $('vision-state').textContent = 'LOADING';
    try {
      const files = await FilesetResolver.forVisionTasks('/assets/vendor');
      const detector = await FaceLandmarker.createFromOptions(files, {baseOptions:{modelAssetPath:'/assets/vendor/face_landmarker.task',delegate:'CPU'},runningMode:'VIDEO',numFaces:2,outputFaceBlendshapes:true});
      if (attempt !== epoch) { detector.close(); return; }
      face = detector; $('vision-state').textContent = 'READY';
    } catch (error) { $('vision-state').textContent = 'UNAVAILABLE'; showError('Face observations could not load. Speech and skill follow-ups still work; camera timing will remain unassessed.'); }
    if (attempt !== epoch) return;
    visionTimer = setInterval(captureFrame, 100);
    status('Devices ready. Record the interviewer’s question first.'); $('next-action').textContent = 'Record interviewer question';
  } catch (error) { showError(error.name === 'NotAllowedError' ? 'Camera or microphone permission was denied. Allow access, then try again.' : error.message); await stopDevices(); }
  finally { preparing = false; controls(); }
});

function captureFrame() {
  if (!face || frameBusy || !stream || $('camera').readyState < 2 || $('camera').currentTime === lastVideoTime) return;
  frameBusy = true;
  try {
    lastVideoTime = $('camera').currentTime;
    const now = performance.now();
    const result = face.detectForVideo($('camera'), now);
    const faces = Math.min(2, result.faceLandmarks.length);
    $('face-count').textContent = String(faces);
    if (turn?.role !== 'answer' || turn.finishing) return;
    const coefficients = Object.fromEntries((result.faceBlendshapes[0]?.categories || []).map(c => [c.categoryName,c.score]));
    const blink = ((coefficients.eyeBlinkLeft || 0) + (coefficients.eyeBlinkRight || 0)) / 2;
    if (faces !== 1) eyeClosedAt = null;
    else if (blink > .55 && eyeClosedAt === null) eyeClosedAt = now;
    else if (blink < .3 && eyeClosedAt !== null) {
      if (now - eyeClosedAt >= 60 && now - eyeClosedAt <= 700) observedBlinks++;
      eyeClosedAt = null;
    }
    $('blink-count').textContent = String(observedBlinks);
    const t = now - turn.started;
    // The first audio chunk arrives after 250 ms; do not treat initialization
    // as stale audio, and never infer timing from WebSocket response times.
    if (t >= 500 && t <= 180000 && samples.length < 1900) samples.push({t,faces,blink,mouth:coefficients.jawOpen || 0,rms:lastRms,audio_fresh:now-lastAudioAt < 400});
  } catch (error) { interrupted = true; $('vision-state').textContent = 'CAPTURE ERROR'; }
  finally { frameBusy = false; }
}

async function startTurn(role) {
  if (!stream || turn || preparing) return;
  clearError(); preparing = true; controls();
  const currentEpoch = epoch;
  if (role === 'question') {
    question = ''; transcript = ''; confidence = null; latestAnalysis = null;
    $('question-text').textContent = 'Listening…'; $('answer-text').textContent = 'No answer recorded.'; renderSkills(); resetCapture();
  } else { transcript = ''; confidence = null; latestAnalysis = null; $('answer-text').textContent = 'Listening…'; renderSkills(); resetCapture(); }
  status('Starting local transcription…');
  try {
    await context.resume();
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/interview/stream`);
    socket = ws;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Local transcription did not start. Check the speech model.')), 20000);
      ws.onopen = () => ws.send(JSON.stringify({synthetic_profile:$('consent').checked}));
      ws.onerror = () => { clearTimeout(timeout); reject(new Error('Cannot connect to local transcription.')); };
      ws.onclose = () => { clearTimeout(timeout); reject(new Error('Transcription connection closed before capture began.')); };
      ws.onmessage = ({data}) => {
        const message = JSON.parse(data);
        if (message.ready) { clearTimeout(timeout); resolve(); }
        if (message.error) { clearTimeout(timeout); reject(new Error(message.error)); }
      };
    });
    if (currentEpoch !== epoch || !stream) { ws.close(); return; }
    turn = {role,started:performance.now(),finishing:false,id:crypto.randomUUID()};
    ws.onmessage = async ({data}) => {
      if (!turn || socket !== ws) return;
      const message = JSON.parse(data);
      if (message.error) { failTurn(message.error); return; }
      if (typeof message.text === 'string') {
        if (role === 'question') { question = message.text; $('question-text').textContent = question || 'Listening…'; }
        else { transcript = message.text; confidence = message.confidence; $('answer-text').textContent = transcript || 'Listening…'; }
      }
      if (message.final) { ws.onclose = null; ws.onerror = null; await completeTurn(); }
    };
    ws.onclose = () => { if (turn && socket === ws) failTurn('Transcription disconnected. This turn is incomplete; record it again.'); };
    ws.onerror = () => { if (turn && socket === ws) failTurn('Transcription connection failed. Record this turn again.'); };
    processor.port.postMessage('start');
    $('recording-label').textContent = role === 'question' ? 'Recording interviewer' : 'Recording candidate'; $('recording-label').hidden = false;
    status(role === 'question' ? 'Listening to the interviewer. Finish the turn before the candidate answers.' : 'Listening to the candidate. Skill conclusions wait until the answer is complete.');
    clockTimer = setInterval(() => {
      if (!turn) return;
      const seconds = Math.floor((performance.now() - turn.started) / 1000);
      $('timer').textContent = `${String(Math.floor(seconds / 60)).padStart(2,'0')}:${String(seconds % 60).padStart(2,'0')}`;
      if (seconds >= 179 && !turn.finishing) finishTurn();
    }, 250);
    if (role === 'answer') analysisTimer = setInterval(() => { if (turn && !turn.finishing) queueAnalysis(false).catch(() => {}); }, 8000);
  } catch (error) { if (currentEpoch === epoch) failTurn(error.message); }
  finally { preparing = false; controls(); }
}
function finishTurn() {
  if (!turn || turn.finishing) return;
  turn.finishing = true; clearInterval(analysisTimer); controls(); status('Finalizing transcript…');
  processor.port.postMessage('finish');
  finishTimer = setTimeout(() => failTurn('Final transcript timed out. This turn is incomplete; record it again.'), 20000);
}
function releaseTurn() {
  clearInterval(clockTimer); clearInterval(analysisTimer); clearTimeout(finishTimer);
  processor?.port.postMessage('cancel');
  const ws = socket; socket = null; turn = null; ws?.close();
  $('recording-label').hidden = true; controls();
}
function failTurn(message) {
  interrupted = true; releaseTurn(); showError(message); status('Capture stopped. Record the turn again before using its transcript.');
  question = ''; transcript = ''; confidence = null; latestAnalysis = null;
  $('question-text').textContent = 'Record question again.'; $('answer-text').textContent = 'Incomplete capture discarded.'; renderSkills(); controls();
}
async function completeTurn() {
  clearTimeout(finishTimer);
  const role = turn.role;
  turn.finishing = true; controls();
  if (role === 'answer') {
    status('Reviewing the completed answer…');
    try { await queueAnalysis(true); if (turn) addHistory(); }
    catch (error) { showError('Could not analyze the answer: ' + error.message); }
  }
  if (!turn) return;
  releaseTurn();
  if (role === 'question') {
    $('question-text').textContent = question || 'No speech recognized. Please record the question again.';
    status(question ? 'Question captured. Record the candidate’s answer now.' : 'No question recognized; try again.');
    $('next-action').textContent = question ? 'Record candidate answer' : 'Record interviewer question';
  } else {
    $('answer-text').textContent = transcript || 'No speech recognized.';
    status('Answer complete. Review the transcript and observations, then record the next question.');
  }
}
function queueAnalysis(final) {
  const turnId = turn?.id, activeEpoch = epoch;
  const payload = {synthetic_profile:true,resume_skills:[...skills],question,transcript,final,speech_confidence:confidence,samples:[...samples],capture_interrupted:interrupted};
  analysisChain = analysisChain.catch(() => {}).then(async () => {
    if (!turn || turn.id !== turnId || epoch !== activeEpoch) return;
    const result = await post('/analyze/live', payload);
    if (!turn || turn.id !== turnId || epoch !== activeEpoch) return;
    latestAnalysis = result; renderSkills(result.skill_assessments);
    $('next-action').textContent = result.next_action;
    const capture = result.capture;
    $('sync-state').textContent = {UNASSESSED:'Unassessed',EXCLUDED_CAPTURE_QUALITY:'Excluded: capture quality',INSUFFICIENT_ACTIVITY:'Insufficient activity',TIMING_REVIEW:'Check device timing',NO_CLEAR_OFFSET_OBSERVED:'No clear offset observed'}[capture.lip_sync_status] || 'Unassessed';
    if (capture.blink_count !== null) $('blink-count').textContent = String(capture.blink_count);
    $('observations').replaceChildren();
    for (const text of capture.observations.length ? capture.observations : ['Insufficient capture for timing observations.']) { const li = document.createElement('li'); li.textContent = text; $('observations').append(li); }
  });
  return analysisChain;
}
function addHistory() {
  if (!latestAnalysis) return;
  historyCount++; $('history-count').textContent = String(historyCount); $('history-empty').hidden = true;
  const entry = document.createElement('details'); entry.className = 'history-item';
  const title = document.createElement('summary'); title.textContent = `Answer ${historyCount} · ${question.slice(0,120) || 'Question not recognized'}`;
  entry.append(title);
  for (const text of [transcript || 'No speech recognized.', ...latestAnalysis.skill_assessments.filter(a => a.status !== 'NOT_ASKED').map(a => `${a.skill}: ${labels[a.status]} — ${a.message}`)]) { const p = document.createElement('p'); p.textContent = text; entry.append(p); }
  $('history').prepend(entry);
  while ($('history').children.length > 10) $('history').lastElementChild.remove();
}
async function stopDevices() {
  stopping = true; epoch++; interrupted = true;
  if (turn) { question = ''; transcript = ''; latestAnalysis = null; $('answer-text').textContent = 'Interrupted capture discarded.'; renderSkills(); }
  releaseTurn(); clearInterval(visionTimer); face?.close(); face = null;
  processor?.disconnect(); processor = null;
  const oldContext = context; context = null; if (oldContext) await oldContext.close().catch(() => {});
  stream?.getTracks().forEach(track => track.stop()); stream = null;
  $('camera').srcObject = null; $('camera-empty').hidden = false; $('vision-state').textContent = 'NOT STARTED';
  preparing = false; stopping = false; status('Devices stopped. Completed session observations remain until reload.'); controls();
}
$('question-start').addEventListener('click', () => startTurn('question'));
$('answer-start').addEventListener('click', () => startTurn('answer'));
$('finish').addEventListener('click', finishTurn);
$('disconnect').addEventListener('click', stopDevices);
window.addEventListener('pagehide', stopDevices);
document.addEventListener('visibilitychange', () => { if (document.hidden && stream) { stopDevices(); showError('Capture stopped when the page was hidden. Reconnect to continue.'); } });
controls();
