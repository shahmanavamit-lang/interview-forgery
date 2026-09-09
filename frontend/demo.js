'use strict';
const form = document.querySelector('#evaluation-form');
const fieldNames = ['gaze_diversion_ratio', 'synthetic_voice_confidence', 'lip_sync_variance_ms', 'network_latency_ms', 'network_jitter_ms', 'packet_loss_percent'];
const normalTranscript = 'In Python I use generators to yield values with an iterator and handle exceptions. In SQL I inspect the query plan, add an index, and use joins within a transaction.';
const highTranscript = 'I usually coordinate meetings and discuss schedules with the team. I cannot explain the technical implementation or describe how any of these systems work in practice.';
const error = document.querySelector('#error');
let revision = 0;
let pending = false;
function clearResult() {
  revision += 1;
  document.querySelector('#result').hidden = true;
  document.querySelector('#empty').hidden = false;
  error.hidden = true;
}
function loadCase(name) {
  document.querySelector('#skills').value = 'Python, SQL';
  document.querySelector('#transcript').value = name === 'high' ? highTranscript : normalTranscript;
  const values = name === 'high' ? [0.9, 0.95, 400, 0, 0, 0] : name === 'edge' ? [0.1, 0, 1500, 500, 200, 15] : [0.1, 0, 20, 0, 0, 0];
  fieldNames.forEach((field, i) => { document.getElementById(field).value = values[i]; });
  document.querySelector('#synthetic').checked = true;
  clearResult();
}
document.querySelectorAll('[data-case]').forEach(button => button.addEventListener('click', () => loadCase(button.dataset.case)));
form.addEventListener('input', clearResult);
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (pending || !form.reportValidity()) return;
  clearResult();
  const skills = document.querySelector('#skills').value.split(',').map(value => value.trim()).filter(Boolean);
  if (!skills.length || skills.length > 30 || skills.some(skill => skill.length > 100)) {
    error.textContent = 'Enter 1–30 comma-separated skills, each no longer than 100 characters.';
    error.hidden = false;
    return;
  }
  const payload = {synthetic_profile: document.querySelector('#synthetic').checked, resume_skills: skills,
    transcript: document.querySelector('#transcript').value.trim(),
    metadata: Object.fromEntries(fieldNames.map(field => [field, Number(document.getElementById(field).value)]))};
  const submittedRevision = revision;
  const button = document.querySelector('#evaluate');
  pending = true;
  button.disabled = true;
  button.textContent = 'Evaluating…';
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch('/evaluate', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload), signal: controller.signal});
    const data = await response.json();
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Unable to evaluate these inputs. Please check the fields.');
    if (revision !== submittedRevision) return;
    const score = data.decision?.risk_score;
    const tier = data.decision?.risk_tier;
    if (!Number.isInteger(score) || score < 0 || score > 100 || !['LOW','MEDIUM','HIGH'].includes(tier) || typeof data.reason !== 'string' || typeof data.next_action !== 'string') throw new Error('The server returned an unexpected response. Please try again.');
    document.querySelector('#score').textContent = score;
    document.querySelector('#tier').textContent = tier;
    document.querySelector('#tier').className = `tier ${tier.toLowerCase()}`;
    document.querySelector('#meter-fill').style.width = `${score}%`;
    document.querySelector('#meter-fill').style.background = {LOW:'#32865e',MEDIUM:'#bf820c',HIGH:'#c34646'}[tier];
    document.querySelector('#reason').textContent = data.reason;
    document.querySelector('#action').textContent = data.next_action;
    document.querySelector('#empty').hidden = true;
    document.querySelector('#result').hidden = false;
  } catch (failure) {
    if (revision === submittedRevision) {
      error.textContent = failure.name === 'AbortError' ? 'Evaluation timed out. Please try again.' : failure instanceof TypeError ? 'Cannot reach the backend. Check that the server is running and try again.' : failure.message;
      error.hidden = false;
    }
  } finally {
    clearTimeout(timeout);
    pending = false;
    button.disabled = false;
    button.textContent = 'Evaluate interview →';
  }
});
// Show the normal synthetic example, but require an explicit first confirmation.
loadCase('normal');
document.querySelector('#synthetic').checked = false;
