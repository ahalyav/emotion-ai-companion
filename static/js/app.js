/**
 * Emotion AI Companion — Interactive Frontend Module
 * Handles: real-time data polling, Chart.js radar + timeline,
 * Web Audio waveform, chat UI, toast notifications, keyboard shortcuts
 */

'use strict';

// -----------------------------------------------------------------------
// Toast Manager
// -----------------------------------------------------------------------
const Toast = {
  container: null,
  init() {
    this.container = document.getElementById('toast-container');
  },
  show(message, type = 'info', duration = 3500) {
    const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${icons[type] || '💬'}</span><span>${message}</span>`;
    if (!this.container) return;
    this.container.appendChild(el);
    setTimeout(() => {
      el.classList.add('removing');
      el.addEventListener('animationend', () => el.remove());
    }, duration);
  }
};

// -----------------------------------------------------------------------
// Emotion Colours & Emojis
// -----------------------------------------------------------------------
const EMOTION_META = {
  happy: { color: '#f5c518', emoji: '😄', cssClass: 'e-happy' },
  sad: { color: '#4da6ff', emoji: '😢', cssClass: 'e-sad' },
  angry: { color: '#f04747', emoji: '😠', cssClass: 'e-angry' },
  anger: { color: '#f04747', emoji: '😠', cssClass: 'e-angry' },
  surprise: { color: '#c27dff', emoji: '😲', cssClass: 'e-surprise' },
  fear: { color: '#36d1c4', emoji: '😰', cssClass: 'e-fear' },
  disgust: { color: '#57d68b', emoji: '🤢', cssClass: 'e-disgust' },
  neutral: { color: '#a0aec0', emoji: '😐', cssClass: 'e-neutral' },
};

function emotionMeta(e) {
  return EMOTION_META[e?.toLowerCase()] || EMOTION_META.neutral;
}

// -----------------------------------------------------------------------
// Confidence Ring renderer
// -----------------------------------------------------------------------
const ConfidenceRing = {
  circumference: 2 * Math.PI * 44,  // r=44
  update(pct) {
    const fill = document.getElementById('ringFill');
    const pctEl = document.getElementById('ringPct');
    if (!fill) return;
    const dash = this.circumference * (1 - pct / 100);
    if (fill) fill.style.strokeDashoffset = dash.toFixed(2);
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
  }
};

// -----------------------------------------------------------------------
// Chart.js — Radar (7-emotion scores)
// -----------------------------------------------------------------------
let radarChart = null;
function initRadarChart() {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;
  const labels = ['Happy', 'Sad', 'Angry', 'Surprise', 'Fear', 'Disgust', 'Neutral'];
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        label: 'Emotion Scores',
        data: Array(7).fill(0),
        backgroundColor: 'rgba(138,75,255,0.15)',
        borderColor: 'rgba(138,75,255,0.8)',
        pointBackgroundColor: 'rgba(138,75,255,1)',
        pointBorderColor: '#fff',
        pointRadius: 4,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      animation: { duration: 400 },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: {
            display: false,
            stepSize: 0.25,
          },
          grid: { color: 'rgba(255,255,255,0.08)' },
          angleLines: { color: 'rgba(255,255,255,0.08)' },
          pointLabels: {
            color: '#a0aec0',
            font: { size: 11, family: 'Inter' },
          }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${(ctx.raw * 100).toFixed(1)}%`
          }
        }
      }
    }
  });
}

function updateRadarChart(scores) {
  if (!radarChart || !scores) return;
  const order = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral'];
  radarChart.data.datasets[0].data = order.map(e => scores[e] || scores[e?.replace('angry', 'anger')] || 0);
  radarChart.update('none');
}

// -----------------------------------------------------------------------
// Chart.js — Timeline (60-second emotion confidence line)
// -----------------------------------------------------------------------
let timelineChart = null;
const TIMELINE_COLORS = {
  happy: '#f5c518', sad: '#4da6ff', angry: '#f04747',
  surprise: '#c27dff', fear: '#36d1c4', disgust: '#57d68b', neutral: '#a0aec0'
};

function initTimelineChart() {
  const ctx = document.getElementById('timelineChart');
  if (!ctx) return;
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Confidence',
        data: [],
        borderColor: 'rgba(138,75,255,0.8)',
        backgroundColor: 'rgba(138,75,255,0.07)',
        fill: true,
        tension: 0.4,
        pointRadius: 2,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      scales: {
        x: {
          display: false,
        },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#a0aec0', font: { size: 10 }, callback: v => `${(v * 100).toFixed(0)}%` }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `${(ctx.raw * 100).toFixed(1)}%`
          }
        }
      }
    }
  });
}

function updateTimelineChart(history) {
  if (!timelineChart || !history?.length) return;
  const recent = history.slice(-60);
  timelineChart.data.labels = recent.map((_, i) => i);
  timelineChart.data.datasets[0].data = recent.map(s => s.confidence || 0);
  // Color the line by dominant emotion
  const lastEmotion = recent[recent.length - 1]?.emotion || 'neutral';
  const col = TIMELINE_COLORS[lastEmotion] || '#a0aec0';
  timelineChart.data.datasets[0].borderColor = col + 'cc';
  timelineChart.data.datasets[0].backgroundColor = col + '15';
  timelineChart.update('none');
}

// -----------------------------------------------------------------------
// Chart.js — Extended Insight Timeline (Full Session history)
// -----------------------------------------------------------------------
let insightTimelineChart = null;

function initInsightTimelineChart() {
  const ctx = document.getElementById('insightTimelineChart');
  if (!ctx) return;
  insightTimelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Emotion Value',
        data: [],
        borderColor: 'rgba(54,209,196,0.8)',
        backgroundColor: 'rgba(54,209,196,0.1)',
        fill: true,
        tension: 0.2,
        pointRadius: 1,
        borderWidth: 1.5,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 200 },
      scales: {
        x: { display: false },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#a0aec0', font: { size: 9 }, callback: v => `${(v * 100).toFixed(0)}%` }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0].raw.emotion,
            label: ctx => `Conf: ${(ctx.raw.y * 100).toFixed(1)}%`
          }
        }
      }
    }
  });
}

function updateInsightTimelineChart(fullHistory) {
  if (!insightTimelineChart || !fullHistory?.length) return;

  insightTimelineChart.data.labels = fullHistory.map((_, i) => i);
  insightTimelineChart.data.datasets[0].data = fullHistory.map(entry => {
    return {
      x: entry.time,
      y: entry.confidence,
      emotion: entry.emotion
    };
  });
  insightTimelineChart.update('none');
}

// -----------------------------------------------------------------------
// Web Audio Waveform Visualizer
// -----------------------------------------------------------------------
const Waveform = {
  ctx: null,
  canvas: null,
  analyser: null,
  dataArray: null,
  animFrame: null,
  stream: null,
  audioCtx: null,

  async start() {
    try {
      this.canvas = document.getElementById('waveformCanvas');
      if (!this.canvas) return;
      this.ctx = this.canvas.getContext('2d');
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const src = this.audioCtx.createMediaStreamSource(this.stream);
      this.analyser = this.audioCtx.createAnalyser();
      this.analyser.fftSize = 256;
      src.connect(this.analyser);
      this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
      this.draw();
      return true;
    } catch (e) {
      console.warn('Waveform: microphone unavailable', e);
      return false;
    }
  },

  draw() {
    this.animFrame = requestAnimationFrame(() => this.draw());
    const c = this.canvas, ctx = this.ctx;
    const W = c.width = c.offsetWidth;
    const H = c.height = c.offsetHeight;
    this.analyser.getByteFrequencyData(this.dataArray);

    ctx.clearRect(0, 0, W, H);
    const barW = Math.max(2, (W / this.dataArray.length) * 2);
    let x = 0;
    const grad = ctx.createLinearGradient(0, H, 0, 0);
    grad.addColorStop(0, 'rgba(138,75,255,0.6)');
    grad.addColorStop(1, 'rgba(54,209,196,0.9)');
    ctx.fillStyle = grad;

    for (let i = 0; i < this.dataArray.length; i++) {
      const barH = (this.dataArray[i] / 255) * H;
      ctx.beginPath();
      ctx.roundRect(x, H - barH, barW - 1, barH, 2);
      ctx.fill();
      x += barW + 1;
    }
  },

  stop() {
    if (this.animFrame) cancelAnimationFrame(this.animFrame);
    if (this.stream) this.stream.getTracks().forEach(t => t.stop());
    if (this.audioCtx) this.audioCtx.close();
    this.analyser = null;
    this.stream = null;
    if (this.ctx && this.canvas) {
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
  }
};

// -----------------------------------------------------------------------
// Chat UI
// -----------------------------------------------------------------------
const Chat = {
  history: [],

  addMessage(role, text) {
    const box = document.getElementById('chatMessages');
    if (!box) return;
    // Remove typing indicator if present
    const typing = box ? box.querySelector('.typing') : null;
    if (typing) typing.remove();

    if (!box) return;
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    bubble.textContent = text;
    box.appendChild(bubble);
    box.scrollTop = box.scrollHeight;
    this.history.push({ role, content: text });
  },

  showTyping() {
    const box = document.getElementById('chatMessages');
    if (!box) return;
    const indicator = document.createElement('div');
    indicator.className = 'chat-bubble assistant typing';
    indicator.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
    box.appendChild(indicator);
    if (box) box.scrollTop = box.scrollHeight;
  },

  async send() {
    const input = document.getElementById('chatInput');
    const msg = input?.value.trim();
    if (!msg) return;
    input.value = '';
    this.addMessage('user', msg);
    this.showTyping();

    try {
      const resp = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: msg })
      });
      const data = await resp.json();
      this.addMessage('assistant', data.response || 'Sorry, I had trouble responding.');
    } catch (e) {
      this.addMessage('assistant', 'Network error. Please try again.');
    }
  }
};

// -----------------------------------------------------------------------
// Emotion State Poller
// -----------------------------------------------------------------------
const EmotionState = {
  current: {},
  scores: {},
  history: [],
  pollInterval: null,
  historyInterval: null,
  recordingActive: false,

  start() {
    this.pollInterval = setInterval(() => this.poll(), 400);
    this.historyInterval = setInterval(() => this.pollHistory(), 2000);
  },

  async poll() {
    try {
      const [emResp, scoreResp, statusResp, timelineResp] = await Promise.all([
        fetch('/api/emotion'),
        fetch('/api/emotion_scores'),
        fetch('/api/process_status'),
        fetch('/api/emotion_timeline') // Added timeline endpoint fetch
      ]);
      this.current = await emResp.json();
      const scoreData = await scoreResp.json();
      const status = await statusResp.json();
      const timelineData = await timelineResp.json();

      this.scores = scoreData.fused_scores || {};
      this.updateUI(status);

      if (timelineData.success && timelineData.timeline) {
        updateInsightTimelineChart(timelineData.timeline);
      }
    } catch (e) {
      // silently ignore network errors
    }
  },

  async pollHistory() {
    try {
      const resp = await fetch('/api/emotion_history');
      this.history = await resp.json();
      updateTimelineChart(this.history);
    } catch (e) { }
  },

  updateUI(status) {
    const d = this.current;
    const meta = emotionMeta(d.current_emotion);

    // Main emotion display
    const emojiEl = document.getElementById('mainEmoji');
    const labelEl = document.getElementById('mainLabel');
    if (emojiEl) emojiEl.textContent = meta.emoji;
    if (labelEl) {
      labelEl.textContent = d.current_emotion || 'neutral';
      labelEl.className = `emotion-label-big ${meta.cssClass}`;
    }
    ConfidenceRing.update((d.confidence || 0) * 100);

    // Mini cards
    this._updateMiniCard('audio', d.audio_emotion, d.audio_confidence);
    this._updateMiniCard('video', d.face_emotion, d.face_confidence);

    // Radar
    updateRadarChart(this.scores);

    // Face badge
    const badge = document.getElementById('faceBadge');
    if (badge) {
      if (d.face_detected) {
        badge.textContent = `${d.face_emotion} detected`;
        badge.classList.add('visible');
      } else {
        badge.classList.remove('visible');
      }
    }

    // Video overlay counters
    this._setText('videoFrames', status?.video_frames_processed || d.video_frames_processed || 0);
    this._setText('audioChunks', status?.audio_chunks_processed || d.audio_chunks_processed || 0);
    this._setText('faceDetected', d.face_detected ? '✅ Yes' : '—');
    this._setText('audioProcessing', status?.audio_processing || '—');

    // Status dots
    this._setDot('audioStatusDot', status?.audio_processing === 'Active');
    this._setDot('videoStatusDot', status?.video_processing === 'Active');
    this._setDot('fusionDot', true, 'ready');
    this._setDot('llmDot', d.llm_available, d.llm_available ? 'active' : 'ready');

    // System status text
    this._setText('systemStatusText', d.system_status || 'Ready');

    // Video feed
    if (d.video_feed) {
      const img = document.getElementById('videoFrame');
      const placeholder = document.getElementById('videoPlaceholder');
      if (img) {
        img.src = `data:image/jpeg;base64,${d.video_feed}`;
        img.style.display = 'block';
      }
      if (placeholder) placeholder.style.display = 'none';
    }
  },

  _updateMiniCard(type, emotion, confidence) {
    const val = document.getElementById(`${type}EmotionVal`);
    const conf = document.getElementById(`${type}ConfVal`);
    const fill = document.getElementById(`${type}ConfFill`);
    const meta = emotionMeta(emotion);
    if (val) { val.textContent = emotion || '—'; val.className = `mini-card-value ${meta.cssClass}`; }
    if (conf) conf.textContent = `${((confidence || 0) * 100).toFixed(1)}%`;
    if (fill) fill.style.width = `${(confidence || 0) * 100}%`;
  },

  _setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  },

  _setDot(id, active, forceClass) {
    const el = document.getElementById(id);
    if (!el) return;
    el.className = 'status-dot ' + (forceClass || (active ? 'active pulse-dot' : 'inactive'));
  }
};

// -----------------------------------------------------------------------
// Recording Timer
// -----------------------------------------------------------------------
let _recordingStart = 0;
let _timerInterval = null;

function startTimer() {
  _recordingStart = Date.now();
  _timerInterval = setInterval(() => {
    const s = Math.floor((Date.now() - _recordingStart) / 1000);
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    const el = document.getElementById('recordingTimer');
    if (el) el.textContent = `${m}:${sec}`;
  }, 1000);
}
function stopTimer() {
  clearInterval(_timerInterval);
  const el = document.getElementById('recordingTimer');
  if (el) el.textContent = '00:00';
}

// -----------------------------------------------------------------------
// Recording Controls
// -----------------------------------------------------------------------
async function startAllRecording() {
  const data = await fetchAPI('/api/start_all');
  if (data?.success) {
    startTimer();
    Waveform.start();
    setRecordingUI(true);
    Toast.show('Recording started — audio & video active', 'success');
  }
}
async function stopAllRecording() {
  const data = await fetchAPI('/api/stop_all');
  if (data?.success) {
    stopTimer();
    Waveform.stop();
    setRecordingUI(false);
    Toast.show('Recording stopped', 'info');
  }
}
async function startAudioRecording() {
  const data = await fetchAPI('/api/start_audio');
  if (data?.success) {
    startTimer();
    Waveform.start();
    setBtnState('btnStartAudio', false);
    setBtnState('btnStopAudio', true);
    Toast.show('Audio recording started 🎤', 'success');
  }
}
async function stopAudioRecording() {
  const data = await fetchAPI('/api/stop_audio');
  if (data?.success) {
    stopTimer();
    Waveform.stop();
    setBtnState('btnStartAudio', true);
    setBtnState('btnStopAudio', false);
    Toast.show('Audio recording stopped', 'info');
  }
}
async function startVideoRecording() {
  const data = await fetchAPI('/api/start_video');
  if (data?.success) {
    startTimer();
    setBtnState('btnStartVideo', false);
    setBtnState('btnStopVideo', true);
    Toast.show('Video recording started 📹', 'success');
  }
}
async function stopVideoRecording() {
  const data = await fetchAPI('/api/stop_video');
  if (data?.success) {
    stopTimer();
    setBtnState('btnStartVideo', true);
    setBtnState('btnStopVideo', false);
    Toast.show('Video recording stopped', 'info');
  }
}

async function getAIFeedback() {
  const feedbackEl = document.getElementById('aiFeedbackText');
  if (feedbackEl) feedbackEl.textContent = '🤔 Analyzing your emotions…';
  const data = await fetchAPI('/api/get_ai_feedback');
  if (feedbackEl) {
    feedbackEl.textContent = data?.ai_feedback || 'Could not get feedback at this time.';
  }
  Toast.show('AI feedback received 🤖', 'success');
}

async function saveAudioSession() {
  const data = await fetchAPI('/api/save_audio_session', 'POST');
  if (data?.success) {
    Toast.show(`Audio session saved — ${data.session.emotion}`, 'success');
    loadSessions();
  } else {
    Toast.show(data?.error || 'Could not save session', 'error');
  }
}
async function saveVideoSession() {
  const data = await fetchAPI('/api/save_video_session', 'POST');
  if (data?.success) {
    Toast.show(`Video session saved — ${data.session.emotion}`, 'success');
    loadSessions();
  } else {
    Toast.show(data?.error || 'Could not save session', 'error');
  }
}

async function loadSessions() {
  const data = await fetchAPI('/api/sessions');
  renderSessions('audioSessionsList', data?.audio_sessions || [], '🎵');
  renderSessions('videoSessionsList', data?.video_sessions || [], '📹');
}

function renderSessions(containerId, sessions, icon) {
  const el = document.getElementById(containerId);
  if (!el) return;
  if (!sessions.length) {
    el.innerHTML = '<p class="text-muted text-sm" style="padding:8px 0">No sessions yet</p>';
    return;
  }
  el.innerHTML = sessions.slice(-10).reverse().map(s => {
    const meta = emotionMeta(s.emotion);
    const dur = s.duration ? `${parseFloat(s.duration).toFixed(1)}s` : '—';
    const ts = s.timestamp ? new Date(s.timestamp).toLocaleTimeString() : '';
    return `
      <div class="session-item">
        <div class="session-icon">${icon}</div>
        <div class="session-info">
          <div class="s-emotion ${meta.cssClass}">${meta.emoji} ${s.emotion}</div>
          <div class="s-meta">${dur} &bull; ${ts}</div>
        </div>
        <span style="font-family:var(--font-mono,monospace);font-size:0.8em;color:var(--text-muted)">${((s.confidence || 0) * 100).toFixed(0)}%</span>
      </div>`;
  }).join('');
}

// -----------------------------------------------------------------------
// Keyboard Shortcuts
// -----------------------------------------------------------------------
function initKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    if (e.target.matches('input, textarea')) return;
    switch (e.key) {
      case '?': toggleShortcutsOverlay(); break;
      case 'Escape':
        closeShortcutsOverlay();
        stopAllRecording();
        break;
      case ' ':
        e.preventDefault();
        toggleAllRecording();
        break;
      case 'a': case 'A': startAudioRecording(); break;
      case 'v': case 'V': startVideoRecording(); break;
      case 'f': case 'F': getAIFeedback(); break;
    }
  });
}
let _isRecording = false;
function toggleAllRecording() {
  if (_isRecording) stopAllRecording(); else startAllRecording();
}
function toggleShortcutsOverlay() {
  const ov = document.getElementById('shortcuts-overlay');
  if (ov) ov.classList.toggle('visible');
}
function closeShortcutsOverlay() {
  const ov = document.getElementById('shortcuts-overlay');
  if (ov) ov.classList.remove('visible');
}

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------
async function fetchAPI(path, method = 'GET') {
  try {
    const resp = await fetch(path, { method });
    return await resp.json();
  } catch (e) {
    console.error(`fetchAPI ${path}:`, e);
    return null;
  }
}

function setBtnState(id, enabled) {
  const el = document.getElementById(id);
  if (el) el.disabled = !enabled;
}

function setRecordingUI(active) {
  _isRecording = active;
  setBtnState('btnStartAll', !active);
  setBtnState('btnStopAll', active);
  setBtnState('btnStartAudio', !active);
  setBtnState('btnStopAudio', active);
  setBtnState('btnStartVideo', !active);
  setBtnState('btnStopVideo', active);
  const wrapper = document.getElementById('appWrapper');
  if (wrapper) {
    if (active) wrapper.classList.add('recording');
    else wrapper.classList.remove('recording');
  }
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  Toast.init();
  initRadarChart();
  initTimelineChart();
  initInsightTimelineChart(); // Added init for new chart
  EmotionState.start();
  initKeyboardShortcuts();
  loadSessions();

  // Chat enter key
  const chatInput = document.getElementById('chatInput');
  if (chatInput) {
    chatInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        Chat.send();
      }
    });
  }
  const chatSendBtn = document.getElementById('chatSendBtn');
  if (chatSendBtn) chatSendBtn.addEventListener('click', () => Chat.send());

  // Nav tabs
  document.querySelectorAll('.nav-item[data-section]').forEach(item => {
    item.addEventListener('click', () => {
      document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
      item.classList.add('active');
    });
  });

  // Overlay close on backdrop click
  const ov = document.getElementById('shortcuts-overlay');
  if (ov) ov.addEventListener('click', e => {
    if (e.target === ov) closeShortcutsOverlay();
  });

  // Initial system status check
  fetchAPI('/api/system_status').then(data => {
    if (data) {
      console.log('System ready:', data);
      Toast.show('Emotion AI Companion ready 🚀', 'success');
    }
  });
});
