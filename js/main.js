/**
 * main.js — orchestration: camera → face landmarks → ROI sampling →
 * pulse / respiration / blood-pressure estimators → UI.
 *
 * Everything runs in the browser; no frame ever leaves the device.
 */

import { createFaceLandmarker, roiBoxes, roiPixels } from './faceTracker.js';
import { RppgEstimator, rgbMeans } from './rppg.js';
import { RespirationEstimator } from './respiration.js';
import { BloodPressureEstimator } from './bloodPressure.js';

const RPPG_FS = 30;

const state = {
  running: false,
  landmarker: null,
  video: null,
  stream: null,
  lastVideoTime: -1,
  rppg: new RppgEstimator({ fs: RPPG_FS, windowSec: 20, updateSec: 0.5 }),
  resp: new RespirationEstimator({ fs: 10, windowSec: 45, updateSec: 1 }),
  bp: new BloodPressureEstimator({ fs: RPPG_FS }),
  samplesSinceRespPush: 0,
  lastFrameT: null,
  motionRaw: 0,
  faceLostSince: null,
};

const els = {};
['startBtn', 'video', 'overlay', 'hrValue', 'rrValue', 'bpValue', 'bpTag', 'qualityBar',
  'qualityText', 'pulseCanvas', 'respCanvas', 'statusText', 'calibrateBtn', 'calDialog',
  'calSbp', 'calDbp', 'calSave', 'calCancel', 'calClear', 'cardHr', 'cardRr', 'cardBp',
  'stabilize', 'stabilizeBar', 'progressFill', 'calibrateBtnLink'].forEach((id) => (els[id] = document.getElementById(id)));

/* ---------------- boot ---------------- */

els.startBtn.addEventListener('click', start);
els.calibrateBtnLink?.addEventListener('click', (e) => { e.preventDefault(); els.calibrateBtn.click(); });
els.calibrateBtn.addEventListener('click', () => {
  els.calSbp.value = state.bp.sbp ?? '';
  els.calDbp.value = state.bp.dbp ?? '';
  els.calDialog.showModal();
});
els.calSave.addEventListener('click', () => {
  const s = parseInt(els.calSbp.value, 10), d = parseInt(els.calDbp.value, 10);
  if (s > 50 && s < 230 && d > 30 && d < 150 && d < s) {
    state.bp.calibrate(s, d);
    setStatus(`Calibrated to cuff reading ${s}/${d} — estimates now tracked relative to this baseline.`);
  }
});
els.calClear.addEventListener('click', () => {
  state.bp.calibration = null;
  try { localStorage.removeItem('vitalscam.bp.calibration.v1'); } catch { /* ignore */ }
  state.bp.uncalibrated = true;
  setStatus('Calibration cleared.');
});

async function start() {
  els.startBtn.disabled = true;
  try {
    setStatus('Requesting camera…');
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user', frameRate: { ideal: 30, max: 60 } },
      audio: false,
    });
    state.video = els.video;
    state.video.srcObject = state.stream;
    await state.video.play();
    setStatus('Loading face model…');
    state.landmarker = await createFaceLandmarker();
    state.running = true;
    els.startBtn.textContent = 'Running';
    setStatus('Hold still, face the camera. Stabilising…');
    loop();
  } catch (err) {
    console.error(err);
    setStatus(`Could not start: ${err.message}. Note: the face model loads from a CDN on first launch.`);
    els.startBtn.disabled = false;
    els.startBtn.textContent = 'Retry';
  }
}

/* ---------------- capture loop ---------------- */

const work = document.createElement('canvas');
const workCtx = work.getContext('2d', { willReadFrequently: true });

function sampleRoi(src, roi, w, h) {
  const px = roiPixels(roi, w, h);
  work.width = px.w; work.height = px.h;
  workCtx.drawImage(src, px.x, px.y, px.w, px.h, 0, 0, px.w, px.h);
  return workCtx.getImageData(0, 0, px.w, px.h);
}

function loop() {
  if (!state.running) return;
  const v = state.video;
  const useRVFC = typeof v.requestVideoFrameCallback === 'function';
  const tick = () => {
    if (!state.running) return;
    processFrame();
    if (useRVFC) v.requestVideoFrameCallback(tick);
    else requestAnimationFrame(tick);
  };
  if (useRVFC) v.requestVideoFrameCallback(tick);
  else requestAnimationFrame(tick);
}

let overlaySetup = false;
function setupOverlay() {
  overlaySetup = true;
  const o = els.overlay;
  o.width = state.video.videoWidth || 640;
  o.height = state.video.videoHeight || 480;
}

function processFrame() {
  const v = state.video;
  if (v.readyState < 2) return;
  if (!overlaySetup) setupOverlay();
  const tMs = v.currentTime * 1000;
  const now = v.currentTime;
  if (now === state.lastFrameT) return;
  const dt = state.lastFrameT == null ? 1 / 30 : Math.min(0.5, now - state.lastFrameT);
  state.lastFrameT = now;

  let result = null;
  try {
    result = state.landmarker.detectForVideo(v, tMs);
  } catch (e) {
    /* occasional GPU hiccup — skip frame */
    return;
  }
  const lm = result?.faceLandmarks?.[0] ?? null;
  const boxes = roiBoxes(lm);
  const w = v.videoWidth, h = v.videoHeight;
  drawOverlay(boxes);

  if (!boxes) {
    if (!state.faceLostSince) state.faceLostSince = now;
    if (now - state.faceLostSince > 1) setStatus('No face — centre your face in the frame.');
    return;
  }
  state.faceLostSince = null;

  const patches = [
    sampleRoi(v, boxes.forehead, w, h),
    sampleRoi(v, boxes.cheekL, w, h),
    sampleRoi(v, boxes.cheekR, w, h),
  ];
  const rgb = rgbMeans(patches);
  if (!rgb) { setStatus('Move closer to the camera — face too small or poorly lit.'); return; }

  state.rppg.push(now, rgb);
  if (state.rppg.wave.length && state.rppg.wave[state.rppg.wave.length - 1] !== undefined && state.rppg.confidence > 0.05) {
    state.bp.push(now, state.rppg.wave[state.rppg.wave.length - 1], state.rppg.confidence);
  }

  // Decimate to 10 Hz for the respiration estimators.
  state.samplesSinceRespPush += dt;
  if (state.samplesSinceRespPush >= 1 / 10) {
    state.samplesSinceRespPush = 0;
    const faceY = lm ? (lm[1].y + lm[152].y) / 2 * h : null;
    const chestRoi = sampleRoi(v, boxes.chest, w, h);
    let chest = 0, cnt = 0;
    for (let i = 1; i < chestRoi.data.length; i += 32) { chest += chestRoi.data[i]; cnt++; }
    state.resp.push(now, { faceY, chest: cnt ? chest / cnt : null });
  }

  updateUi();
}

/* ---------------- drawing ---------------- */

function drawOverlay(boxes) {
  const o = els.overlay;
  const ctx = o.getContext('2d');
  ctx.clearRect(0, 0, o.width, o.height);
  if (!boxes) return;
  ctx.lineWidth = 2;
  const path = (roi, color) => {
    ctx.strokeStyle = color;
    ctx.strokeRect(roi.x * o.width, roi.y * o.height, roi.w * o.width, roi.h * o.height);
  };
  path(boxes.forehead, 'rgba(0,255,170,0.9)');
  path(boxes.cheekL, 'rgba(0,200,255,0.8)');
  path(boxes.cheekR, 'rgba(0,200,255,0.8)');
  path(boxes.chest, 'rgba(255,190,60,0.55)');
}

function drawWave(canvas, data, color, fillUpTo = null) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(255,255,255,0.04)';
  ctx.fillRect(0, 0, W, H);
  if (!data || data.length < 2) {
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font = `${12 * devicePixelRatio}px system-ui`;
    ctx.fillText('waiting for signal…', 12, H / 2);
    return;
  }
  let lo = Infinity, hi = -Infinity;
  for (const v of data) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const span = Math.max(1e-6, hi - lo);
  ctx.beginPath();
  const step = W / (data.length - 1);
  data.forEach((v, i) => {
    const x = i * step;
    const y = H - ((v - lo) / span) * (H - 8) - 4;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6 * devicePixelRatio;
  ctx.stroke();
  if (fillUpTo != null) {
    ctx.lineTo(W, H); ctx.lineTo(0, H); ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, H);
    grad.addColorStop(0, color.replace('rgb', 'rgba').replace(')', ',0.25)'));
    grad.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = grad;
    ctx.fill();
  }
}

/* ---------------- UI ---------------- */

function qualityColor(c) {
  return c > 0.6 ? '#34d399' : c > 0.3 ? '#fbbf24' : '#f87171';
}

let lastUiUpdate = 0;
function updateUi() {
  const now = performance.now();
  if (now - lastUiUpdate < 250) return;
  lastUiUpdate = now;

  const { rppg, resp, bp } = state;
  const conf = (rppg.confidence + (resp.confidence || 0) + bp.quality) / 3;
  els.qualityBar.style.width = `${Math.round(conf * 100)}%`;
  els.qualityBar.style.background = qualityColor(conf);
  els.qualityText.textContent = conf > 0.6 ? 'Good signal' : conf > 0.3 ? 'Weak — reduce motion / improve light' : 'Poor — hold still';

  if (rppg.hr != null) {
    els.hrValue.textContent = rppg.hr;
    els.cardHr.dataset.state = conf > 0.35 ? 'live' : 'weak';
  }
  if (resp.rr != null) {
    els.rrValue.textContent = resp.rr;
    els.cardRr.dataset.state = conf > 0.35 ? 'live' : 'weak';
  }
  if (bp.sbp != null) {
    els.bpValue.textContent = `${bp.sbp}/${bp.dbp}`;
    els.bpTag.textContent = bp.uncalibrated
      ? 'uncalibrated estimate — tap CALIBRATE with a cuff reading'
      : `calibrated · MAP ≈ ${bp.map} mmHg`;
  }

  // Stabilisation progress: time since start of adequate signal.
  const stable = rppg.confidence > 0.3 && rppg.history.length > 2;
  const elapsed = rppg.history.length ? (rppg.history[rppg.history.length - 1].t - rppg.history[0].t) : 0;
  const pct = Math.min(100, Math.round((elapsed / 15) * 100));
  els.progressFill.style.width = `${pct}%`;
  els.stabilize.textContent = stable && pct >= 100 ? 'Estimate stable — breathe normally' : `Stabilising… ${pct}%`;

  const lastT = rppg.history.length ? rppg.history[rppg.history.length - 1].t : now / 1000;
  const t0 = rppg.history.length ? rppg.history[0].t : 0;
  const wave = rppg.wave;
  drawWave(els.pulseCanvas, wave, 'rgb(52,211,153)');
  drawWave(els.respCanvas, resp.wave, 'rgb(96,165,250)');
  void lastT; void t0;
}

function setStatus(msg) { els.statusText.textContent = msg; }

window.addEventListener('beforeunload', () => {
  state.running = false;
  state.stream?.getTracks().forEach((t) => t.stop());
});
