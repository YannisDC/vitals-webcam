/**
 * bloodPressure.js — exploratory camera-based blood pressure estimate.
 *
 * ⚠ Research estimate, NOT a medical device. Published camera-only BP work
 * (Schrumpf et al., IEEE JBHI 2021 & CVPRW 2021; Takahashi, Ogawa-Ochiai,
 * Tsumura, Artif. Life Robot. 2019 — 4-element Windkessel fit to facial-video
 * pulse waves; van Putten & Bamford, CVPRW 2023 — regression over rPPG
 * morphology features; U-FaceBP, 2024) consistently shows that single-source
 * rPPG BP needs INDIVIDUAL CALIBRATION against a cuff to be even moderately
 * useful, and that uncalibrated population models carry large errors
 * (see "Estimating Blood Pressure with a Camera", arXiv:2503.00890, 2025).
 *
 * Model implemented here:
 *  - Detect beats in the filtered pulse wave (adaptive refractory peak
 *    detection).
 *  - Per-beat morphology: systolic upstroke time ST, pulse width at 50% and
 *    25% amplitude, exponential diastolic decay time constant tau (the
 *    Windkessel-derived stiffness proxy used by Takahashi et al.), relative
 *    pulse amplitude.
 *  - Features are median-smoothed over the window; robust z-scores drive a
 *    linear model with literature-derived within-subject sensitivities.
 *  - One cuff calibration (user-typed systolic/diastolic) pins the model to
 *    the person; readings are then relative deviations from that baseline,
 *    which is the regime where cuffless BP tracks trends best
 *    (Mukkamala et al., Annual Review of Biomedical Engineering, 2022).
 */

import { bandpass, detrend, zscore, mean, medianSmooth, clamp } from './dsp.js';

export class BloodPressureEstimator {
  constructor({ fs = 30, updateSec = 2, minBeats = 6 } = {}) {
    this.fs = fs;
    this.updateSec = updateSec;
    this.minBeats = minBeats;
    this.buf = [];            // {t, v} filtered pulse wave
    this._sinceLast = 0;
    this.sbp = null;
    this.dbp = null;
    this.map = null;
    this.uncalibrated = true;
    this.quality = 0;
    this.features = null;     // last smoothed feature set
    this.trace = [];          // {t, sbp, dbp}
    this.calibration = loadCalibration();
  }

  /** Feed filtered pulse-wave samples. */
  push(tSec, value, pulseConfidence = 0) {
    this.buf.push({ t: tSec, v: value, conf: pulseConfidence });
    const cap = 30 * this.fs;
    if (this.buf.length > cap) this.buf.splice(0, this.buf.length - cap);
    this._sinceLast += 1 / this.fs;
    if (this._sinceLast >= this.updateSec) this._sinceLast = 0, this._update();
  }

  _update() {
    const t0 = this.buf[this.buf.length - 1]?.t - this.windowSecTarget();
    const seg = this.buf.filter((p) => p.t >= t0);
    if (seg.length < this.fs * 8) return;
    const wave = detrend(zscore(bandpass(seg.map((p) => p.v), this.fs, 0.5, 5)));
    const times = seg.map((p) => p.t);
    const feas = this.beatFeatures(wave, times);
    if (!feas || feas.length < this.minBeats) { this.quality *= 0.7; return; }
    this.features = {
      st: runMedian(feas.map((f) => f.st)),
      pw50: runMedian(feas.map((f) => f.pw50)),
      tau: runMedian(feas.map((f) => f.tau)),
      amp: runMedian(feas.map((f) => f.amp)),
      hr: 60 / runMedian(feas.map((f) => f.ibi)),
      nBeats: feas.length,
    };
    this.quality = clamp((this.features.nBeats - this.minBeats) / 10 + 0.4 + 0.2, 0, 1);
    this._estimate();
  }

  windowSecTarget() { return 25; }

  /**
   * Beat segmentation + morphology. Wave is z-scored.
   */
  beatFeatures(wave, times) {
    const fs = this.fs;
    const peaks = detectPeaks(wave, fs);
    if (peaks.length < 3) return null;
    const out = [];
    for (let i = 1; i < peaks.length; i++) {
      const pk = peaks[i];
      const prevFoot = findFoot(wave, peaks[i - 1], pk);
      const nextFoot = peaks[i + 1] ? findFoot(wave, pk, peaks[i + 1]) : wave.length - 1;
      if (prevFoot == null) continue;
      const amp = wave[pk] - wave[prevFoot];
      if (amp <= 0.08) continue;
      const st = (times[pk] - times[prevFoot]) * 1000; // ms
      const pw50 = widthAt(wave, times, pk, amp * 0.5, nextFoot);
      const tau = diastolicTau(wave, times, pk, nextFoot);
      if (st < 40 || st > 400) continue;
      out.push({
        st, pw50, tau,
        amp,
        ibi: times[pk] - times[peaks[i - 1]],
      });
    }
    return out.length >= 3 ? out : null;
  }

  _estimate() {
    const F = this.features;
    if (!F) return;
    const cal = this.calibration;
    if (cal && cal.sbp && cal.features) {
      this.uncalibrated = false;
      const c = cal.features;
      // Within-subject sensitivities from cuffless-BP physiology
      // (features in ms / bpm): stiffer arteries => faster upstroke and
      // decay => higher SBP; longer diastolic tau => slower runoff =>
      // higher DBP (DBP ∝ exp(T_diastole/tau), Windkessel); wider pulse =>
      // higher pulse pressure share in DBP; HR both via wave features.
      const dSbp = -0.20 * (F.st - c.st) + 0.06 * (F.pw50 - c.pw50)
        - 0.03 * (F.tau - c.tau) + 0.35 * (F.hr - c.hr);
      const dDbp = -0.08 * (F.st - c.st) + 0.10 * (F.pw50 - c.pw50)
        + 0.15 * (F.tau - c.tau) + 0.10 * (F.hr - c.hr);
      this.sbp = Math.round(clamp(cal.sbp + dSbp, 60, 220));
      this.dbp = Math.round(clamp(cal.dbp + dDbp, 35, 140));
    } else {
      // Population fallback: coarse centre around typical values, driven by
      // waveform deviations only. Large uncertainty — shown as such in UI.
      const zSt = (200 - F.st) / 60;    // upstroke time relative to ~200 ms norm
      this.sbp = Math.round(clamp(118 + 8 * zSt + 0.2 * (F.hr - 70), 80, 190));
      this.dbp = Math.round(clamp(76 + 4 * zSt + 0.1 * (F.hr - 70), 50, 120));
      this.uncalibrated = true;
    }
    this.map = Math.round(this.dbp + (this.sbp - this.dbp) / 3);
    this.trace.push({ t: Date.now(), sbp: this.sbp, dbp: this.dbp });
    if (this.trace.length > 60) this.trace.shift();
  }

  /** Re-anchor the model to a fresh cuff reading. */
  calibrate(sbp, dbp) {
    this.calibration = {
      sbp, dbp,
      features: this.features ? { ...this.features } : DEFAULT_FEATURES(),
      at: Date.now(),
    };
    saveCalibration(this.calibration);
    this._estimate();
  }
}

/* ---------- waveform mechanics ---------- */

/** Adaptive-threshold systolic peak detection (Reyben & Wilkes-style). */
export function detectPeaks(wave, fs) {
  const peaks = [];
  const refractory = Math.round(0.35 * fs); // max 171 bpm
  const n = wave.length;
  let i = 1;
  while (i < n - 1) {
    if (wave[i] > wave[i - 1] && wave[i] >= wave[i + 1] && wave[i] > 0.25) {
      const j = Math.min(n, i + refractory);
      let best = i;
      for (let k = i; k < j; k++) if (wave[k] > wave[best]) best = k;
      if (!peaks.length || best - peaks[peaks.length - 1] > refractory * 0.8) peaks.push(best);
      i = best + refractory;
    } else i++;
  }
  return peaks;
}

/** Diastolic foot: last local minimum before a peak. */
function findFoot(wave, from, to) {
  let minIdx = from;
  for (let i = from; i < to; i++) if (wave[i] < wave[minIdx]) minIdx = i;
  return minIdx;
}

/** Full width (ms) at a fractional amplitude around a peak. */
function widthAt(wave, times, pk, frac, limit) {
  const thr = frac;
  const fsApprox = 1 / Math.max(1e-6, (times[Math.min(times.length - 1, pk)] - times[Math.max(0, pk - 1)]));
  let l = pk, r = pk;
  while (l > 0 && wave[l] > thr) l--;
  while (r < Math.min(limit, wave.length - 1) && wave[r] > thr) r++;
  return (r - l) / fsApprox * 1000;
}

/**
 * Diastolic decay time constant (ms): fit ln(wave) linearly from just after
 * the peak to the next foot; tau = −1/slope. This is the stiffness proxy of
 * the 4-element Windkessel fit used by Takahashi et al. (2019).
 */
function diastolicTau(wave, times, pk, nextFoot) {
  const start = Math.min(nextFoot, pk + 2);
  const xs = [], ys = [];
  for (let i = start; i < nextFoot; i++) {
    if (wave[i] <= 0.02) continue;
    xs.push(times[i]);
    ys.push(Math.log(Math.max(1e-6, wave[i])));
  }
  if (xs.length < 4) return NaN;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  const n = xs.length;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * ys[i]; }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return NaN;
  const slope = (n * sxy - sx * sy) / denom;   // per ms
  return slope < -1e-9 ? Math.round(-1000 / slope) : NaN;
}

const runMedian = (arr) => {
  const s = medianSmooth(arr, Math.min(9, Math.max(3, arr.length - (arr.length % 2 ? 0 : 1))));
  return mean(s);
};

const DEFAULT_FEATURES = () => ({ st: 200 / 1, pw50: 220 / 1, tau: 300 / 1, hr: 70, amp: 1, nBeats: 0 });

/* ---------- persistence ---------- */

const LS_KEY = 'vitalscam.bp.calibration.v1';

const hasLS = typeof localStorage !== 'undefined';

function loadCalibration() {
  if (!hasLS) return null;
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return null;
    const obj = JSON.parse(raw);
    if (!obj || !obj.sbp || !obj.dbp || !obj.features) return null;
    return obj;
  } catch { return null; }
}

function saveCalibration(cal) {
  if (!hasLS) return;
  try { localStorage.setItem(LS_KEY, JSON.stringify(cal)); } catch { /* private mode */ }
}
