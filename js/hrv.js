/**
 * hrv.js — heart-rate variability from the rPPG pulse wave.
 *
 * Beat intervals are extracted from the filtered pulse wave with systolic
 * peak detection plus sub-sample parabolic interpolation of peak positions
 * (32–33 ms sampling grid refinement). Metrics follow standard short-term
 * HRV practice (Task Force of the European Society of Cardiology, Eur.
 * Heart J. 17:354, 1996): SDNN, RMSSD, pNN50 over a sliding window.
 *
 * Caveat baked into the confidence: webcam rPPG at 30 fps quantises the
 * pulse waveform more coarsely than a contact PPG, so absolute HRV values
 * are less accurate than the pulse rate itself — trends and session-level
 * comparisons are the intended use (cf. rPPG HRV literature: correlations
 * are moderate and depend on signal quality).
 */

import { std, mean } from './dsp.js';
import { detectPeaks } from './bloodPressure.js';

export class HRVEstimator {
  constructor({ fs = 30, updateSec = 1.5, minBeats = 20 } = {}) {
    this.fs = fs;
    this.updateSec = updateSec;
    this.minBeats = minBeats;
    this.buf = [];          // {t, v}
    this._sinceLast = 0;
    this.rmssd = null;      // ms
    this.sdnn = null;       // ms
    this.pnn50 = null;      // 0..1
    this.meanHr = null;     // bpm from IBIs
    this.nBeats = 0;
    this.quality = 0;
    this.rejectRate = 0;
  }

  /** @param {number} value filtered pulse-wave sample */
  push(tSec, value) {
    this.buf.push({ t: tSec, v: value });
    const cap = 75 * this.fs;
    if (this.buf.length > cap) this.buf.splice(0, this.buf.length - cap);
    this._sinceLast += 1 / this.fs;
    if (this._sinceLast >= this.updateSec) {
      this._sinceLast = 0;
      this._update();
    }
  }

  _update() {
    if (this.buf.length < this.fs * 20) return;
    const t0 = this.buf[this.buf.length - 1].t - this.windowTarget();
    const seg = this.buf.filter((p) => p.t >= t0);
    const times = refinedPeakTimes(seg.map((p) => p.v), seg.map((p) => p.t), this.fs);
    if (times.length < this.minBeats) { this.quality *= 0.7; return; }

    const ibis = [];
    for (let i = 1; i < times.length; i++) {
      const ibi = (times[i] - times[i - 1]) * 1000;
      ibis.push(ibi);
    }
    const medianIbi = median(ibis);
    const good = [];
    let rejected = 0;
    for (const ibi of ibis) {
      // Physiologic + ectopic screening: 1.5 s refractory cap and ±35% dev
      // from the running median (standard ectopic-beat rule of thumb).
      if (ibi >= 350 && ibi <= 1500 && Math.abs(ibi - medianIbi) < 0.35 * medianIbi) good.push(ibi);
      else rejected++;
    }
    this.rejectRate = rejected / ibis.length;
    if (good.length < this.minBeats - 1) { this.quality *= 0.7; return; }

    const diffs = [];
    for (let i = 1; i < good.length; i++) diffs.push(good[i] - good[i - 1]);
    this.sdnn = Math.round(std(good));
    this.rmssd = Math.round(Math.sqrt(mean(diffs.map((d) => d * d))));
    this.pnn50 = Math.round(100 * diffs.filter((d) => Math.abs(d) > 50).length / diffs.length);
    this.meanHr = Math.round(60 / (mean(good) / 1000));
    this.nBeats = good.length;
    const enough = seg.length >= this.windowTarget() * this.fs * 0.8;
    this.quality = Math.max(0, Math.min(1,
      (enough ? 0.6 : 0.3) + (1 - this.rejectRate) * 0.4));
  }

  windowTarget() { return 60; }
}

/**
 * Systolic peak times with parabolic sub-sample refinement (seconds).
 */
export function refinedPeakTimes(wave, times, fs) {
  const peaks = detectPeaks(wave, fs);
  const out = [];
  for (const k of peaks) {
    if (k <= 0 || k >= wave.length - 1) { out.push(times[k]); continue; }
    const y0 = wave[k - 1], y1 = wave[k], y2 = wave[k + 1];
    const denom = y0 - 2 * y1 + y2;
    const delta = denom !== 0 ? 0.5 * (y0 - y2) / denom : 0;
    const deltaClamped = Math.max(-0.5, Math.min(0.5, delta));
    const dt = times[k + 1] !== undefined ? times[k + 1] - times[k] : 1 / fs;
    out.push(times[k] + deltaClamped * dt);
  }
  return out;
}

function median(arr) {
  const s = arr.slice().sort((a, b) => a - b);
  return s[s.length >> 1];
}
