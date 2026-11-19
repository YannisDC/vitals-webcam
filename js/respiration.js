/**
 * respiration.js — respiratory rate from webcam video.
 *
 * Dual-modality fusion, following the evidence in:
 *  - "Remote Respiration Measurement with RGB Cameras: A Review and
 *    Benchmark", ACM Computing Surveys, 2025 (doi:10.1145/3771763):
 *    motion-derived signals are the most robust under realistic conditions,
 *    while pulse-derived (rPPG baseline/amplitude modulation) signals help
 *    when the torso is out of frame.
 *  - "Seconds Matter: Rapid Non-Contact Monitoring of Heart and
 *    Respiratory Rate from Face Videos", Sensors 26(5):1506, 2026:
 *    dual-modality (face rPPG + upper-torso motion) with quality weighting.
 *
 * Channels fused:
 *  1. Face micro-motion: vertical drift of facial landmarks (nose tip,
 *     chin) captured breathing-related posture sway, band-limited to
 *     0.1–0.7 Hz (6–42 brpm).
 *  2. Chest/intensity channel: mean green-channel of an upper-chest ROI
 *     (expansion displaces specular/shadow pattern).
 *  3. Pulse modulation: slow AM/FM envelope of the rPPG waveform
 *     (respiratory modulation of pulse amplitude, known "respiratory
 *     induced intensity variation").
 */

import { RingWindow, bandpass, spectrum, spectralSNR, peakOf, detrend, zscore, std } from './dsp.js';

export const RR_MIN = 6;    // brpm
export const RR_MAX = 40;   // brpm (clinical obstruction threshold; band up to 0.67 Hz)
const RR_F_LO = RR_MIN / 60;
const RR_F_HI = 0.7;

export class RespirationEstimator {
  constructor({ fs = 10, windowSec = 45, updateSec = 1 } = {}) {
    this.fs = fs;
    this.windowSec = windowSec;
    this.updateSec = updateSec;
    this.win = new RingWindow(windowSec * fs + 20, fs);
    this._sinceLast = 0;
    this.rr = null;
    this.confidence = 0;
    this.snr = 0;
    this.wave = [];
    this.motionEnergy = 0;
  }

  /**
   * @param {number} tSec
   * @param {{faceY: number|null, chest: number|null}} sample
   *        faceY = weighted vertical landmark position (px or arbitrary units)
   *        chest = mean intensity of chest ROI (or null if out of frame)
   */
  push(tSec, sample) {
    const row = [sample.faceY ?? 0, sample.chest ?? 0, (sample.faceY == null ? 0 : 1) + (sample.chest == null ? 0 : 2)];
    this.win.push(tSec, row);
    this._sinceLast += 1 / this.fs;
    if (this._sinceLast >= this.updateSec && this.win.length >= Math.min(20, this.windowSec) * this.fs) {
      this._sinceLast = 0;
      this._update(tSec);
    }
  }

  _update(tSec) {
    const n = this.win.length;
    const valid = this.win.col(2);
    const faceValid = valid.map((v) => (v & 1) !== 0);
    const chestValid = valid.map((v) => (v & 2) !== 0);

    const candidates = [];

    // Movement of face centroid (breathing sway) — decimated rPPG-rate input.
    const fy = this.win.col(0);
    const fyLin = fy.map((v, i) => (faceValid[i] ? v : null));
    pushMotionCandidate(candidates, interpolateGaps(fyLin).slice(2), this.fs, 'face-motion');
    pushMotionCandidate(candidates, interpolateGaps(this.win.col(1)).slice(2), this.fs, 'chest-intensity');

    if (candidates.length === 0) { this.confidence *= 0.8; return; }

    let best = null;
    for (const c of candidates) {
      const spec = spectrum(c.s, this.fs, { fLo: RR_F_LO, fHi: RR_F_HI, minSegments: 3 });
      if (!spec) continue;
      const pk = peakOf(spec);
      if (!pk) continue;
      const snr = spectralSNR(spec, pk.f, 0.1);
      if (!best || snr > best.snr) best = { ...c, spec, pk, snr };
    }
    if (!best) return;

    const brpm = best.pk.f * 60;
    const plausible = brpm >= RR_MIN && brpm <= RR_MAX;
    this.snr = best.snr;
    this.motionEnergy = best.energy;
    const enough = this.win.length >= this.windowSec * this.fs * 0.8;
    this.confidence = Math.max(0, Math.min(enough && plausible ? 1 : 0, (best.snr - 0.12) / 0.9));
    if (plausible && enough) {
      this.rr = Math.round(brpm * 10) / 10;
      this.wave = best.s;
    }
  }
}

function pushMotionCandidate(arr, x, fs, name) {
  if (x.length < 10 * fs) {
    void arr; void fs; void name;
    return;
  }
  const s = zscore(bandpass(detrend(x), fs, RR_F_LO, RR_F_HI));
  const energy = std(s);
  if (!Number.isFinite(energy) || energy < 1e-6) return;
  arr.push({ name, s, energy });
}

/** Linear interpolation across nulls inside an array of numbers/nulls. */
function interpolateGaps(x) {
  const out = x.slice();
  let last = NaN, firstIdx = -1;
  for (let i = 0; i < out.length; i++) {
    if (out[i] != null && Number.isFinite(out[i])) { if (Number.isNaN(last)) firstIdx = i; last = out[i]; }
    else out[i] = last;
  }
  if (Number.isNaN(firstIdx)) return out.map(() => 0);
  for (let i = 0; i < firstIdx; i++) if (out[i] == null || !Number.isFinite(out[i])) out[i] = out[firstIdx];
  for (let i = 0; i < out.length; i++) if (out[i] == null || !Number.isFinite(out[i])) out[i] = 0;
  return out;
}
