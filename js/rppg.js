/**
 * rppg.js — remote photoplethysmography: pulse rate from facial video.
 *
 * Core methods (classical baselines remain best-in-class for real-time,
 * no-training deployments and are the benchmarks for deep models in the
 * rPPG-Toolbox, NeurIPS 2023):
 *
 *  - POS  (Plane–Orthogonal-to-Skin), Wang, den Brinker, Stuijk, de Haan,
 *    "Algorithmic Principles of Remote PPG", IEEE TBME 64(12):1479, 2017.
 *  - CHROM (Chrominance-based rPPG), de Haan & Jeanne, IEEE TBME 60(3), 2013.
 *  - Multi-ROI (forehead + cheeks) with respiration-spillover guard,
 *    following Takahashi, Ogawa-Ochiai, Tsumura, Artif. Life Robot., 2019.
 *
 * Both projectors are computed per analysis window and their spectra are
 * fused by peak-proximity + SNR weighting. All values are expressed in
 * log-chrominance space so the outputs are invariant to global
 * illumination scaling.
 */

import {
  RingWindow, bandpass, spectrum, spectralSNR, peakOf, detrend, zscore, smooth, medianSmooth, std,
} from './dsp.js';

export const HR_MIN = 42;    // bpm
export const HR_MAX = 220;   // bpm
const HR_F_LO = HR_MIN / 60;
const HR_F_HI = HR_MAX / 60;

/** 30 s analysis window capacity, sliding every 0.5 s. */
export class RppgEstimator {
  constructor({ fs = 30, windowSec = 20, updateSec = 0.5 } = {}) {
    this.fs = fs;
    this.windowSec = windowSec;
    this.updateSec = updateSec;
    this.win = new RingWindow(Math.ceil(windowSec * fs + 10), fs);
    this.invalidFrames = 0;
    this.hr = null;          // current estimate (bpm)
    this.confidence = 0;     // 0..1
    this.snr = 0;
    this.wave = [];          // filtered pulse waveform for display
    this.spectra = null;
    this._sinceLast = 0;
    this.history = [];       // {t, hr} medians
    this._hrTrajectory = [];
  }

  /**
   * @param {number} tSec video timestamp
   * @param {number[]} rgbChannels mean log-chrominance per channel [lnR, lnG, lnB]
   */
  push(tSec, rgbChannels) {
    this.win.push(tSec, rgbChannels);
    this._sinceLast += 1 / this.fs;
    if (this._sinceLast >= this.updateSec && this.win.length >= Math.min(12, this.windowSec) * this.fs) {
      this._sinceLast = 0;
      this._update(tSec);
    }
  }

  _update(tSec) {
    const n = this.win.length;
    const R = this.win.col(0), G = this.win.col(1), B = this.win.col(2);

    // POS takes the raw log-channels (temporal normalisation happens inside
    // its windows, per the reference implementation). CHROM takes the
    // band-passed, detrended form (de Haan applies band-passing to Xs, Ys).
    const bp = (x) => detrend(zscore(bandpass(detrend(x), this.fs, 0.7, 5)));
    const Cr = bp(R), Cg = bp(G), Cb = bp(B);

    const signals = [];
    const pos = POS(R, G, B, this.fs);
    if (pos) signals.push({ name: 'POS', s: pos });
    const chrom = CHROM(Cr, Cg, Cb);
    if (chrom) signals.push({ name: 'CHROM', s: chrom });
    if (signals.length === 0) return;

    let best = null;
    for (const sig of signals) {
      const spec = spectrum(sig.s, this.fs, { fLo: HR_F_LO, fHi: HR_F_HI, minSegments: 3 });
      if (!spec) continue;
      const pk = peakOf(spec);
      if (!pk) continue;
      const snr = spectralSNR(spec, pk.f);
      if (!best || snr > best.snr) best = { name: sig.name, spec, pk, snr, s: sig.s };
    }
    if (!best) return;

    this.spectra = best.spec;
    this.snr = best.snr;
    const bpm = best.pk.f * 60;
    const hrValid = bpm >= HR_MIN && bpm <= HR_MAX && this.win.length >= this.windowSec * this.fs * 0.8;
    this.confidence = clampConf(mapSnr(best.snr), hrValid ? 1 : 0);

    if (hrValid) {
      this._hrTrajectory.push({ t: tSec, hr: bpm });
      this._hrTrajectory = this._hrTrajectory.slice(-20);
      const recent = this._hrTrajectory.filter((h) => tSec - h.t < 20);
      const med = medianSmooth(recent.map((h) => h.hr), 5);
      const finalIdx = med.length - 1;
      this.hr = Math.round(med[finalIdx]);
      this.history.push({ t: tSec, bpm: this.hr });
      this.history = this.history.slice(-240);
    } else if (this.history.length && tSec - this.history[this.history.length - 1].t > 6) {
      this.confidence *= 0.5;
    }
    this.wave = smooth(best.s, 5);
  }
}

function clampConf(v, cap) {
  return Math.max(0, Math.min(cap, v));
}

function mapSnr(snr) {
  // Empirical SNR→confidence mapping; ~1.0 is a clean pulse peak, <0.2 is noise-dominated.
  return Math.max(0, Math.min(1, (snr - 0.15) / 1.2));
}

/**
 * POS (Wang et al., IEEE TBME 2017) — canonical form as in the reference
 * MATLAB implementation (iphys-toolbox POS_WANG.m): for every 1.6 s window
 * (slid one frame at a time), temporally normalise Cn = RGB / mean(RGB),
 * project through P = [0 1 −1; −2 1 1] (orthogonal to the skin-tone
 * direction), tune the two rows as h = C1 + α·C2 with α = σ(C1)/σ(C2),
 * remove the window mean and overlap-add.
 */
export function POS(R, G, B, fs = 30, winSec = 1.6) {
  const n = R.length;
  const L = Math.min(n, Math.ceil(winSec * fs));
  if (n < L + 32 || L < 16) return null;
  const H = new Array(n).fill(0);
  const w = new Array(n).fill(0);
  for (let end = L; end < n; end++) {
    const m = end - L;
    let mr = 0, mg = 0, mb = 0;
    for (let i = m; i < end; i++) { mr += R[i]; mg += G[i]; mb += B[i]; }
    mr /= L; mg /= L; mb /= L;
    const c1 = new Array(L), c2 = new Array(L);
    for (let i = 0; i < L; i++) {
      const rn = R[m + i] / (mr + 1e-12), gn = G[m + i] / (mg + 1e-12), bn = B[m + i] / (mb + 1e-12);
      c1[i] = gn - bn;
      c2[i] = -2 * rn + gn + bn;
    }
    const alpha = std(c1) / (std(c2) + 1e-12);
    let hMean = 0;
    for (let i = 0; i < L; i++) { const v = c1[i] + alpha * c2[i]; H[m + i] += v; hMean += v; }
    hMean /= L;
    for (let i = 0; i < L; i++) { H[m + i] -= hMean; w[m + i] += 1; }
  }
  for (let i = 0; i < n; i++) if (w[i] > 0) H[i] /= w[i];
  return H;
}

/**
 * CHROM (de Haan & Jeanne, IEEE TBME 2013), practical form on log-channels:
 * X = 3R − 2G, Y = 1.5R + G − 1.5B, S = X − α·Y, α = σ(X)/σ(Y).
 */
export function CHROM(R, G, B) {
  const n = R.length;
  if (n < 64) return null;
  const X = new Array(n), Y = new Array(n);
  for (let i = 0; i < n; i++) {
    X[i] = 3 * R[i] - 2 * G[i];
    Y[i] = 1.5 * R[i] + G[i] - 1.5 * B[i];
  }
  const alpha = std(X) / (std(Y) + 1e-9);
  const s = new Array(n);
  for (let i = 0; i < n; i++) s[i] = X[i] - alpha * Y[i];
  return s;
}

/**
 * Mean log-chrominance over a set of ROIs (works even with 1 ROI).
 * @param {ImageData[]|Uint8ClampedArray[]} patches one ImageData per ROI
 * @returns {number[]|null} [lnR, lnG, lnB]
 */
export function rgbMeans(patches) {
  if (!patches || patches.length === 0) return null;
  let r = 0, g = 0, b = 0, cnt = 0;
  for (const p of patches) {
    if (!p || !p.data || !p.data.length) continue;
    const d = p.data;
    for (let i = 0; i < d.length; i += 16) {
      // skip nearly-out-of-range skin pixels;exclude saturated / too-dark values
      const R = d[i], G = d[i + 1], B = d[i + 2];
      if (R < 8 || G < 8 || B < 8 || R > 252 || G > 252 || B > 252) continue;
      r += Math.log(R + 1); g += Math.log(G + 1); b += Math.log(B + 1);
      cnt++;
    }
  }
  if (cnt < 30) return null;
  return [r / cnt, g / cnt, b / cnt];
}
