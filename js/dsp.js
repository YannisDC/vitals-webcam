/**
 * dsp.js — signal-processing primitives for camera-based vital signs.
 *
 * Implementations follow the methods used in the rPPG literature:
 *  - Hann-windowed Welch-style spectra for rate estimation.
 *  - RBJ audio-EQ biquad coefficients standing in for Butterworth
 *    band-pass sections (cascade of HP + LP) — the same filtering role
 *    Butterworth plays in Wang et al., "Algorithmic Principles of Remote
 *    PPG", IEEE TBME 64(12), 2016.
 *  - Robust per-beat waveform features for cuffless BP research
 *    (see bloodPressure.js / van Putten & Bamford, CVPRW 2023).
 */

/* ---------- buffers ---------- */

/** Uniform-rate sliding window with timestamp drift resampling. */
export class RingWindow {
  /**
   * @param {number} capacity max samples kept
   * @param {number} fs target uniform sampling rate (Hz)
   */
  constructor(capacity, fs) {
    this.capacity = capacity;
    this.fs = fs;
    this.t = [];     // uniform grid times (s)
    this.data = [];  // rows aligned with t (arrays of channel values)
    this._lastSourceT = null;
    this._lastSourceRow = null;
  }

  get length() { return this.t.length; }

  /** Push one frame; tSec monotonic source timestamps are resampled onto the uniform grid. */
  push(tSec, row) {
    const fs = this.fs;
    if (this._lastSourceT === null) {
      this._seedGrid(tSec, row);
      this._lastSourceT = tSec;
      this._lastSourceRow = row;
      return;
    }
    const dt = tSec - this._lastSourceT;
    if (dt <= 0) return; // ignore frame reordering
    const gridStart = this.t.length ? this.t[this.t.length - 1] + 1 / fs : tSec;
    const steps = Math.floor((tSec - (gridStart - 1 / fs)) * fs);
    for (let i = 0; i < steps; i++) {
      const gt = gridStart + i / fs;
      if (gt > tSec) break;
      const alpha = (gt - this._lastSourceT) / dt;
      const interp = this._lastSourceRow.map((v, c) => v + alpha * (row[c] - this._lastSourceRow[c]));
      this.t.push(gt);
      this.data.push(interp);
    }
    this._lastSourceT = tSec;
    this._lastSourceRow = row;
    this._trim();
  }

  _seedGrid(tSec, row) {
    for (let i = 0; i < 2; i++) {
      this.t.push(tSec - (1 - i) / this.fs);
      this.data.push(row.slice());
    }
  }

  _trim() {
    while (this.t.length > this.capacity) {
      this.t.shift();
      this.data.shift();
    }
  }

  /** Column copy for one channel. */
  col(c) {
    return this.data.map((r) => r[c]);
  }

  slice(count) {
    const start = Math.max(0, this.t.length - count);
    return { t: this.t.slice(start), data: this.data.slice(start) };
  }
}

/* ---------- preprocessing ---------- */

/** Remove per-channel mean and linear trend (de Haan rPPG practice). */
export function detrend(x) {
  const n = x.length;
  if (n < 3) return x.slice();
  const sx = (n - 1) * n / 2, sxx = (n - 1) * n * (2 * n - 1) / 6;
  let sy = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sy += x[i]; sxy += i * x[i]; }
  const denom = n * sxx - sx * sx;
  const a = denom !== 0 ? (n * sxy - sx * sy) / denom : 0;
  const b = denom !== 0 ? (sy - a * sx) / n : sy / n;
  return x.map((v, i) => v - (a * i + b));
}

/** Normalise (x - mean) / std, guarding zero variance. */
export function zscore(x) {
  const n = x.length;
  let m = 0;
  for (const v of x) m += v;
  m /= n;
  let s = 0;
  for (const v of x) s += (v - m) * (v - m);
  s = Math.sqrt(s / n);
  return s > 1e-9 ? x.map((v) => (v - m) / s) : x.map(() => 0);
}

/** Clip magnitude (robust artifact suppression, Rppg-Toolbox style). */
export function clipTail(x, zMax = 3) {
  const z = zscore(x);
  let m = 0; for (const v of x) m += v; m /= x.length;
  return x.map((v, i) => Math.min(zMax, Math.max(-zMax, z[i])) * 0.5 + m);
}

/* ---------- biquad filtering (RBJ cookbook: 12 dB/oct HP + LP) ---------- */

function biquadCoef(type, f0, fs, Q = Math.SQRT1_2) {
  const w0 = 2 * Math.PI * f0 / fs;
  const cw = Math.cos(w0), sw = Math.sin(w0);
  const alpha = sw / (2 * Q);
  let a0, a1, a2, b0, b1, b2;
  if (type === 'highpass') {
    b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = b0;
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
  } else if (type === 'lowpass') {
    b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = b0;
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
  } else {
    throw new Error('unsupported type');
  }
  return { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0 };
}

/** One-pass second-order IIR filter. */
export function biquad(x, coef) {
  let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
  const y = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const xi = x[i];
    y[i] = coef.b0 * xi + coef.b1 * x1 + coef.b2 * x2 - coef.a1 * y1 - coef.a2 * y2;
    x2 = x1; x1 = xi; y2 = y1; y1 = y[i];
  }
  return y;
}

/** Butterworth-like band-pass, zero-phase (forward + backward biquad cascade, filtfilt-style). */
export function bandpass(x, fs, fLo, fHi) {
  if (x.length < 6 || fHi <= fLo) return x.slice();
  const rev = (a) => a.slice().reverse();
  const hp = biquadCoef('highpass', fLo, fs);
  const lp = biquadCoef('lowpass', fHi, fs);
  const fb = (a, c) => biquad(rev(biquad(rev(a), c)), c); // forward + backward
  return fb(fb(x, hp), lp);
}

/** Moving-average smoother. */
export function smooth(x, k) {
  const half = Math.floor(k / 2);
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(x.length - 1, i + half); j++) { s += x[j]; c++; }
    out[i] = s / c;
  }
  return out;
}

/** Moving median (robust heart-rate trajectory stabiliser). */
export function medianSmooth(x, k) {
  const half = Math.floor(k / 2);
  const out = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const win = x.slice(Math.max(0, i - half), Math.min(x.length, i + half + 1)).sort((a, b) => a - b);
    out[i] = win[win.length >> 1];
  }
  return out;
}

/* ---------- spectral analysis ---------- */

export function hann(n) {
  return new Array(n).fill(0).map((_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)));
}

/** In-place iterative radix-2 FFT. `real`/`imag` length must be a power of two. */
export function fft(real, imag) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = real[i + k], ui = imag[i + k];
        const vr = real[i + k + len / 2] * cr - imag[i + k + len / 2] * ci;
        const vi = real[i + k + len / 2] * ci + imag[i + k + len / 2] * cr;
        real[i + k] = ur + vr; imag[i + k] = ui + vi;
        real[i + k + len / 2] = ur - vr; imag[i + k + len / 2] = ui - vi;
        const ncr = cr * wr - ci * wi;
        ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}

/** Welch spectrum: mean of Hann-windowed, 50%-overlapped segments. Non-zero freqs interpolated and normalised so sum(power)=1 over the band. */
export function spectrum(x, fs, opts = {}) {
  const { fLo = 0.1, fHi = null, minSegments = 4 } = opts;
  const n = x.length;
  if (n < 64) return null;
  let seg = 1 << Math.floor(Math.log2(n));
  while (seg > 64 && Math.floor((n - seg) / (seg >> 1)) + 1 < Math.min(minSegments, 2)) seg >>= 1;
  const hop = seg >> 1;
  const segments = Math.max(1, Math.floor((n - seg) / hop) + 1);
  if (segments < Math.min(minSegments, 2)) return null;
  const w = hann(seg);
  const hi = fHi ?? fs / 2;
  const nBins = seg >> 1;
  const acc = new Float64Array(nBins);
  for (let s = 0; s < segments; s++) {
    const off = s * hop;
    if (off + seg > n) break;
    const re = new Float64Array(seg), im = new Float64Array(seg);
    for (let i = 0; i < seg; i++) re[i] = x[off + i] * w[i];
    fft(re, im);
    for (let k = 0; k < nBins; k++) acc[k] += re[k] * re[k] + im[k] * im[k];
  }
  const df = fs / seg;
  const out = [];
  for (let k = 1; k < nBins; k++) {
    const f = k * df;
    if (f < fLo || f > hi) continue;
    out.push({ f, p: acc[k] / segments });
  }
  const total = out.reduce((s, b) => s + b.p, 0);
  if (total > 0) for (const b of out) b.p /= total;
  return { bins: out, df, segLen: seg };
}

/** Highest power bin, refined with a 3-point parabolic interpolation for sub-bin accuracy. */
export function peakOf(spec) {
  if (!spec || spec.bins.length === 0) return null;
  let best = 0;
  for (let i = 1; i < spec.bins.length; i++) if (spec.bins[i].p > spec.bins[best].p) best = i;
  const b = spec.bins;
  if (best === 0 || best === b.length - 1) return b[best];
  const { f: fm, p: pm } = b[best - 1];
  const { f: f0, p: p0 } = b[best];
  const { f: fp, p: pp } = b[best + 1];
  const denom = pm - 2 * p0 + pp;
  if (Math.abs(denom) < 1e-12) return b[best];
  const delta = 0.5 * (pm - pp) / denom * (fp - fm);
  return { f: f0 + delta, p: p0 };
}

/** Spectral SNR: power within ±win Hz of f0 against the rest of the band (Wang 2016 metric). */
export function spectralSNR(spec, f0, win = 0.2) {
  if (!spec) return 0;
  let sig = 0, noise = 0;
  for (const b of spec.bins) {
    if (Math.abs(b.f - f0) <= win) sig += b.p;
    else if (Math.abs(b.f - 2 * f0) < win) noise += 0.3 * b.p; // 2nd harmonic treated as signal-consistent
    else noise += b.p;
  }
  return sig / (noise + 1e-9);
}

/* ---------- generic helpers ---------- */

export function mean(x) { let s = 0; for (const v of x) s += v; return x.length ? s / x.length : 0; }

export function std(x) {
  const m = mean(x);
  let s = 0;
  for (const v of x) s += (v - m) * (v - m);
  return Math.sqrt(s / Math.max(1, x.length - 1));
}

export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
