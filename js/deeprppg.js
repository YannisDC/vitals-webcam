/**
 * deeprppg.js — OPTIONAL on-device deep rPPG via ONNX Runtime Web.
 *
 * Deep end-to-end models (PhysNet, TS-CAN, EfficientPhys, PhysFormer) beat
 * classical rPPG under motion and challenging light when trained per the
 * rPPG-Toolbox protocol (Liu et al., NeurIPS 2023). Running the big multi-
 * frame backbones in a browser is heavy, so this module supports the
 * lightweight **TS-CAN / EfficientPhys** contract:
 *
 *   input  : float32 [1, T=32, 3, 9, 9]  — 32 stacked frame *difference*
 *            crops (9×9 mean over the forehead patch; EfficientPhys-style
 *            motion-difference input, per-channel chunk standardisation)
 *   output : float32 [1, T] BVP sequence (or [1,1] per-chunk scalar)
 *
 * The weights are NOT bundled. Export a TS-CAN/EfficientPhys checkpoint to
 * ONNX (e.g. from the rPPG-Toolbox) and load it in the settings panel —
 * the model then runs locally via WebGPU (with a WASM fallback) and its
 * estimate is fused with the classical POS/CHROM result.
 */

import { RingWindow, spectrum, peakOf, spectralSNR, zscore } from './dsp.js';

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.bundle.min.mjs';
export const T = 32;

/** Load onnxruntime-web (WebGPU with automatic WASM fallback). */
export async function loadOrt() {
  const ort = await import(/* @vite-ignore */ ORT_URL);
  ort.env.wasm.numThreads = 1; // no cross-origin-isolation in plain http.server
  return ort;
}

export class DeepRppg {
  constructor({ fs = 30 } = {}) {
    this.fs = fs;
    this.ready = false;
    this.hr = null;
    this.confidence = 0;
    this.snr = 0;
    this.bvp = [];
    this._session = null;
    this._ort = null;
    this._frames = new RingWindow(T + 2, fs); // 9×9×3 rows
    this._prev = null;
    this._ring = [];        // diff frames waiting for a full chunk
    this._busy = false;
    this.error = null;
  }

  /**
   * @param {File|Uint8Array|string} source ONNX model
   */
  async load(source) {
    const ort = await loadOrt();
    this._ort = ort;
    const opts = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' };
    this._session = await ort.InferenceSession.create(source, opts);
    this.ready = true;
    // Validate the contract with a zero chunk; report shape mismatches early.
    const probe = await this._infer(new Float32Array(T * 3 * 9 * 9));
    if (!probe.ok) {
      this.ready = false;
      throw new Error(`model output rejected: ${probe.reason}`);
    }
  }

  get inputLayout() { return 'NTHWC: [1,32,3,9,9] float32'; }

  /** Feed one 9×9×3 mean-pooled forehead patch (Uint8ClampedArray-like). */
  async push(tSec, patch) {
    if (!this.ready || this._busy) return;
    const flat = new Float32Array(3 * 9 * 9);
    for (let i = 0; i < 9 * 9; i++) {
      flat[i * 3] = patch[i * 4];
      flat[i * 3 + 1] = patch[i * 4 + 1];
      flat[i * 3 + 2] = patch[i * 4 + 2];
    }
    this._frames.push(tSec, Array.from(flat));
    if (this._frames.length < T + 1) return;

    // EfficientPhys motion difference of the last two aligned frames.
    const { data } = this._frames.slice(T + 1);
    const prev = data[data.length - 2], cur = data[data.length - 1];
    const diff = new Float32Array(flat.length);
    for (let i = 0; i < flat.length; i++) diff[i] = (cur[i] - prev[i]) / 255;
    standardiseChunk(diff);
    this._ring.push(diff);
    if (this._ring.length < T) return;
    const chunk = this._ring.splice(0, T);

    this._busy = true;
    try {
      const tensor = new Float32Array(T * 3 * 9 * 9);
      chunk.forEach((f, ti) => tensor.set(f, ti * 3 * 9 * 9));
      const res = await this._infer(tensor);
      if (res.ok && res.bvp) this._absorbBvp(res.bvp);
    } catch (e) {
      this.error = String(e?.message ?? e);
      this.confidence = 0;
    } finally {
      this._busy = false;
    }
  }

  async _infer(chunk) {
    try {
      const input = new this._ort.Tensor('float32', chunk, [1, T, 3, 9, 9]);
      const feeds = {};
      feeds[this._session.inputNames[0]] = input;
      const out = await this._session.run(feeds);
      const first = this._session.outputNames[0];
      const o = out[first] ?? Object.values(out)[0];
      const dims = o.dims ?? [];
      const d = o.data;
      if (dims.length === 2 && d.length === 1) {
        // [1,1] per-chunk scalar: expand into a T-length pseudo-BVP? Accept
        // only heart-rate readout from bvp accumulation — treat as point.
        return { ok: true, bvp: null, scalar: d[0] };
      }
      if (d.length >= T) return { ok: true, bvp: Array.from(d.slice(0, T)) };
      return { ok: false, reason: `unexpected dims ${JSON.stringify(dims)}` };
    } catch (e) {
      return { ok: false, reason: String(e?.message ?? e) };
    }
  }

  _absorbBvp(bvpVals) {
    this.bvp = this.bvp.concat(bvpVals).slice(-this.fs * 30);
    if (this.bvp.length < 10 * this.fs) return;
    const spec = spectrum(this.bvp, this.fs, { fLo: 0.7, fHi: 3.667, minSegments: 3 });
    const pk = peakOf(spec);
    if (!pk) return;
    const snr = spectralSNR(spec, pk.f);
    const bpm = pk.f * 60;
    this.snr = snr;
    if (bpm >= 42 && bpm <= 220) {
      this.hr = Math.round(bpm);
      this.confidence = Math.max(0, Math.min(1, (snr - 0.15) / 1.2));
    }
  }
}

/** Per-channel standardisation across the chunk (EfficientPhys pre-proc). */
export function standardiseChunk(diff) {
  for (let c = 0; c < 3; c++) {
    const col = [];
    for (let i = 0; i < diff.length / 3; i++) col.push(diff[i * 3 + c]);
    const z = zscore(col);
    for (let i = 0; i < z.length; i++) diff[i * 3 + c] = z[i];
  }
  return diff;
}

/**
 * Fusion of classical and deep estimates: agreement boosts confidence;
 * disagreement falls back to whichever source is more confident.
 */
export function fuseHR(classical, deep) {
  if (!deep || deep.hr == null) return { hr: classical?.hr ?? null, confidence: classical?.confidence ?? 0, source: 'classic' };
  if (!classical || classical.hr == null) return { hr: deep.hr, confidence: deep.confidence * 0.8, source: 'deep' };
  const agree = Math.abs(deep.hr - classical.hr) <= 3;
  if (agree) {
    const w = deep.confidence / (deep.confidence + classical.confidence + 1e-9);
    return {
      hr: Math.round(deep.hr * w + classical.hr * (1 - w)),
      confidence: Math.min(1, 0.5 * (deep.confidence + classical.confidence) + 0.35),
      source: 'fused',
    };
  }
  return classical.confidence >= deep.confidence
    ? { hr: classical.hr, confidence: classical.confidence, source: 'classic' }
    : { hr: deep.hr, confidence: deep.confidence * 0.8, source: 'deep' };
}
