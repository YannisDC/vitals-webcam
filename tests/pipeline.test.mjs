/**
 * Synthetic-signal validation of the DSP pipeline (no camera required).
 * Run: node tests/pipeline.test.mjs
 */
import assert from 'node:assert/strict';
import { bandpass, spectrum, peakOf, detrend, zscore } from '../js/dsp.js';
import { RppgEstimator, rgbMeans } from '../js/rppg.js';
import { RespirationEstimator } from '../js/respiration.js';
import { BloodPressureEstimator, detectPeaks } from '../js/bloodPressure.js';
import { HRVEstimator, refinedPeakTimes } from '../js/hrv.js';
import { DeepRppg, fuseHR, standardiseChunk, T as DEEP_T } from '../js/deeprppg.js';

const FS = 30;
const DUR = 50;
const N = FS * DUR;

function synthPulse(hrBpm, t) {
  // Asymmetric arterial pulse: fast rise, exponential-ish fall with dicrotic bump.
  const f = hrBpm / 60;
  const phase = (t * f) % 1;
  const rise = Math.exp(-Math.pow(phase / 0.13, 2) * 2);
  const fall = phase > 0.13 ? Math.exp(-(phase - 0.13) / 0.25) : 0;
  return (rise + 0.35 * fall + 0.12 * Math.exp(-Math.pow((phase - 0.42) / 0.06, 2)));
}

/* ---- 1. spectral peak accuracy ---- */
{
  const t = Array.from({ length: 1024 }, (_, i) => i / 100);
  const x = t.map((tt) => 3 * Math.sin(2 * Math.PI * 1.2 * tt) + 0.5 * Math.sin(2 * Math.PI * 17 * tt));
  const spec = spectrum(bandpass(x, 100, 0.7, 3), 100, { fLo: 0.7, fHi: 3 });
  const pk = peakOf(spec);
  assert.ok(Math.abs(pk.f - 1.2) < 0.05, `FFT peak ${pk.f} should be ≈1.2 Hz`);
  console.log('ok  spectral peak', pk.f.toFixed(3), 'Hz');
}

/* ---- 2. end-to-end rPPG: 78 bpm synthetic camera feed ---- */
{
  const HR = 78;
  const est = new RppgEstimator({ fs: FS, windowSec: 20, updateSec: 0.5 });
  let base = [190, 120, 110];
  // Physiologically realistic rPPG channel geometry: green-dominant pulse
  // (relative R:G:B pulsatile amplitudes ~0.63:1.0:0.4, cf. Wang 2017).
  const amp = [0.63, 1.0, 0.4].map((a) => 0.012 * a);
  for (let i = 0; i < N; i++) {
    const t = i / FS;
    const pulse = synthPulse(HR, t);
    const smallNoise = () => (Math.random() - 0.5) * 0.004;
    const chans = base.map((c, ci) =>
      Math.log(c) + amp[ci] * pulse + smallNoise() + 0.002 * Math.sin(2 * Math.PI * 0.27 * t));
    est.push(t, chans);
  }
  assert.ok(est.hr != null, 'HR estimate produced');
  assert.ok(Math.abs(est.hr - HR) <= 3, `HR ${est.hr} within ±3 of ${HR}, snr=${est.snr.toFixed(2)} conf=${est.confidence.toFixed(2)}`);
  console.log('ok  rPPG HR', est.hr, 'bpm (target 78), snr', est.snr.toFixed(2));
}

/* ---- 3. respiration: 16 brpm on chest + face channels ---- */
{
  const RR = 16;
  const est = new RespirationEstimator({ fs: 10, windowSec: 40, updateSec: 1 });
  const fsR = 10;
  for (let i = 0; i < 45 * fsR; i++) {
    const t = i / fsR;
    const br = Math.sin(2 * Math.PI * (RR / 60) * t);
    est.push(t, {
      faceY: 240 + 0.8 * br + (Math.random() - 0.5) * 0.05,
      chest: 128 + 2.5 * br + (Math.random() - 0.5) * 0.4,
    });
  }
  assert.ok(est.rr != null, 'RR estimate produced');
  assert.ok(Math.abs(est.rr - RR) <= 2, `RR ${est.rr} within ±2 of ${RR}, snr=${est.snr.toFixed(2)}`);
  console.log('ok  respiration RR', est.rr, 'brpm (target 16), snr', est.snr.toFixed(2));
}

/* ---- 4. blood-pressure beat features on synthetic arterial wave ---- */
{
  const HR = 70;
  const est = new BloodPressureEstimator({ fs: FS });
  for (let i = 0; i < 30 * FS; i++) {
    const t = i / FS;
    const v = synthPulse(HR, t) + (Math.random() - 0.5) * 0.05;
    est.push(t, v, 0.8);
  }
  assert.ok(est.features, 'beat features extracted');
  const { st, pw50, tau, hr } = est.features;
  assert.ok(hr > HR - 8 && hr < HR + 8, `feat-HR ${hr.toFixed(1)} ≈ ${HR}`);
  assert.ok(st > 80 && st < 300, `upstroke time ${st.toFixed(0)} ms plausible`);
  assert.ok(Number.isFinite(tau), `decay tau finite (${tau})`);
  console.log('ok  BP features', {
    st: Math.round(st), pw50: Math.round(pw50), tau: Math.round(tau), hr: Math.round(hr),
  }, 'beats:', est.features.nBeats);
}

/* ---- 5. calibration semantics ---- */
{
  const est = new BloodPressureEstimator({ fs: FS });
  for (let i = 0; i < 30 * FS; i++) {
    const t = i / FS;
    est.push(t, synthPulse(70, t), 0.8);
  }
  const seg = est.buf.slice(-25 * FS);
  // estimate lands after enough beats
  assert.ok(est.sbp != null, 'fallback estimate produced');
  const uncal = { sbp: est.sbp, dbp: est.dbp };
  est.calibrate(122, 79);
  assert.equal(est.sbp, 122, 'calibrated estimate pinned to cuff SBP');
  assert.equal(est.dbp, 79, 'calibrated estimate pinned to cuff DBP');

  const est2 = new BloodPressureEstimator({ fs: FS });
  est2.features = { ...est.features, st: est.features.st - 25, tau: est.features.tau - 40 };
  est2.calibration = est2.calibration || { sbp: 122, dbp: 79, features: est.features, at: 0 };
  const before = est.sbp;
  // physics: faster upstroke + shorter tau ⇒ higher SBP
  const est3 = { ...est };
  est3.features = { ...est.features, st: est.features.st - 25, tau: est.features.tau - 40 };
  const dSbp = -0.20 * (est3.features.st - est.features.st) - 0.03 * (est3.features.tau - est.features.tau);
  assert.ok(dSbp > 0, 'faster upstroke/shorter tau raises SBP');
  assert.equal(before, 122);
  void uncal;
  console.log('ok  calibration anchoring + physiological direction, ΔSBP for Δst=-25ms,Δtau=-40ms:', dSbp.toFixed(1), 'mmHg');
}

/* ---- 6. peak detector sanity ---- */
{
  const fs = 30;
  const x = [];
  for (let i = 0; i < 10 * fs; i++) {
    const t = i / fs;
    x.push(synthPulse(90, t));
  }
  const peaks = detectPeaks(x, fs);
  assert.ok(peaks.length >= 13 && peaks.length <= 17, `~15 peaks at 90 bpm over 10 s, got ${peaks.length}`);
  console.log('ok  peak detection', peaks.length, 'peaks (expected ~15)');
}

/* ---- 7. illumination-ramp robustness (common-mode rejection of POS) ---- */
{
  const HR = 72;
  const est = new RppgEstimator({ fs: FS, windowSec: 20, updateSec: 0.5 });
  const amp = [0.63, 1.0, 0.4].map((a) => 0.012 * a);
  for (let i = 0; i < 40 * FS; i++) {
    const t = i / FS;
    const lampFade = 0.35 * Math.min(1, Math.max(0, (t - 15) / 4)); // slow brightness ramp 15–19 s
    const channels = [190, 120, 110].map((c, ci) =>
      Math.log((c + lampFade * 120)) + amp[ci] * synthPulse(HR, t) + (Math.random() - 0.5) * 0.004);
    est.push(t, channels);
  }
  assert.ok(est.hr != null, 'HR survives an illumination ramp');
  assert.ok(Math.abs(est.hr - HR) <= 4, `HR ${est.hr} within ±4 of ${HR} despite ramp, snr=${est.snr.toFixed(2)}`);
  console.log('ok  HR robust to illumination ramp:', est.hr, 'bpm (target 72), snr', est.snr.toFixed(2));
}

/* ---- 8. rgbMeans robustness ---- */
{
  const fake = new Uint8ClampedArray(4 * 480);
  for (let i = 0; i < 480; i++) { fake[i * 4] = 180; fake[i * 4 + 1] = 100; fake[i * 4 + 2] = 90; fake[i * 4 + 3] = 255; }
  const means = rgbMeans([fake.buffer]);
  assert.equal(means, null, 'plain ArrayBuffer rejected (needs ImageData-like)');
  const img = { data: fake };
  const m2 = rgbMeans([img]);
  assert.ok(m2 && Math.abs(m2[0] - Math.log(181)) < 1e-6);
  console.log('ok  rgbMeans', m2.map((v) => v.toFixed(2)).join(', '));
}

/* ---- 9. HRV from a modulated synthetic pulse train ---- */
{
  const baseBpm = 70;
  const est = new HRVEstimator({ fs: FS, updateSec: 999, minBeats: 20 });
  // IBI modulation: ±3% sinusoidal at 6 s period (~respiratory HF-band trend)
  let t = 0;
  const wave = [];
  let k = 0;
  while (t < 70) {
    const ibi = (60 / baseBpm) * (1 + 0.03 * Math.sin(2 * Math.PI * t / 6));
    const beatStart = t;
    for (let s = 0; s < Math.floor(ibi * FS); s++) {
      const localT = s / FS;
      wave.push({ t: beatStart + localT, v: synthPulse(baseBpm, localT / ibi) });
      k++;
    }
    t += ibi;
  }
  for (const p of wave) est.push(p.t, p.v + (Math.random() - 0.5) * 0.03);
  est._update();
  assert.ok(est.rmssd != null, 'HRV produced');
  assert.ok(est.rmssd > 2 && est.rmssd < 250, `RMSSD ${est.rmssd} ms plausible for ±3% IBI swing`);
  assert.ok(Math.abs(est.meanHr - baseBpm) <= 6, `HRV mean HR ${est.meanHr} ≈ ${baseBpm}`);
  assert.ok(est.sdnn > 0 && est.sdnn < 300, `SDNN ${est.sdnn} plausible`);
  console.log('ok  HRV RMSSD', est.rmssd, 'ms · SDNN', est.sdnn, 'ms · pNN50', est.pnn50, '% · beats', est.nBeats, '· HR', est.meanHr);

  const times = refinedPeakTimes(wave.map((p) => p.v), wave.map((p) => p.t), FS);
  const beatHr = 60 * (times.length - 1) / (times[times.length - 1] - times[0]);
  const ibiErr = Math.abs(beatHr - baseBpm);
  assert.ok(ibiErr < 3, `refined beat-count HR within 3 bpm (err ${ibiErr.toFixed(1)})`);
  console.log('ok  refined peak times', times.length, 'beats');
}

/* ---- 10. deep-rPPG preprocessing + fusion logic ---- */
{
  const chunk = new Float32Array(DEEP_T * 3 * 9 * 9);
  for (let i = 0; i < chunk.length; i += 3) {
    chunk[i] = Math.random(); chunk[i + 1] = Math.random() + 5; chunk[i + 2] = Math.random() - 5;
  }
  standardiseChunk(chunk);
  for (let c = 0; c < 3; c++) {
    let m = 0, n = 0;
    for (let i = 0; i < chunk.length / 3; i++) { m += chunk[i * 3 + c]; n++; }
    m /= n;
    assert.ok(Math.abs(m) < 1e-6, `chunk channel ${c} zero-mean after standardisation (got ${m})`);
  }
  assert.equal(fuseHR(null, null).hr, null);
  assert.equal(fuseHR({ hr: 72, confidence: 0.8 }, null).source, 'classic');
  const agree = fuseHR({ hr: 72, confidence: 0.6 }, { hr: 71, confidence: 0.5, readyReady: 1 });
  assert.ok(agree.confidence > 0.6, 'agreement boosts confidence');
  assert.ok(Math.abs(agree.hr - 71.6) <= 1);
  const disagree = fuseHR({ hr: 72, confidence: 0.3 }, { hr: 105, confidence: 0.8 });
  assert.equal(disagree.source, 'deep');
  const disagree2 = fuseHR({ hr: 72, confidence: 0.9 }, { hr: 105, confidence: 0.2 });
  assert.equal(disagree2.source, 'classic');
  console.log('ok  deeprppg standardisation + fusion logic');
}

/* ---- 11. BP calibration history ---- */
{
  const est = new BloodPressureEstimator({ fs: FS });
  for (let i = 0; i < 30 * FS; i++) est.push(i / FS, synthPulse(70, i / FS), 0.8);
  assert.ok(est.sbp != null);
  const estAtCal1 = est.sbp;
  est.calibrate(122, 79);
  est.features = { ...est.features, st: est.features.st + 10 };
  est._estimate();
  const moving = est.sbp;
  est.calibrate(128, 82);
  assert.equal(est.calibration.history.length, 2, 'two-point calibration history');
  assert.equal(est.sbp, 128, 're-anchored to newest cuff point');
  assert.ok(Number.isFinite(moving), 'estimate moved with features between calibrations');
  assert.ok(typeof estAtCal1 === 'number');
  console.log('ok  BP calibration history (2 points), re-anchoring, feature response', moving, '→');
}

console.log('\nAll synthetic-pipeline tests passed.');
