# VitalsCam — webcam heart rate, respiration & blood pressure

Open-source, 100% client-side web app that estimates **heart rate**, **respiratory rate**
and an **exploratory blood-pressure estimate** from a regular webcam feed. No servers,
no uploads — every frame stays in your browser.

> **Not a medical device.** This is a research/wellness tool following the published
> rPPG literature. The blood-pressure number is experimental and only meaningful
> relative to a calibration cuff reading. Never make treatment decisions from it.

## Run

```bash
cd vitals-webcam
python3 -m http.server 8080        # or: npx serve
# open http://localhost:8080 (Chrome/Edge recommended)
```

Allow the camera, sit ~50–80 cm from the webcam in steady, even lighting, stay still,
and breathe normally. The first launch downloads the MediaPipe face-landmark model from
a CDN (~3 MB, then browser-cached); it works offline afterwards.

Run the synthetic-signal validation suite (no camera needed):

```bash
npm test          # equivalently: node tests/pipeline.test.mjs
```

The suite reconstructs known physiological signals (78 bpm pulse with realistic
green-dominant channel geometry, 16 brpm breathing, a windkessel-shaped arterial wave)
and asserts that every stage of the pipeline recovers the ground truth.

## What's inside (methods & the research behind them)

### Heart rate — remote photoplethysmography (rPPG)

- **ROIs:** forehead + both cheeks, tracked with MediaPipe Face Landmarker
  (478-point dense landmarks; Kartynnik et al., arXiv:1907.06724) — modern landmarks
  instead of the legacy Viola-Jones/SSD detectors, per the 2024 toolkit literature.
- **Signal extraction:** canonical **POS** (Plane–Orthogonal-to-Skin), Wang, den Brinker,
  Stuijk, de Haan, *IEEE TBME* 64(12):1479, 2017 — per-1.6 s-window temporal
  normalisation, projection through `[0 1 −1; −2 1 1]`, σ-ratio tuning, overlap-add —
  plus **CHROM** (de Haan & Jeanne, *IEEE TBME* 60(3), 2013) on band-passed
  log-chrominance channels. The stronger spectrum wins each round.
- **Rate readout:** Welch spectrum (Hann, 50% overlap, parabolic peak refinement),
  spectral-SNR-gated confidence, moving-median trajectory smoothing — the evaluation
  protocol used by the **rPPG-Toolbox** (Liu et al., *NeurIPS* 2023), the reference
  benchmark where deep models (PhysNet, TS-CAN, PhysFormer, RhythmFormer, PhysMamba)
  and classical POS/CHROM are compared.
- **Known limits from the robustness literature** (built into the confidence scoring):
  accuracy degrades with low illumination and elevated heart rates (*npj Digital
  Medicine*, 2025), motion and compression artifacts (Nguyen et al., *Comput. Biol.
  Med.* 179:108873, 2024), and can show skin-tone-related bias (Nowara et al.,
  CVPR Workshops 2020 meta-analysis). Typical seated, well-lit error: ~1–4 bpm MAE.

### Respiratory rate — dual modality

Following the 2025 *ACM Computing Surveys* review/benchmark (doi:10.1145/3771763) and
the dual-modality design of *Sensors* 26(5):1506, 2026, three channels are fused and
the most confident 0.1–0.7 Hz spectrum (6–42 brpm) wins:

1. **Facial micro-motion** — vertical drift of nasal/chin landmarks (breathing sway);
2. **Upper-chest intensity** — torso expansion modulates shading/specular pattern;
3. **Pulse modulation** — respiratory AM of the rPPG waveform.

### Blood pressure — experimental, calibration-anchored

Camera-only BP remains an open research problem; the 2025 exploratory clinical study
(arXiv:2503.00890) and *Annual Review of Biomedical Engineering* 24 (2022) both
conclude that single-source cuffless BP needs **individual calibration** and should be
treated as a trend instrument. This app implements the tractable end of that research
line (Takahashi, Ogawa-Ochiai & Tsumura 2019; van Putten & Bamford, CVPRW 2023):

- per-beat morphology from the filtered pulse wave: systolic upstroke time, pulse width,
  **Windkessel diastolic-decay time constant τ** (stiffness proxy), intersystolic interval;
- Enter a real cuff reading once (**CALIBRATE**): estimates are pinned to that baseline
  and track waveform-driven deviations with literature-derived within-subject
  sensitivities (stiffer/faster upstroke ⇒ higher SBP; longer diastolic τ ⇒ higher DBP);
- Without calibration it shows a coarse population estimate and says so, loudly.

### Heart-rate variability (RMSSD / SDNN / pNN50)

Beat times are taken from the filtered pulse wave with sub-sample parabolic
interpolation of systolic peaks; ectopic/implausible intervals (±35% from the
running median, standard screening) are rejected before computing SDNN,
RMSSD and pNN50 over a sliding 60 s window (Task Force standards, *Eur. Heart
J.* 17:354, 1996). Webcam rPPG quantises beats far more coarsely than a
tactile PPG, so HRV values carry reduced confidence — track trends, not
absolute readings.

### Optional deep rPPG (TS-CAN / EfficientPhys via ONNX Runtime Web)

The ⚙ model panel lets you load any ONNX model matching the lightweight
diff-frame contract used by EfficientPhys/TS-CAN:

- input `[1, 32, 3, 9, 9]` float32 — 32 stacked frame *difference* crops of
  the forehead patch (motion-difference input, per-channel chunk
  standardisation),
- output `[1, 32]` BVP sequence (per-chunk scalar `[1,1]` also accepted).

Weights are deliberately not bundled. Export a checkpoint from the
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) to ONNX, load it,
and inference runs locally on WebGPU (WASM fallback). The deep estimate is
fused with classical POS/CHROM: agreement boosts confidence, disagreement
falls back to the more reliable source. Deep models help most under motion
and poor light (rPPG-Toolbox, NeurIPS 2023).

### BP calibration history & drift plot

Every cuff calibration is stored (localStorage) with its waveform features.
The BP card shows a session drift plot (SBP red / DBP blue; dashed markers
where cuffs were taken) plus the running cuff history — the two-point regime
where cuffless BP behaves best as a trend instrument (Mukkamala et al. 2022).

### Signal-quality gating

Every displayed value carries a live confidence (spectral SNR → color-coded quality bar,
stabilisation progress). Estimates only display when plausibility gates pass — the
webcam rPPG equivalent of the quality labels used in the iBVP dataset (Joshi et al.,
*Electronics* 13:1334, 2024).

## Privacy

The camera stream is processed in-page by vanilla JS + MediaPipe; **no pixel, sample,
or derived value is ever transmitted**. The only network requests are the MediaPipe
model files on first load (self-hostable — swap the URLs in `js/faceTracker.js`).
The single stored datum is your optional BP calibration, in `localStorage`.

## Accuracy tips (from the literature)

- Diffuse daylight or a lamp in front of you; avoid backlight and flickering light.
- 30 fps camera, face filling ≥ 25% of the frame, hold still, breathe normally.
- Metal frames/glasses catch specular highlights; darker skin tones need more light
  to reach the same SNR (Nowara et al. 2020) — improve lighting rather than exposure.
- For BP: always calibrate with a validated upper-arm cuff at rest; recalibrate after
  medication or posture changes. Watch trends, not absolute numbers.

## Roadmap

- [x] Optional on-device deep rPPG (EfficientPhys/TS-CAN via ONNX Runtime Web) for
      motion robustness — done: `js/deeprppg.js`, load any ONNX export in the ⚙ panel.
- [x] HRV (RMSSD/SDNN/pNN50) from beat intervals — done: `js/hrv.js`.
- [x] BP calibration history with drift plot — done: multi-point cuff history, drift
      chart on the BP card.
- [ ] SpO2 and stress metrics (experimental, needs multi-wavelength signals or careful pupil/HRV features).
- [ ] Session recording/export of derived signals (CSV) for research use.

## License

MIT (see `LICENSE`). The MediaPipe face-landmarker model is Apache-2.0 (Google).
