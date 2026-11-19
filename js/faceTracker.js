/**
 * faceTracker.js — MediaPipe Face Landmarker (tasks-vision) wrapper.
 *
 * MediaPipe FaceMesh provides 478 landmarks in real time (Kartynnik et al.,
 * "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs",
 * arXiv:1907.06724) — the same localisation stack used by modern rPPG
 * pipelines instead of legacy Viola-Jones/SSD detectors.
 *
 * Model + wasm are loaded from CDN on first use and cached by the browser.
 */

import { FilesetResolver, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const WASM_ROOT = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

export async function createFaceLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_ROOT);
  let opts = {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numFaces: 1,
  };
  try {
    return await FaceLandmarker.createFromOptions(fileset, opts);
  } catch (e) {
    console.warn('GPU delegate unavailable, falling back to CPU', e);
    opts.baseOptions.delegate = 'CPU';
    return await FaceLandmarker.createFromOptions(fileset, opts);
  }
}

/**
 * Landmark indices for rPPG skin patches (FaceMesh topology):
 *  10  — top of forehead (glabella-to-hairline centre)
 *  71/139/301/339 — forehead sides
 *  50  — left cheek, 280 — right cheek
 */
export const LM = { foreheadTop: 10, foreheadL: 71, foreheadR: 301, cheekL: 50, cheekR: 280, chin: 152, nose: 1 };

/**
 * Compute ROI rectangles (in normalized 0..1 coords) for the estimator.
 * Forehead patch sits between the hairline (lm 10) and the brows; cheeks are
 * centred on the cheek landmarks. Returns null when no landmark list given.
 */
export function roiBoxes(landmarks) {
  if (!landmarks || landmarks.length === 0) return null;
  const p = (i) => landmarks[i];
  const top = p(10), chin = p(152), l = p(234), r = p(454);
  const faceH = Math.abs(chin.y - top.y);
  const faceW = Math.abs(r.x - l.x);
  const forehead = {
    x: (p(LM.foreheadL).x + p(LM.foreheadR).x) / 2 - faceW * 0.17,
    y: top.y + faceH * 0.05,
    w: faceW * 0.34,
    h: faceH * 0.10,
  };
  const cheek = (idx) => ({
    x: p(idx).x - faceW * 0.10,
    y: p(idx).y - faceH * 0.05,
    w: faceW * 0.20,
    h: faceH * 0.12,
  });
  const chest = {
    x: (l.x + r.x) / 2 - faceW * 0.55,
    y: chin.y + faceH * 0.25,
    w: faceW * 1.10,
    h: faceH * 0.6,
  };
  return { forehead, cheekL: cheek(LM.cheekL), cheekR: cheek(LM.cheekR), chest, face: { x: l.x, y: top.y, w: faceW, h: faceH } };
}

/** Clamp + convert normalized rect to integer image pixels. */
export function roiPixels(roi, w, h) {
  const x0 = Math.max(0, Math.min(w - 2, Math.floor(roi.x * w)));
  const y0 = Math.max(0, Math.min(h - 2, Math.floor(roi.y * h)));
  return {
    x: x0, y: y0,
    w: Math.max(2, Math.min(roi.w * w, w - x0)) | 0,
    h: Math.max(2, Math.min(roi.h * h, h - y0)) | 0,
  };
}
