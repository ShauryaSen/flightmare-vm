// reward_math.js — exact port of C++ QuadrotorEnv reward logic
// Matches: quadrotor_env.cpp buildTrackSpline() + spline_progress()

export const CONFIG = {
  startPos: [17.4, -5.0, 6.31],
  gateWidth: 5,
  splineDeviationRatio: 1.5,
  deviationPenaltyCoeff: 0.01,
  bodyRateCoeff: 0.0005,
  safetyCoeff: 0.0,
  safetyDMax: 1.5,
  gates: [
    { pos: [17.25, 3.63, 6.31], quat: [0.7399242044980668, 0.011581198096193812, -0.012742889663570194, 0.6724698255026628] },
    { pos: [15.81, 11.49, 4.87], quat: [0.4178, -0.0773, 0.0357, 0.9046] },
    { pos: [8.28, 14.18, 3.43], quat: [0.0894, 0.0848, 0.0076, -0.9924] },
    { pos: [2.20, 9.00, 1.99], quat: [0.6767607538136899, -0.020071138933028868, -0.018463233851244758, -0.7356977235762059] },
    { pos: [-7.31, -12.14, 3.23], quat: [0.5489392989243054, -0.005591023608941385, -0.0036719339571304396, -0.8358354523783439] },
    { pos: [-7.61, -28.47, 2.64], quat: [0.8675806243752933, 0.017934150987913768, 0.03137074171758188, -0.4959819583422445] },
    { pos: [-0.01, -33.91, 2.10], quat: [0.998425348623773, 0.00029584413315685845, -0.005289170362916548, 0.05584586268129906] },
    { pos: [5.87, -28.44, 2.57], quat: [0.8606956628855412, 0.013573947951699823, -0.02297911752200398, 0.50841979110272] },
    { pos: [7.25, -11.79, 2.55], quat: [0.5425898966059537, -0.014531641495382797, 0.009388590420559003, 0.8398195579209294] },
    { pos: [-10.23, 7.77, 2.02], quat: [0.5201707829903365, 0.004834159805478233, -0.002944333232843624, 0.8540435107908785] },
    { pos: [-10.05, 30.62, 2.90], quat: [0.7753291124908203, 0.01491979592960568, -0.018329070193316095, 0.6311150546449233] },
  ]
};

// Vector math
export function v3sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
export function v3add(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
export function v3scale(v, s) { return [v[0] * s, v[1] * s, v[2] * s]; }
export function v3dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
export function v3norm(v) { return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
export function v3normalize(v) { const n = v3norm(v); return n > 1e-8 ? v3scale(v, 1 / n) : [0, 0, 0]; }

export function quatToR(w, x, y, z) {
  const n = Math.sqrt(w * w + x * x + y * y + z * z); w /= n; x /= n; y /= n; z /= n;
  return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]];
}
function matCol(R, c) { return [R[0][c], R[1][c], R[2][c]]; }

// Cubic Hermite — matches C++ hermite_eval exactly
function hermiteEval(p0, m0, p1, m1, t) {
  const t2 = t * t, t3 = t2 * t;
  const h00 = 2 * t3 - 3 * t2 + 1, h10 = t3 - 2 * t2 + t, h01 = -2 * t3 + 3 * t2, h11 = t3 - t2;
  return [h00 * p0[0] + h10 * m0[0] + h01 * p1[0] + h11 * m1[0],
  h00 * p0[1] + h10 * m0[1] + h01 * p1[1] + h11 * m1[1],
  h00 * p0[2] + h10 * m0[2] + h01 * p1[2] + h11 * m1[2]];
}

const SAMPLES_PER_SEG = 200; // matches C++ kSplineSamplesPerSegment

// Builds the exact same spline as C++ buildTrackSpline()
export function buildTrackSpline(config = CONFIG) {
  const { startPos, gates } = config;
  const numGates = gates.length;
  const ctrlPts = [startPos];
  const tangentDirs = [v3normalize(v3sub(gates[0].pos, startPos))];
  for (const g of gates) {
    ctrlPts.push(g.pos);
    const R = quatToR(...g.quat);
    tangentDirs.push(v3normalize(matCol(R, 0)));
  }
  const normals = gates.map(g => { const R = quatToR(...g.quat); return v3normalize(matCol(R, 0)); });
  const frames = gates.map(g => quatToR(...g.quat));

  const segments = [];
  let globalArc = 0;
  for (let si = 0; si < numGates; si++) {
    const p0 = ctrlPts[si], p1 = ctrlPts[si + 1];
    const chord = v3norm(v3sub(p1, p0));
    const m0 = v3scale(tangentDirs[si], chord);
    const m1 = v3scale(tangentDirs[si + 1], chord);
    const samples = [], arcLengths = [];
    samples.push([...p0]); arcLengths.push(globalArc);
    const sStart = globalArc;
    for (let j = 1; j < SAMPLES_PER_SEG; j++) {
      const t = j / (SAMPLES_PER_SEG - 1);
      const pt = hermiteEval(p0, m0, p1, m1, t);
      globalArc += v3norm(v3sub(pt, samples[j - 1]));
      samples.push(pt); arcLengths.push(globalArc);
    }
    segments.push({ sStart, sEnd: globalArc, samples, arcLengths });
  }
  return { segments, normals, frames, totalArc: globalArc };
}

// Exact port of C++ spline_progress()
export function splineProgress(pos, targetGateIdx, spline, config = CONFIG) {
  if (targetGateIdx < 0 || targetGateIdx >= spline.segments.length) return 0;
  const seg = spline.segments[targetGateIdx];
  if (!seg.samples.length) return 0;

  let bestDistSq = Infinity, bestIdx = 0;
  for (let j = 0; j < seg.samples.length; j++) {
    const d2 = v3sub(pos, seg.samples[j]).reduce((s, v) => s + v * v, 0);
    if (d2 < bestDistSq) { bestDistSq = d2; bestIdx = j; }
  }
  let bestArc = seg.arcLengths[bestIdx];
  let bestDist = Math.sqrt(bestDistSq);

  for (const offset of [-1, 1]) {
    const j2 = bestIdx + offset;
    if (j2 < 0 || j2 >= seg.samples.length) continue;
    const a = seg.samples[bestIdx], b = seg.samples[j2];
    const ab = v3sub(b, a);
    const abLen2 = v3dot(ab, ab);
    if (abLen2 < 1e-12) continue;
    let t = v3dot(v3sub(pos, a), ab) / abLen2;
    t = Math.max(0, Math.min(1, t));
    const proj = v3add(a, v3scale(ab, t));
    const d = v3norm(v3sub(pos, proj));
    if (d < bestDist) {
      bestDist = d;
      bestArc = seg.arcLengths[bestIdx] + t * (seg.arcLengths[j2] - seg.arcLengths[bestIdx]);
    }
  }
  bestArc = Math.max(bestArc, seg.sStart);
  bestArc = Math.min(bestArc, seg.sEnd);
  return (bestArc - seg.sStart) - config.splineDeviationRatio * bestDist;
}

// Compute reward delta when moving from prevPos to currPos
export function computeRewardDelta(prevPos, currPos, targetGateIdx, spline, config = CONFIG) {
  const sPrev = splineProgress(prevPos, targetGateIdx, spline, config);
  const sCurr = splineProgress(currPos, targetGateIdx, spline, config);
  const lateralDev = getNearestDist(currPos, targetGateIdx, spline);

  // Per-step quadratic deviation penalty beyond gate radius
  let deviationPenalty = 0;
  const threshold = config.gateWidth / 2;
  if (config.deviationPenaltyCoeff > 0 && lateralDev > threshold) {
    const excess = lateralDev - threshold;
    deviationPenalty = -config.deviationPenaltyCoeff * excess * excess;
  }

  const progress = sCurr - sPrev + deviationPenalty;
  return { progress, sCurr, sPrev, lateralDev, deviationPenalty };
}

function getNearestDist(pos, idx, spline) {
  if (idx < 0 || idx >= spline.segments.length) return 0;
  const seg = spline.segments[idx];
  let best = Infinity;
  for (const s of seg.samples) { const d = v3norm(v3sub(pos, s)); if (d < best) best = d; }
  return best;
}

// Check gate pass (plane crossing)
export function checkGatePass(prevPos, currPos, gateIdx, spline, config = CONFIG) {
  const g = config.gates[gateIdx];
  const n = spline.normals[gateIdx];
  const dnPrev = v3dot(v3sub(prevPos, g.pos), n);
  const dnCurr = v3dot(v3sub(currPos, g.pos), n);
  const dist = v3norm(v3sub(currPos, g.pos));
  if (dnPrev < 0 && dnCurr >= 0 && dist < config.gateWidth * 2) {
    const denom = dnPrev - dnCurr;
    const tCross = Math.abs(denom) > 1e-9 ? (dnPrev / denom) : 0.5;
    const crossPos = v3add(prevPos, v3scale(v3sub(currPos, prevPos), tCross));
    const offset = v3sub(crossPos, g.pos);
    const R = spline.frames[gateIdx];
    const gRight = matCol(R, 1), gUp = matCol(R, 2);
    const offR = Math.abs(v3dot(offset, gRight));
    const offU = Math.abs(v3dot(offset, gUp));
    const hw = config.gateWidth / 2;
    return offR < hw && offU < hw;
  }
  return false;
}
