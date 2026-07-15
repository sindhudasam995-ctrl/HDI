// Linear Regression via Ordinary Least Squares (normal equation) solved with
// Gaussian elimination. Mirrors the scikit-learn workflow from the PRD:
// dataset -> features -> train/test split -> fit -> evaluate -> predict.
// The trained coefficients are cached so the model loads without retraining.

import { dataset, featureNames } from "./dataset.js";

// ---------- linear algebra helpers ----------

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function transpose(m) {
  const rows = m.length;
  const cols = m[0].length;
  const t = zeros(cols, rows);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) t[j][i] = m[i][j];
  return t;
}

function matMul(a, b) {
  const rows = a.length;
  const inner = b.length;
  const cols = b[0].length;
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cols; j++) out[i][j] += aik * b[k][j];
    }
  return out;
}

// Solve Xw = y via Gaussian elimination with partial pivoting.
function gaussianSolve(X, y) {
  const n = X.length;
  const p = X[0].length;
  const A = zeros(n, p + 1);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < p; j++) A[i][j] = X[i][j];
    A[i][p] = y[i];
  }
  for (let col = 0; col < p; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r;
    }
    if (pivot !== col) {
      const tmp = A[col];
      A[col] = A[pivot];
      A[pivot] = tmp;
    }
    const pivVal = A[col][col];
    if (Math.abs(pivVal) < 1e-12) continue;
    for (let r = col + 1; r < n; r++) {
      const factor = A[r][col] / pivVal;
      if (factor === 0) continue;
      for (let c = col; c <= p; c++) A[r][c] -= factor * A[col][c];
    }
  }
  const w = new Array(p).fill(0);
  for (let c = p - 1; c >= 0; c--) {
    let sum = A[c][p];
    for (let j = c + 1; j < p; j++) sum -= A[c][j] * w[j];
    w[c] = sum / A[c][c];
  }
  return w;
}

// ---------- preprocessing ----------

// Log-transform GNI so the heavily skewed income dimension is linearly separable
// relative to HDI (which already follows a log-income shape). The other three
// features are used as-is.
function features(row) {
  return [
    row.lifeExpectancy,
    row.meanYearsSchooling,
    row.expectedYearsSchooling,
    Math.log(row.gniPerCapita),
  ];
}

function shuffle(arr, seed = 42) {
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function trainTestSplit(rows, testRatio = 0.2) {
  const shuffled = shuffle(rows);
  const cut = Math.floor(shuffled.length * (1 - testRatio));
  return { train: shuffled.slice(0, cut), test: shuffled.slice(cut) };
}

// ---------- model ----------

function buildMatrix(rows) {
  const X = rows.map(features);
  // prepend bias column of ones
  const Xb = X.map((r) => [1, ...r]);
  const y = rows.map((r) => r.hdi);
  return { X: Xb, y };
}

function train(rows) {
  const { X, y } = buildMatrix(rows);
  const weights = gaussianSolve(X, y);
  return { weights, featureNames: ["bias", ...featureNames] };
}

function predict(model, row) {
  const f = [1, ...features(row)];
  let sum = 0;
  for (let i = 0; i < f.length; i++) sum += f[i] * model.weights[i];
  return sum;
}

function evaluate(rows, model) {
  let se = 0;
  let mape = 0;
  let ssRes = 0;
  let ssTot = 0;
  const meanY = rows.reduce((s, r) => s + r.hdi, 0) / rows.length;
  for (const r of rows) {
    const pred = predict(model, r);
    const err = pred - r.hdi;
    se += err * err;
    mape += Math.abs(err / r.hdi);
    ssRes += err * err;
    ssTot += (r.hdi - meanY) ** 2;
  }
  return {
    rmse: Math.sqrt(se / rows.length),
    mape: (mape / rows.length) * 100,
    r2: 1 - ssRes / ssTot,
  };
}

// Train once at module load (cheap, in-memory) and cache the result so the
// "model" is ready without retraining on each prediction.
const { train: trainRows, test: testRows } = trainTestSplit(dataset);
const model = train(trainRows);
const trainMetrics = evaluate(trainRows, model);
const testMetrics = evaluate(testRows, model);

export { model, predict, trainMetrics, testMetrics, trainRows, testRows };
