import {
  MountainCarEnv,
  makeBins,
  discretise,
  velocityBinMidpoints,
  RandomPolicy,
  MixturePolicy,
  GreedyPolicy,
} from "../index.js";
import { createSeededRng, splitSeed } from "../utils/rng.js";
import { MountainCarRenderer, drawHeatmap, drawQValueHeatmap, drawGreedyActionMap } from "./renderCanvas.js";

// ─────────────────────────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────────────────────────

function make3D(n, fill = 0) {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => Array(3).fill(fill)),
  );
}

function make2D(size, fill = 0) {
  return Array.from({ length: size }, () => Array(size).fill(fill));
}

function fmt(n, digits = 3) {
  if (typeof n !== "number" || Number.isNaN(n)) return "-";
  return n.toFixed(digits);
}

function readNumber(el, fallback) {
  const v = Number.parseFloat(el.value);
  return Number.isFinite(v) ? v : fallback;
}
function readInt(el, fallback) {
  const v = Number.parseInt(el.value, 10);
  return Number.isInteger(v) ? v : fallback;
}

function logLine(msg) {
  const now = new Date().toLocaleTimeString();
  ui.trainLog.value += `[${now}] ${msg}\n`;
  ui.trainLog.scrollTop = ui.trainLog.scrollHeight;
}

// ─────────────────────────────────────────────────────────────────────────────
// Q-table validation helpers (reused from original)
// ─────────────────────────────────────────────────────────────────────────────

function validateQTable(qTable) {
  if (!Array.isArray(qTable) || qTable.length === 0)
    throw new Error("Q-table must be a non-empty 3D array.");
  const nBins = qTable.length;
  for (let i = 0; i < nBins; i++) {
    if (!Array.isArray(qTable[i]) || qTable[i].length !== nBins)
      throw new Error("Q-table must have shape [n_bins][n_bins][3].");
    for (let j = 0; j < nBins; j++) {
      if (!Array.isArray(qTable[i][j]) || qTable[i][j].length !== 3)
        throw new Error("Q-table last dimension must have 3 actions.");
      for (let a = 0; a < 3; a++) {
        if (typeof qTable[i][j][a] !== "number" || Number.isNaN(qTable[i][j][a]))
          throw new Error("Q-table entries must be numeric.");
      }
    }
  }
  return nBins;
}

function extractQTablePayload(json) {
  if (Array.isArray(json)) return { qTable: json, nBinsHint: json.length };
  if (json && typeof json === "object") {
    if (Array.isArray(json.q_table))
      return { qTable: json.q_table, nBinsHint: Number.isInteger(json.n_bins) ? json.n_bins : json.q_table.length };
    if (Array.isArray(json.qTable))
      return { qTable: json.qTable, nBinsHint: Number.isInteger(json.nBins) ? json.nBins : json.qTable.length };
  }
  throw new Error("JSON must contain `q_table` or be a 3D array.");
}

// ─────────────────────────────────────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────────────────────────────────────

const ui = {
  // Pipeline bar
  saveEpisodesBtn: document.getElementById("saveEpisodesBtn"),
  saveQTableBtn: document.getElementById("saveQTableBtn"),
  loadEpisodesFile: document.getElementById("loadEpisodesFile"),
  loadQTableFile: document.getElementById("loadQTableFile"),
  pipelineStatusBadge: document.getElementById("pipelineStatusBadge"),

  // Section badges
  collectStatusBadge: document.getElementById("collectStatusBadge"),
  qInitStatusBadge: document.getElementById("qInitStatusBadge"),
  trainStatusBadge: document.getElementById("trainStatusBadge"),
  testStatusBadge: document.getElementById("testStatusBadge"),

  // Section locks
  sectionQTable: document.getElementById("section-qtable"),
  sectionTrain: document.getElementById("section-train"),
  sectionTest: document.getElementById("section-test"),

  // ── Step 1: Collection ──
  collectPolicy: document.getElementById("collectPolicy"),
  pSmart: document.getElementById("pSmart"),
  episodesTarget: document.getElementById("episodesTarget"),
  maxSteps: document.getElementById("maxSteps"),
  stepsPerFrame: document.getElementById("stepsPerFrame"),
  stepsPerFrameLabel: document.getElementById("stepsPerFrameLabel"),
  seed: document.getElementById("seed"),
  startCollectBtn: document.getElementById("startCollectBtn"),
  pauseCollectBtn: document.getElementById("pauseCollectBtn"),
  resetCollectBtn: document.getElementById("resetCollectBtn"),
  envCanvas: document.getElementById("envCanvas"),
  heatmapCanvas: document.getElementById("heatmapCanvas"),
  // Collection metrics
  mEpisodes: document.getElementById("mEpisodes"),
  mSuccesses: document.getElementById("mSuccesses"),
  mSuccessRate: document.getElementById("mSuccessRate"),
  mAvgReturn: document.getElementById("mAvgReturn"),
  mAvgLength: document.getElementById("mAvgLength"),
  mCurStep: document.getElementById("mCurStep"),
  mCurReturn: document.getElementById("mCurReturn"),
  mLastAction: document.getElementById("mLastAction"),
  mBehaviorProb: document.getElementById("mBehaviorProb"),
  mPosition: document.getElementById("mPosition"),
  mVelocity: document.getElementById("mVelocity"),
  mState: document.getElementById("mState"),
  mA0: document.getElementById("mA0"),
  mA1: document.getElementById("mA1"),
  mA2: document.getElementById("mA2"),
  mCollectStatus: document.getElementById("mCollectStatus"),

  // ── Step 2: Q-Table Init ──
  nBins: document.getElementById("nBins"),
  gamma: document.getElementById("gamma"),
  initQTableBtn: document.getElementById("initQTableBtn"),
  qInitStatus: document.getElementById("qInitStatus"),

  // Q-viz panel
  qVizPanel: document.getElementById("qVizPanel"),
  qValueCanvas: document.getElementById("qValueCanvas"),
  greedyMapCanvas: document.getElementById("greedyMapCanvas"),
  pass1Progress: document.getElementById("pass1Progress"),
  pass2Progress: document.getElementById("pass2Progress"),
  pass1Pct: document.getElementById("pass1Pct"),
  pass2Pct: document.getElementById("pass2Pct"),
  startTrainBtn: document.getElementById("startTrainBtn"),
  trainLog: document.getElementById("trainLog"),
  trainStatus: document.getElementById("trainStatus"),

  // ── Step 4: Test ──
  testEpisodesTarget: document.getElementById("testEpisodesTarget"),
  testMaxSteps: document.getElementById("testMaxSteps"),
  testStepsPerFrame: document.getElementById("testStepsPerFrame"),
  testStepsPerFrameLabel: document.getElementById("testStepsPerFrameLabel"),
  startTestBtn: document.getElementById("startTestBtn"),
  pauseTestBtn: document.getElementById("pauseTestBtn"),
  resetTestBtn: document.getElementById("resetTestBtn"),
  testCanvas: document.getElementById("testCanvas"),
  tEpisodes: document.getElementById("tEpisodes"),
  tSuccesses: document.getElementById("tSuccesses"),
  tSuccessRate: document.getElementById("tSuccessRate"),
  tAvgReturn: document.getElementById("tAvgReturn"),
  tAvgLength: document.getElementById("tAvgLength"),
  tCurStep: document.getElementById("tCurStep"),
  tCurReturn: document.getElementById("tCurReturn"),
  mTestStatus: document.getElementById("mTestStatus"),
};

// ─────────────────────────────────────────────────────────────────────────────
// Renderers
// ─────────────────────────────────────────────────────────────────────────────

const collectRenderer = new MountainCarRenderer(ui.envCanvas);
const testRenderer = new MountainCarRenderer(ui.testCanvas);

// ─────────────────────────────────────────────────────────────────────────────
// Central Pipeline State
// ─────────────────────────────────────────────────────────────────────────────

const pipeline = {
  // Checkpoint data
  rawEpisodes: [],        // collected trajectories
  qTableConfig: null,     // { nBins, gamma }
  trainedQTable: null,    // 3D array after training

  // Phase: 'idle' | 'collecting' | 'collected' | 'qtable_ready' | 'trained' | 'testing'
  phase: "idle",
};

// ─────────────────────────────────────────────────────────────────────────────
// Collection Simulation State
// ─────────────────────────────────────────────────────────────────────────────

const collect = {
  running: false,
  rafId: null,
  env: null,
  bins: null,
  behaviorPolicy: null,
  nBins: 20,
  maxSteps: 200,
  targetEpisodes: 200,
  policyName: "mixture",
  pSmart: 0.5,
  stepsPerFrame: 8,
  episodes: [],
  visits: make2D(20, 0),
  actionCounts: [0, 0, 0],
  completedEpisodes: 0,
  successes: 0,
  totalReturn: 0,
  totalLength: 0,
  curSteps: 0,
  curReturn: 0,
  curObs: null,
  curState: null,
  curTrajectory: [],
  lastAction: null,
  lastBehaviorProb: null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Test Simulation State
// ─────────────────────────────────────────────────────────────────────────────

const test = {
  running: false,
  rafId: null,
  env: null,
  bins: null,
  greedyPolicy: null,
  nBins: 20,
  maxSteps: 200,
  targetEpisodes: 20,
  stepsPerFrame: 8,
  completedEpisodes: 0,
  successes: 0,
  totalReturn: 0,
  totalLength: 0,
  curSteps: 0,
  curReturn: 0,
  curObs: null,
};

// ─────────────────────────────────────────────────────────────────────────────
// Badge & lock helpers
// ─────────────────────────────────────────────────────────────────────────────

function setBadge(el, text, mode = "") {
  el.textContent = text;
  el.classList.remove("badge-ok", "badge-running", "badge-warn");
  if (mode) el.classList.add("badge-" + mode);
}

function setPipelineBadge(text) {
  ui.pipelineStatusBadge.textContent = text;
}

function unlockSection(sectionEl, badgeEl, badgeText) {
  sectionEl.classList.remove("section-locked");
  setBadge(badgeEl, badgeText);
}

// ─────────────────────────────────────────────────────────────────────────────
// Collection: metrics
// ─────────────────────────────────────────────────────────────────────────────

function updateCollectMetrics() {
  ui.mEpisodes.textContent = String(collect.completedEpisodes);
  ui.mSuccesses.textContent = String(collect.successes);
  ui.mSuccessRate.textContent = `${fmt(
    collect.completedEpisodes > 0 ? (100 * collect.successes / collect.completedEpisodes) : 0, 2,
  )}%`;
  ui.mAvgReturn.textContent = fmt(
    collect.completedEpisodes > 0 ? collect.totalReturn / collect.completedEpisodes : 0, 2,
  );
  ui.mAvgLength.textContent = fmt(
    collect.completedEpisodes > 0 ? collect.totalLength / collect.completedEpisodes : 0, 2,
  );
  ui.mCurStep.textContent = String(collect.curSteps);
  ui.mCurReturn.textContent = fmt(collect.curReturn, 2);
  ui.mLastAction.textContent = collect.lastAction === null ? "-" : String(collect.lastAction);
  ui.mBehaviorProb.textContent = collect.lastBehaviorProb === null ? "-" : fmt(collect.lastBehaviorProb, 4);
  ui.mPosition.textContent = collect.curObs ? fmt(collect.curObs[0], 4) : "-";
  ui.mVelocity.textContent = collect.curObs ? fmt(collect.curObs[1], 5) : "-";
  ui.mState.textContent = collect.curState ? `[${collect.curState[0]}, ${collect.curState[1]}]` : "-";
  ui.mA0.textContent = String(collect.actionCounts[0]);
  ui.mA1.textContent = String(collect.actionCounts[1]);
  ui.mA2.textContent = String(collect.actionCounts[2]);
}

function updateTestMetrics() {
  ui.tEpisodes.textContent = String(test.completedEpisodes);
  ui.tSuccesses.textContent = String(test.successes);
  ui.tSuccessRate.textContent = `${fmt(
    test.completedEpisodes > 0 ? (100 * test.successes / test.completedEpisodes) : 0, 2,
  )}%`;
  ui.tAvgReturn.textContent = fmt(
    test.completedEpisodes > 0 ? test.totalReturn / test.completedEpisodes : 0, 2,
  );
  ui.tAvgLength.textContent = fmt(
    test.completedEpisodes > 0 ? test.totalLength / test.completedEpisodes : 0, 2,
  );
  ui.tCurStep.textContent = String(test.curSteps);
  ui.tCurReturn.textContent = fmt(test.curReturn, 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// Collection: core logic
// ─────────────────────────────────────────────────────────────────────────────

function buildBehaviorPolicy(policyRng) {
  if (collect.policyName === "mixture") {
    const mids = velocityBinMidpoints(
      collect.nBins, collect.env.observationLow[1], collect.env.observationHigh[1],
    );
    collect.behaviorPolicy = new MixturePolicy(3, mids, { pSmart: collect.pSmart, rng: policyRng });
    return;
  }
  collect.behaviorPolicy = new RandomPolicy(3, policyRng);
}

function resetCollectEpisode() {
  const { observation } = collect.env.reset();
  collect.curObs = observation;
  collect.curState = discretise(observation, collect.bins);
  collect.curSteps = 0;
  collect.curReturn = 0;
  collect.curTrajectory = [];
  collect.lastAction = null;
  collect.lastBehaviorProb = null;
}

function clearCollectStats() {
  collect.episodes = [];
  collect.visits = make2D(collect.nBins, 0);
  collect.actionCounts = [0, 0, 0];
  collect.completedEpisodes = 0;
  collect.successes = 0;
  collect.totalReturn = 0;
  collect.totalLength = 0;
  collect.curObs = null;
  collect.curState = null;
  collect.curTrajectory = [];
  collect.lastAction = null;
  collect.lastBehaviorProb = null;
}

function initCollectSim() {
  collect.nBins = Math.max(2, readInt(ui.nBins, 20));
  collect.targetEpisodes = Math.max(1, readInt(ui.episodesTarget, 200));
  collect.maxSteps = Math.max(1, readInt(ui.maxSteps, 200));
  collect.stepsPerFrame = Math.max(1, readInt(ui.stepsPerFrame, 8));
  collect.policyName = ui.collectPolicy.value;
  collect.pSmart = Math.max(0, Math.min(1, readNumber(ui.pSmart, 0.5)));

  let envRng = Math.random;
  let policyRng = Math.random;
  const seed = readInt(ui.seed, Number.NaN);
  if (Number.isInteger(seed)) {
    const [envSeed, policySeed] = splitSeed(seed);
    envRng = createSeededRng(envSeed);
    policyRng = createSeededRng(policySeed);
  }

  collect.env = new MountainCarEnv({ maxEpisodeSteps: collect.maxSteps, rng: envRng });
  collect.bins = makeBins(collect.nBins, collect.env.observationLow, collect.env.observationHigh);
  buildBehaviorPolicy(policyRng);

  clearCollectStats();
  resetCollectEpisode();
  updateCollectMetrics();
  drawHeatmap(ui.heatmapCanvas, collect.visits, collect.nBins);
  collectRenderer.draw(collect.curObs, { action: null });
  ui.mCollectStatus.textContent = "Ready";
  setBadge(ui.collectStatusBadge, "Ready");
}

function completeCollectEpisode(success) {
  collect.completedEpisodes += 1;
  if (success) collect.successes += 1;
  collect.totalReturn += collect.curReturn;
  collect.totalLength += collect.curSteps;
  collect.episodes.push(collect.curTrajectory);

  if (collect.completedEpisodes >= collect.targetEpisodes) {
    collect.running = false;
    pipeline.rawEpisodes = collect.episodes;
    pipeline.phase = "collected";

    const msg = `Done — ${collect.completedEpisodes} episodes stored.`;
    ui.mCollectStatus.textContent = msg;
    setBadge(ui.collectStatusBadge, `✓ ${collect.completedEpisodes} episodes`, "ok");
    setPipelineBadge("Step 1 ✓");

    // Unlock Step 2
    unlockSection(ui.sectionQTable, ui.qInitStatusBadge, "Ready");
    ui.initQTableBtn.disabled = false;

    // Enable save button
    ui.saveEpisodesBtn.disabled = false;
    return;
  }
  resetCollectEpisode();
}

function collectStepOnce() {
  const state = discretise(collect.curObs, collect.bins);
  const action = collect.behaviorPolicy.selectAction(state);
  const prob = collect.behaviorPolicy.prob(state, action);
  const result = collect.env.step(action);

  collect.curTrajectory.push([state, action, result.reward, prob]);
  collect.curObs = result.observation;
  collect.curState = discretise(collect.curObs, collect.bins);
  collect.curSteps += 1;
  collect.curReturn += result.reward;
  collect.actionCounts[action] += 1;
  collect.visits[state[0]][state[1]] += 1;
  collect.lastAction = action;
  collect.lastBehaviorProb = prob;

  if (result.terminated || result.truncated) {
    completeCollectEpisode(result.terminated);
  }
}

function collectTick() {
  if (!collect.running) {
    collect.rafId = null;
    updateCollectMetrics();
    collectRenderer.draw(collect.curObs, { action: collect.lastAction });
    drawHeatmap(ui.heatmapCanvas, collect.visits, collect.nBins);
    return;
  }

  for (let i = 0; i < collect.stepsPerFrame; i++) {
    if (!collect.running) break;
    collectStepOnce();
  }

  updateCollectMetrics();
  collectRenderer.draw(collect.curObs, { action: collect.lastAction });
  drawHeatmap(ui.heatmapCanvas, collect.visits, collect.nBins);

  if (collect.running) {
    const msg = `Episode ${collect.completedEpisodes + 1} / ${collect.targetEpisodes}`;
    ui.mCollectStatus.textContent = msg;
    setBadge(ui.collectStatusBadge, msg, "running");
    setPipelineBadge("Collecting…");
  }

  collect.rafId = window.requestAnimationFrame(collectTick);
}

function startCollect() {
  if (!collect.env) {
    try { initCollectSim(); }
    catch (err) { ui.mCollectStatus.textContent = `Error: ${err.message}`; return; }
  }
  if (collect.running) return;
  collect.running = true;
  setBadge(ui.collectStatusBadge, "Running…", "running");
  if (!collect.rafId) {
    collect.rafId = window.requestAnimationFrame(collectTick);
  }
}

function pauseCollect() {
  collect.running = false;
  setBadge(ui.collectStatusBadge, "Paused");
  ui.mCollectStatus.textContent = "Paused";
}

function resetCollect() {
  collect.running = false;
  if (collect.rafId) { window.cancelAnimationFrame(collect.rafId); collect.rafId = null; }
  try { initCollectSim(); }
  catch (err) { ui.mCollectStatus.textContent = `Error: ${err.message}`; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Initialize Q-Table
// ─────────────────────────────────────────────────────────────────────────────

function initQTable() {
  const nBins = Math.max(2, readInt(ui.nBins, 20));
  const gamma = Math.max(0, Math.min(1, readNumber(ui.gamma, 0.99)));

  pipeline.qTableConfig = { nBins, gamma };
  pipeline.phase = "qtable_ready";

  const msg = `Q-Table initialized: ${nBins}×${nBins}×3, γ=${gamma.toFixed(3)}`;
  ui.qInitStatus.textContent = msg;
  ui.qInitStatus.className = "status ok";
  setBadge(ui.qInitStatusBadge, "✓ Initialized", "ok");
  setPipelineBadge("Step 2 ✓");

  // Unlock Step 3
  unlockSection(ui.sectionTrain, ui.trainStatusBadge, "Ready");
  ui.startTrainBtn.disabled = false;
  logLine(`Q-Table allocated (${nBins}×${nBins}×3), γ=${gamma}`);
  ui.trainLog.value = "";
  logLine(`Waiting to train on ${pipeline.rawEpisodes.length} episodes…`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Training (stub for Phase 1 — real IS logic added in Phase 2)
// ─────────────────────────────────────────────────────────────────────────────

async function runTraining() {
  ui.startTrainBtn.disabled = true;
  setBadge(ui.trainStatusBadge, "Running…", "running");
  setPipelineBadge("Training…");

  const { nBins, gamma } = pipeline.qTableConfig;
  const episodes = pipeline.rawEpisodes;
  const nEps = episodes.length;

  logLine(`Starting Pass 1 (Ordinary MC) over ${nEps} episodes, γ=${gamma}…`);

  // ── Pass 1: Ordinary MC (EVERY-VISIT, incremental mean) ─────────────────
  // We compute Q^b: expected return under behavior policy.
  // Every-visit: every (state, action, G) tuple in the backward sweep
  // contributes to the mean, matching train_is.py exactly.
  const Qb = make3D(nBins, 0);
  const count = make3D(nBins, 0);

  const BATCH = 500; // episodes per animation frame yield
  for (let ep = 0; ep < nEps; ep++) {
    const traj = episodes[ep];
    let G = 0;
    for (let t = traj.length - 1; t >= 0; t--) {
      const [state, action, reward] = traj[t];
      G = reward + gamma * G;
      // Every-visit: no first-visit guard — all timesteps update Q^b
      const [pi, vi] = state;
      count[pi][vi][action] += 1;
      Qb[pi][vi][action] += (G - Qb[pi][vi][action]) / count[pi][vi][action];
    }

    if (ep % BATCH === 0 || ep === nEps - 1) {
      const pct = Math.round((ep + 1) / nEps * 100);
      ui.pass1Progress.value = pct;
      ui.pass1Pct.textContent = `${pct}%`;
      // Yield to the event loop so progress bar repaints
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  logLine("Pass 1 complete.");

  // ── Pass 2: Weighted Importance Sampling (backward sweep) ───────────────
  logLine("Starting Pass 2 (Weighted IS)…");

  const Q = make3D(nBins, 0);        // IS-weighted Q
  const C = make3D(nBins, 0);        // cumulative IS denominator

  // Pre-compute fixed greedy map from Q^b (argmax per state), matching Python's
  // `greedy_map = np.argmax(Q_b, axis=-1)` before the Pass 2 loop.
  const greedyMap = Array.from({ length: nBins }, (_, pi) =>
    Array.from({ length: nBins }, (_, vi) => {
      const row = Qb[pi][vi];
      let best = 0;
      for (let a = 1; a < 3; a++) if (row[a] > row[best]) best = a;
      return best;
    }),
  );

  for (let ep = 0; ep < nEps; ep++) {
    const traj = episodes[ep];
    let G = 0;
    let W = 1.0;
    // S&B Algorithm 5.7 backward sweep
    for (let t = traj.length - 1; t >= 0; t--) {
      const [state, action, reward, bProb] = traj[t];
      G = reward + gamma * G;
      const [pi, vi] = state;
      // Update Q^π FIRST (before the break check) — matches Python exactly
      C[pi][vi][action] += W;
      Q[pi][vi][action] += (W / C[pi][vi][action]) * (G - Q[pi][vi][action]);
      // Break if action deviates from fixed greedy target
      if (action !== greedyMap[pi][vi]) break;
      // Action matched target: π(a|s)=1, so ratio = 1/b(a|s)
      W *= 1.0 / bProb;
    }

    if (ep % BATCH === 0 || ep === nEps - 1) {
      const pct = Math.round((ep + 1) / nEps * 100);
      ui.pass2Progress.value = pct;
      ui.pass2Pct.textContent = `${pct}%`;
      await new Promise((r) => setTimeout(r, 0));
    }
  }
  logLine("Pass 2 complete. Training finished.");

  // Final Q selected is Q^b — matches Python: `Q_final = Q_b`.
  // Q^b has much better coverage (every visited state has a value);
  // greedy(Q^b) is the off-policy improvement result.
  // Q (Q^π) is sparser and is kept only for IS-correction analysis.
  pipeline.trainedQTable = Qb;
  pipeline.phase = "trained";

  setBadge(ui.trainStatusBadge, "✓ Trained", "ok");
  ui.trainStatus.textContent = "Training complete.";
  ui.trainStatus.className = "status ok";
  setPipelineBadge("Step 3 ✓");

  // Enable save Q-table
  ui.saveQTableBtn.disabled = false;

  // Unlock Step 4
  unlockSection(ui.sectionTest, ui.testStatusBadge, "Ready");
  ui.startTestBtn.disabled = false;
  ui.pauseTestBtn.disabled = false;
  ui.resetTestBtn.disabled = false;

  // Reveal Q-value visualizations
  ui.qVizPanel.classList.remove("hidden");
  drawQValueHeatmap(ui.qValueCanvas, Qb, nBins);
  drawGreedyActionMap(ui.greedyMapCanvas, Qb, nBins);

  logLine(`Q-table saved. Ready to test.`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Test simulation
// ─────────────────────────────────────────────────────────────────────────────

function initTestSim() {
  test.nBins = pipeline.trainedQTable.length;
  test.targetEpisodes = Math.max(1, readInt(ui.testEpisodesTarget, 20));
  test.maxSteps = Math.max(1, readInt(ui.testMaxSteps, 200));
  test.stepsPerFrame = Math.max(1, readInt(ui.testStepsPerFrame, 8));

  test.env = new MountainCarEnv({ maxEpisodeSteps: test.maxSteps });
  test.bins = makeBins(test.nBins, test.env.observationLow, test.env.observationHigh);
  test.greedyPolicy = new GreedyPolicy(pipeline.trainedQTable, 3);

  test.completedEpisodes = 0;
  test.successes = 0;
  test.totalReturn = 0;
  test.totalLength = 0;
  test.curSteps = 0;
  test.curReturn = 0;
  const { observation } = test.env.reset();
  test.curObs = observation;

  updateTestMetrics();
  testRenderer.draw(test.curObs, { action: null });
  ui.mTestStatus.textContent = "Ready";
  setBadge(ui.testStatusBadge, "Ready");
}

function testStepOnce() {
  const state = discretise(test.curObs, test.bins);
  const action = test.greedyPolicy.selectAction(state);
  const result = test.env.step(action);

  test.curObs = result.observation;
  test.curSteps += 1;
  test.curReturn += result.reward;

  if (result.terminated || result.truncated) {
    test.completedEpisodes += 1;
    if (result.terminated) test.successes += 1;
    test.totalReturn += test.curReturn;
    test.totalLength += test.curSteps;
    test.curSteps = 0;
    test.curReturn = 0;

    if (test.completedEpisodes >= test.targetEpisodes) {
      test.running = false;
      setBadge(ui.testStatusBadge, `✓ ${test.completedEpisodes} episodes`, "ok");
      ui.mTestStatus.textContent = "Done";
      setPipelineBadge("Step 4 ✓");
      return;
    }
    const { observation } = test.env.reset();
    test.curObs = observation;
  }
}

function testTick() {
  if (!test.running) {
    test.rafId = null;
    updateTestMetrics();
    testRenderer.draw(test.curObs, { action: null });
    return;
  }
  for (let i = 0; i < test.stepsPerFrame; i++) {
    if (!test.running) break;
    testStepOnce();
  }
  updateTestMetrics();
  testRenderer.draw(test.curObs, { action: null });
  if (test.running) {
    const msg = `Episode ${test.completedEpisodes + 1} / ${test.targetEpisodes}`;
    setBadge(ui.testStatusBadge, msg, "running");
    ui.mTestStatus.textContent = msg;
    setPipelineBadge("Testing…");
  }
  test.rafId = window.requestAnimationFrame(testTick);
}

function startTest() {
  if (!test.env || test.completedEpisodes >= test.targetEpisodes) {
    try { initTestSim(); }
    catch (err) { ui.mTestStatus.textContent = `Error: ${err.message}`; return; }
  }
  if (test.running) return;
  test.running = true;
  setBadge(ui.testStatusBadge, "Running…", "running");
  if (!test.rafId) test.rafId = window.requestAnimationFrame(testTick);
}

function pauseTest() {
  test.running = false;
  setBadge(ui.testStatusBadge, "Paused");
  ui.mTestStatus.textContent = "Paused";
}

function resetTest() {
  test.running = false;
  if (test.rafId) { window.cancelAnimationFrame(test.rafId); test.rafId = null; }
  try { initTestSim(); }
  catch (err) { ui.mTestStatus.textContent = `Error: ${err.message}`; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint: Save
// ─────────────────────────────────────────────────────────────────────────────

function saveEpisodes() {
  if (!pipeline.rawEpisodes.length) return;
  const payload = {
    format: "mountaincar-js-episodes-v1",
    n_bins: collect.nBins,
    max_steps: collect.maxSteps,
    policy: collect.policyName,
    p_smart: collect.pSmart,
    episodes: pipeline.rawEpisodes,
  };
  downloadJSON(payload, `episodes-${ts()}.json`);
}

function saveQTable() {
  if (!pipeline.trainedQTable) return;
  const payload = {
    format: "mountaincar-js-qtable-v1",
    n_bins: pipeline.qTableConfig.nBins,
    gamma: pipeline.qTableConfig.gamma,
    q_table: pipeline.trainedQTable,
  };
  downloadJSON(payload, `q_table-${ts()}.json`);
}

function downloadJSON(obj, filename) {
  const blob = new Blob([JSON.stringify(obj)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function ts() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint: Load
// ─────────────────────────────────────────────────────────────────────────────

async function handleLoadEpisodes(file) {
  if (!file) return;
  try {
    const json = JSON.parse(await file.text());
    if (!Array.isArray(json.episodes)) throw new Error("Expected `episodes` array.");
    pipeline.rawEpisodes = json.episodes;
    pipeline.phase = "collected";
    collect.nBins = json.n_bins ?? 20;
    collect.maxSteps = json.max_steps ?? 200;
    collect.policyName = json.policy ?? "mixture";
    collect.pSmart = json.p_smart ?? 0.5;
    collect.completedEpisodes = json.episodes.length;
    collect.episodes = json.episodes;

    // Update metric display
    ui.mEpisodes.textContent = String(collect.completedEpisodes);
    ui.mCollectStatus.textContent = `Loaded ${collect.completedEpisodes} episodes from file.`;
    setBadge(ui.collectStatusBadge, `✓ ${collect.completedEpisodes} episodes (loaded)`, "ok");
    setPipelineBadge("Step 1 ✓ (loaded)");
    ui.saveEpisodesBtn.disabled = false;

    // Unlock step 2
    unlockSection(ui.sectionQTable, ui.qInitStatusBadge, "Ready");
    ui.initQTableBtn.disabled = false;
    ui.nBins.value = String(collect.nBins);
  } catch (err) {
    alert(`Failed to load trajectories: ${err.message}`);
  }
}

async function handleLoadQTable(file) {
  if (!file) return;
  try {
    const json = JSON.parse(await file.text());
    const { qTable, nBinsHint } = extractQTablePayload(json);
    const nBins = validateQTable(qTable);
    pipeline.trainedQTable = qTable;
    pipeline.qTableConfig = { nBins, gamma: json.gamma ?? 0.99 };
    pipeline.phase = "trained";

    setBadge(ui.qInitStatusBadge, "✓ Loaded from file", "ok");
    setBadge(ui.trainStatusBadge, "✓ Loaded from file", "ok");
    setPipelineBadge("Steps 1–3 ✓ (loaded)");
    ui.saveQTableBtn.disabled = false;

    // Unlock steps 2–4
    unlockSection(ui.sectionQTable, ui.qInitStatusBadge, "Loaded");
    ui.initQTableBtn.disabled = false;
    unlockSection(ui.sectionTrain, ui.trainStatusBadge, "Loaded");
    ui.startTrainBtn.disabled = false;
    unlockSection(ui.sectionTest, ui.testStatusBadge, "Ready");
    ui.startTestBtn.disabled = false;
    ui.pauseTestBtn.disabled = false;
    ui.resetTestBtn.disabled = false;
    ui.nBins.value = String(nBinsHint ?? nBins);
    logLine(`Q-Table loaded from file: ${file.name} (n_bins=${nBins})`);
    // Reveal Q-value visualizations for the loaded table
    ui.qVizPanel.classList.remove("hidden");
    drawQValueHeatmap(ui.qValueCanvas, qTable, nBins);
    drawGreedyActionMap(ui.greedyMapCanvas, qTable, nBins);
  } catch (err) {
    alert(`Failed to load Q-table: ${err.message}`);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Event wiring
// ─────────────────────────────────────────────────────────────────────────────

function bindEvents() {
  // Step 1
  ui.startCollectBtn.addEventListener("click", startCollect);
  ui.pauseCollectBtn.addEventListener("click", pauseCollect);
  ui.resetCollectBtn.addEventListener("click", resetCollect);
  ui.stepsPerFrame.addEventListener("input", () => {
    ui.stepsPerFrameLabel.textContent = ui.stepsPerFrame.value;
    collect.stepsPerFrame = Math.max(1, readInt(ui.stepsPerFrame, 8));
  });

  // Step 2
  ui.initQTableBtn.addEventListener("click", initQTable);

  // Step 3
  ui.startTrainBtn.addEventListener("click", () => {
    runTraining().catch((err) => {
      logLine(`Error: ${err.message}`);
      setBadge(ui.trainStatusBadge, "Error", "warn");
    });
  });

  // Step 4
  ui.startTestBtn.addEventListener("click", startTest);
  ui.pauseTestBtn.addEventListener("click", pauseTest);
  ui.resetTestBtn.addEventListener("click", resetTest);
  ui.testStepsPerFrame.addEventListener("input", () => {
    ui.testStepsPerFrameLabel.textContent = ui.testStepsPerFrame.value;
    test.stepsPerFrame = Math.max(1, readInt(ui.testStepsPerFrame, 8));
  });

  // Pipeline bar: save
  ui.saveEpisodesBtn.addEventListener("click", saveEpisodes);
  ui.saveQTableBtn.addEventListener("click", saveQTable);

  // Pipeline bar: load
  ui.loadEpisodesFile.addEventListener("change", async (e) => {
    const [f] = e.target.files;
    await handleLoadEpisodes(f);
    e.target.value = ""; // allow re-load of same file
  });
  ui.loadQTableFile.addEventListener("change", async (e) => {
    const [f] = e.target.files;
    await handleLoadQTable(f);
    e.target.value = "";
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Boot
// ─────────────────────────────────────────────────────────────────────────────

function boot() {
  ui.stepsPerFrameLabel.textContent = ui.stepsPerFrame.value;
  ui.testStepsPerFrameLabel.textContent = ui.testStepsPerFrame.value;
  bindEvents();

  // Initial state: all sections beyond Step 1 are locked
  // (already locked via `section-locked` class in HTML)

  // Draw blank canvases
  collect.env = new MountainCarEnv({ maxEpisodeSteps: 200 });
  collect.bins = makeBins(20, collect.env.observationLow, collect.env.observationHigh);
  resetCollectEpisode();
  collectRenderer.draw(collect.curObs, { action: null });
  drawHeatmap(ui.heatmapCanvas, collect.visits, 20);
  collect.env = null; // will be re-initialized on Start
}

boot();
