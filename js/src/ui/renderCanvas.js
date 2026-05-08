export class MountainCarRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.minPosition = -1.2;
    this.maxPosition = 0.6;
    this.goalPosition = 0.5;
    this._configureHiDpi();
  }

  _configureHiDpi() {
    const { canvas } = this;
    const ratio = Math.max(1, window.devicePixelRatio || 1);
    const displayW = canvas.width;
    const displayH = canvas.height;
    canvas.width = Math.floor(displayW * ratio);
    canvas.height = Math.floor(displayH * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = displayW;
    this.height = displayH;
  }

  _height(x) {
    return (Math.sin(3 * x) * 0.45) + 0.55;
  }

  _worldToCanvasX(x) {
    return ((x - this.minPosition) / (this.maxPosition - this.minPosition)) * this.width;
  }

  _worldToCanvasY(yNorm) {
    return this.height - (yNorm * (this.height * 0.84));
  }

  draw(state, meta = {}) {
    if (!Array.isArray(state) || state.length !== 2) {
      return;
    }
    const [position] = state;
    const action = meta.action ?? null;

    this.ctx.clearRect(0, 0, this.width, this.height);

    const sky = this.ctx.createLinearGradient(0, 0, 0, this.height);
    sky.addColorStop(0, "#fdf6e7");
    sky.addColorStop(1, "#dbe8df");
    this.ctx.fillStyle = sky;
    this.ctx.fillRect(0, 0, this.width, this.height);

    const path = new Path2D();
    const points = 120;
    for (let i = 0; i <= points; i += 1) {
      const x = this.minPosition + ((this.maxPosition - this.minPosition) * i / points);
      const y = this._height(x);
      const cx = this._worldToCanvasX(x);
      const cy = this._worldToCanvasY(y);
      if (i === 0) {
        path.moveTo(cx, cy);
      } else {
        path.lineTo(cx, cy);
      }
    }
    this.ctx.lineWidth = 3;
    this.ctx.strokeStyle = "#28413f";
    this.ctx.stroke(path);

    this._drawGoalFlag();
    this._drawCar(position, action);
  }

  _drawGoalFlag() {
    const x = this._worldToCanvasX(this.goalPosition);
    const baseY = this._worldToCanvasY(this._height(this.goalPosition));
    const topY = baseY - 56;

    this.ctx.strokeStyle = "#1a2529";
    this.ctx.lineWidth = 3;
    this.ctx.beginPath();
    this.ctx.moveTo(x, baseY);
    this.ctx.lineTo(x, topY);
    this.ctx.stroke();

    this.ctx.fillStyle = "#f7b733";
    this.ctx.beginPath();
    this.ctx.moveTo(x, topY);
    this.ctx.lineTo(x + 26, topY + 8);
    this.ctx.lineTo(x, topY + 16);
    this.ctx.closePath();
    this.ctx.fill();
  }

  _drawCar(position, action) {
    const x = this._worldToCanvasX(position);
    const y = this._worldToCanvasY(this._height(position)) - 14;
    const rot = -Math.cos(3 * position);
    const carW = 46;
    const carH = 20;

    this.ctx.save();
    this.ctx.translate(x, y);
    this.ctx.rotate(rot);

    let bodyColor = "#2f4f4f";
    if (action === 0) {
      bodyColor = "#2c6e8f";
    } else if (action === 2) {
      bodyColor = "#ad5b2e";
    }

    this.ctx.fillStyle = bodyColor;
    this.ctx.fillRect(-carW / 2, -carH, carW, carH);

    this.ctx.fillStyle = "#7a7d82";
    this.ctx.beginPath();
    this.ctx.arc(-carW * 0.24, 0, 6.5, 0, Math.PI * 2);
    this.ctx.arc(carW * 0.24, 0, 6.5, 0, Math.PI * 2);
    this.ctx.fill();

    this.ctx.restore();
  }
}

export function drawHeatmap(canvas, counts, nBins) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fffdf8";
  ctx.fillRect(0, 0, w, h);

  if (!Array.isArray(counts) || nBins <= 0) {
    return;
  }

  const maxCount = counts.reduce(
    (outerMax, row) => Math.max(outerMax, row.reduce((innerMax, c) => Math.max(innerMax, c), 0)),
    0,
  );
  const denom = maxCount > 0 ? maxCount : 1;
  const cell = Math.floor(Math.min(w, h) / nBins);
  const startX = Math.floor((w - (cell * nBins)) / 2);
  const startY = Math.floor((h - (cell * nBins)) / 2);

  for (let x = 0; x < nBins; x += 1) {
    for (let y = 0; y < nBins; y += 1) {
      const ratio = counts[x][y] / denom;
      const hue = 38 - (38 * ratio);
      const sat = 90;
      const light = 97 - (56 * ratio);
      ctx.fillStyle = `hsl(${hue}deg ${sat}% ${light}%)`;
      const drawY = startY + ((nBins - 1 - y) * cell);
      ctx.fillRect(startX + (x * cell), drawY, cell, cell);
    }
  }

  ctx.strokeStyle = "rgba(17, 28, 30, 0.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(startX, startY, cell * nBins, cell * nBins);
}

// ─── Shared helpers for Q-viz functions ────────────────────────────────────

// MountainCar-v0 observation space bounds
const MC_POS_MIN = -1.2;
const MC_POS_MAX = 0.6;
const MC_VEL_MIN = -0.07;
const MC_VEL_MAX = 0.07;

/**
 * Return the world-space midpoint of bin `i` along an axis [worldMin, worldMax].
 */
function binMidpoint(i, nBins, worldMin, worldMax) {
  return worldMin + (worldMax - worldMin) * (i + 0.5) / nBins;
}

/**
 * Draw a complete chart frame: background, grid cells (via fillCell callback),
 * tick marks + numeric labels on both axes, colorbar (optional), and axis titles.
 *
 * Layout (all in canvas logical pixels):
 *   marginLeft   – room for y-axis tick labels + title
 *   marginBottom – room for x-axis tick labels + title
 *   marginTop    – breathing room at top
 *   marginRight  – colorbar width (if any) + right padding
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} w  canvas logical width
 * @param {number} h  canvas logical height
 * @param {number} nBins
 * @param {object} opts
 * @param {function} opts.fillCell  (ctx, pi, vi, cx, cy, cw, ch) → fills the cell
 * @param {string}  opts.xTitle
 * @param {string}  opts.yTitle
 * @param {number[]} opts.xTicks  world-space values to mark on x-axis
 * @param {number[]} opts.yTicks  world-space values to mark on y-axis
 * @param {function|null} opts.drawColorbar  (ctx, x, y, w, h) → optional
 * @returns {{ gridX, gridY, gridW, gridH, cellW, cellH }}  grid geometry
 */
function drawChartFrame(ctx, w, h, nBins, opts) {
  const {
    fillCell,
    xTitle,
    yTitle,
    xTicks,
    yTicks,
    drawColorbar = null,
  } = opts;

  const FONT_SIZE = 10;   // tick label size (px)
  const TITLE_SIZE = 11;   // axis title size (px)
  const TICK_LEN = 4;    // tick mark length (px)
  const CBAR_W = drawColorbar ? 16 : 0;
  const CBAR_GAP = drawColorbar ? 8 : 0;
  const CBAR_LABEL_W = drawColorbar ? 32 : 0;

  const marginLeft = 55;
  const marginBottom = opts.marginBottom ?? 44;   // caller can override
  const marginTop = 12;
  const marginRight = CBAR_W + CBAR_GAP + CBAR_LABEL_W + 8;

  const gridW = w - marginLeft - marginRight;
  const gridH = h - marginTop - marginBottom;
  const gridX = marginLeft;
  const gridY = marginTop;

  // Cell dimensions (may be fractional — use floats for crisp rendering)
  const cellW = gridW / nBins;
  const cellH = gridH / nBins;

  // Background
  ctx.fillStyle = "#0d1117";
  ctx.fillRect(0, 0, w, h);

  // ── Fill cells ──────────────────────────────────────────────────────────
  for (let pi = 0; pi < nBins; pi++) {
    for (let vi = 0; vi < nBins; vi++) {
      const cx = gridX + pi * cellW;
      const cy = gridY + (nBins - 1 - vi) * cellH;
      fillCell(ctx, pi, vi, cx, cy, cellW, cellH);
    }
  }

  // ── Subtle cell grid lines ───────────────────────────────────────────────
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= nBins; i++) {
    const x = gridX + i * cellW;
    const y = gridY + i * cellH;
    ctx.beginPath(); ctx.moveTo(x, gridY); ctx.lineTo(x, gridY + gridH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(gridX, y); ctx.lineTo(gridX + gridW, y); ctx.stroke();
  }

  // ── Grid border ─────────────────────────────────────────────────────────
  ctx.strokeStyle = "rgba(255,255,255,0.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(gridX, gridY, gridW, gridH);

  // ── X-axis ticks + labels ───────────────────────────────────────────────
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = 1;
  ctx.font = `${FONT_SIZE}px monospace`;
  ctx.textBaseline = "top";
  ctx.textAlign = "center";

  for (const worldVal of xTicks) {
    // Bin whose left edge contains worldVal (clamp to grid)
    const tNorm = (worldVal - MC_POS_MIN) / (MC_POS_MAX - MC_POS_MIN);
    const px = gridX + tNorm * gridW;
    if (px < gridX - 1 || px > gridX + gridW + 1) continue;

    // Tick mark
    ctx.beginPath();
    ctx.moveTo(px, gridY + gridH);
    ctx.lineTo(px, gridY + gridH + TICK_LEN);
    ctx.stroke();

    // Label
    ctx.fillText(worldVal.toFixed(2), px, gridY + gridH + TICK_LEN + 2);
  }

  // X-axis title
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = `${TITLE_SIZE}px sans-serif`;
  ctx.textBaseline = "bottom";
  ctx.fillText(xTitle, gridX + gridW / 2, h - 2);

  // ── Y-axis ticks + labels ───────────────────────────────────────────────
  ctx.font = `${FONT_SIZE}px monospace`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "right";

  for (const worldVal of yTicks) {
    const tNorm = (worldVal - MC_VEL_MIN) / (MC_VEL_MAX - MC_VEL_MIN);
    // y=0 is at top; bin 0 (min vel) is at the bottom of the grid
    const py = gridY + gridH - tNorm * gridH;
    if (py < gridY - 1 || py > gridY + gridH + 1) continue;

    // Tick mark
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.beginPath();
    ctx.moveTo(gridX - TICK_LEN, py);
    ctx.lineTo(gridX, py);
    ctx.stroke();

    // Label (3 significant digits)
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillText(worldVal.toFixed(3), gridX - TICK_LEN - 3, py);
  }

  // Y-axis title (rotated)
  ctx.save();
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = `${TITLE_SIZE}px sans-serif`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  ctx.translate(9, gridY + gridH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yTitle, 0, 0);
  ctx.restore();

  // ── Colorbar ─────────────────────────────────────────────────────────────
  if (drawColorbar) {
    const cbX = gridX + gridW + CBAR_GAP;
    const cbY = gridY;
    const cbH = gridH;
    drawColorbar(ctx, cbX, cbY, CBAR_W, cbH, FONT_SIZE);
  }

  return { gridX, gridY, gridW, gridH, cellW, cellH };
}

// Default tick locations
const POS_TICKS = [-1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6];
const VEL_TICKS = [-0.07, -0.035, 0.0, 0.035, 0.07];

/** Q-value color for a normalised t in [0,1] */
function qColor(t) {
  const hue = 260 - t * 200; // indigo → teal → yellow-green
  const sat = 70 + t * 20;
  const light = 18 + t * 52;
  return `hsl(${hue}deg ${sat}% ${light}%)`;
}

/**
 * Draw a heatmap of max Q(s,·) across actions.
 * Color scale: deep indigo (low) → teal → yellow-green (high).
 * Axes show the real MountainCar state ranges.
 */
export function drawQValueHeatmap(canvas, qTable, nBins) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  if (!Array.isArray(qTable) || nBins <= 0) {
    ctx.clearRect(0, 0, w, h);
    return;
  }

  // Precompute max-Q per cell and global min/max
  let globalMin = Infinity;
  let globalMax = -Infinity;
  const maxQ = Array.from({ length: nBins }, (_, pi) =>
    Array.from({ length: nBins }, (_, vi) => {
      const m = Math.max(...qTable[pi][vi]);
      if (m < globalMin) globalMin = m;
      if (m > globalMax) globalMax = m;
      return m;
    }),
  );
  const range = globalMax - globalMin || 1;

  drawChartFrame(ctx, w, h, nBins, {
    xTitle: "Position",
    yTitle: "Velocity",
    xTicks: POS_TICKS,
    yTicks: VEL_TICKS,

    fillCell(c, pi, vi, cx, cy, cw, ch) {
      const t = (maxQ[pi][vi] - globalMin) / range;
      c.fillStyle = qColor(t);
      c.fillRect(cx, cy, cw + 0.5, ch + 0.5); // +0.5 closes hairline gaps
    },

    drawColorbar(c, cbX, cbY, cbW, cbH, fontSize) {
      // Gradient strip
      for (let py = 0; py < cbH; py++) {
        const t = 1 - py / cbH;
        c.fillStyle = qColor(t);
        c.fillRect(cbX, cbY + py, cbW, 1);
      }
      // Border
      c.strokeStyle = "rgba(255,255,255,0.3)";
      c.lineWidth = 1;
      c.strokeRect(cbX, cbY, cbW, cbH);
      // Max/min labels
      c.fillStyle = "rgba(255,255,255,0.7)";
      c.font = `${fontSize}px monospace`;
      c.textAlign = "left";
      c.textBaseline = "top";
      c.fillText(globalMax.toFixed(0), cbX + cbW + 4, cbY);
      c.textBaseline = "bottom";
      c.fillText(globalMin.toFixed(0), cbX + cbW + 4, cbY + cbH);
      // Middle
      c.textBaseline = "middle";
      const mid = (globalMin + globalMax) / 2;
      c.fillText(mid.toFixed(0), cbX + cbW + 4, cbY + cbH / 2);
    },
  });
}

/**
 * Draw the greedy action per state (argmax over Q-values).
 * Colors: push-left = blue (#2e86de), coast = gray (#7f8c8d), push-right = orange (#e67e22).
 * Axes show the real MountainCar state ranges.
 */
export function drawGreedyActionMap(canvas, qTable, nBins) {
  const ACTION_COLORS = ["#2e86de", "#7f8c8d", "#e67e22"]; // left, coast, right
  const ACTION_ARROWS = ["←", "·", "→"];

  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  if (!Array.isArray(qTable) || nBins <= 0) {
    ctx.clearRect(0, 0, w, h);
    return;
  }

  const { gridX, gridY, gridW, gridH, cellW, cellH } = drawChartFrame(ctx, w, h, nBins, {
    xTitle: "Position",
    yTitle: "Velocity",
    xTicks: POS_TICKS,
    yTicks: VEL_TICKS,
    drawColorbar: null,
    marginBottom: 64,   // extra room for legend row between tick labels and axis title

    fillCell(c, pi, vi, cx, cy, cw, ch) {
      const row = qTable[pi][vi];
      let best = 0;
      for (let a = 1; a < 3; a++) if (row[a] > row[best]) best = a;

      c.fillStyle = ACTION_COLORS[best] + "bb";
      c.fillRect(cx, cy, cw + 0.5, ch + 0.5);

      // Arrow glyph inside cell
      const minDim = Math.min(cw, ch);
      if (minDim >= 9) {
        const fs = Math.max(7, Math.floor(minDim * 0.52));
        c.fillStyle = "rgba(255,255,255,0.85)";
        c.font = `bold ${fs}px sans-serif`;
        c.textAlign = "center";
        c.textBaseline = "middle";
        c.fillText(ACTION_ARROWS[best], cx + cw / 2, cy + ch / 2);
      }
    },
  });

  // ── Action legend: sits between the x-axis tick labels and the axis title ──
  // gridY + gridH = bottom edge of grid; add TICK_LEN + tick font + gap to
  // clear the tick labels, then centre the legend row in the remaining space.
  const TICK_LEN = 4;
  const TICK_FONT = 10;
  const legendTopClearance = TICK_LEN + TICK_FONT + 8;          // px below grid edge
  const legendFontSize = 10;
  const legendY = gridY + gridH + legendTopClearance + legendFontSize / 2 + 2;
  const spacing = gridW / 3;
  const labels = ["← push-left", "· coast", "→ push-right"];

  ctx.font = `bold ${legendFontSize}px sans-serif`;
  ctx.textBaseline = "middle";
  ctx.textAlign = "center";
  for (let i = 0; i < 3; i++) {
    // Color swatch
    ctx.fillStyle = ACTION_COLORS[i] + "cc";
    ctx.fillRect(gridX + spacing * i + spacing / 2 - 28, legendY - 6, 12, 12);
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 0.5;
    ctx.strokeRect(gridX + spacing * i + spacing / 2 - 28, legendY - 6, 12, 12);
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillText(labels[i], gridX + spacing * i + spacing / 2 + 6, legendY);
  }
}
