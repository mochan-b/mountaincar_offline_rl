export function linspace(start, stop, count) {
  if (!Number.isInteger(count) || count < 2) {
    throw new Error(`count must be an integer >= 2. Got: ${count}`);
  }
  const out = new Array(count);
  const step = (stop - start) / (count - 1);
  for (let i = 0; i < count; i += 1) {
    out[i] = start + (step * i);
  }
  return out;
}

export function makeBins(nBins, low, high) {
  if (!Number.isInteger(nBins) || nBins < 2) {
    throw new Error(`nBins must be an integer >= 2. Got: ${nBins}`);
  }
  if (!Array.isArray(low) || !Array.isArray(high) || low.length !== high.length) {
    throw new Error("low/high must be arrays of equal length");
  }

  const bins = [];
  for (let i = 0; i < low.length; i += 1) {
    const edges = linspace(low[i], high[i], nBins + 1);
    bins.push(edges.slice(1, -1));
  }
  return bins;
}

function digitize(value, cuts) {
  let lo = 0;
  let hi = cuts.length;

  // np.digitize(value, cuts, right=False):
  // returns the first index where cuts[idx] > value.
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (value < cuts[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

export function discretise(obs, bins) {
  if (!Array.isArray(obs) || !Array.isArray(bins) || obs.length !== bins.length) {
    throw new Error("obs and bins must be arrays of equal length");
  }
  return obs.map((value, idx) => digitize(value, bins[idx]));
}

export function velocityBinMidpoints(nBins, velLow, velHigh) {
  const edges = linspace(velLow, velHigh, nBins + 1);
  const mids = new Array(nBins);
  for (let i = 0; i < nBins; i += 1) {
    mids[i] = 0.5 * (edges[i] + edges[i + 1]);
  }
  return mids;
}
