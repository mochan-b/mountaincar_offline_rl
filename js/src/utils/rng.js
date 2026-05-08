function normalizeSeed(seed) {
  if (!Number.isFinite(seed)) {
    throw new Error(`Invalid seed: ${seed}`);
  }
  return (Math.trunc(seed) >>> 0);
}

export function createSeededRng(seed) {
  let state = normalizeSeed(seed);
  return function next() {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomInt(rng, maxExclusive) {
  if (!Number.isInteger(maxExclusive) || maxExclusive <= 0) {
    throw new Error(`maxExclusive must be a positive integer. Got: ${maxExclusive}`);
  }
  return Math.floor(rng() * maxExclusive);
}

export function splitSeed(seed) {
  const base = normalizeSeed(seed);
  return [base ^ 0x9e3779b9, base ^ 0x85ebca6b];
}
