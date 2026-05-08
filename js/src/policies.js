import { randomInt } from "./utils/rng.js";

function argmax(values) {
  if (!Array.isArray(values) || values.length === 0) {
    throw new Error("argmax requires a non-empty array");
  }
  let bestIdx = 0;
  let bestVal = values[0];
  for (let i = 1; i < values.length; i += 1) {
    if (values[i] > bestVal) {
      bestVal = values[i];
      bestIdx = i;
    }
  }
  return bestIdx;
}

export class RandomPolicy {
  constructor(nActions, rng = Math.random) {
    if (!Number.isInteger(nActions) || nActions <= 0) {
      throw new Error(`nActions must be a positive integer. Got: ${nActions}`);
    }
    this.nActions = nActions;
    this._rng = rng;
    this._prob = 1.0 / nActions;
  }

  selectAction(_state) {
    return randomInt(this._rng, this.nActions);
  }

  prob(_state, _action) {
    return this._prob;
  }
}

export class GreedyPolicy {
  constructor(qTable, nActions) {
    this.qTable = qTable;
    this.nActions = nActions;
  }

  selectAction(state) {
    const [posBin, velBin] = state;
    return argmax(this.qTable[posBin][velBin]);
  }

  prob(state, action) {
    return action === this.selectAction(state) ? 1.0 : 0.0;
  }

  update(qTable) {
    this.qTable = qTable;
  }
}

export class MixturePolicy {
  constructor(nActions, velBinMidpoints, options = {}) {
    if (!Number.isInteger(nActions) || nActions <= 0) {
      throw new Error(`nActions must be a positive integer. Got: ${nActions}`);
    }
    this.nActions = nActions;
    this.velBinMidpoints = velBinMidpoints;
    this.pSmart = options.pSmart ?? 0.5;
    this._rng = options.rng ?? Math.random;
    this._pRandom = 1.0 / nActions;
  }

  _smartAction(velBin) {
    const v = velBin < this.velBinMidpoints.length ? this.velBinMidpoints[velBin] : 0.0;
    return v >= 0 ? 2 : 0;
  }

  selectAction(state) {
    const velBin = state[1];
    if (this._rng() < this.pSmart) {
      return this._smartAction(velBin);
    }
    return randomInt(this._rng, this.nActions);
  }

  prob(state, action) {
    const velBin = state[1];
    const smartAction = this._smartAction(velBin);
    const smart = action === smartAction ? this.pSmart : 0.0;
    const random = (1.0 - this.pSmart) * this._pRandom;
    return smart + random;
  }
}
