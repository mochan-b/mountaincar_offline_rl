import { createSeededRng } from "../utils/rng.js";

export class MountainCarEnv {
  constructor(options = {}) {
    this.minPosition = -1.2;
    this.maxPosition = 0.6;
    this.maxSpeed = 0.07;
    this.goalPosition = 0.5;
    this.goalVelocity = options.goalVelocity ?? 0.0;
    this.force = 0.001;
    this.gravity = 0.0025;
    this.maxEpisodeSteps = options.maxEpisodeSteps ?? 200;
    this._resetLow = options.resetLow ?? -0.6;
    this._resetHigh = options.resetHigh ?? -0.4;
    this._rng = options.rng ?? Math.random;

    this.state = null;
    this.stepCount = 0;
  }

  setSeed(seed) {
    this._rng = createSeededRng(seed);
  }

  get observationLow() {
    return [this.minPosition, -this.maxSpeed];
  }

  get observationHigh() {
    return [this.maxPosition, this.maxSpeed];
  }

  reset(options = {}) {
    if (options.seed !== undefined) {
      this.setSeed(options.seed);
    }

    const low = options.low ?? this._resetLow;
    const high = options.high ?? this._resetHigh;
    const position = low + this._rng() * (high - low);
    this.state = [position, 0.0];
    this.stepCount = 0;
    return { observation: [...this.state], info: {} };
  }

  step(action) {
    if (!Number.isInteger(action) || action < 0 || action > 2) {
      throw new Error(`Invalid action: ${action}. Expected integer in [0, 2].`);
    }
    if (this.state === null) {
      throw new Error("Environment must be reset() before step().");
    }

    let [position, velocity] = this.state;

    velocity += ((action - 1) * this.force) + (Math.cos(3 * position) * (-this.gravity));
    velocity = Math.min(Math.max(velocity, -this.maxSpeed), this.maxSpeed);

    position += velocity;
    position = Math.min(Math.max(position, this.minPosition), this.maxPosition);
    if (position === this.minPosition && velocity < 0) {
      velocity = 0.0;
    }

    const terminated = position >= this.goalPosition && velocity >= this.goalVelocity;
    this.stepCount += 1;
    const truncated = this.stepCount >= this.maxEpisodeSteps;
    const reward = -1.0;

    this.state = [position, velocity];
    return {
      observation: [...this.state],
      reward,
      terminated,
      truncated,
      info: {},
    };
  }
}
