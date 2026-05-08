import test from "node:test";
import assert from "node:assert/strict";

import { MountainCarEnv } from "../src/env/mountainCarEnv.js";

function closeTo(actual, expected, eps = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`,
  );
}

test("reset initializes position in range and velocity at zero", () => {
  const env = new MountainCarEnv();
  const { observation } = env.reset({ seed: 123 });
  assert.ok(observation[0] >= -0.6 && observation[0] <= -0.4);
  closeTo(observation[1], 0.0);
});

test("step follows MountainCar transition dynamics", () => {
  const env = new MountainCarEnv();
  env.state = [-0.5, 0.0];

  const action = 2;
  const expectedVelocity = 0.0 + ((action - 1) * env.force) + (Math.cos(3 * -0.5) * (-env.gravity));
  const expectedPosition = -0.5 + expectedVelocity;

  const out = env.step(action);
  closeTo(out.observation[1], expectedVelocity);
  closeTo(out.observation[0], expectedPosition);
  assert.equal(out.reward, -1.0);
  assert.equal(out.terminated, false);
  assert.equal(out.truncated, false);
});

test("left wall collision is inelastic (velocity reset to zero)", () => {
  const env = new MountainCarEnv();
  env.state = [-1.2, -0.02];

  const out = env.step(0);
  assert.equal(out.observation[0], -1.2);
  assert.equal(out.observation[1], 0.0);
});

test("episode truncates at maxEpisodeSteps", () => {
  const env = new MountainCarEnv({ maxEpisodeSteps: 3 });
  env.state = [-0.5, 0.0];
  assert.equal(env.step(1).truncated, false);
  assert.equal(env.step(1).truncated, false);
  assert.equal(env.step(1).truncated, true);
});

test("termination occurs when position reaches goal", () => {
  const env = new MountainCarEnv();
  env.state = [0.4995, 0.01];

  const out = env.step(2);
  assert.equal(out.terminated, true);
});
