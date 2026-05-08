import test from "node:test";
import assert from "node:assert/strict";

import { GreedyPolicy, MixturePolicy, RandomPolicy } from "../src/policies.js";

test("RandomPolicy exposes uniform probability", () => {
  const policy = new RandomPolicy(3, () => 0.123);
  assert.equal(policy.prob([0, 0], 0), 1 / 3);
  assert.equal(policy.prob([0, 0], 1), 1 / 3);
  assert.equal(policy.prob([0, 0], 2), 1 / 3);
  assert.equal(policy.selectAction([0, 0]), 0);
});

test("MixturePolicy probability matches python formula", () => {
  const policy = new MixturePolicy(3, [-0.02, 0.02], { pSmart: 0.5, rng: () => 0.0 });
  const stateLeft = [10, 0];
  assert.equal(policy.prob(stateLeft, 0), 2 / 3);
  assert.equal(policy.prob(stateLeft, 1), 1 / 6);
  assert.equal(policy.prob(stateLeft, 2), 1 / 6);
  assert.equal(policy.selectAction(stateLeft), 0);
});

test("GreedyPolicy picks lowest-index action on ties", () => {
  const q = [
    [
      [1.0, 1.0, 0.0],
    ],
  ];
  const policy = new GreedyPolicy(q, 3);
  assert.equal(policy.selectAction([0, 0]), 0);
  assert.equal(policy.prob([0, 0], 0), 1.0);
  assert.equal(policy.prob([0, 0], 1), 0.0);
});
