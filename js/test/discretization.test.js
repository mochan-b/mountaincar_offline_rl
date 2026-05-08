import test from "node:test";
import assert from "node:assert/strict";

import { makeBins, discretise, velocityBinMidpoints } from "../src/env/discretization.js";

function closeTo(actual, expected, eps = 1e-12) {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `Expected ${actual} to be within ${eps} of ${expected}`,
  );
}

test("makeBins returns interior cut points only", () => {
  const bins = makeBins(5, [-1.0, -0.5], [1.0, 0.5]);
  assert.equal(bins.length, 2);
  assert.equal(bins[0].length, 4);
  assert.equal(bins[1].length, 4);
  assert.deepEqual(bins[0], [-0.6, -0.19999999999999996, 0.20000000000000018, 0.6000000000000001]);
});

test("discretise matches np.digitize style semantics", () => {
  const bins = [
    [-1.0, 0.0, 1.0],
    [-0.1, 0.1, 0.2],
  ];
  assert.deepEqual(discretise([-2.0, -0.2], bins), [0, 0]);
  assert.deepEqual(discretise([-1.0, -0.1], bins), [1, 1]);
  assert.deepEqual(discretise([0.0, 0.15], bins), [2, 2]);
  assert.deepEqual(discretise([5.0, 1.0], bins), [3, 3]);
});

test("velocityBinMidpoints mirrors python midpoint logic", () => {
  const mids = velocityBinMidpoints(4, -0.08, 0.08);
  assert.equal(mids.length, 4);
  closeTo(mids[0], -0.06);
  closeTo(mids[1], -0.02);
  closeTo(mids[2], 0.02);
  closeTo(mids[3], 0.06);
});
