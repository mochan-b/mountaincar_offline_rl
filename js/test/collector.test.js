import test from "node:test";
import assert from "node:assert/strict";

import { collectEpisodes } from "../src/sim/collector.js";

test("collector returns trajectory schema compatible with python format", () => {
  const { episodes, stats } = collectEpisodes({
    nEpisodes: 5,
    maxSteps: 20,
    nBins: 20,
    policyName: "random",
    seed: 123,
  });

  assert.equal(episodes.length, 5);
  assert.equal(stats.nEpisodes, 5);
  assert.equal(stats.policyName, "random");

  for (const ep of episodes) {
    assert.ok(ep.length >= 1 && ep.length <= 20);
    for (const step of ep) {
      assert.equal(step.length, 4);
      const [state, action, reward, behaviorProb] = step;
      assert.equal(state.length, 2);
      assert.ok(Number.isInteger(state[0]));
      assert.ok(Number.isInteger(state[1]));
      assert.ok(action >= 0 && action <= 2);
      assert.equal(reward, -1.0);
      assert.equal(behaviorProb, 1 / 3);
    }
  }
});

test("collector is deterministic for fixed seed", () => {
  const runA = collectEpisodes({
    nEpisodes: 3,
    maxSteps: 10,
    nBins: 20,
    policyName: "mixture",
    pSmart: 0.5,
    seed: 42,
  });
  const runB = collectEpisodes({
    nEpisodes: 3,
    maxSteps: 10,
    nBins: 20,
    policyName: "mixture",
    pSmart: 0.5,
    seed: 42,
  });

  assert.deepEqual(runA.episodes, runB.episodes);
  assert.deepEqual(runA.stats, runB.stats);
});
