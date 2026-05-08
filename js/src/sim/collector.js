import { makeBins, discretise, velocityBinMidpoints } from "../env/discretization.js";
import { MountainCarEnv } from "../env/mountainCarEnv.js";
import { MixturePolicy, RandomPolicy } from "../policies.js";
import { createSeededRng, splitSeed } from "../utils/rng.js";

function buildPolicy(policyName, nActions, nBins, env, pSmart, rng) {
  if (policyName === "mixture") {
    const mids = velocityBinMidpoints(nBins, env.observationLow[1], env.observationHigh[1]);
    return new MixturePolicy(nActions, mids, { pSmart, rng });
  }
  if (policyName === "random") {
    return new RandomPolicy(nActions, rng);
  }
  throw new Error(`Unknown policyName '${policyName}'. Expected 'random' or 'mixture'.`);
}

export function collectEpisodes(options = {}) {
  const nEpisodes = options.nEpisodes ?? 50_000;
  const maxSteps = options.maxSteps ?? 200;
  const nBins = options.nBins ?? 20;
  const policyName = options.policyName ?? "mixture";
  const pSmart = options.pSmart ?? 0.5;
  const onEpisode = options.onEpisode ?? null;

  let envRng = Math.random;
  let policyRng = Math.random;
  if (options.seed !== undefined) {
    const [envSeed, policySeed] = splitSeed(options.seed);
    envRng = createSeededRng(envSeed);
    policyRng = createSeededRng(policySeed);
  }

  const env = new MountainCarEnv({
    maxEpisodeSteps: maxSteps,
    rng: envRng,
  });
  const bins = makeBins(nBins, env.observationLow, env.observationHigh);
  const policy = buildPolicy(policyName, 3, nBins, env, pSmart, policyRng);

  const episodes = [];
  let nSuccess = 0;
  let totalSteps = 0;

  for (let epIdx = 0; epIdx < nEpisodes; epIdx += 1) {
    let { observation } = env.reset();
    const trajectory = [];
    let epSuccess = false;

    for (let step = 0; step < maxSteps; step += 1) {
      const state = discretise(observation, bins);
      const action = policy.selectAction(state);
      const behaviorProb = policy.prob(state, action);

      const result = env.step(action);
      trajectory.push([state, action, result.reward, behaviorProb]);
      observation = result.observation;

      if (result.terminated) {
        epSuccess = true;
        break;
      }
      if (result.truncated) {
        break;
      }
    }

    if (epSuccess) {
      nSuccess += 1;
    }
    totalSteps += trajectory.length;
    episodes.push(trajectory);

    if (typeof onEpisode === "function") {
      onEpisode({
        episodeIndex: epIdx,
        trajectory,
        success: epSuccess,
        successes: nSuccess,
      });
    }
  }

  return {
    episodes,
    stats: {
      nEpisodes,
      nSuccess,
      avgLength: nEpisodes > 0 ? (totalSteps / nEpisodes) : 0.0,
      successRate: nEpisodes > 0 ? (nSuccess / nEpisodes) : 0.0,
      policyName,
      nBins,
      maxSteps,
      pSmart,
    },
  };
}
