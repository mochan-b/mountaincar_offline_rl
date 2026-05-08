# MountainCar Off-Policy Importance Sampling

This project implements **Off-Policy Monte Carlo Control with Weighted Importance Sampling** (Sutton & Barto, Algorithm 5.7) to solve the classic `MountainCar-v0` environment from Gymnasium. It has two independent but algorithmically identical implementations:

1. **Python backend** — Batch data collection and offline training (50K+ episodes, ~373 MB datasets).
2. **JavaScript/HTML frontend** — Interactive browser UI for live visualization, training, and testing.

## Big Picture

```
Phase 1: Data Collection          Phase 2: Offline Training       Phase 3: Evaluation
┌──────────────────────┐         ┌──────────────────────┐        ┌──────────────────┐
│ Behavior Policy      │         │ Pass 1: Ordinary MC  │        │ Greedy Policy    │
│ (random or mixture)  │ ──────► │   → Q^b              │ ─────► │ over Q^b         │
│                      │         │                      │        │                  │
│ Output: episodes.pkl │         │ Pass 2: Weighted IS  │        │ Goal: reach flag │
│ (S, A, R, prob)      │         │   → Q^π with fixed   │        │ at x=0.5 before  │
│                      │         │     greedy(Q^b)      │        │ 200 steps expire │
└──────────────────────┘         └──────────────────────┘        └──────────────────┘
```

A **two-pass approach** avoids the "zero-update problem" of pure single-pass IS with a deterministic greedy target: Pass 1 builds a meaningful Q^b via ordinary every-visit MC, then Pass 2 applies weighted IS correction using a fixed greedy target derived from Q^b. The final policy uses greedy(Q^b) — the off-policy improvement result — because it has complete state-action coverage.

## Repository Structure

### Python Backend (`src/`)

| File | Purpose |
|---|---|
| `src/collect_data.py` | **Phase 1:** Runs episodes under a behavior policy (random or mixture), records `(discrete_state, action, reward, behavior_prob)` tuples, and pickles them to `data/`. CLI flags: `--policy`, `--p_smart`, `--n_episodes`, `--n_bins`, `--out`, `--seed`. |
| `src/train_is.py` | **Phase 2:** Loads episodes from Phase 1, runs two-pass training (ordinary MC → Q^b, then weighted IS → Q^π), saves the Q-table to `results/`, and runs a 100-episode greedy evaluation. CLI flags: `--data`, `--out_q`, `--n_bins`, `--gamma`, `--n_eval`. |
| `src/env_utils.py` | State discretization: `make_bins()` creates bin cut points from the observation space bounds, `discretise()` maps a continuous observation to integer bin indices. |
| `src/policies.py` | Three policy classes with `select_action(state)` and `prob(state, action)` methods: `RandomPolicy` (uniform), `MixturePolicy` (velocity-oscillation heuristic + random), and `GreedyPolicy` (deterministic argmax over a Q-table). |
| `src/debug_qtable.py` | Diagnostic script: loads a saved Q-table and prints per-action statistics, greedy action distribution, and sample state values. |

### JavaScript Frontend (`js/`)

| File | Purpose |
|---|---|
| `js/index.html` | Single-page app with a 4-step pipeline UI: (1) Collect Data, (2) Build Q-Table, (3) Train Model, (4) Test Model. Includes canavs rendering, metrics panels, progress bars, and checkpoint save/load. |
| `js/package.json` | Node.js package. `npm start` serves via Python HTTP server, `npm test` runs the native Node.js test runner. |
| `js/src/env/discretization.js` | JS port of `env_utils.py` — `makeBins()`, `digitize()`, `discretise()`, `velocityBinMidpoints()`. |
| `js/src/env/mountainCarEnv.js` | Stand-alone JS reimplementation of MountainCar-v0 physics (gravity with cos(3x) terrain, inelastic left-wall collision, goal termination, step truncation). Supports seeded PRNG. |
| `js/src/policies.js` | JS port of all three policy classes (Random, Mixture, Greedy) with configurable RNG. |
| `js/src/sim/collector.js` | JS port of `collect_data.py` — runs the collection loop, supports a callback for live progress and returns episodes + stats. |
| `js/src/utils/rng.js` | Seeded PRNG (Lehmer/Marsaglia style) with `randomInt()` and `splitSeed()` for deterministic reproducibility. |
| `js/src/ui/app.js` | Main UI application (~917 lines): manages pipeline state, wires form controls, runs async training (Pass 1 + Pass 2 with event loop yielding), renders canvases, and handles JSON checkpointing. |
| `js/src/ui/renderCanvas.js` | Canvas rendering: mountain car scene (terrain profile, car with wheels, goal flag), state visitation heatmap, Q-value max heatmap, and greedy action map with arrow glyphs. |
| `js/src/ui/styles.css` | Full stylesheet: card-based layout, sticky pipeline bar, locked-section overlays, animated blobs, responsive design. |
| `js/src/index.js` | Barrel re-export module. |
| `js/test/` | Node.js test suite covering the collector, discretization, environment physics, and policies. All tests verify parity with the Python implementation. |

### Planning Documents (`plans/`)

| File | Purpose |
|---|---|
| `plans/prompt.md` | Original project brief: "make a mountaincar RL model using importance sampling" with a random behavior policy and deterministic greedy target, split into data collection then training. |
| `plans/web_implementation_plan.md` | 4-phase plan for the web UI matching the final implementation. |

### Data & Results

| Directory | Contents |
|---|---|
| `data/` | Pickled episode trajectories at various sizes (1K, 5K, 10K, 50K episodes) from both random and mixture behavior policies. |
| `results/` | Trained Q-table pickles containing `q_table` (final Q^b), `q_b`, `q_pi`, `n_bins`, `n_actions`, and `gamma`. |

## Running the Project

### Python Pipeline

```bash
pip install -r requirements.txt

# Phase 1: Collect data (50K episodes with mixture policy)
python src/collect_data.py --policy mixture --n_episodes 50000 --out data/episodes.pkl

# Phase 2: Train with importance sampling
python src/train_is.py --data data/episodes.pkl --out_q results/q_table.pkl
```

### Web Visualization

```bash
cd js && npm start
# Open http://localhost:8000
```

### JS Tests

```bash
cd js && npm test
```
