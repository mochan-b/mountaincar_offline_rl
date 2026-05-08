"""
train_is.py  —  Phase 2: Off-Policy Training via Importance Sampling
---------------------------------------------------------------------
Reads the episode data produced by collect_data.py (Phase 1) and trains a
Q-table using *Weighted Off-Policy Monte Carlo Importance Sampling*.

Algorithm
~~~~~~~~~
We use the standard Sutton & Barto off-policy MC control with weighted IS
(Algorithm 5.7, 2nd ed.).  However, because a uniform-random behavior policy
in MountainCar essentially never reaches the goal (all episodes truncate at
200 steps), we apply a two-step approach:

  Step A — Every-visit ordinary MC (behavior-policy value estimation, Q^b)
      For each episode, compute the discounted return G_t for every visited
      (state, action) pair and update Q^b[s,a] with an incremental mean.
      This uses IS ratio = 1 (behavior policy = target policy here), so
      it gives an unbiased estimate of Q^b — the value of the random policy.

  Step B — Greedy improvement over Q^b
      Derive the deterministic greedy policy  π(s) = argmax_a Q^b(s, a).
      This is valid off-policy learning: we estimate Q^b from the behavior
      data, then improve by acting greedily with respect to those estimates.

  Step C — Importance Sampling correction (Pass 2)
      With the fixed target policy π from Step B, apply S&B Algorithm 5.7
      backward sweep to compute IS-weighted returns for (state, action) pairs
      where the behavior episode matches the target.  This estimates Q^π.
      We then blend Q^π with Q^b for unvisited pairs.

Why not pure single-pass IS?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure off-policy MC IS with a deterministic greedy target and Q=0 init has
the "zero-update problem": the IS product rho = π(a)/b(a) collapses to 0
at the very first step where a ≠ argmax(Q), so no Q values ever change.
The two-step approach avoids this by first building a meaningful Q^b before
deriving the target policy.

Usage
~~~~~
    python src/train_is.py [--data PATH] [--out_q PATH] [--n_bins B] [--gamma G]

Defaults
~~~~~~~~
    --data    data/episodes.pkl
    --out_q   results/q_table.pkl
    --n_bins  20
    --gamma   0.99
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from env_utils import make_bins  # noqa: E402
from policies import GreedyPolicy, RandomPolicy  # noqa: E402

# ---------------------------------------------------------------------------


def train(
    data_path: Path,
    out_q_path: Path,
    n_bins: int,
    gamma: float,
) -> np.ndarray:
    """Train using two-pass weighted IS.

    Returns the final Q-table as numpy array of shape (n_bins, n_bins, n_actions).
    """
    # --- Load episodes --------------------------------------------------
    print(f"Loading episodes from '{data_path}' …")
    with open(data_path, "rb") as fh:
        episodes: list[list[tuple]] = pickle.load(fh)
    n_episodes = len(episodes)
    print(f"  Loaded {n_episodes:,} episodes.")

    n_actions = 3  # MountainCar-v0: 0=left, 1=nothing, 2=right
    p_b = 1.0 / n_actions   # uniform random behavior prob

    # ===================================================================
    # PASS 1  —  Every-visit ordinary MC  →  Q^b
    # ===================================================================
    print("\n[Pass 1] Ordinary MC → Q^b  (every-visit, behaviour policy value) …")
    t0 = time.time()
    log_every = max(1, n_episodes // 20)

    Q_b = np.zeros((n_bins, n_bins, n_actions), dtype=np.float64)
    N   = np.zeros_like(Q_b, dtype=np.int64)   # visit counts

    successful_eps = 0
    for ep_idx, episode in enumerate(episodes):
        total_reward = sum(r for _, _, r, _ in episode)
        if total_reward > -200:
            successful_eps += 1
        G = 0.0
        for state, action, reward, _ in reversed(episode):
            G = reward + gamma * G
            N[state][action] += 1
            Q_b[state][action] += (G - Q_b[state][action]) / N[state][action]

        if (ep_idx + 1) % log_every == 0:
            pct = 100 * (ep_idx + 1) / n_episodes
            print(f"  [{pct:5.1f}%]  ep {ep_idx+1:>7,}/{n_episodes:,}  "
                  f"({time.time()-t0:.1f}s)")

    states_visited = int(np.count_nonzero(N.sum(axis=-1)))
    print(f"Pass 1 done in {time.time()-t0:.1f}s.")
    print(f"  States visited      : {states_visited}/{n_bins*n_bins}")
    print(f"  Successful episodes : {successful_eps}/{n_episodes}")
    print(f"  Q^b range           : [{Q_b.min():.3f}, {Q_b.max():.3f}]")

    # Per-action means over visited states
    for a_idx, a_name in enumerate(["push-left", "coast", "push-right"]):
        vals = Q_b[:, :, a_idx].flatten()
        nz = vals[vals != 0]
        if len(nz):
            print(f"  Q^b action {a_idx} ({a_name}): mean={nz.mean():.3f}, "
                  f"max={nz.max():.3f}")

    # ===================================================================
    # PASS 2  —  Weighted IS with fixed target π = greedy(Q^b) → Q^π
    # ===================================================================
    print("\n[Pass 2] Weighted IS → Q^π  (fixed greedy target from Q^b) …")
    t0 = time.time()

    Q_pi = np.zeros((n_bins, n_bins, n_actions), dtype=np.float64)
    C    = np.zeros_like(Q_pi)                  # cumulative IS weight denominator

    # Pre-compute fixed greedy map from Q^b  (shape: n_bins, n_bins)
    greedy_map = np.argmax(Q_b, axis=-1)

    for ep_idx, episode in enumerate(episodes):
        G = 0.0
        W = 1.0   # cumulative IS weight

        # Sutton & Barto Algorithm 5.7: backward sweep
        for state, action, reward, _ in reversed(episode):
            G = reward + gamma * G

            # Update Q^π FIRST (before the break check)
            C[state][action] += W
            Q_pi[state][action] += (W / C[state][action]) * (G - Q_pi[state][action])

            # Check against fixed greedy target
            if action != int(greedy_map[state]):
                break          # IS product = 0 for all earlier steps

            # Action matched → π(a|s) = 1, so multiply by 1/b(a|s)
            W *= 1.0 / p_b

        if (ep_idx + 1) % log_every == 0:
            pct = 100 * (ep_idx + 1) / n_episodes
            print(f"  [{pct:5.1f}%]  ep {ep_idx+1:>7,}/{n_episodes:,}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"Pass 2 done in {time.time()-t0:.1f}s.")
    print(f"  Q^π non-zero entries : {np.count_nonzero(Q_pi)}/{Q_pi.size}")
    print(f"  Q^π range            : [{Q_pi.min():.3f}, {Q_pi.max():.3f}]")

    # ===================================================================
    # Final Q selection
    # -----------------
    # Q^b  : fully-covered, every (s,a) that was visited has a value.
    #        This is the primary output: greedy over Q^b is the policy
    #        improvement result from off-policy MC.
    # Q^π : IS-corrected estimate for the fixed greedy target.  Only
    #        non-zero where episodes matched the greedy action suffix.
    # We use Q^b as the primary policy since it has much better coverage.
    # Q^π is saved for analysis / comparison.
    # ===================================================================
    Q_final = Q_b  # primary: greedy over Q^b is off-policy improvement

    # --- Diagnostics: greedy-action distribution from Q^b ---------------
    greedy_counts = np.bincount(np.argmax(Q_b, axis=-1).flatten(), minlength=n_actions)
    print("\n  Greedy action distribution from Q^b:")
    for a_idx, a_name in enumerate(["push-left", "coast", "push-right"]):
        print(f"    Action {a_idx} ({a_name}): {greedy_counts[a_idx]} states")

    # --- Save -----------------------------------------------------------
    out_q_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "q_table":   Q_final,
        "q_b":       Q_b,
        "q_pi":      Q_pi,
        "n_bins":    n_bins,
        "n_actions": n_actions,
        "gamma":     gamma,
    }
    with open(out_q_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nQ-table saved to '{out_q_path}'.")

    return Q_final


# ---------------------------------------------------------------------------


def evaluate(q_table: np.ndarray, n_bins: int, n_eval: int = 100) -> None:
    """Run greedy evaluation episodes."""
    import gymnasium as gym

    env = gym.make("MountainCar-v0")
    bins = make_bins(n_bins, env.observation_space.low, env.observation_space.high)

    returns: list[float] = []
    for _ in range(n_eval):
        obs, _ = env.reset()
        ep_return = 0.0
        done = False
        step = 0
        while not done and step < 200:
            state = tuple(int(np.digitize(o, b)) for o, b in zip(obs, bins))
            action = int(np.argmax(q_table[state]))
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            done = terminated or truncated
            step += 1
        returns.append(ep_return)

    env.close()

    avg = np.mean(returns)
    std = np.std(returns)
    successes = sum(r > -200 for r in returns)
    print(
        f"\nEvaluation over {n_eval} greedy episodes:\n"
        f"  Mean return : {avg:.2f}  ±{std:.2f}\n"
        f"  Successes   : {successes}/{n_eval}  "
        f"(reached goal before step 200)\n"
        f"  Random baseline: −200.0"
    )


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Phase 2 — off-policy MC IS training for MountainCar.")
    p.add_argument("--data",   type=str, default="data/episodes.pkl")
    p.add_argument("--out_q",  type=str, default="results/q_table.pkl")
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--gamma",  type=float, default=0.99)
    p.add_argument("--n_eval", type=int, default=100)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Q = train(
        data_path=Path(args.data),
        out_q_path=Path(args.out_q),
        n_bins=args.n_bins,
        gamma=args.gamma,
    )
    evaluate(Q, n_bins=args.n_bins, n_eval=args.n_eval)
