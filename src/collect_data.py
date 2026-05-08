"""
collect_data.py  —  Phase 1: Data Collection
----------------------------------------------
Runs N episodes in MountainCar-v0 under a *random* (behavior) policy and
stores each trajectory to disk so that Phase 2 (train_is.py) can load and
train from them without needing to interact with the environment again.

Stored format
~~~~~~~~~~~~~
A Python list of episodes, where each episode is itself a list of steps:

    episode = [ (state, action, reward, behavior_prob), ... ]

    state         : tuple[int, int]  — discretised (pos_bin, vel_bin)
    action        : int              — 0 (push left), 1 (no push), 2 (push right)
    reward        : float            — always -1.0 per step in MountainCar
    behavior_prob : float            — P(action|state) under behavior policy

Behavior policies
~~~~~~~~~~~~~~~~~
    random   — Uniform over 3 actions.  P(a|s) = 1/3 always.
               Pure random; rarely reaches the goal in MountainCar.

    mixture  — With prob p_smart, selects a velocity-oscillation action
               (push right when velocity > 0, push left when velocity < 0);
               otherwise selects uniformly at random.
               This occasionally swings the car to the goal while keeping
               all IS probabilities well-defined and nonzero.
               Recommended for practical IS learning.

Usage
~~~~~
    python src/collect_data.py --policy mixture --n_episodes 50000 --out data/episodes.pkl

Defaults
~~~~~~~~
    --policy      mixture  (random or mixture)
    --n_episodes  50000
    --max_steps   200       (Gymnasium default for MountainCar-v0)
    --n_bins      20        (20×20 discrete state grid)
    --out         data/episodes.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Allow imports from the project root when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent))

from env_utils import discretise, make_bins  # noqa: E402
from policies import MixturePolicy, RandomPolicy  # noqa: E402


# ---------------------------------------------------------------------------

def collect(
    n_episodes: int,
    max_steps: int,
    n_bins: int,
    out_path: Path,
    policy_name: str = "mixture",
    p_smart: float = 0.5,
    seed: int | None = None,
) -> None:
    """Collect episodes under the chosen behavior policy and pickle them.

    Args:
        n_episodes:  Number of episodes to collect.
        max_steps:   Maximum steps per episode.
        n_bins:      Number of bins per dimension for state discretisation.
        out_path:    Destination .pkl file.
        policy_name: 'random' or 'mixture'.
        p_smart:     For 'mixture' policy: probability of using the
                     velocity-oscillation action (default 0.5).
        seed:        Optional random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)

    env = gym.make("MountainCar-v0")
    n_actions = env.action_space.n  # 3
    bins = make_bins(n_bins, env.observation_space.low, env.observation_space.high)

    if policy_name == "mixture":
        # Compute velocity bin midpoints for the mixture policy
        vel_low, vel_high = env.observation_space.low[1], env.observation_space.high[1]
        vel_edges = np.linspace(vel_low, vel_high, n_bins + 1)
        vel_midpoints = 0.5 * (vel_edges[:-1] + vel_edges[1:])  # shape: (n_bins,)
        behavior_policy = MixturePolicy(n_actions, vel_midpoints, p_smart=p_smart)
        print(f"Behavior policy : mixture  (p_smart={p_smart})")
    else:
        behavior_policy = RandomPolicy(n_actions)
        print("Behavior policy : random (uniform)")

    episodes: list[list[tuple]] = []
    t_start = time.time()
    n_success = 0

    print(f"Collecting {n_episodes:,} episodes …")
    log_every = max(1, n_episodes // 20)  # progress every 5 %

    for ep_idx in range(n_episodes):
        obs, _ = env.reset(seed=None)
        trajectory: list[tuple] = []
        ep_success = False

        for _ in range(max_steps):
            state = discretise(obs, bins)
            action = behavior_policy.select_action(state)
            p_b = behavior_policy.prob(state, action)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            trajectory.append((state, action, float(reward), p_b))
            obs = obs_next

            if terminated:   # reached the goal
                ep_success = True
                break
            if truncated:
                break

        if ep_success:
            n_success += 1
        episodes.append(trajectory)

        if (ep_idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            pct = 100 * (ep_idx + 1) / n_episodes
            print(f"  [{pct:5.1f}%]  episode {ep_idx + 1:>7,} / {n_episodes:,}  "
                  f"successes so far: {n_success}  ({elapsed:.1f}s elapsed)")

    env.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump(episodes, fh, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = time.time() - t_start
    avg_len = np.mean([len(e) for e in episodes])
    print(
        f"\nDone. Saved {len(episodes):,} episodes to '{out_path}'  "
        f"(avg length {avg_len:.1f} steps, {n_success} successes, {elapsed:.1f}s total)"
    )


# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 — collect MountainCar episodes under a behavior policy."
    )
    parser.add_argument("--policy",     type=str, default="mixture",
                        choices=["random", "mixture"],
                        help="Behavior policy: 'random' or 'mixture' (default: mixture)")
    parser.add_argument("--p_smart",    type=float, default=0.5,
                        help="(mixture only) prob of oscillation action (default: 0.5)")
    parser.add_argument("--n_episodes", type=int, default=50_000,
                        help="Number of episodes to collect (default: 50000)")
    parser.add_argument("--max_steps",  type=int, default=200,
                        help="Max steps per episode (default: 200)")
    parser.add_argument("--n_bins",     type=int, default=20,
                        help="Bins per dimension for state discretisation (default: 20)")
    parser.add_argument("--out",        type=str, default="data/episodes.pkl",
                        help="Output pickle file (default: data/episodes.pkl)")
    parser.add_argument("--seed",       type=int, default=None,
                        help="Optional random seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect(
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        n_bins=args.n_bins,
        out_path=Path(args.out),
        policy_name=args.policy,
        p_smart=args.p_smart,
        seed=args.seed,
    )
