"""
policies.py
-----------
Behavior policy  — RandomPolicy (uniform over actions)
Target  policy   — GreedyPolicy (deterministic argmax over Q)

Both expose the same interface:
    select_action(state) -> int
    prob(state, action)  -> float
"""

import numpy as np


class RandomPolicy:
    """Uniform random policy over a discrete action space.

    Args:
        n_actions: Number of discrete actions.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._prob = 1.0 / n_actions

    # ------------------------------------------------------------------
    def select_action(self, state) -> int:  # state unused for random policy
        return np.random.randint(self.n_actions)

    def prob(self, state, action: int) -> float:  # noqa: ARG002
        """Return P(action | state) = 1/n_actions for all (state, action)."""
        return self._prob


class GreedyPolicy:
    """Deterministic greedy policy derived from a Q-table.

    The policy always selects the action with the highest Q-value.  Ties are
    broken by picking the lowest-index action (argmax default).

    Args:
        q_table:  numpy array of shape (*state_shape, n_actions).  Can be
                  updated in place; the policy reads the array by reference.
        n_actions: Number of discrete actions.
    """

    def __init__(self, q_table: np.ndarray, n_actions: int):
        self.q_table = q_table
        self.n_actions = n_actions

    # ------------------------------------------------------------------
    def select_action(self, state) -> int:
        return int(np.argmax(self.q_table[state]))

    def prob(self, state, action: int) -> float:
        """Return 1.0 if action is greedy, 0.0 otherwise."""
        greedy_action = int(np.argmax(self.q_table[state]))
        return 1.0 if action == greedy_action else 0.0

    def update(self, q_table: np.ndarray):
        """Point the policy at an updated Q-table (or update in-place)."""
        self.q_table = q_table


class MixturePolicy:
    """Mixture of a random policy and a velocity-based oscillation policy.

    With probability `p_smart`, selects the oscillation action (push in the
    direction of current velocity to build momentum).  With probability
    `1 - p_smart`, selects uniformly at random.

    This keeps all action probabilities strictly positive (so IS ratios are
    always finite) while occasionally producing episodes that reach the goal.

    The `vel_bin_midpoints` array maps the discretised velocity bin index to
    an approximate velocity sign (+1 or -1), used to determine the smart action.

    Args:
        n_actions:        Number of discrete actions (3 for MountainCar).
        p_smart:          Probability of using the oscillation action (default 0.5).
        vel_bin_midpoints: 1-D array mapping velocity bin → midpoint value.
                          Positive midpoint → action 2 (push right).
                          Negative midpoint → action 0 (push left).
    """

    def __init__(
        self,
        n_actions: int,
        vel_bin_midpoints: np.ndarray,
        p_smart: float = 0.5,
    ):
        self.n_actions = n_actions
        self.vel_bin_midpoints = vel_bin_midpoints
        self.p_smart = p_smart
        self.p_random = 1.0 / n_actions  # uniform component

    def _smart_action(self, vel_bin: int) -> int:
        """Return action that pushes in the direction of current velocity."""
        v = self.vel_bin_midpoints[vel_bin] if vel_bin < len(self.vel_bin_midpoints) else 0.0
        return 2 if v >= 0 else 0  # push right if moving right, else push left

    def select_action(self, state) -> int:
        vel_bin = state[1]
        if np.random.random() < self.p_smart:
            return self._smart_action(vel_bin)
        return np.random.randint(self.n_actions)

    def prob(self, state, action: int) -> float:
        """Return P(action | state) under the mixture policy."""
        vel_bin = state[1]
        smart_action = self._smart_action(vel_bin)
        # Mixture probability:  p_smart * I[a == smart] + (1-p_smart) * (1/n_actions)
        smart_contribution = self.p_smart if action == smart_action else 0.0
        random_contribution = (1.0 - self.p_smart) * self.p_random
        return smart_contribution + random_contribution
