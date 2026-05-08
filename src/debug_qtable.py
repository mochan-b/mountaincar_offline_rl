"""Check Q^b values and what the greedy-over-Q^b policy does."""
import pickle
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

data = pickle.load(open('results/q_table.pkl', 'rb'))
Q_b  = data['q_b']
Q_pi = data['q_pi']
Q    = data['q_table']

print("=== Q^b stats ===")
print(f"  Non-zero: {np.count_nonzero(Q_b)} / {Q_b.size}")
print(f"  Range:    [{Q_b.min():.3f}, {Q_b.max():.3f}]")

# Per-action averages for visited states
for a_name, a in [("left(0)", 0), ("none(1)", 1), ("right(2)", 2)]:
    vals = Q_b[:, :, a]
    nz = vals[vals != 0]
    if len(nz):
        print(f"  Action {a_name}: mean={nz.mean():.3f}, max={nz.max():.3f}, n={len(nz)}")

print("\n=== Greedy action distribution (from Q^b) ===")
greedy = np.argmax(Q_b, axis=-1).flatten()
for a in range(3):
    nonzero_states = np.count_nonzero(Q_b.sum(axis=-1))
    count = (greedy == a).sum()
    print(f"  Action {a}: {count} states")

print("\n=== Q^b for a sample of states (non-zero rows) ===")
nonzero_mask = Q_b.sum(axis=-1) != 0
idxs = np.argwhere(nonzero_mask)[:10]
for s in idxs:
    s_t = tuple(s)
    print(f"  state {s_t}: Q_b={Q_b[s_t]}")
