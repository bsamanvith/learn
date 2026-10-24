"""1.2 — Consequences of the axioms, checked by simulation.

Run: python3 subjects/probability/sims/axioms.py
"""

import numpy as np

rng = np.random.default_rng(0)
N = 1_000_000

# First head on an even-numbered flip: P = 1/4 + 1/16 + ... = 1/3.
first_head = rng.geometric(0.5, size=N)  # flip number of the first head
print(f"P(first head on an even flip): simulated {np.mean(first_head % 2 == 0):.4f}, exact {1/3:.4f}")

# Darts in the unit square, three random circles A, B, C.
pts = rng.random((N, 2))
circles = [((0.35, 0.45), 0.25), ((0.6, 0.5), 0.22), ((0.5, 0.7), 0.2)]
A, B, C = [((pts - c) ** 2).sum(axis=1) < r**2 for c, r in circles]
p = lambda e: e.mean()

lhs = p(A | B | C)
rhs = p(A) + p(B) + p(C) - p(A & B) - p(A & C) - p(B & C) + p(A & B & C)
print(f"\nThree circles, darts: P(A∪B∪C) = {lhs:.4f}; inclusion–exclusion gives {rhs:.4f}")
# On the same darts this is an identity of counts, not just an approximation:
assert abs(lhs - rhs) < 1e-12

print("\nUnion bound vs the truth:")
print(f"  P(A)+P(B)+P(C) = {p(A) + p(B) + p(C):.4f}  >=  P(A∪B∪C) = {lhs:.4f}")

# Union bound for rare events is nearly tight: 20 independent events, each 1%.
rare = rng.random((N, 20)) < 0.01
print(f"  20 rare events (1% each): bound 0.2000, truth {rare.any(axis=1).mean():.4f}, exact {1 - 0.99**20:.4f}")
# ...and useless for common ones.
common = rng.random((N, 20)) < 0.2
print(f"  20 common events (20% each): bound 4.0000 (!), truth {common.any(axis=1).mean():.4f}")
