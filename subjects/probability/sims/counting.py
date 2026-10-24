"""1.3 — Poker hands and birthday rooms: exact, approximate, simulated.

Run: python3 subjects/probability/sims/counting.py
"""

from math import comb, exp, prod
import numpy as np

rng = np.random.default_rng(0)
total = comb(52, 5)
exact = {
    "full house": 13 * 4 * 12 * 6 / total,
    "two pair": comb(13, 2) * 6**2 * 44 / total,
    "one pair": 13 * 6 * comb(12, 3) * 4**3 / total,
}

# Deal hands in chunks: shuffle keys, keep the 5 smallest -> 5 distinct cards per row.
N, CHUNK = 1_000_000, 100_000
hits = {name: 0 for name in exact}
for _ in range(N // CHUNK):
    hands = np.argpartition(rng.random((CHUNK, 52)), 5, axis=1)[:, :5]
    per_rank = (hands[:, :, None] // 4 == np.arange(13)).sum(axis=1)  # cards of each rank
    top = -np.sort(-per_rank, axis=1)[:, :2]                          # two biggest groups
    hits["full house"] += np.sum((top[:, 0] == 3) & (top[:, 1] == 2))
    hits["two pair"] += np.sum((top[:, 0] == 2) & (top[:, 1] == 2))
    hits["one pair"] += np.sum((top[:, 0] == 2) & (top[:, 1] == 1))
print(f"Poker, {N:,} simulated hands:")
for name in exact:
    print(f"  {name:10}  exact {exact[name]:.5f}   simulated {hits[name] / N:.5f}")

print("\nBirthday problem: exact, e^(-pairs/365), simulated (100,000 rooms each)")
for n in [10, 23, 30, 50, 70]:
    ex = 1 - prod((365 - k) / 365 for k in range(n))
    approx = 1 - exp(-n * (n - 1) / (2 * 365))
    rooms = rng.integers(0, 365, size=(100_000, n))
    srt = np.sort(rooms, axis=1)
    sim = (np.diff(srt, axis=1) == 0).any(axis=1).mean()
    print(f"  n={n:3}  exact {ex:.4f}  approx {approx:.4f}  simulated {sim:.4f}")

print("\nDe Méré: P(at least one 6 in 4 rolls) =", round(1 - (5 / 6) ** 4, 4),
      "  P(at least one double-6 in 24 rolls of two dice) =", round(1 - (35 / 36) ** 24, 4))
