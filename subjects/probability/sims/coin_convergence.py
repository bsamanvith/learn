"""0.1 — Relative frequency settles down, and the error shrinks like 1/sqrt(n).

Run: python3 subjects/probability/sims/coin_convergence.py
"""

import numpy as np

rng = np.random.default_rng(0)
RUNS = 2000  # independent coin-flipping experiments
P = 0.5

flips = rng.random((RUNS, 100_000)) < P
running = flips.cumsum(axis=1) / np.arange(1, flips.shape[1] + 1)

print("One run's running frequency of heads:")
for n in [10, 100, 1_000, 10_000, 100_000]:
    print(f"  after {n:>7,} flips: {running[0, n - 1]:.4f}")

print("\nTypical error |freq - 0.5| across", RUNS, "runs, against the 1/sqrt(n) rule:")
print(f"  {'n':>7}  {'mean |error|':>12}  {'0.4/sqrt(n)':>11}  {'error * sqrt(n)':>15}")
for n in [10, 100, 1_000, 10_000, 100_000]:
    err = np.abs(running[:, n - 1] - P).mean()
    # E|freq - p| ~ sd * sqrt(2/pi) = 0.5/sqrt(n) * 0.798 ≈ 0.399/sqrt(n)
    print(f"  {n:>7,}  {err:12.5f}  {0.399 / np.sqrt(n):11.5f}  {err * np.sqrt(n):15.3f}")

print("\n100x more flips buys only 10x more accuracy: error * sqrt(n) stays near 0.40.")

inside = (np.abs(running[:, -1] - P) <= 2 * 0.5 / np.sqrt(100_000)).mean()
print(f"Share of runs within ±2 sd after 100,000 flips: {inside:.3f} (theory ≈ 0.954)")
