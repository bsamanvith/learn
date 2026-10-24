"""0.2 — The problem of points, three ways; and Galileo's three dice.

A needs `a` more wins, B needs `b`; A wins each round with probability p.
Run: python3 subjects/probability/sims/problem_of_points.py
"""

from functools import lru_cache
from itertools import product
from math import comb
import random


def fermat(a, b, p=0.5):
    """Imagine all a+b-1 remaining rounds are played; A needs at least a of them."""
    n = a + b - 1
    return sum(comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(a, n + 1))


@lru_cache(maxsize=None)
def pascal(a, b, p=0.5):
    """Value of the position = average of the two positions one round later."""
    if a == 0:
        return 1.0
    if b == 0:
        return 0.0
    return p * pascal(a - 1, b, p) + (1 - p) * pascal(a, b - 1, p)


def simulate(a, b, p=0.5, trials=100_000, seed=1):
    rng = random.Random(seed)
    wins = 0
    for _ in range(trials):
        x, y = a, b
        while x and y:
            if rng.random() < p:
                x -= 1
            else:
                y -= 1
        wins += x == 0
    return wins / trials


print("Share of the pot owed to A (A needs a, B needs b):")
print(f"  {'a':>2} {'b':>2} {'p':>4}  {'Fermat':>8} {'Pascal':>8} {'simulated':>9}")
for a, b, p in [(1, 2, 0.5), (1, 3, 0.5), (2, 3, 0.5), (3, 5, 0.5), (2, 2, 0.6)]:
    f, q, s = fermat(a, b, p), pascal(a, b, p), simulate(a, b, p)
    assert abs(f - q) < 1e-12
    print(f"  {a:>2} {b:>2} {p:>4}  {f:8.4f} {q:8.4f} {s:9.4f}")
print("  The historic case (a=1, b=2): A gets 3/4 of the pot, i.e. 3:1.")

print("\nGalileo: totals of three dice, counted over 6^3 = 216 ordered rolls")
counts = {9: 0, 10: 0}
for roll in product(range(1, 7), repeat=3):
    if sum(roll) in counts:
        counts[sum(roll)] += 1
print(f"  ways to make 9:  {counts[9]}   (P = {counts[9] / 216:.4f})")
print(f"  ways to make 10: {counts[10]}   (P = {counts[10] / 216:.4f})")
print("  Both totals have 6 unordered patterns, but 10 wins, 27 to 25.")
assert counts == {9: 25, 10: 27}
