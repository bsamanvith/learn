"""Problem book — the ménage problem, three ways, and how slowly it reaches e^-2.

Women sit in seats 0, 2, 4, ... (woman i in seat 2i); man j sits between women j and j+1.
Man i must avoid women i and i-1, i.e. seat j for j = i and j = i-1 (mod n).

Run: .venv/bin/python subjects/probability/sims/menage.py
"""

from itertools import combinations, permutations
from math import comb, exp, factorial
import numpy as np


def circle_choices(m, k):
    """Ways to choose k of m points on a circle with no two neighbours: m/(m-k) * C(m-k, k)."""
    return 1 if k == 0 else m * comb(m - k, k) // (m - k)


def touchard(n):
    """Seatings of the men with nobody next to his partner, by inclusion–exclusion."""
    return sum((-1) ** k * circle_choices(2 * n, k) * factorial(n - k) for k in range(n + 1))


def brute_force(n):
    return sum(all(p[j] != j and p[j] != (j + 1) % n for j in range(n)) for p in permutations(range(n)))


def circle_brute(m, k):
    def ok(c):
        gaps = all(b - a > 1 for a, b in zip(c, c[1:]))
        wrap = not (k > 1 and c[0] == 0 and c[-1] == m - 1)
        return gaps and wrap
    return sum(ok(c) for c in combinations(range(m), k))


# The lemma: non-adjacent choices on a circle.
for m, k in [(6, 2), (6, 3), (8, 3), (10, 4), (12, 5)]:
    assert circle_choices(m, k) == circle_brute(m, k)
print("circle lemma m/(m-k)·C(m-k,k) checked by brute force")

print(f"\n{'n':>3} {'Touchard':>12} {'brute':>8} {'P = U_n/n!':>11} {'e^-2(1-1/n)':>12}")
for n in range(2, 13):
    u = touchard(n)
    bf = brute_force(n) if n <= 8 else None
    assert bf is None or bf == u
    print(f"{n:>3} {u:>12} {bf if bf is not None else '-':>8} {u / factorial(n):11.5f} {exp(-2) * (1 - 1 / n):12.5f}")
for n in (20, 50, 100):
    print(f"{n:>3} {'':>12} {'':>8} {touchard(n) / factorial(n):11.5f} {exp(-2) * (1 - 1 / n):12.5f}")
print(f"e^-2 = {exp(-2):.5f}")

# Total number of ways to seat n couples alternately at a round table with 2n numbered seats.
n = 5
print(f"\nn = {n}: 2·n!·U_n = {2 * factorial(n) * touchard(n):,} seatings out of {2 * factorial(n) ** 2:,} alternating ones")

# The tempting wrong answer: treat the men's constraints as independent.
print("\nIndependent guess (1 - 2/n)^n against the truth:")
for n in (5, 10, 20):
    print(f"  n={n:>2}  guess {(1 - 2 / n) ** n:.4f}   true {touchard(n) / factorial(n):.4f}")

# Simulation.
rng = np.random.default_rng(1)
n, N = 10, 200_000
p = np.argsort(rng.random((N, n)), axis=1)
j = np.arange(n)
ok = ((p != j) & (p != (j + 1) % n)).all(axis=1)
print(f"\nSimulated n = {n}, {N:,} random seatings: {ok.mean():.4f}  (exact {touchard(n) / factorial(n):.4f})")
